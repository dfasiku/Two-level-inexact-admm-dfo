
import numpy as np
from scipy.linalg import block_diag
from concurrent.futures import ThreadPoolExecutor
from dfo_tr2 import dfo_tr2  # Import the adapted DFO solver
import os



def dfo_solver(objective, x0, delta_tol, max_iter):
    """
    Solves the subproblem using the Trust-Region DFO method.
    """
    # Define options for the DFO solver
    options = {
        "tol_delta": delta_tol,     # Tolerance for delta
        "maxfev": max_iter,         # Maximum function evaluations
        "verbosity": 0              # Verbosity level (1 for output)
    }
    
    # Run the DFO solver
    result = dfo_tr2(objective, x0.reshape(-1, 1), options)
    
    # Return the optimized point and norm of the gradient
    return result.x.flatten(), result.normg, result.func_eval  # <<< ADDED: return func_eval as third item

def solve_xi_parallel(i, x, A, B, x_bar, z, y, rho_k, f_list, epsilon4_k_r, max_inner_inner_iter):
    
    """Parallel subproblem solver (ONLY NEW FUNCTION ADDED)"""
    def subproblem_objective(x_i):
        x_i = x_i.reshape(-1)
        x_all = x.copy()
        x_all[i] = x_i
        x_stacked = np.concatenate(x_all)
        residual = A @ x_stacked + B @ x_bar + z + y / rho_k

        tau_i = 0.000  # adjust based on sensitivity
        prox_term = (tau_i / 2) * np.linalg.norm(x_i - x[i])**2

        return f_list[i](x_i) + (rho_k / 2) * np.linalg.norm(residual) ** 2 + prox_term



        

    
    return i, dfo_solver(subproblem_objective, x[i], epsilon4_k_r, max_inner_inner_iter)
    

def two_level_inexact_admm_dfo(N, A_consesus, B, f_list, g, epsilon, c, a, max_outer_iter, max_inner_iter, max_inner_inner_iter, x, x_bar, z, y, lambda_, beta):
    """
    Implements the algorithm 
    """
    # Unpack tolerances and constants
    epsilon1, epsilon2, epsilon3, epsilon4, epsilon5 = epsilon
    c1, c2, c4, c5 = c
    a1, a2, a4, a5 = a

    # Fixed algorithm parameters
    gamma = 1.005  # Amplifying ratio for beta
    omega = 0.75  # Shrinking ratio for slack variables

    

    A = A_consesus
    history = {
        'epsilon1': [],  # e1^{k,r}
        'epsilon2': [],  # e2^{k,r}
        'epsilon3': [],  # e3^{k,r}
        'epsilon4': [],  # e4^{k,r}
        'epsilon5': [],  # e5^{k,r}
        'outer_iter': [],  # Outer iteration index k
        'inner_iter': [],   # Inner iteration index r
        'L_aug': []         # <<< ADDED: augmented Lagrangian per (k,r)
    }

    # Store the previous z for the outer iteration
    z_prev = z  # z^{k-1} for the first outer iteration

    # Track the number of outer and inner iterations
    outer_iter_count = 0
    inner_iter_count = 0
    
    # NEW: Track inner iterations per outer iteration
    inner_iter_per_outer = []

    # <<< ADDED: per-block function-evaluation counters (accumulated across all inner iterations)
    eval_counts = np.zeros(N, dtype=int)

    # Debugging: Print initial setup
    #print("\n=== Initial Setup ===")
    #print(f"N: {N}")
    print(A_consesus)
    print(B)
    print(A_consesus.shape)
    print(f"A_consesus shapes: {[A_i.shape for A_i in A_consesus]}")
    print(f"B shape: {B.shape}")
    print(f"x shapes: {[x_i.shape for x_i in x]}")
    print(f"x_bar shape: {x_bar.shape}")
    print(f"z shape: {z.shape}")
    print(f"y shape: {y.shape}")
    print(f"lambda shape: {lambda_.shape}")
   

    # Outer loop
    for k in range(max_outer_iter):
        outer_iter_count += 1  # Increment outer iteration count
        rho_k = 2 * beta  # Penalty parameter for outer iteration k
        inner_iter_this_outer = 0  # NEW: Counter for inner iterations in this outer iteration

        # Compute outer loop tolerances
        epsilon1_k = epsilon4_k = max(c1 / (a1 ** (k)), epsilon1)
        epsilon2_k = epsilon5_k = max(c2 / (a2 ** (k)), epsilon2)
        epsilon3_k = (epsilon3 / epsilon1) * epsilon1_k

        # Print outer loop information
        print(f"\n=== Outer Iteration {k + 1} ===")
        print(f"rho_k: {rho_k}")
        print(f"epsilon1_k: {epsilon1_k}, epsilon2_k: {epsilon2_k}, epsilon3_k: {epsilon3_k}")

        # Inner loop
        for r in range(max_inner_iter):
            inner_iter_count += 1  # Increment inner iteration count
            inner_iter_this_outer += 1  # NEW: Increment counter for this outer iteration

            # Store the previous values of x_bar and z
            x_bar_prev = x_bar.copy()  # x_bar^{k,r}
            z_prev = z.copy()  # z^{k,r}

            # Initialize epsilon1_k_r, epsilon2_k_r, and epsilon3_k_r for r = 0
            if r == 0:
                epsilon1_k_r = 0.2  # Placeholder value 
                epsilon2_k_r = 0.2  # Placeholder value 
                epsilon3_k_r = 0.2  # Placeholder value 
            else:
                # For r >= 1, compute e1^{k,r}, e2^{k,r}, e3^{k,r} after updates
                epsilon1_k_r = np.linalg.norm(rho_k * A.T @ (B @ x_bar + z - B @ x_bar_prev - z_prev))
                epsilon2_k_r = np.linalg.norm(rho_k * B.T @ (z - z_prev))
                epsilon3_k_r = np.linalg.norm(A @ np.hstack([x[i] for i in range(N)]) + B @ x_bar + z)
         

            # Compute epsilon4^{k,r} and epsilon5^{k,r}
            if r == 0:
                epsilon4_k_r = 0.2  
                epsilon5_k_r = 0.2  
            else:
                epsilon4_k_r = max(c4 * (epsilon1_k_r ** a4), epsilon4_k)
                              
                epsilon5_k_r=0

            
            # Solve subproblems for x_i IN PARALLEL
            with ThreadPoolExecutor(max_workers=min(N, os.cpu_count())) as executor:
                futures = []
                for i in range(N):
                    futures.append(executor.submit(
                        solve_xi_parallel,
                        i, x.copy(), A, B, x_bar.copy(), z.copy(), y.copy(),
                        rho_k, f_list, epsilon4_k_r, max_inner_inner_iter
                    ))
                
                for future in futures:
                    i, (x_i, normg_x, fev_i) = future.result()  
                    x[i] = x_i
                    eval_counts[i] += int(fev_i)                
           

            # Solve coupling problem for x_bar
            def coupling_objective(x_bar_new):
                 # Ensure x_bar_new is a 1D array
                 x_bar_new = x_bar_new.reshape(-1)

                  # Compute term1
                 term1 = g(x_bar_new)

                    # Step 1: Stack all x[i]
                 x_stacked = np.concatenate(x)

                  # Step 2: Compute A @ x and residual
                 Ax = A @ x_stacked
                 residual = Ax + B @ x_bar_new + z + y / rho_k
                              
                    # Step 3: Compute term2
                 term2 = (rho_k / 2) * np.linalg.norm(residual) ** 2

                 return term1 + term2       

            # Compute C term (needed for both approaches)
            x_stacked = np.concatenate(x)
            C = A @ x_stacked + z + y / rho_k 
            # ===== Closed-form solution (use when g(x̄)=0) =====
            x_bar = -np.linalg.solve(B.T @ B, B.T @ C) 


            # Update z and y using the correct formula
            z_new = -(rho_k / (rho_k + beta)) * (A @ np.hstack([x[i] for i in range(N)]) + B @ x_bar + y / rho_k) - (1 / (rho_k + beta)) * lambda_
            y_new = y + rho_k * (A @ np.hstack([x[i] for i in range(N)]) + B @ x_bar + z_new)

            # Update z and y for the next iteration
            z = z_new
            y = y_new

            # <<< ADDED: track the *augmented Lagrangian* after updates at (k,r)
            # L(x, x_bar, z, y; lambda, rho, beta) = sum_i f_i(x_i) + g(x_bar) + y^T r + (rho/2)||r||^2 + lambda^T z + (beta/2)||z||^2
            r_vec = A @ np.hstack([x[i] for i in range(N)]) + B @ x_bar + z
            sum_f = 0.0
            for i_block in range(N):
                sum_f += float(f_list[i_block](x[i_block]))
            L_aug_val = (
                sum_f
                + float(g(x_bar))
                + float(np.dot(y, r_vec))
                + 0.5 * float(rho_k) * float(np.linalg.norm(r_vec) ** 2)
                + float(np.dot(lambda_, z))
                + 0.5 * float(beta) * float(np.linalg.norm(z) ** 2)
            )
            if r >= 1: 
                history['L_aug'].append(L_aug_val)
            if r >= 1:
                epsilon1_k_r = np.linalg.norm(rho_k * A.T @ (B @ x_bar + z - B @ x_bar_prev - z_prev))
                epsilon2_k_r = np.linalg.norm(rho_k * B.T @ (z - z_prev))
                epsilon3_k_r = np.linalg.norm(A @ np.hstack([x[i] for i in range(N)]) + B @ x_bar + z)               
                epsilon4_k_r = max(c4 * (epsilon2_k_r ** a4), epsilon4_k)
                epsilon5_k_r=0


                # Store history for r >= 1
                history['epsilon1'].append(epsilon1_k_r)
                history['epsilon2'].append(epsilon2_k_r)
                history['epsilon3'].append(epsilon3_k_r)
                history['epsilon4'].append(epsilon4_k_r)
                history['epsilon5'].append(epsilon5_k_r)
                history['outer_iter'].append(k)
                history['inner_iter'].append(r)

            

            # Check stopping criteria for inner loop
            if (epsilon4_k_r <= epsilon4_k and epsilon5_k_r <= epsilon5_k and epsilon1_k_r <= epsilon1_k and epsilon2_k_r <= epsilon2_k and epsilon3_k_r <= epsilon3_k ) or r >= max_inner_iter - 1:
                print("\n--- Inner Loop Stopping Criteria ---")
                if r >= max_inner_inner_iter - 1:
                    print("Inner loop stopped because the maximum number of iterations was reached.")
                else:
                    print("Inner loop stopped because the tolerance inequalities were met:")
                    print(f"epsilon1_k_r ({epsilon1_k_r}) <= epsilon1_k ({epsilon1_k})")
                    print(f"epsilon2_k_r ({epsilon2_k_r}) <= epsilon2_k ({epsilon2_k})")
                    print(f"epsilon3_k_r ({epsilon3_k_r}) <= epsilon3_k ({epsilon3_k})")
                    print(f"epsilon4_k_r ({epsilon4_k_r}) <= epsilon4_k ({epsilon4_k})")
                    print(f"epsilon5_k_r ({epsilon5_k_r}) <= epsilon5_k ({epsilon5_k})")
                break

        # NEW: Store inner iteration count for this outer iteration
        inner_iter_per_outer.append(inner_iter_this_outer)

        # Update lambda and beta
        if np.linalg.norm(z) > omega * np.linalg.norm(z_prev):  # Compare with z^{k-1}
            beta = gamma * beta
        else:
            beta = beta

        # Update z_prev for the next outer iteration
        z_prev = z

       
        
        lambda_ = np.clip(lambda_ + beta * z, -500000000000.0, 5000000000000.0)  # One-line projection

        # Print inner loop information in tabular form
        print(f"\n--- Inner Iteration {r + 1} ---")
        print(f"epsilon4_k: {epsilon4_k}, epsilon5_k: {epsilon5_k}")

        if (epsilon4_k <= epsilon4 and 
            epsilon5_k <= epsilon5 and 
            0 <= 1 and 
            0 <= 1 and 
            0 <= 1) or k >= max_outer_iter - 1:
            
            print("\n--- Outer Loop Stopping Criteria ---")
            if k >= max_outer_iter - 1:
                print("Outer loop stopped because the maximum number of iterations was reached.")
            else:
                print("Outer loop stopped because the tolerance inequalities were met:")                
                print(f"epsilon4_k ({epsilon4_k:.2e}) <= epsilon4 ({epsilon4:.2e})")
                print(f"epsilon5_k ({epsilon5_k:.2e}) <= epsilon5 ({epsilon5:.2e})")
            break

    # NEW: Print iteration count table
    print("\n=== Iteration Count Table ===")
    print("Outer Iter | Inner Iters")
    print("-----------------------")
    for k, count in enumerate(inner_iter_per_outer):
        print(f"{k+1:^10}|{count:^12}")

    # <<< ADDED: compute final function-evaluation summary
    max_eval = int(np.max(eval_counts)) if eval_counts.size > 0 else 0

    # Prepare final results
    final_results = {
        'x': x,
        'x_bar': x_bar,
        'z': z,
        'y': y,
        'lambda': lambda_,
        'beta': beta,
        'inner_iter_per_outer': inner_iter_per_outer,  # NEW: Add iteration counts
        'eval_counts': eval_counts,                    # <<< ADDED: per-block totals
        'max_eval': max_eval                           # <<< ADDED: fairness metric (max across blocks)
    }

    return final_results, history, outer_iter_count, inner_iter_count


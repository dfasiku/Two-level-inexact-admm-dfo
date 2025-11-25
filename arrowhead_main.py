


import numpy as np
import matplotlib.pyplot as plt
from two_level_inexact_admm_dfo import two_level_inexact_admm_dfo as admm_dfo
import time
from scipy.linalg import block_diag
from typing import List, Dict, Callable
start_time = time.time()

import numpy as np

def define_problem(total_dim=1200, block_size=5, seed: int = 1011):
    """
    Unpadded decomposition:
      - Each full block has (block_size-1) locals + 1 shared copy.
      - The LAST block has exactly the remainder locals + 1 shared copy (no padding).
    """
   

    D = int(total_dim)
    k = block_size - 1          # locals per full block
    L = D - 1                   # total locals to distribute (x1..x_{D-1})

    full_blocks, remainder = divmod(L, k)
    block_local_sizes = [k] * full_blocks + ([remainder] if remainder > 0 else [])
    N = len(block_local_sizes)

    # --- Variable-width blocks: local_dim is the true concatenated length
    block_lengths = [s + 1 for s in block_local_sizes]   # locals + shared copy
    local_dim = int(np.sum(block_lengths))

    # --- Consensus matrices (one row per block; pick last entry of each block)
    A_consensus = np.zeros((N, local_dim))
    B = -np.ones((N, 1))  # one global scalar x_bar; -1 per constraint
    col = 0
    for j, blen in enumerate(block_lengths):
        A_consensus[j, col + blen - 1] = 1.0  # select the block's shared copy
        col += blen

    # --- Block objectives: 
    def make_fi(s_loc: int):
        def fi(x_block: np.ndarray) -> float:
            x_block = np.asarray(x_block, dtype=float).reshape(-1)
            x_shared = x_block[-1]
            xi = x_block[:-1]                 # exactly s_loc locals by construction
            # (xi^2 + x_shared^2)^2 - 4*xi + 3 summed over this block's locals
            vals = (xi*xi + x_shared*x_shared)**2 - 4.0*xi + 3.0
            return float(np.sum(vals))
        return fi

    f_list = [make_fi(s) for s in block_local_sizes]

    # --- Per-block identities match the variable block sizes
    A_L = [np.eye(blen) for blen in block_lengths]

    # --- Global function on x_bar is zero
    g = lambda x_bar: 0.0

    # --- Initializations
    np.random.seed(seed)
    q = 1          # x_bar is scalar
    p = N          # one consensus eq per block

    x = [np.zeros(blen) for blen in block_lengths]   # variable-length blocks
    x_bar = np.zeros(q)
    z = np.random.uniform(0, 1, p)
    lambda_ = np.random.uniform(0, 1, p)

    beta = 20.0
    y = -lambda_ - beta * z

    epsilon = [5e-6] * 5
    c = [1, 1, 1, 1]
    a = [2.0, 2.0, 1.5, 1.5]

    max_outer_iter = 10_000_000_000_000
    max_inner_iter = 100_000_000_000
    max_inner_inner_iter = 1_000_000_000_000

    return (N, A_consensus, B, f_list, g, epsilon, c, a,
            max_outer_iter, max_inner_iter, max_inner_inner_iter,
            x, x_bar, z, y, lambda_, beta, block_size)


def compute_objective(x_i_results, total_dim=None, use_mean_shared=False):
    """
    Reconstruct the global solution from variable-size blocks and compute
    the Arrowhead objective:
        f(x) = sum_{i=1}^{D-1} [ (x_i^2 + x_D^2)^2 - 4*x_i + 3 ].

    Parameters
    ----------
    x_i_results : list of 1D arrays
        Each entry is a block vector [locals..., shared_copy].
    total_dim : int or None
        If provided, sanity-checked against the inferred D.
        If None, D is inferred as (sum of all locals) + 1.
    use_mean_shared : bool
        If True, use the mean of block shared copies; else use the first block's
        shared copy (matches your previous behavior).

    Returns
    -------
    f_val : float
    """
    # Infer locals per block and total locals L
    block_local_sizes = []
    for blk in x_i_results:
        blk = np.asarray(blk).reshape(-1)
        s = max(0, blk.size - 1)  # locals = length minus shared copy
        block_local_sizes.append(s)
    L = sum(block_local_sizes)

    # Infer / check total dimension D
    D_inferred = L + 1
    if total_dim is None:
        D = D_inferred
    else:
        D = int(total_dim)
        if D != D_inferred:
            raise ValueError(
                f"total_dim mismatch: provided D={total_dim} but blocks imply D={D_inferred} "
                f"(sum locals {L} + 1 shared)."
            )

    # Reconstruct global x = [x_1,...,x_{D-1}, x_D]
    x_global = np.zeros(D, dtype=float)

    pos = 0
    shared_vals = []
    for blk, s in zip(x_i_results, block_local_sizes):
        blk = np.asarray(blk, dtype=float).reshape(-1)
        if s > 0:
            x_global[pos:pos+s] = blk[:s]
            pos += s
        shared_vals.append(blk[-1])

    # Choose shared value (consensus should make them equal)
    x_shared = (np.mean(shared_vals) if use_mean_shared else shared_vals[0])
    x_global[-1] = x_shared

    # Arrowhead objective
    xi = x_global[:-1]
    xn2 = x_global[-1] * x_global[-1]
    f_val = float(np.sum((xi*xi + xn2)**2 - 4.0*xi + 3.0))
    return f_val



def generate_plots(history):
    """Generates convergence plots."""
    if not history:
        print("No history data to plot")
        return

    plt.figure(figsize=(10, 8))

    # Outer iteration ids 
    outer_iters = sorted(set(history.get('outer_iter', [])))
    num_outer_iters = len(outer_iters)

    # Build a smooth color gradient per outer
    colors = []
    for i in range(num_outer_iters):
        if i < num_outer_iters / 2:
            ratio = i / max(1, (num_outer_iters / 2))
            r = 0.0 + 1.0 * ratio
            g = 0.3 + 0.5 * ratio
            b = 1.0 - 1.0 * ratio
        else:
            ratio = (i - num_outer_iters / 2) / max(1, (num_outer_iters / 2))
            r = 1.0
            g = 0.8 - 0.8 * ratio
            b = 0.0
        colors.append((r, g, b))

    # Global xmax 
    if history.get('inner_iter'):
        xmax = max(1, max(history['inner_iter']))
    else:
        xmax = 1

    # === (1,1) Augmented Lagrangian (linear y, log x), one curve per outer ===
    ax11 = plt.subplot(2, 2, 1)
    if 'L_aug' in history and history['L_aug']:

        L_all = list(history['L_aug'])

        if num_outer_iters > 0 and history.get('outer_iter'):
            # For each outer k, count epsilon entries (they are for r >= 1)
            counts_eps_per_outer = []
            for k in outer_iters:
                m_eps = sum(1 for ok in history['outer_iter'] if ok == k)  # length of eps curves for this outer
                counts_eps_per_outer.append(m_eps)

            # Slice L_all by outer; L has exactly m_eps points per outer (since we stored r>=1 only)
            pos = 0
            for m_eps, color in zip(counts_eps_per_outer, colors):
                if m_eps > 0 and pos + m_eps <= len(L_all):
                    L_seg = L_all[pos:pos + m_eps]     # length == m_eps
                    r_vals = list(range(1, m_eps + 1))  # 1..m_eps
                    ax11.plot(r_vals, L_seg, linewidth=2.0, color=color)
                    pos += m_eps
                else:
                    break
        else:
            # Fallback: plot sequentially if outer_iter info is missing
            r_vals = list(range(1, len(L_all) + 1))
            ax11.plot(r_vals, L_all, linewidth=2.0)

        ax11.set_xscale('log')
        ax11.xaxis.set_major_locator(plt.LogLocator(base=10))
        ax11.set_xlim(left=1, right=max(1, xmax))
        ax11.set_xlabel('Inner iteration ($r$)')
        ax11.set_ylabel('Augmented Lagrangian $L$')  # linear y
        #ax11.set_title('Augmented Lagrangian')
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'No Lagrangian history', ha='center', va='center')
        ax11.set_axis_off()

    # Helper for ε-plots (log–log), one curve per outer using your color schedule
    def plot_eps(ax, eps_key, y_label, title):
        for k, color in zip(outer_iters, colors):
            indices = [i for i, ok in enumerate(history['outer_iter']) if ok == k]
            if indices:
                r_vals = [history['inner_iter'][i] for i in indices]
                if eps_key in history:
                    eps_vals = [history[eps_key][i] for i in indices]
                    ax.plot(r_vals, eps_vals, color=color, linewidth=2.0)
        ax.set_xlim(left=1, right=max(1, xmax))
        ax.xaxis.set_major_locator(plt.LogLocator(base=10))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Inner iteration ($r$)')
        ax.set_ylabel(y_label)
        #ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # === (1,2) ε1 ===
    ax12 = plt.subplot(2, 2, 2)
    plot_eps(ax12, 'epsilon1', r'$\epsilon_{1}^{k,r}$', r'$\epsilon_{1}^{k,r}$')

    # === (2,1) ε2 ===
    ax21 = plt.subplot(2, 2, 3)
    plot_eps(ax21, 'epsilon2', r'$\epsilon_{2}^{k,r}$', r'$\epsilon_{2}^{k,r}$')

    # === (2,2) ε3 ===
    ax22 = plt.subplot(2, 2, 4)
    plot_eps(ax22, 'epsilon3', r'$\epsilon_{3}^{k,r}$', r'$\epsilon_{3}^{k,r}$')

    plt.tight_layout()
    plt.show()




def main():
    try:
        print("Starting main()...")     
        # Get block_size from define_problem
        N, A_consensus, B, f_list, g, epsilon, c, a, max_outer_iter, max_inner_iter, max_inner_inner_iter, x, x_bar, z, y, lambda_, beta, block_size = define_problem()

        print(f"Problem setup: {N} blocks, {block_size} variables per block")
        print(f"Total dimension: {N * (block_size - 1) + 1}")
        print("Executing algorithm...")

        
         # Capture start time of algorithm
        algo_start_time = time.time()

        final_results, history, outer_iter_count, inner_iter_count = admm_dfo(
            N, A_consensus, B, f_list, g, epsilon, c, a, max_outer_iter,
            max_inner_iter, max_inner_inner_iter, x, x_bar, z, y, lambda_, beta
        )

        # Capture end time of algorithm
        algo_end_time = time.time()
        algorithm_time = algo_end_time - algo_start_time


         # PRINT TIME BEFORE PLOTS
        print(f"\nAlgorithm execution time: {algorithm_time:.4f} seconds")




        print("\nFinal Results:")
        
        print(f"x_bar: {final_results['x_bar']}")
        print(f"x: {final_results['x']}")

        print(f"z: {final_results['z']}")  # <-- ADD THIS LINE TO PRINT z
        print(f"Iterations: {outer_iter_count} outer, {inner_iter_count} inner")



        # --- NEW: Function-evaluation accounting ---
        eval_counts = final_results.get('eval_counts', None)
        max_eval    = final_results.get('max_eval', None)

        if eval_counts is not None:
            print("\nFunction evaluations per block:")
            print(eval_counts)  # e.g., [102  88 110 ...]
            print(f"Sum across all blocks: {int(eval_counts.sum())}")
        if max_eval is not None:
            print(f"Fair cost (max across blocks): {int(max_eval)}")


       
        # Pass num_blocks and block_size to compute_objective
        obj_value = compute_objective(final_results['x'],1200)

        theoretical_min = 0
        
        print(f"\nObjective at solution: {obj_value:.6f}")
        print(f"Theoretical minimum: {theoretical_min:.6f}")
        print(f"Difference: {obj_value - theoretical_min:.6e}")

        # Check consensus constraints
        if 'x_bar' in final_results and 'x' in final_results:
            print("\nConsensus verification:")
            max_diff = 0
            for i in range(N):
                x_n_i = final_results['x'][i][-1]  # x_n from block i
                x_bar_n = final_results['x_bar'][0]  # global x̄_n
                diff = abs(x_n_i - x_bar_n)
                max_diff = max(max_diff, diff)
                print(f"Block {i}: |x_n_{i} - x̄_n| = {diff:.2e}")
            print(f"Maximum consensus error: {max_diff:.2e}")

        if history:
            print("\nGenerating convergence plots...")
            generate_plots(history)
        else:
            print("No history data available for plotting")

    except Exception as e:
        print(f"Error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print(f"Elapsed time: {time.time()-start_time:.4f} seconds")

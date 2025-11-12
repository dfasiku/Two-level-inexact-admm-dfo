
import numpy as np
import matplotlib.pyplot as plt
from two_level_inexact_admm_dfo import two_level_inexact_admm_dfo as admm_dfo
import time
from typing import List, Callable

start_time = time.time()

def define_problem(total_dim: int =500, block_size: int = 4, seed: int = 1011):
    """
    Unpadded chain decomposition of the D-dim *standard* Rosenbrock:
        F(x) = sum_{i=1}^{D-1} [ 100*(x_{i+1} - x_i^2)**2 + (1 - x_i)**2 ].

    Blocks:
      - Let k = block_size - 1 be the #new locals per full block (k >= 1).
      - Split L = D - 1 locals into chunks of size k; last chunk may be smaller.
      - Each block vector is [overlap, locals...] of length s_j + 1 (>= 2).
    """

    D = int(total_dim)
    k = block_size - 1           # locals per full block
    L = D - 1                    # locals to distribute

    # Partition locals: q full blocks of k, plus remainder r
    q, r = divmod(L, k)
    block_local_sizes: List[int] = [k] * q + ([r] if r > 0 else [])
    N = len(block_local_sizes)

    # Variable block lengths (overlap + s_j locals)
    block_lengths = [s + 1 for s in block_local_sizes]
    local_dim = int(np.sum(block_lengths))

    # Chain consensus
    num_overlaps = max(0, N - 1)
    num_constraints = 2 * num_overlaps
    A_consensus = np.zeros((num_constraints, local_dim))
    B = np.zeros((num_constraints, num_overlaps))

    # Column slices per block (cumulative offsets)
    block_slices: List[slice] = []
    col = 0
    for blen in block_lengths:
        block_slices.append(slice(col, col + blen))
        col += blen

    # For each boundary o: last(block o) = first(block o+1) = x_bar[o]
    for o in range(num_overlaps):
        sl_o = block_slices[o]
        sl_n = block_slices[o + 1]
        last_col_o  = sl_o.stop - 1
        first_col_n = sl_n.start
        A_consensus[2*o,     last_col_o]  = 1.0
        A_consensus[2*o + 1, first_col_n] = 1.0
        B[2*o, o] = -1.0
        B[2*o + 1, o] = -1.0

    # Per-block objective (uniform; no terminal (1 - v_last)^2)
    def make_f_block(s_loc: int) -> Callable[[np.ndarray], float]:
        def f_block(x_block: np.ndarray) -> float:
            xb = np.asarray(x_block, dtype=float).ravel()
            val = 0.0
            for t in range(xb.size - 1):
                val += 100.0 * (xb[t+1] - xb[t]**2)**2 + (1.0 - xb[t])**2
            return float(val)
        return f_block

    f_list = [make_f_block(s) for s in block_local_sizes]

    
    g = lambda x_bar: 0.0

    # Initialization
    np.random.seed(seed)
    q_bar = B.shape[1]        # N-1
    p = A_consensus.shape[0]  # 2*(N-1)

    x = [np.zeros(blen) for blen in block_lengths]
    x_bar = np.zeros(q_bar)
    z = np.zeros(p)
    lambda_ = np.zeros(p)
    beta = 20.0
    y = -lambda_ - beta * z

    # Your existing parameters
    epsilon = [1e-6] * 5
    c = [1, 1, 1, 1]
    a = [2.0, 2.0, 1.5, 1.5]
    max_outer_iter = 100000000000000
    max_inner_iter = 1000000000000000
    max_inner_inner_iter = 1000000000000000

    return (N, A_consensus, B, f_list, g, epsilon, c, a,
            max_outer_iter, max_inner_iter, max_inner_inner_iter,
            x, x_bar, z, y, lambda_, beta, block_size)

def compute_objective_rosenbrock(x_i_results, total_dim: int = None) -> float:
    """
    Reconstruct global x from variable-size chain blocks and compute the *standard* Rosenbrock:
        F = sum_{i=1}^{D-1} [ 100*(x_{i+1} - x_i**2)**2 + (1 - x_i)**2 ].
    """
    lengths = [np.asarray(b).ravel().size for b in x_i_results]
    L = sum(l - 1 for l in lengths)     # total locals
    D_inf = L + 1
    D = int(D_inf if total_dim is None else total_dim)
    if D != D_inf:
        raise ValueError(f"total_dim mismatch: provided D={total_dim}, inferred D={D_inf} from blocks.")

    # Reconstruct: take full block 0, then append each subsequent block without its first (overlap)
    x0 = np.asarray(x_i_results[0], dtype=float).ravel()
    pieces = [x0]
    for b in x_i_results[1:]:
        bb = np.asarray(b, dtype=float).ravel()
        pieces.append(bb[1:])
    xg = np.concatenate(pieces)
    if xg.size != D:
        raise RuntimeError(f"Reconstruction produced {xg.size} entries, expected D={D}.")

    f = 0.0
    for i in range(D - 1):
        f += 100.0 * (xg[i+1] - xg[i]**2)**2 + (1.0 - xg[i])**2
    return float(f)

def compute_objective_from_blocks(x_i_results, f_list) -> float:
    """Exact objective by summing block objectives (with satisfied chain constraints)."""
    return float(sum(fi(b) for fi, b in zip(f_list, x_i_results)))




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

    # Global xmax (largest inner-iteration index across all eps entries)
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

    # Helper for ε-plots (log–log)
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

        # >>> Choose your problem size here <<<
        D_request = 500
        B_request = 4

        # Build problem
        N, A_consensus, B, f_list, g, epsilon, c, a, \
        max_outer_iter, max_inner_iter, max_inner_inner_iter, \
        x, x_bar, z, y, lambda_, beta, block_size = define_problem(
            total_dim=D_request, block_size=B_request
        )

        # Show setup using the *actual* block lengths (unpadded)
        block_lengths = [len(b) for b in x]
        D_inferred = sum(bl - 1 for bl in block_lengths) + 1
        print(f"Problem setup: {N} blocks")
        print(f"Block lengths: {block_lengths}")
        print(f"Total dimension (requested): {D_request}")
        print(f"Total dimension (inferred):  {D_inferred}")
        print("Executing algorithm...")

        # Run algorithm
        algo_start_time = time.time()
        final_results, history, outer_iter_count, inner_iter_count = admm_dfo(
            N, A_consensus, B, f_list, g, epsilon, c, a,
            max_outer_iter, max_inner_iter, max_inner_inner_iter,
            x, x_bar, z, y, lambda_, beta
        )
        algo_end_time = time.time()
        algorithm_time = algo_end_time - algo_start_time
        print(f"\nAlgorithm execution time: {algorithm_time:.4f} seconds")

        # Results
        print("\nFinal Results:")
        print(f"x_bar: {final_results.get('x_bar')}")
        print(f"x: {final_results.get('x')}")
        print(f"z:     {final_results.get('z')}")
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





        # Objective (two equivalent ways)
        obj_blocks = compute_objective_from_blocks(final_results['x'], f_list)
        obj_global = compute_objective_rosenbrock(final_results['x'], total_dim=D_inferred)
        print(f"\nObjective (sum of blocks): {obj_blocks:.6f}")
        print(f"Objective (reconstructed): {obj_global:.6f}")
        print(f"Difference: {abs(obj_blocks - obj_global):.3e}")
        print(f"Theoretical minimum: 0.000000")
        print(f"Gap to 0: {obj_global - 0.0:.6e}")

        # Chain consensus verification
        if 'x_bar' in final_results and 'x' in final_results:
            print("\nConsensus verification:")
            xb = final_results['x_bar']
            X = final_results['x']
            for i in range(N - 1):
                last_var_block_i  = X[i][-1]
                first_var_block_i1 = X[i + 1][0]
                consensus_val = xb[i]
                diff1 = abs(last_var_block_i - consensus_val)
                diff2 = abs(first_var_block_i1 - consensus_val)
                print(f"Block {i}-{i+1}: max_diff = {max(diff1, diff2):.2e}")

        # Plots
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
    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")

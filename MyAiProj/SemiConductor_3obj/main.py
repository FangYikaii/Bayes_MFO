import argparse
from src.tkg_optimizer import TraceAwareKGOptimizer
from config import OUTPUT_DIR, FIGURE_DIR


def main(n_iter):
    output_dir = OUTPUT_DIR
    fig_dir = FIGURE_DIR
    optimizer = TraceAwareKGOptimizer(output_dir=output_dir, fig_dir=fig_dir, seed=42)
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    optimizer.plot_pareto_front()
    optimizer.plot_hypervolume_convergence()

    print(f"Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize formula with Trace-Aware Knowledge Gradient.")
    parser.add_argument('--n_iter', type=int, default=5,
                        help="Number of iterations for optimization (default: 5)")
    args = parser.parse_args()
    main(args.n_iter)


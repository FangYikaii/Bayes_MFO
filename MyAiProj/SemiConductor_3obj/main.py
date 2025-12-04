import argparse
import torch
from src.tkg_optimizer import TraceAwareKGOptimizer
from config import OUTPUT_DIR, FIGURE_DIR


def detect_gpus():
    """
    Detect available GPUs and return their count and names
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return gpu_count, gpu_names
    else:
        return 0, []


def main(n_iter, device=None):
    """
    Main optimization function with device support
    
    Args:
        n_iter: Number of iterations for optimization
        device: Device to use (cuda, cuda:0, cpu, or auto for automatic detection)
    """
    output_dir = OUTPUT_DIR
    fig_dir = FIGURE_DIR
    
    # Detect available GPUs
    gpu_count, gpu_names = detect_gpus()
    print(f"\n=== GPU Detection ===")
    print(f"Available GPUs: {gpu_count}")
    if gpu_count > 0:
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
    
    # Handle device selection
    if device == "auto":
        # Auto-detect device (uses GPU if available)
        final_device = None  # Let optimizer auto-detect
    elif device is None:
        # Default to auto-detect
        final_device = None
    else:
        # Use specified device
        final_device = device
    
    # Initialize and run optimizer with fallback to CPU if GPU fails
    print(f"\n=== Optimization Setup ===")
    
    try:
        # Try to use the specified device
        optimizer = TraceAwareKGOptimizer(output_dir=output_dir, fig_dir=fig_dir, seed=42, device=final_device)
        print(f"Using device: {optimizer.device}")
        
        # Run optimization
        optimizer.optimize(n_iter=n_iter, simulation_flag=True)
        
        # Generate plots
        print(f"\n=== Generating Visualizations ===")
        optimizer.plot_pareto_front()
        optimizer.plot_hypervolume_convergence()

        print(f"\n=== Optimization Complete ===")
        print(f"Total iterations: {n_iter}")
        print(f"Final hypervolume: {optimizer.hypervolume_history[-1]:.4f}")
        print(f"Total samples: {optimizer.X.shape[0]}")
        
    except RuntimeError as e:
        # Check if it's a CUDA compatibility error
        if "CUDA error" in str(e) or "no kernel image" in str(e):
            print(f"【WARNING】GPU compatibility error: {e}")
            print("【INFO】Falling back to CPU...")
            
            # Re-initialize with CPU
            optimizer = TraceAwareKGOptimizer(output_dir=output_dir, fig_dir=fig_dir, seed=42, device="cpu")
            print(f"Using device: {optimizer.device}")
            
            # Run optimization on CPU
            optimizer.optimize(n_iter=n_iter, simulation_flag=True)
            
            # Generate plots
            print(f"\n=== Generating Visualizations ===")
            optimizer.plot_pareto_front()
            optimizer.plot_hypervolume_convergence()

            print(f"\n=== Optimization Complete ===")
            print(f"Total iterations: {n_iter}")
            print(f"Final hypervolume: {optimizer.hypervolume_history[-1]:.4f}")
            print(f"Total samples: {optimizer.X.shape[0]}")
        else:
            # Re-raise other runtime errors
            raise


def main_multi_gpu(n_iter):
    """
    Main optimization function for multi-GPU usage
    
    Args:
        n_iter: Number of iterations for optimization
    """
    gpu_count, gpu_names = detect_gpus()
    
    if gpu_count == 0:
        print("No GPUs available, running on CPU")
        main(n_iter, device="cpu")
        return
    
    print(f"\n=== Multi-GPU Optimization ===")
    print(f"Using {gpu_count} GPUs for parallel optimization")
    
    # For multi-GPU, we can run multiple optimizations in parallel on different GPUs
    # or implement data/model parallelism
    # Currently, we'll run sequential optimizations on different GPUs for demonstration
    for gpu_idx in range(gpu_count):
        device = f"cuda:{gpu_idx}"
        print(f"\n--- Running optimization on {device} ({gpu_names[gpu_idx]}) ---")
        
        # Create separate output directories for each GPU run
        output_dir = f"{OUTPUT_DIR}/gpu_{gpu_idx}"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run optimizer on specific GPU
        optimizer = TraceAwareKGOptimizer(output_dir=output_dir, fig_dir=FIGURE_DIR, seed=42 + gpu_idx, device=device)
        optimizer.optimize(n_iter=n_iter // gpu_count, simulation_flag=True)
        optimizer.plot_pareto_front()
        optimizer.plot_hypervolume_convergence()

    print(f"\n=== Multi-GPU Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize formula with Trace-Aware Knowledge Gradient.")
    parser.add_argument('--n_iter', type=int, default=20,
                        help="Number of iterations for optimization (default: 20)")
    parser.add_argument('--device', type=str, default="auto",
                        help="Device to use: auto, cpu, cuda, or cuda:0 (default: auto)")
    parser.add_argument('--multi_gpu', action='store_true',
                        help="Use all available GPUs for parallel optimization")
    args = parser.parse_args()
    
    if args.multi_gpu:
        main_multi_gpu(args.n_iter)
    else:
        main(args.n_iter, device=args.device)


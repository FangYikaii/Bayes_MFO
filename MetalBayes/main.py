#!/usr/bin/env python3
"""
主程序：运行完整的优化流程
参考 models.py 的 main.py 实现
"""
import argparse
import os
import sys
import importlib.util
from datetime import datetime

# 添加 src 目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# 动态导入优化器类（避免 Pylance 警告）
# 先导入 TkgOptimizer 并注册到 sys.modules
tkg_spec = importlib.util.spec_from_file_location(
    "TkgOptimizer", 
    os.path.join(src_path, "TkgOptimizer.py")
)
tkg_module = importlib.util.module_from_spec(tkg_spec)
sys.modules['TkgOptimizer'] = tkg_module
tkg_spec.loader.exec_module(tkg_module)
TraceAwareKGOptimizer = tkg_module.TraceAwareKGOptimizer

# 导入 OptimizerManager（处理相对导入）
# 读取文件内容并替换相对导入
opt_mgr_path = os.path.join(src_path, "OptimizerManager.py")
with open(opt_mgr_path, 'r', encoding='utf-8') as f:
    opt_mgr_code = f.read()
# 替换相对导入为绝对导入
opt_mgr_code = opt_mgr_code.replace('from .TkgOptimizer import', 'from TkgOptimizer import')
# 创建模块并执行代码
opt_mgr_module = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("OptimizerManager", opt_mgr_path)
)
exec(opt_mgr_code, opt_mgr_module.__dict__)
sys.modules['OptimizerManager'] = opt_mgr_module
OptimizerManager = opt_mgr_module.OptimizerManager


def main(n_iter, phase='phase_1_oxide', use_manager=False, test_phase_switching=False):
    """
    主函数：运行优化流程
    
    Args:
        n_iter: 优化迭代次数
        phase: 优化阶段 ('phase_1_oxide', 'phase_1_organic', 'phase_2')
        use_manager: 是否使用 OptimizerManager（多阶段管理）或直接使用 TkgOptimizer
        test_phase_switching: 是否测试阶段切换功能（自动运行三个阶段）
    """
    # 设置输出目录
    output_dir = os.path.join(project_root, 'data', 'output')
    fig_dir = os.path.join(project_root, 'data', 'figures')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    if test_phase_switching:
        # 测试阶段切换功能
        print("=" * 80)
        print("=== Testing Phase Switching with OptimizerManager ===")
        print("=" * 80)
        
        # 创建 OptimizerManager，设置较小的最大迭代次数以便快速测试
        manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu',
            phase_1_oxide_max_iterations=2,  # 第一阶段最大迭代2次
            phase_1_organic_max_iterations=2  # 第二阶段最大迭代2次
        )
        
        print(f"\n初始阶段: {manager.current_phase}")
        print(f"Phase 1 Oxide 最大迭代次数: {manager.phase_1_oxide_max_iterations}")
        print(f"Phase 1 Organic 最大迭代次数: {manager.phase_1_organic_max_iterations}")
        
        # 运行迭代直到达到总迭代次数
        # 计算所有阶段的迭代次数总和
        def get_total_iterations():
            return sum(manager.phase_iterations.values())
        
        print(f"目标总迭代次数: {n_iter}")
        
        while True:
            # 计算当前总迭代次数
            current_total_iterations = get_total_iterations()
            
            # 如果达到目标总迭代次数，停止
            if current_total_iterations >= n_iter:
                print(f"\n已达到目标总迭代次数 {n_iter}（当前: {current_total_iterations}），停止优化")
                break
            
            current_phase = manager.current_phase
            phase_iter = manager.phase_iterations[current_phase]
            
            print(f"\n{'=' * 80}")
            print(f"当前阶段: {current_phase}")
            print(f"阶段迭代次数: {phase_iter}")
            print(f"总迭代次数: {current_total_iterations}/{n_iter}")
            print(f"{'=' * 80}")
            
            # 运行单次迭代
            result = manager.run_single_iteration(simulation_flag=True)
            
            # 更新总迭代次数
            new_total_iterations = get_total_iterations()
            
            # 显示迭代结果
            optimizer = result['optimizer']
            print(f"迭代 {result['iteration']} 完成")
            print(f"当前样本数: {optimizer.X.shape[0]}")
            print(f"超体积: {result['hypervolume']:.6f}")
            print(f"总迭代次数: {new_total_iterations}/{n_iter}")
            
            # 检查阶段切换
            if result.get('should_switch_phase', False):
                old_phase = result.get('old_phase', current_phase)
                new_phase = result.get('new_phase', manager.current_phase)
                print(f"\n【阶段切换】{old_phase} -> {new_phase}")
                
                # 如果是切换到 Phase 2，显示初始样本信息
                if new_phase == OptimizerManager.PHASE_2:
                    new_optimizer = manager.get_current_optimizer()
                    print(f"Phase 2 初始样本数: {new_optimizer.X.shape[0]}")
                    print(f"Phase 2 初始样本参数形状: {new_optimizer.X.shape}")
                    print(f"Phase 2 初始目标值形状: {new_optimizer.Y.shape}")
                    
                    # 显示前几个初始样本的参数
                    if new_optimizer.X.shape[0] > 0:
                        print(f"\nPhase 2 前3个初始样本参数:")
                        for i in range(min(3, new_optimizer.X.shape[0])):
                            print(f"  样本 {i+1}: {new_optimizer.X[i].cpu().numpy()}")
                        print(f"\nPhase 2 前3个初始样本目标值:")
                        for i in range(min(3, new_optimizer.Y.shape[0])):
                            print(f"  样本 {i+1}: Uniformity={new_optimizer.Y[i, 0]:.4f}, "
                                  f"Coverage={new_optimizer.Y[i, 1]:.4f}, "
                                  f"Adhesion={new_optimizer.Y[i, 2]:.4f}")
                
                # 阶段切换后，检查是否达到总迭代次数
                if get_total_iterations() >= n_iter:
                    print(f"\n阶段切换后已达到目标总迭代次数 {n_iter}，停止优化")
                    break
        
        # 显示最终状态
        final_total_iterations = get_total_iterations()
        print(f"\n{'=' * 80}")
        print("=== 优化完成 ===")
        print(f"{'=' * 80}")
        print(f"总迭代次数: {final_total_iterations}/{n_iter}")
        print(f"\n各阶段迭代次数:")
        for phase_name, iterations in manager.phase_iterations.items():
            print(f"  {phase_name}: {iterations} 次")
        
        print(f"\n各阶段样本数:")
        for phase_name, optimizer in manager.optimizers.items():
            print(f"  {phase_name}: {optimizer.X.shape[0]} 个样本")
        
        # 显示各阶段的最优参数
        print(f"\n各阶段最优参数:")
        for phase_name in [OptimizerManager.PHASE_1_OXIDE, OptimizerManager.PHASE_1_ORGANIC]:
            if phase_name in manager.optimizers:
                optimizer = manager.optimizers[phase_name]
                if optimizer.X.shape[0] > 0:
                    pareto_x, pareto_y = optimizer.get_pareto_front()
                    if pareto_x.shape[0] > 0:
                        print(f"\n{phase_name} 帕累托前沿 ({pareto_x.shape[0]} 个解):")
                        for i in range(min(3, pareto_x.shape[0])):
                            print(f"  解 {i+1}: 参数={pareto_x[i].cpu().numpy()}, "
                                  f"目标值={pareto_y[i].cpu().numpy()}")
    
    elif use_manager:
        # 使用 OptimizerManager（多阶段管理）
        print("=== Using OptimizerManager (Multi-phase) ===")
        manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu'  # 可以改为 'cuda' 如果有 GPU
        )
        
        # 运行指定阶段的优化
        print(f"\n=== Running optimization for phase: {phase} ===")
        optimizer = manager.get_current_optimizer()
        
        # 如果当前阶段不是目标阶段，切换到目标阶段
        if manager.current_phase != phase:
            print(f"Switching from {manager.current_phase} to {phase}")
            # 这里需要手动切换阶段（简化版本）
            manager.current_phase = phase
            optimizer = manager.get_current_optimizer()
        
        # 运行优化
        optimizer.optimize(n_iter=n_iter, simulation_flag=True)
        
    else:
        # 直接使用 TkgOptimizer（单阶段）
        print(f"=== Using TkgOptimizer directly (Phase: {phase}) ===")
        
        # 创建 OptimizerManager 以获取参数空间配置
        manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu'
        )
        
        # 获取指定阶段的参数空间
        param_space = manager._get_phase_parameter_space(phase)
        
        # 创建优化器
        optimizer = TraceAwareKGOptimizer(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu',
            phase=phase,
            param_space=param_space
        )
        
        # 运行优化
        optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    print(f"\n=== Optimization completed successfully! ===")
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization with Trace-aware Knowledge Gradient."
    )
    parser.add_argument(
        '--n_iter', 
        type=int, 
        default=5,
        help="Number of optimization iterations (default: 5)"
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='phase_1_oxide',
        choices=['phase_1_oxide', 'phase_1_organic', 'phase_2'],
        help="Optimization phase (default: phase_1_oxide)"
    )
    parser.add_argument(
        '--use_manager',
        action='store_true',
        help="Use OptimizerManager for multi-phase optimization"
    )
    parser.add_argument(
        '--test_phase_switching',
        action='store_true',
        help="Test phase switching functionality (automatically runs through all phases)"
    )
    
    args = parser.parse_args()
    main(n_iter=args.n_iter, phase=args.phase, use_manager=args.use_manager, 
         test_phase_switching=args.test_phase_switching)

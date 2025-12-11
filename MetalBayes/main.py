#!/usr/bin/env python3
"""
主程序：运行完整的优化流程
参考 models.py 的 main.py 实现
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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


def save_phase_parameter_space(optimizer, phase_name, output_dir, iteration_count):
    """
    保存阶段退出时的参数空间到 CSV 文件
    
    Args:
        optimizer: TraceAwareKGOptimizer 实例
        phase_name: 阶段名称
        output_dir: 输出目录
        iteration_count: 该阶段的迭代次数
    """
    if optimizer.X.shape[0] == 0:
        logger.warning(f"阶段 {phase_name} 没有数据可保存")
        return
    
    # 将参数数据转换为 numpy 数组
    X_cpu = optimizer.X.cpu().numpy()
    Y_cpu = optimizer.Y.cpu().numpy()
    
    # 创建 DataFrame
    data_dict = {}
    for i, param_name in enumerate(optimizer.param_names):
        data_dict[param_name] = X_cpu[:, i]
    
    # 添加目标值
    data_dict['Uniformity'] = Y_cpu[:, 0]
    data_dict['Coverage'] = Y_cpu[:, 1]
    data_dict['Adhesion'] = Y_cpu[:, 2]
    
    # 添加迭代信息
    data_dict['Phase'] = [phase_name] * X_cpu.shape[0]
    data_dict['Iteration'] = list(range(X_cpu.shape[0]))
    data_dict['Phase_Iteration_Count'] = [iteration_count] * X_cpu.shape[0]
    data_dict['Timestamp'] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * X_cpu.shape[0]
    
    df = pd.DataFrame(data_dict)
    
    # 保存到 CSV 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"phase_parameter_space_{phase_name}_{timestamp}.csv")
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logger.info(f"阶段 {phase_name} 的参数空间已保存到: {filename}")
    logger.info(f"共保存 {X_cpu.shape[0]} 个样本，{len(optimizer.param_names)} 个参数")


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
        # 分阶段执行：前20次氧化物，21-40有机物，41-50混合
        logger.info("=" * 80)
        logger.info("=== 分阶段优化执行 ===")
        logger.info("=" * 80)
        logger.info(f"总迭代次数: {n_iter}")
        logger.info(f"阶段分配:")
        logger.info(f"  - Phase 1 Oxide (氧化物): 迭代 1-20")
        logger.info(f"  - Phase 1 Organic (有机物): 迭代 21-40")
        logger.info(f"  - Phase 2 (混合): 迭代 41-{n_iter}")
        logger.info("=" * 80)
        
        # 创建 OptimizerManager
        manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu',
            phase_1_oxide_max_iterations=20,  # 第一阶段最大迭代20次
            phase_1_organic_max_iterations=20  # 第二阶段最大迭代20次
        )
        
        # 定义阶段迭代范围
        phase_ranges = {
            OptimizerManager.PHASE_1_OXIDE: (1, 20),
            OptimizerManager.PHASE_1_ORGANIC: (21, 40),
            OptimizerManager.PHASE_2: (41, n_iter)
        }
        
        # 计算所有阶段的迭代次数总和
        def get_total_iterations():
            return sum(manager.phase_iterations.values())
        
        # 执行各阶段优化
        for phase_name, (start_iter, end_iter) in phase_ranges.items():
            if start_iter > n_iter:
                logger.info(f"跳过阶段 {phase_name}（起始迭代 {start_iter} > 总迭代次数 {n_iter}）")
                continue
            
            # 切换到目标阶段
            if manager.current_phase != phase_name:
                logger.info(f"切换到阶段: {phase_name}")
                # 如果是从其他阶段切换，需要手动设置
                if phase_name == OptimizerManager.PHASE_1_ORGANIC:
                    # 从 Phase 1 Oxide 切换到 Phase 1 Organic
                    manager.current_phase = OptimizerManager.PHASE_1_ORGANIC
                    manager._initialize_optimizer(OptimizerManager.PHASE_1_ORGANIC)
                elif phase_name == OptimizerManager.PHASE_2:
                    # 切换到 Phase 2 需要特殊处理
                    # 确保 Phase 1 Organic 已经初始化
                    if OptimizerManager.PHASE_1_ORGANIC not in manager.optimizers:
                        manager.current_phase = OptimizerManager.PHASE_1_ORGANIC
                        manager._initialize_optimizer(OptimizerManager.PHASE_1_ORGANIC)
                    # 然后切换到 Phase 2
                    manager._switch_to_next_phase()  # 这会从 Phase 1 Organic 切换到 Phase 2
            
            # 计算该阶段需要执行的迭代次数
            phase_iterations_needed = min(end_iter, n_iter) - start_iter + 1
            current_phase_iterations = manager.phase_iterations[phase_name]
            iterations_to_run = phase_iterations_needed - current_phase_iterations
            
            logger.info("=" * 80)
            logger.info(f"阶段: {phase_name}")
            logger.info(f"迭代范围: {start_iter}-{min(end_iter, n_iter)}")
            logger.info(f"当前已执行: {current_phase_iterations} 次")
            logger.info(f"需要执行: {iterations_to_run} 次迭代")
            logger.info("=" * 80)
            
            # 执行该阶段的迭代
            while manager.phase_iterations[phase_name] < phase_iterations_needed:
                # 确保当前阶段正确
                if manager.current_phase != phase_name:
                    logger.warning(f"当前阶段 {manager.current_phase} 与目标阶段 {phase_name} 不一致，调整中...")
                    manager.current_phase = phase_name
                
                current_phase_iter = manager.phase_iterations[phase_name]
                global_iter = get_total_iterations() + 1
                
                logger.info(f"迭代 {global_iter}/{n_iter} - 阶段 {phase_name} 第 {current_phase_iter + 1} 次迭代")
                
                # 运行单次迭代
                result = manager.run_single_iteration(simulation_flag=True)
                
                # 显示迭代结果
                optimizer = result['optimizer']
                logger.info(f"  样本数: {optimizer.X.shape[0]}")
                logger.info(f"  超体积: {result['hypervolume']:.6f}")
                
                # 检查是否达到该阶段的迭代次数上限
                if manager.phase_iterations[phase_name] >= phase_iterations_needed:
                    logger.info(f"阶段 {phase_name} 已完成 {phase_iterations_needed} 次迭代")
                    break
            
            # 阶段结束时立即保存参数空间（使用当前阶段的优化器）
            if phase_name in manager.optimizers:
                phase_optimizer = manager.optimizers[phase_name]
                phase_iter_count = manager.phase_iterations[phase_name]
                logger.info(f"阶段 {phase_name} 结束，保存参数空间...")
                save_phase_parameter_space(phase_optimizer, phase_name, output_dir, phase_iter_count)
        
        # 显示最终状态
        final_total_iterations = get_total_iterations()
        logger.info("=" * 80)
        logger.info("=== 优化完成 ===")
        logger.info("=" * 80)
        logger.info(f"总迭代次数: {final_total_iterations}/{n_iter}")
        logger.info("各阶段迭代次数:")
        for phase_name, iterations in manager.phase_iterations.items():
            logger.info(f"  {phase_name}: {iterations} 次")
        
        logger.info("各阶段样本数:")
        for phase_name, optimizer in manager.optimizers.items():
            logger.info(f"  {phase_name}: {optimizer.X.shape[0]} 个样本")
        
        # 显示各阶段的最优参数
        logger.info("各阶段最优参数:")
        for phase_name in [OptimizerManager.PHASE_1_OXIDE, OptimizerManager.PHASE_1_ORGANIC]:
            if phase_name in manager.optimizers:
                optimizer = manager.optimizers[phase_name]
                if optimizer.X.shape[0] > 0:
                    pareto_x, pareto_y = optimizer.get_pareto_front()
                    if pareto_x.shape[0] > 0:
                        logger.info(f"{phase_name} 帕累托前沿 ({pareto_x.shape[0]} 个解):")
                        for i in range(min(3, pareto_x.shape[0])):
                            logger.info(f"  解 {i+1}: 参数={pareto_x[i].cpu().numpy()}, "
                                  f"目标值={pareto_y[i].cpu().numpy()}")
    
    elif use_manager:
        # 使用 OptimizerManager（多阶段管理）
        logger.info("=== Using OptimizerManager (Multi-phase) ===")
        manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=42,
            device='cpu'  # 可以改为 'cuda' 如果有 GPU
        )
        
        # 运行指定阶段的优化
        logger.info(f"=== Running optimization for phase: {phase} ===")
        optimizer = manager.get_current_optimizer()
        
        # 如果当前阶段不是目标阶段，切换到目标阶段
        if manager.current_phase != phase:
            logger.info(f"Switching from {manager.current_phase} to {phase}")
            # 这里需要手动切换阶段（简化版本）
            manager.current_phase = phase
            optimizer = manager.get_current_optimizer()
        
        # 运行优化
        optimizer.optimize(n_iter=n_iter, simulation_flag=True)
        
    else:
        # 直接使用 TkgOptimizer（单阶段）
        logger.info(f"=== Using TkgOptimizer directly (Phase: {phase}) ===")
        
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
    
    logger.info("=== Optimization completed successfully! ===")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Figures saved to: {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization with Trace-aware Knowledge Gradient."
    )
    parser.add_argument(
        '--n_iter', 
        type=int, 
        default=50,
        help="Number of optimization iterations (default: 50)"
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

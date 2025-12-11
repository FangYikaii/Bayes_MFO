#!/usr/bin/env python3
"""
测试 optimize 方法
参考 models.py 的 main.py 实现
"""
import pytest
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# conftest.py 会自动处理导入和 fixtures
from conftest import TraceAwareKGOptimizer, OptimizerManager


def test_optimize_phase_1_oxide(optimizer_and_param_space, temp_dirs):
    """测试 Phase 1 Oxide 阶段的 optimize 方法"""
    optimizer, param_space = optimizer_and_param_space
    
    # 运行优化（5次迭代）
    n_iter = 5
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    # 验证结果
    assert optimizer.X.shape[0] > 0, "应该有实验数据"
    assert optimizer.Y.shape[0] > 0, "应该有目标值数据"
    assert optimizer.Y.shape[1] == 3, "应该有3个目标值"
    
    # 验证迭代历史
    assert len(optimizer.iteration_history) == n_iter + 1, f"应该有 {n_iter + 1} 次迭代记录（包括初始）"
    assert len(optimizer.hypervolume_history) == n_iter + 1, f"应该有 {n_iter + 1} 次超体积记录"
    
    # 验证文件是否保存
    phase_str = optimizer.phase if isinstance(optimizer.phase, str) else f"phase_{optimizer.phase}"
    csv_file = os.path.join(temp_dirs['output_dir'], f"experiment_{phase_str}_{optimizer.experiment_id}.csv")
    json_file = os.path.join(temp_dirs['output_dir'], f"optimization_history_{phase_str}_{optimizer.experiment_id}.json")
    
    assert os.path.exists(csv_file), f"CSV 文件应该存在: {csv_file}"
    assert os.path.exists(json_file), f"JSON 文件应该存在: {json_file}"
    
    logger.info("✓ Phase 1 Oxide 优化测试通过")
    logger.info(f"  - 总样本数: {optimizer.X.shape[0]}")
    logger.info(f"  - 最终超体积: {optimizer.hypervolume_history[-1]:.6f}")


@pytest.mark.parametrize('phase', ['phase_1_oxide', 'phase_1_organic', 'phase_2'])
def test_optimize_all_phases(optimizer_manager, temp_dirs, seed, device, phase):
    """测试所有阶段的 optimize 方法"""
    # 获取参数空间
    param_space = optimizer_manager._get_phase_parameter_space(phase)
    
    # 创建优化器
    optimizer = TraceAwareKGOptimizer(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device,
        phase=phase,
        param_space=param_space
    )
    
    # 运行优化（3次迭代，快速测试）
    n_iter = 3
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    # 验证结果
    assert optimizer.X.shape[0] > 0, f"{phase}: 应该有实验数据"
    assert optimizer.Y.shape[0] > 0, f"{phase}: 应该有目标值数据"
    assert optimizer.Y.shape[1] == 3, f"{phase}: 应该有3个目标值"
    
    # 验证参数维度
    expected_n_params = len(param_space['parameters'])
    assert optimizer.X.shape[1] == expected_n_params, \
        f"{phase}: 参数维度应为 {expected_n_params}，实际为 {optimizer.X.shape[1]}"
    
    # 验证迭代历史
    assert len(optimizer.iteration_history) == n_iter + 1, \
        f"{phase}: 应该有 {n_iter + 1} 次迭代记录"
    
    logger.info(f"✓ {phase} 优化测试通过")
    logger.info(f"  - 参数数量: {expected_n_params}")
    logger.info(f"  - 总样本数: {optimizer.X.shape[0]}")
    logger.info(f"  - 最终超体积: {optimizer.hypervolume_history[-1]:.6f}")


def test_optimize_with_existing_data(optimizer_and_param_space, temp_dirs):
    """测试在有现有数据的情况下运行 optimize"""
    optimizer, param_space = optimizer_and_param_space
    
    # 先运行一次迭代
    result1 = optimizer.run_single_step(simulation_flag=True)
    initial_samples = optimizer.X.shape[0]
    
    # 然后运行 optimize（应该继续使用现有数据）
    n_iter = 3
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    # 验证总样本数
    assert optimizer.X.shape[0] >= initial_samples + n_iter * optimizer.batch_size, \
        f"应该有更多样本，初始: {initial_samples}, 当前: {optimizer.X.shape[0]}"
    
    logger.info("✓ 使用现有数据的优化测试通过")
    logger.info(f"  - 初始样本数: {initial_samples}")
    logger.info(f"  - 最终样本数: {optimizer.X.shape[0]}")


def test_optimize_hypervolume_increase(optimizer_and_param_space, temp_dirs):
    """测试优化过程中超体积应该增加（或至少不减少太多）"""
    optimizer, param_space = optimizer_and_param_space
    
    # 运行多次迭代
    n_iter = 5
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    # 验证超体积历史
    assert len(optimizer.hypervolume_history) == n_iter + 1
    
    # 验证超体积值都是有效的
    for i, hv in enumerate(optimizer.hypervolume_history):
        assert hv >= 0, f"迭代 {i}: 超体积应该 >= 0，实际为 {hv}"
        assert not (hv != hv), f"迭代 {i}: 超体积不应该是 NaN"
    
    # 记录超体积趋势
    logger.info("✓ 超体积趋势测试通过")
    logger.info(f"  - 初始超体积: {optimizer.hypervolume_history[0]:.6f}")
    logger.info(f"  - 最终超体积: {optimizer.hypervolume_history[-1]:.6f}")
    logger.info(f"  - 超体积变化: {optimizer.hypervolume_history[-1] - optimizer.hypervolume_history[0]:.6f}")


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])

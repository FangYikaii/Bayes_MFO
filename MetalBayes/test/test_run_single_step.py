#!/usr/bin/env python3

import pytest
import torch

# conftest.py 会自动处理导入和 fixtures
from conftest import TraceAwareKGOptimizer, OptimizerManager


# ==================== 测试函数 ====================

@pytest.mark.parametrize('phase', ['phase_1_oxide', 'phase_1_organic', 'phase_2'])
def test_initial_samples(optimizer_manager, temp_dirs, seed, device, phase):
    """测试不同阶段的初始样本生成（第一次调用 run_single_step）"""
    param_space = optimizer_manager._get_phase_parameter_space(phase)
    optimizer = TraceAwareKGOptimizer(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device,
        phase=phase,
        param_space=param_space
    )
    
    # 验证初始状态
    assert optimizer.X.shape[0] == 0, "初始时 X 应该为空"
    assert optimizer.Y.shape[0] == 0, "初始时 Y 应该为空"
    assert optimizer.current_iteration == 0, "初始迭代次数应该为 0"
    
    # 运行第一次迭代（应该生成初始样本）
    result = optimizer.run_single_step(simulation_flag=True)
    
    # 验证结果
    assert 'iteration' in result, "结果应包含 'iteration' 键"
    assert 'candidates' in result, "结果应包含 'candidates' 键"
    assert 'hypervolume' in result, "结果应包含 'hypervolume' 键"
    
    assert result['iteration'] == 0, f"第一次迭代应该返回 iteration=0，实际为 {result['iteration']}"
    assert optimizer.X.shape[0] == optimizer.n_init, \
        f"X 应该有 {optimizer.n_init} 个样本，实际有 {optimizer.X.shape[0]} 个"
    assert optimizer.Y.shape[0] == optimizer.n_init, \
        f"Y 应该有 {optimizer.n_init} 个样本，实际有 {optimizer.Y.shape[0]} 个"
    assert optimizer.Y.shape[1] == 3, "Y 应该有 3 个目标值"
    
    # 验证候选样本的形状
    expected_n_params = len(param_space['parameters'])
    assert result['candidates'].shape[1] == expected_n_params, \
        f"候选样本参数维度应为 {expected_n_params}，实际为 {result['candidates'].shape[1]}"
    assert result['candidates'].shape[0] == optimizer.n_init, \
        f"候选样本数量应为 {optimizer.n_init}，实际为 {result['candidates'].shape[0]}"


@pytest.mark.parametrize('n_iterations', [1, 2, 3])
def test_subsequent_iterations(optimizer_and_param_space, n_iterations):
    """测试后续迭代（参数化测试）"""
    optimizer, param_space = optimizer_and_param_space
    
    # 第一次迭代：生成初始样本
    result1 = optimizer.run_single_step(simulation_flag=True)
    initial_sample_count = optimizer.X.shape[0]
    initial_iteration = optimizer.current_iteration
    
    assert result1['iteration'] == 0, "第一次迭代应该返回 iteration=0"
    assert initial_sample_count == optimizer.n_init, \
        f"初始样本数应为 {optimizer.n_init}，实际为 {initial_sample_count}"
    
    # 后续迭代
    for i in range(n_iterations):
        result = optimizer.run_single_step(simulation_flag=True)
        
        expected_iteration = initial_iteration + i + 1
        expected_sample_count = initial_sample_count + (i + 1) * optimizer.batch_size
        
        assert result['iteration'] == expected_iteration, \
            f"第 {i+2} 次迭代应该返回 iteration={expected_iteration}，实际为 {result['iteration']}"
        assert optimizer.X.shape[0] == expected_sample_count, \
            f"第 {i+2} 次迭代后应该有 {expected_sample_count} 个样本，实际有 {optimizer.X.shape[0]} 个"
        assert optimizer.Y.shape[0] == expected_sample_count, \
            f"第 {i+2} 次迭代后应该有 {expected_sample_count} 个目标值，实际有 {optimizer.Y.shape[0]} 个"
        assert result['candidates'].shape[0] == optimizer.batch_size, \
            f"候选样本数应为 {optimizer.batch_size}，实际为 {result['candidates'].shape[0]}"


def test_return_values_structure(optimizer_and_param_space):
    """测试返回值的结构和类型"""
    optimizer, param_space = optimizer_and_param_space
    
    # 第一次迭代
    result1 = optimizer.run_single_step(simulation_flag=True)
    
    # 检查返回值的键
    required_keys = ['iteration', 'candidates', 'hypervolume']
    for key in required_keys:
        assert key in result1, f"返回值应包含 '{key}' 键"
    
    # 检查返回值类型
    assert isinstance(result1['iteration'], int), "iteration 应为整数"
    assert isinstance(result1['candidates'], torch.Tensor), "candidates 应为 torch.Tensor"
    assert isinstance(result1['hypervolume'], (float, int)), "hypervolume 应为数值"
    
    # 检查超体积值是否合理（应该 >= 0）
    assert result1['hypervolume'] >= 0, f"超体积应该 >= 0，实际为 {result1['hypervolume']}"
    
    # 第二次迭代
    result2 = optimizer.run_single_step(simulation_flag=True)
    assert result2['hypervolume'] >= 0, f"第二次迭代的超体积应该 >= 0"


def test_data_consistency(optimizer_and_param_space):
    """测试数据一致性（X 和 Y 的样本数应该一致）"""
    optimizer, param_space = optimizer_and_param_space
    
    # 多次迭代
    for i in range(3):
        result = optimizer.run_single_step(simulation_flag=True)
        
        # X 和 Y 的样本数应该一致
        assert optimizer.X.shape[0] == optimizer.Y.shape[0], \
            f"第 {i+1} 次迭代后，X 和 Y 的样本数不一致: X={optimizer.X.shape[0]}, Y={optimizer.Y.shape[0]}"
        
        # Y 应该有 3 个目标值
        assert optimizer.Y.shape[1] == 3, \
            f"Y 应该有 3 个目标值，实际有 {optimizer.Y.shape[1]} 个"
        
        # X 的参数维度应该正确
        expected_n_params = len(param_space['parameters'])
        assert optimizer.X.shape[1] == expected_n_params, \
            f"X 的参数维度应为 {expected_n_params}，实际为 {optimizer.X.shape[1]}"


def test_hypervolume_monotonicity(optimizer_and_param_space):
    """测试超体积的单调性（随着样本增加，超体积应该非递减）"""
    optimizer, param_space = optimizer_and_param_space
    
    hypervolumes = []
    
    # 多次迭代
    for i in range(3):
        result = optimizer.run_single_step(simulation_flag=True)
        hypervolumes.append(result['hypervolume'])
    
    # 超体积应该非递减（由于是模拟数据，可能相同，但不应该减少）
    for i in range(1, len(hypervolumes)):
        assert hypervolumes[i] >= hypervolumes[i-1] - 1e-6, \
            f"超体积不应该减少: {hypervolumes[i-1]} -> {hypervolumes[i]}"


def test_simulation_flag(optimizer_and_param_space):
    """测试 simulation_flag 参数"""
    optimizer, param_space = optimizer_and_param_space
    
    # 使用 simulation_flag=True（默认）
    result1 = optimizer.run_single_step(simulation_flag=True)
    assert result1['iteration'] == 0
    
    # 重置优化器
    optimizer.X = torch.empty((0, 0), dtype=torch.float64, device=optimizer.device)
    optimizer.Y = torch.empty((0, 3), dtype=torch.float64, device=optimizer.device)
    optimizer.current_iteration = 0
    
    # 使用 simulation_flag=False（应该抛出 NotImplementedError）
    with pytest.raises(NotImplementedError):
        optimizer.run_single_step(simulation_flag=False)


# ==================== 测试类（用于组织相关测试）====================

class TestRunSingleStep:
    """测试 run_single_step 函数的测试类"""
    
    def test_phase_1_oxide_initial(self, optimizer_manager, temp_dirs, seed, device):
        """测试 phase_1_oxide 阶段的初始样本生成"""
        phase = 'phase_1_oxide'
        param_space = optimizer_manager._get_phase_parameter_space(phase)
        optimizer = TraceAwareKGOptimizer(
            output_dir=temp_dirs['output_dir'],
            fig_dir=temp_dirs['fig_dir'],
            seed=seed,
            device=device,
            phase=phase,
            param_space=param_space
        )
        
        result = optimizer.run_single_step(simulation_flag=True)
        
        assert result['iteration'] == 0
        assert optimizer.X.shape[0] == optimizer.n_init
        assert optimizer.Y.shape[0] == optimizer.n_init
        assert result['candidates'].shape[0] == optimizer.n_init
    
    def test_phase_1_organic_initial(self, optimizer_manager, temp_dirs, seed, device):
        """测试 phase_1_organic 阶段的初始样本生成"""
        phase = 'phase_1_organic'
        param_space = optimizer_manager._get_phase_parameter_space(phase)
        optimizer = TraceAwareKGOptimizer(
            output_dir=temp_dirs['output_dir'],
            fig_dir=temp_dirs['fig_dir'],
            seed=seed,
            device=device,
            phase=phase,
            param_space=param_space
        )
        
        result = optimizer.run_single_step(simulation_flag=True)
        
        assert result['iteration'] == 0
        assert optimizer.X.shape[0] == optimizer.n_init
        assert optimizer.Y.shape[0] == optimizer.n_init
    
    def test_phase_2_initial(self, optimizer_manager, temp_dirs, seed, device):
        """测试 phase_2 阶段的初始样本生成"""
        phase = 'phase_2'
        param_space = optimizer_manager._get_phase_parameter_space(phase)
        optimizer = TraceAwareKGOptimizer(
            output_dir=temp_dirs['output_dir'],
            fig_dir=temp_dirs['fig_dir'],
            seed=seed,
            device=device,
            phase=phase,
            param_space=param_space
        )
        
        result = optimizer.run_single_step(simulation_flag=True)
        
        assert result['iteration'] == 0
        assert optimizer.X.shape[0] == optimizer.n_init
        assert optimizer.Y.shape[0] == optimizer.n_init
    
    def test_multiple_iterations(self, optimizer_and_param_space):
        """测试多次迭代"""
        optimizer, param_space = optimizer_and_param_space
        
        # 第一次迭代
        result1 = optimizer.run_single_step(simulation_flag=True)
        assert result1['iteration'] == 0
        
        # 第二次迭代
        result2 = optimizer.run_single_step(simulation_flag=True)
        assert result2['iteration'] == 1
        assert optimizer.X.shape[0] == optimizer.n_init + optimizer.batch_size
        
        # 第三次迭代
        result3 = optimizer.run_single_step(simulation_flag=True)
        assert result3['iteration'] == 2
        assert optimizer.X.shape[0] == optimizer.n_init + 2 * optimizer.batch_size
    
    def test_using_factory_fixture(self, optimizer_factory):
        """使用工厂 fixture 的示例"""
        optimizer, param_space, manager = optimizer_factory('phase_1_oxide')
        
        result = optimizer.run_single_step(simulation_flag=True)
        
        assert result['iteration'] == 0
        assert optimizer.X.shape[0] == optimizer.n_init
        assert result['candidates'].shape[0] == optimizer.n_init

class TestRunSingleStepIntegration:
    """集成测试：测试完整的 run_single_step 流程"""
    
    @pytest.mark.parametrize('phase', ['phase_1_oxide', 'phase_1_organic', 'phase_2'])
    def test_full_workflow(self, optimizer_manager, temp_dirs, seed, device, phase):
        """测试完整的工作流程"""
        param_space = optimizer_manager._get_phase_parameter_space(phase)
        optimizer = TraceAwareKGOptimizer(
            output_dir=temp_dirs['output_dir'],
            fig_dir=temp_dirs['fig_dir'],
            seed=seed,
            device=device,
            phase=phase,
            param_space=param_space
        )
        
        # 初始迭代
        result1 = optimizer.run_single_step(simulation_flag=True)
        assert result1['iteration'] == 0
        assert optimizer.X.shape[0] == optimizer.n_init
        
        # 后续迭代
        for i in range(2):
            result = optimizer.run_single_step(simulation_flag=True)
            expected_iteration = i + 1
            expected_samples = optimizer.n_init + (i + 1) * optimizer.batch_size
            
            assert result['iteration'] == expected_iteration
            assert optimizer.X.shape[0] == expected_samples
            assert optimizer.Y.shape[0] == expected_samples
            assert result['candidates'].shape[0] == optimizer.batch_size

#!/usr/bin/env python3
"""
pytest 单元测试：测试 generate_initial_samples 函数

使用方法:
    # 运行所有测试
    conda activate bayes
    pytest MetalBayes/test/test_generate_initial_samples.py -v
    
    # 运行特定测试
    pytest MetalBayes/test/test_generate_initial_samples.py::test_phase_samples -v
    
    # 显示详细输出
    pytest MetalBayes/test/test_generate_initial_samples.py -v -s
"""

import pytest
import torch

# conftest.py 会自动处理导入和 fixtures
from conftest import TraceAwareKGOptimizer, OptimizerManager, verify_samples


# ==================== 测试函数 ====================

@pytest.mark.parametrize('phase', ['phase_1_oxide', 'phase_1_organic', 'phase_2'])
def test_phase_samples(optimizer_manager, temp_dirs, seed, device, phase):
    """测试不同阶段的初始样本生成（参数化测试）"""
    param_space = optimizer_manager._get_phase_parameter_space(phase)
    optimizer = TraceAwareKGOptimizer(
    output_dir=temp_dirs['output_dir'],
    fig_dir=temp_dirs['fig_dir'],
        seed=seed,
    device=device,
        phase=phase,
        param_space=param_space
    )
    n_init = 5
        
    # 生成样本
    samples = optimizer.generate_initial_samples(n_init=n_init)
        
    # 验证样本形状
    assert samples.shape[0] == n_init, f"样本数量不匹配: 期望 {n_init}, 实际 {samples.shape[0]}"
    assert samples.shape[1] == len(param_space['parameters']), \
        f"参数维度不匹配: 期望 {len(param_space['parameters'])}, 实际 {samples.shape[1]}"
        
    # 验证边界约束和离散化
    verify_samples(samples, param_space)


@pytest.mark.parametrize('n_init', [1, 3, 5, 10])
def test_different_n_init(optimizer_and_param_space, n_init):
    """测试不同的初始样本数量（参数化测试）"""
    optimizer, param_space = optimizer_and_param_space
    
    samples = optimizer.generate_initial_samples(n_init=n_init)
    
    assert samples.shape[0] == n_init, f"n_init={n_init} 时样本数量不匹配"
    assert samples.shape[1] == len(param_space['parameters']), \
        f"n_init={n_init} 时参数维度不匹配"


def test_sample_uniqueness(optimizer_and_param_space):
    """测试样本的唯一性"""
    optimizer, param_space = optimizer_and_param_space
    n_init = 10
    
    samples = optimizer.generate_initial_samples(n_init=n_init)
    
    # 检查是否有重复样本
    unique_samples = torch.unique(samples, dim=0)
    assert unique_samples.shape[0] == samples.shape[0], "存在重复样本"


def test_deterministic_with_same_seed(temp_dirs, seed, device):
    """测试相同随机种子下结果的可重复性"""
    manager1 = OptimizerManager(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device
    )
    manager2 = OptimizerManager(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device
    )
    
    phase = 'phase_1_oxide'
    param_space1 = manager1._get_phase_parameter_space(phase)
    param_space2 = manager2._get_phase_parameter_space(phase)
    
    optimizer1 = TraceAwareKGOptimizer(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device,
        phase=phase,
        param_space=param_space1
    )
    optimizer2 = TraceAwareKGOptimizer(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device,
        phase=phase,
        param_space=param_space2
    )
    
    n_init = 5
    
    # 使用相同的种子生成两次样本
    samples1 = optimizer1.generate_initial_samples(n_init=n_init)
    samples2 = optimizer2.generate_initial_samples(n_init=n_init)
    
    # 应该得到相同的结果（因为种子相同）
    assert torch.allclose(samples1, samples2), "相同种子下生成的样本不一致"


def test_boundary_constraints(optimizer_and_param_space):
    """测试边界约束"""
    optimizer, param_space = optimizer_and_param_space
    n_init = 20  # 生成更多样本以更好地测试边界
    
    samples = optimizer.generate_initial_samples(n_init=n_init)
    
    for i, param_name in enumerate(param_space['parameters']):
        param_values = samples[:, i]
        min_bound = param_space['bounds'][0, i].item()
        max_bound = param_space['bounds'][1, i].item()
        
        assert torch.all(param_values >= min_bound), \
            f"{param_name}: 有值小于最小值 {min_bound}"
        assert torch.all(param_values <= max_bound), \
            f"{param_name}: 有值大于最大值 {max_bound}"
            

def test_discretization(optimizer_and_param_space):
    """测试离散化"""
    optimizer, param_space = optimizer_and_param_space
    n_init = 20
    
    samples = optimizer.generate_initial_samples(n_init=n_init)
    
    for i, param_name in enumerate(param_space['parameters']):
        param_values = samples[:, i]
        step = param_space['steps'][i].item()
        
        if step > 0:
        # 检查值是否是步长的倍数
            remainders = (param_values / step) % 1.0
            # 允许小的浮点误差
            is_discrete = torch.all(torch.abs(remainders) < 1e-6) or \
                            torch.all(torch.abs(remainders - 1.0) < 1e-6)
        assert is_discrete, \
            f"{param_name}: 值未正确离散化，步长={step}"


def test_sample_statistics(optimizer_and_param_space):
    """测试样本统计信息"""
    optimizer, param_space = optimizer_and_param_space
    n_init = 10
    
    samples = optimizer.generate_initial_samples(n_init=n_init)
    
    for i, param_name in enumerate(param_space['parameters']):
        param_values = samples[:, i]
        min_bound = param_space['bounds'][0, i].item()
        max_bound = param_space['bounds'][1, i].item()
        
        # 检查统计值是否合理
        assert param_values.min().item() >= min_bound, \
            f"{param_name}: 最小值小于边界"
        assert param_values.max().item() <= max_bound, \
            f"{param_name}: 最大值大于边界"
        assert param_values.std().item() > 0, \
            f"{param_name}: 所有值相同，缺乏多样性"


# ==================== 测试类（用于组织相关测试）====================

class TestGenerateInitialSamples:
    """测试 generate_initial_samples 函数的测试类"""
    
    def test_phase_1_oxide(self, optimizer_manager, temp_dirs, seed, device):
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
        n_init = 5
        
        samples = optimizer.generate_initial_samples(n_init=n_init)
        
        assert samples.shape[0] == n_init
        assert samples.shape[1] == len(param_space['parameters'])
        verify_samples(samples, param_space)
    
    def test_phase_1_organic(self, optimizer_manager, temp_dirs, seed, device):
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
        n_init = 5
        
        samples = optimizer.generate_initial_samples(n_init=n_init)
        
        assert samples.shape[0] == n_init
        assert samples.shape[1] == len(param_space['parameters'])
        verify_samples(samples, param_space)
    
    def test_phase_2(self, optimizer_manager, temp_dirs, seed, device):
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
        n_init = 5
        
        samples = optimizer.generate_initial_samples(n_init=n_init)
        
        assert samples.shape[0] == n_init
        assert samples.shape[1] == len(param_space['parameters'])
        verify_samples(samples, param_space)
    
    def test_using_factory_fixture(self, optimizer_factory):
        """使用工厂 fixture 的示例"""
        optimizer, param_space, manager = optimizer_factory('phase_1_oxide')
        samples = optimizer.generate_initial_samples(n_init=5)
        assert samples.shape[0] == 5
        verify_samples(samples, param_space)


class TestGenerateInitialSamplesIntegration:
    """集成测试：测试完整的初始样本生成流程"""
    
    def test_full_workflow(self, optimizer_manager, temp_dirs, seed, device):
        """测试完整的工作流程"""
        # 测试所有阶段
        for phase in ['phase_1_oxide', 'phase_1_organic', 'phase_2']:
            param_space = optimizer_manager._get_phase_parameter_space(phase)
            optimizer = TraceAwareKGOptimizer(
                output_dir=temp_dirs['output_dir'],
                fig_dir=temp_dirs['fig_dir'],
                seed=seed,
                device=device,
                phase=phase,
                param_space=param_space
            )
    
            # 生成样本
            samples = optimizer.generate_initial_samples(n_init=5)
            
            # 基本验证
            assert samples.shape[0] == 5
            assert samples.shape[1] == len(param_space['parameters'])

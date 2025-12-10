"""
pytest 配置文件：提供共享的 fixtures 和工具函数
"""
import pytest
import torch
import tempfile
import shutil
import os
import sys

# 添加 src 目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# 导入模块（处理相对导入）
import importlib.util
import types

# 先加载 TkgOptimizer
tkg_spec = importlib.util.spec_from_file_location(
    "TkgOptimizer", 
    os.path.join(src_path, "TkgOptimizer.py")
)
tkg_module = importlib.util.module_from_spec(tkg_spec)
sys.modules['TkgOptimizer'] = tkg_module
tkg_spec.loader.exec_module(tkg_module)

# 然后加载 OptimizerManager（替换相对导入）
opt_mgr_path = os.path.join(src_path, "OptimizerManager.py")
with open(opt_mgr_path, 'r', encoding='utf-8') as f:
    opt_mgr_code = f.read()
opt_mgr_code = opt_mgr_code.replace('from .TkgOptimizer import', 'from TkgOptimizer import')
opt_mgr_module = types.ModuleType('OptimizerManager')
exec(opt_mgr_code, opt_mgr_module.__dict__)
sys.modules['OptimizerManager'] = opt_mgr_module

# 导出类
TraceAwareKGOptimizer = tkg_module.TraceAwareKGOptimizer
OptimizerManager = opt_mgr_module.OptimizerManager


def verify_samples(samples: torch.Tensor, param_space: dict):
    """
    验证样本是否符合参数空间的约束
    
    Args:
        samples: 样本张量，shape 为 (n_samples, n_params)
        param_space: 参数空间字典，包含 'bounds' 和 'steps' 键
    """
    bounds = param_space['bounds']
    steps = param_space['steps']
    
    # 验证边界
    for i in range(samples.shape[1]):
        param_values = samples[:, i]
        min_bound = bounds[0, i].item()
        max_bound = bounds[1, i].item()
        
        assert torch.all(param_values >= min_bound), \
            f"参数 {i}: 有值小于最小值 {min_bound}"
        assert torch.all(param_values <= max_bound), \
            f"参数 {i}: 有值大于最大值 {max_bound}"
        
        # 验证离散化（如果步长 > 0）
        if steps[i].item() > 0:
            step = steps[i].item()
            remainders = (param_values / step) % 1.0
            is_discrete = torch.all(torch.abs(remainders) < 1e-6) or \
                         torch.all(torch.abs(remainders - 1.0) < 1e-6)
            assert is_discrete, \
                f"参数 {i}: 值未正确离散化，步长={step}"


# ==================== pytest fixtures ====================

@pytest.fixture
def seed():
    """随机种子 fixture"""
    return 42


@pytest.fixture
def device():
    """设备 fixture"""
    return 'cpu'


@pytest.fixture
def temp_dirs():
    """临时目录 fixture"""
    output_dir = tempfile.mkdtemp(prefix='test_output_')
    fig_dir = tempfile.mkdtemp(prefix='test_fig_')
    
    yield {
        'output_dir': output_dir,
        'fig_dir': fig_dir
    }
    
    # 清理
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(fig_dir, ignore_errors=True)


@pytest.fixture
def optimizer_manager(temp_dirs, seed, device):
    """OptimizerManager fixture"""
    return OptimizerManager(
        output_dir=temp_dirs['output_dir'],
        fig_dir=temp_dirs['fig_dir'],
        seed=seed,
        device=device
    )


@pytest.fixture
def optimizer_and_param_space(optimizer_manager, temp_dirs, seed, device):
    """优化器和参数空间的 fixture（默认使用 phase_1_oxide）"""
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
    return optimizer, param_space


@pytest.fixture
def optimizer_factory(optimizer_manager, temp_dirs, seed, device):
    """优化器工厂 fixture，可以根据阶段创建优化器"""
    def _factory(phase: str):
        param_space = optimizer_manager._get_phase_parameter_space(phase)
        optimizer = TraceAwareKGOptimizer(
            output_dir=temp_dirs['output_dir'],
            fig_dir=temp_dirs['fig_dir'],
            seed=seed,
            device=device,
            phase=phase,
            param_space=param_space
        )
        return optimizer, param_space, optimizer_manager
    
    return _factory

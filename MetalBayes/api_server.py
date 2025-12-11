import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import torch
import logging

# 尝试使用 colorlog，如果不可用则使用标准 logging
USE_COLORLOG = False
try:
    import colorlog  # type: ignore
    USE_COLORLOG = True
except ImportError:
    pass

def setup_logging():
    """配置带颜色的日志系统"""
    # 清除现有的处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if USE_COLORLOG:
        # 使用 colorlog 配置彩色日志
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red,bold',
                'CRITICAL': 'red,bg_white,bold',
            },
            secondary_log_colors={},
            style='%'
        ))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
    else:
        # 使用标准 logging，但添加简单的 ANSI 颜色（如果终端支持）
        class ColoredFormatter(logging.Formatter):
            """简单的彩色日志格式化器"""
            COLORS = {
                'DEBUG': '\033[36m',      # Cyan
                'INFO': '\033[32m',       # Green
                'WARNING': '\033[33m',    # Yellow
                'ERROR': '\033[31m',      # Red
                'CRITICAL': '\033[31;1m', # Bold Red
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{log_color}{record.levelname}{self.RESET}"
                return super().format(record)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# 添加项目路径以便导入模型
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# 导入 OptimizerManager
from src.OptimizerManager import OptimizerManager

# 创建FastAPI应用
app = FastAPI(title="MetalBayes Optimization API", version="1.0.0")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局优化器管理器实例
global_optimizer_manager: OptimizerManager = None
# 优化器运行状态锁
global_optimizer_running = False

# 初始化优化器请求模型
class InitOptimizerRequest(BaseModel):
    seed: int = 42
    phase_1_oxide_max_iterations: int = 5
    phase_1_organic_max_iterations: int = 5
    device: Optional[str] = None  # 如果为 None，自动检测 CUDA；可以指定 'cuda' 或 'cpu'

# 初始化优化器
@app.post("/api/init")
async def init_optimizer(request: InitOptimizerRequest = InitOptimizerRequest()):
    """初始化优化器管理器"""
    global global_optimizer_manager, global_optimizer_running
    
    try:
        # 设置输出目录
        output_dir = os.path.join(project_root, 'data', 'output')
        fig_dir = os.path.join(project_root, 'data', 'figures')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        
        # 创建新的优化器管理器实例，重置所有状态
        global_optimizer_manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=request.seed,
            device=request.device,  # 如果为 None，会自动检测 CUDA
            phase_1_oxide_max_iterations=request.phase_1_oxide_max_iterations,
            phase_1_organic_max_iterations=request.phase_1_organic_max_iterations
        )
        global_optimizer_running = False
        
        # 获取当前优化器以获取参数信息
        current_optimizer = global_optimizer_manager.get_current_optimizer()
        
        # 获取设备信息
        device_info = {
            "device": str(global_optimizer_manager.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        return {
            "success": True,
            "message": "Optimizer initialized successfully",
            "param_names": current_optimizer.param_names,
            "bounds": current_optimizer.param_bounds.cpu().numpy().tolist(),
            "phase": global_optimizer_manager.current_phase,
            "device_info": device_info
        }
    except Exception as e:
        logger.error(f"初始化优化器失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize optimizer: {str(e)}")

# 定义请求体模型
class OptimizeStepRequest(BaseModel):
    simulation_flag: bool = True
    total_iterations: int = 5  # 用于前端显示总迭代次数

def save_recommended_parameters(candidates, optimizer, phase_name, phase_iteration, global_iteration, output_dir):
    """
    保存推荐的参数到 CSV 文件（按阶段分文件保存）
    
    Args:
        candidates: 推荐的参数，torch.Tensor 或 numpy.ndarray，shape 为 (n_candidates, n_params)
        optimizer: TraceAwareKGOptimizer 实例
        phase_name: 阶段名称
        phase_iteration: 当前阶段的迭代次数
        global_iteration: 全局迭代次数
        output_dir: 输出目录
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换 candidates 为 numpy 数组
        if isinstance(candidates, torch.Tensor):
            candidates_np = candidates.cpu().numpy()
        else:
            candidates_np = np.array(candidates)
        
        # 如果没有推荐的参数，跳过保存
        if candidates_np.shape[0] == 0:
            return
        
        # 获取参数名称
        param_names = optimizer.param_names
        
        # 创建 DataFrame
        data_dict = {}
        for i, param_name in enumerate(param_names):
            if i < candidates_np.shape[1]:
                param_values = candidates_np[:, i]
                # 对于 organic_concentration，只保留两位小数
                if param_name == 'organic_concentration':
                    param_values = np.round(param_values, 2)
                data_dict[param_name] = param_values
            else:
                # 如果参数索引超出范围，填充 NaN
                data_dict[param_name] = [np.nan] * candidates_np.shape[0]
        
        # 添加元数据
        data_dict['Phase'] = [phase_name] * candidates_np.shape[0]
        data_dict['Phase_Iteration'] = [phase_iteration] * candidates_np.shape[0]
        data_dict['Global_Iteration'] = [global_iteration] * candidates_np.shape[0]
        data_dict['Timestamp'] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * candidates_np.shape[0]
        data_dict['Candidate_Index'] = list(range(candidates_np.shape[0]))
        
        df = pd.DataFrame(data_dict)
        
        # 构建文件名：按阶段分文件，追加模式
        phase_file_map = {
            OptimizerManager.PHASE_1_OXIDE: 'phase_1_oxide_recommended_params.csv',
            OptimizerManager.PHASE_1_ORGANIC: 'phase_1_organic_recommended_params.csv',
            OptimizerManager.PHASE_2: 'phase_2_recommended_params.csv'
        }
        
        filename = phase_file_map.get(phase_name, f'{phase_name}_recommended_params.csv')
        filepath = os.path.join(output_dir, filename)
        
        # 追加到 CSV 文件（如果文件已存在则追加，否则创建新文件）
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(filepath, mode='w', header=True, index=False, encoding='utf-8-sig')
            
    except Exception as e:
        logger.error(f"保存推荐参数失败: {str(e)}", exc_info=True)

# 运行单步优化迭代
@app.post("/api/optimize/step")
async def optimize_step(request: OptimizeStepRequest):
    """运行单次优化迭代"""
    global global_optimizer_manager, global_optimizer_running
    
    if global_optimizer_manager is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    if global_optimizer_running:
        raise HTTPException(status_code=400, detail="Optimizer is already running")
    
    try:
        global_optimizer_running = True
        
        # 使用 OptimizerManager 运行单次迭代
        result = global_optimizer_manager.run_single_iteration(simulation_flag=request.simulation_flag)
        
        # 获取当前优化器
        optimizer = result['optimizer']
        
        # 获取阶段信息
        current_phase = result['phase']
        phase_iteration = result['phase_iteration']
        should_switch_phase = result['should_switch_phase']
        
        # 保存推荐的参数到 CSV 文件（每次迭代后保存）
        candidates = result.get('candidates')
        if candidates is not None:
            # 判断是否是初始样本生成（iteration == 0）
            is_initial_samples = result.get('iteration', 0) == 0
            if not is_initial_samples:
                # 只有真正的迭代才保存推荐的参数（初始样本生成不算迭代）
                # 计算全局迭代次数
                global_iteration = result.get('iteration', phase_iteration)
                save_recommended_parameters(
                    candidates=candidates,
                    optimizer=optimizer,
                    phase_name=current_phase,
                    phase_iteration=phase_iteration,
                    global_iteration=global_iteration,
                    output_dir=os.path.join(project_root, 'data', 'output')
                )
        
        # 计算总迭代次数（所有阶段的迭代次数总和）
        total_iterations = sum(global_optimizer_manager.phase_iterations.values())
        
        # 转换阶段为前端期望的格式
        phase_map = {
            OptimizerManager.PHASE_1_OXIDE: 1,
            OptimizerManager.PHASE_1_ORGANIC: 1,
            OptimizerManager.PHASE_2: 2
        }
        phase_number = phase_map.get(current_phase, 1)
        
        # 获取 phase_1_subphase 信息
        phase_1_subphase = None
        if current_phase == OptimizerManager.PHASE_1_OXIDE:
            phase_1_subphase = 'oxide'
        elif current_phase == OptimizerManager.PHASE_1_ORGANIC:
            phase_1_subphase = 'organic'
        
        # 获取帕累托前沿
        pareto_x, pareto_y = optimizer.get_pareto_front()
        
        # 获取超体积历史（从当前优化器）
        hypervolume_history = optimizer.hypervolume_history
        
        # 获取迭代历史
        latest_iteration = optimizer.iteration_history[-1] if optimizer.iteration_history else None
        
        # 构建响应
        response = {
            "success": True,
            "message": f"Iteration {result['iteration']} completed",
            "iteration": result['iteration'],
            "total_samples": optimizer.X.shape[0],
            "hypervolume": result['hypervolume'],
            "current_hypervolume": result['hypervolume'],
            "hypervolume_history": hypervolume_history,
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist()
            },
            "phase": phase_number,
            "phase_1_subphase": phase_1_subphase,
            "should_switch_phase": should_switch_phase,  # 反馈前端是否应该切换阶段
            "phase_iteration": phase_iteration,  # 当前阶段的迭代次数
            "total_iterations": total_iterations,  # 所有阶段的总迭代次数
            "current_phase": current_phase  # 当前阶段名称
        }
        
        # 如果有阶段切换，添加新阶段信息
        if should_switch_phase and 'new_phase' in result:
            new_phase = result['new_phase']
            new_phase_number = phase_map.get(new_phase, 1)
            new_phase_1_subphase = None
            if new_phase == OptimizerManager.PHASE_1_OXIDE:
                new_phase_1_subphase = 'oxide'
            elif new_phase == OptimizerManager.PHASE_1_ORGANIC:
                new_phase_1_subphase = 'organic'
            
            response['new_phase'] = new_phase_number
            response['new_phase_1_subphase'] = new_phase_1_subphase
            response['old_phase'] = result.get('old_phase', current_phase)
            
            # 如果是切换到 Phase 2，添加初始样本信息
            if new_phase == OptimizerManager.PHASE_2:
                new_optimizer = global_optimizer_manager.get_current_optimizer()
                if new_optimizer.X.shape[0] > 0:
                    response['phase_2_initial_samples'] = {
                        "count": new_optimizer.X.shape[0],
                        "X": new_optimizer.X.cpu().numpy().tolist()[:5],  # 只返回前5个样本
                        "Y": new_optimizer.Y.cpu().numpy().tolist()[:5]
                    }
        
        # 添加迭代结果详情
        if latest_iteration:
            # 确保 latest_iteration 包含所有必要字段
            iteration_result = latest_iteration.copy()
            # 确保 candidates 和 Y 存在
            if 'candidates' not in iteration_result or not iteration_result['candidates']:
                iteration_result['candidates'] = result['candidates'].cpu().numpy().tolist() if hasattr(result['candidates'], 'cpu') else result['candidates']
            if 'Y' not in iteration_result or not iteration_result['Y']:
                iteration_result['Y'] = optimizer.Y.cpu().numpy().tolist()
            response['iteration_result'] = iteration_result
        else:
            # 如果没有 latest_iteration，从当前优化器状态构建
            candidates_list = result['candidates'].cpu().numpy().tolist() if hasattr(result['candidates'], 'cpu') else result['candidates']
            response['iteration_result'] = {
                "iteration": result['iteration'],
                "candidates": candidates_list,
                "X": optimizer.X.cpu().numpy().tolist(),
                "Y": optimizer.Y.cpu().numpy().tolist(),
                "hypervolume": result['hypervolume'],
                "phase": phase_number,
                "phase_1_subphase": phase_1_subphase
            }
        
        return response
        
    except Exception as e:
        logger.error(f"优化迭代失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization step failed: {str(e)}")
    finally:
        global_optimizer_running = False

# 获取优化状态
@app.get("/api/status")
async def get_status():
    """获取优化状态"""
    global global_optimizer_manager, global_optimizer_running
    if global_optimizer_manager is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        # 获取当前状态
        status = global_optimizer_manager.get_status()
        optimizer = status['optimizer']
        
        # 获取帕累托前沿
        pareto_x, pareto_y = optimizer.get_pareto_front()
        
        # 获取阶段信息
        current_phase = status['current_phase']
        phase_map = {
            OptimizerManager.PHASE_1_OXIDE: ("OXIDE ONLY", 1),
            OptimizerManager.PHASE_1_ORGANIC: ("ORGANIC OPTIMIZATION", 1),
            OptimizerManager.PHASE_2: ("HYBRID GLOBAL SEARCH", 2)
        }
        phase_str, phase_number = phase_map.get(current_phase, ("UNKNOWN", 0))
        
        # 获取 phase_1_subphase 信息
        phase_1_subphase = None
        if current_phase == OptimizerManager.PHASE_1_OXIDE:
            phase_1_subphase = 'oxide'
        elif current_phase == OptimizerManager.PHASE_1_ORGANIC:
            phase_1_subphase = 'organic'
        
        # 计算总迭代次数（所有阶段的迭代次数总和）
        total_iterations = sum(global_optimizer_manager.phase_iterations.values())
        
        # 计算多目标结果
        max_adhesion = 0.0
        max_uniformity = 0.0
        max_coverage = 0.0
        hypervolume = 0.0
        
        if optimizer.Y.shape[0] > 0:
            max_adhesion = float(optimizer.Y[:, 2].max().item())
            max_uniformity = float(optimizer.Y[:, 0].max().item())
            max_coverage = float(optimizer.Y[:, 1].max().item())
            hypervolume = float(status['hypervolume'])
        
        return {
            "success": True,
            "is_running": global_optimizer_running,
            "phase": phase_str,
            "phase_number": phase_number,
            "phase_1_subphase": phase_1_subphase,
            "iteration": status['phase_iteration'],
            "total_iterations": total_iterations,  # 所有阶段的总迭代次数
            "multi_objective_results": {
                "max_adhesion": max_adhesion,
                "max_uniformity": max_uniformity,
                "max_coverage": max_coverage,
                "hypervolume": hypervolume
            },
            "total_samples": status['total_samples'],
            "current_iteration": status['phase_iteration'],
            "hypervolume_history": optimizer.hypervolume_history,
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist()
            },
            "experiment_stats": {
                "total_experiments": optimizer.X.shape[0],
                "total_iterations": total_iterations,  # 所有阶段的总迭代次数
                "current_phase_iterations": len(optimizer.iteration_history),  # 当前阶段的迭代次数
                "phase_iterations": {  # 各阶段的迭代次数
                    phase_name: iterations 
                    for phase_name, iterations in global_optimizer_manager.phase_iterations.items()
                },
                "pareto_solutions": len(pareto_x),
                "current_phase": current_phase,
                "hypervolume": hypervolume,
                "objectives": {
                    "uniformity": {
                        "min": float(optimizer.Y[:, 0].min().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "max": float(optimizer.Y[:, 0].max().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "mean": float(optimizer.Y[:, 0].mean().item()) if optimizer.Y.shape[0] > 0 else 0.0
                    },
                    "coverage": {
                        "min": float(optimizer.Y[:, 1].min().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "max": float(optimizer.Y[:, 1].max().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "mean": float(optimizer.Y[:, 1].mean().item()) if optimizer.Y.shape[0] > 0 else 0.0
                    },
                    "adhesion": {
                        "min": float(optimizer.Y[:, 2].min().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "max": float(optimizer.Y[:, 2].max().item()) if optimizer.Y.shape[0] > 0 else 0.0,
                        "mean": float(optimizer.Y[:, 2].mean().item()) if optimizer.Y.shape[0] > 0 else 0.0
                    }
                }
            },
            "algorithm_info": {
                "name": "Trace-Aware Knowledge Gradient (taKG)",
                "acquisition_function": "qLogExpectedHypervolumeImprovement",
                "phase": current_phase,
                "hyperparameters": {
                    "batch_size": optimizer.batch_size,
                    "num_restarts": optimizer.num_restarts,
                    "raw_samples": optimizer.raw_samples,
                    "n_init": optimizer.n_init
                },
                "optimization_objectives": ["Uniformity", "Coverage", "Adhesion"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

# 获取算法策略信息
@app.get("/api/algorithm_strategy")
async def get_algorithm_strategy():
    """获取算法策略信息"""
    global global_optimizer_manager
    if global_optimizer_manager is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        current_phase = global_optimizer_manager.current_phase
        
        # 构建符合前端期望的算法策略格式
        algorithm_strategy = {
            "phases": [
                {
                    "id": 1,
                    "name": "Marginal Search (Oxide)",
                    "description": "Finding optimal parameters for the pure oxide formula.",
                    "status": "active" if current_phase == OptimizerManager.PHASE_1_OXIDE else "completed" if current_phase in [OptimizerManager.PHASE_1_ORGANIC, OptimizerManager.PHASE_2] else "pending"
                },
                {
                    "id": 2,
                    "name": "Marginal Search (Organic)",
                    "description": "Finding optimal parameters for the pure organic formula.",
                    "status": "active" if current_phase == OptimizerManager.PHASE_1_ORGANIC else "completed" if current_phase == OptimizerManager.PHASE_2 else "pending"
                },
                {
                    "id": 3,
                    "name": "Hybrid Global Search",
                    "description": "Optimizing both oxide and organic layers individually (marginals) before searching the complex Hybrid space.",
                    "status": "active" if current_phase == OptimizerManager.PHASE_2 else "pending"
                }
            ],
            "current_phase": current_phase,
            "algorithm_name": "Trace-Aware Knowledge Gradient (taKG)",
            "acquisition_function": "qLogExpectedHypervolumeImprovement"
        }
        
        return {
            "success": True,
            "algorithm_strategy": algorithm_strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm strategy: {str(e)}")

# 获取参数空间信息
@app.get("/api/parameter_space")
async def get_parameter_space():
    """获取参数空间信息"""
    global global_optimizer_manager
    if global_optimizer_manager is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        optimizer = global_optimizer_manager.get_current_optimizer()
        return {
            "success": True,
            "param_names": optimizer.param_names,
            "bounds": optimizer.param_bounds.cpu().numpy().tolist(),
            "steps": optimizer.param_steps.cpu().numpy().tolist(),
            "pH_safety_constraints": global_optimizer_manager.pH_safety_constraints
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parameter space: {str(e)}")

# 更新阶段最大迭代次数
@app.post("/api/update_max_iterations")
async def update_max_iterations(
    phase_1_oxide_max_iterations: int = Query(None, description="Phase 1 Oxide max iterations"),
    phase_1_organic_max_iterations: int = Query(None, description="Phase 1 Organic max iterations")
):
    """更新阶段的最大迭代次数（由前端动态设置）"""
    global global_optimizer_manager
    if global_optimizer_manager is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        global_optimizer_manager.update_max_iterations(
            phase_1_oxide_max_iterations=phase_1_oxide_max_iterations,
            phase_1_organic_max_iterations=phase_1_organic_max_iterations
        )
        return {
            "success": True,
            "message": "Max iterations updated successfully",
            "phase_1_oxide_max_iterations": global_optimizer_manager.phase_1_oxide_max_iterations,
            "phase_1_organic_max_iterations": global_optimizer_manager.phase_1_organic_max_iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update max iterations: {str(e)}")

# 重置优化器
@app.post("/api/reset")
async def reset_optimizer(seed: int = Query(42, description="Random seed for reproducibility")):
    """重置优化器"""
    global global_optimizer_manager
    try:
        output_dir = os.path.join(project_root, 'data', 'output')
        fig_dir = os.path.join(project_root, 'data', 'figures')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        
        global_optimizer_manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=seed
        )
        return {
            "success": True,
            "message": "Optimizer reset successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset optimizer: {str(e)}")

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)

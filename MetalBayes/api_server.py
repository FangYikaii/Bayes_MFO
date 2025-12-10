from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime

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
            phase_1_oxide_max_iterations=request.phase_1_oxide_max_iterations,
            phase_1_organic_max_iterations=request.phase_1_organic_max_iterations
        )
        global_optimizer_running = False
        
        # 获取当前优化器以获取参数信息
        current_optimizer = global_optimizer_manager.get_current_optimizer()
        
        return {
            "success": True,
            "message": "Optimizer initialized successfully",
            "param_names": current_optimizer.param_names,
            "bounds": current_optimizer.param_bounds.cpu().numpy().tolist(),
            "phase": global_optimizer_manager.current_phase
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"【ERROR】Failed to initialize optimizer: {str(e)}")
        print(f"【ERROR】Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize optimizer: {str(e)}")

# 定义请求体模型
class OptimizeStepRequest(BaseModel):
    simulation_flag: bool = True
    total_iterations: int = 5  # 用于前端显示总迭代次数

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
        import traceback
        error_trace = traceback.format_exc()
        print(f"【ERROR】Optimization step failed: {str(e)}")
        print(f"【ERROR】Traceback: {error_trace}")
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

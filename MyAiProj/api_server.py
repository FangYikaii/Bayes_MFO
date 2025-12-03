from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import subprocess
from datetime import datetime
import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import unnormalize

# 添加项目路径以便导入模型
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemiConductor_3obj'))

# 导入优化器类
from src.tkg_optimizer import TraceAwareKGOptimizer
from config import OUTPUT_DIR, FIGURE_DIR

# 创建FastAPI应用
app = FastAPI(title="Bayesian Optimization API", version="1.0.0")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局优化器实例
global_optimizer = None
# 优化器运行状态锁
global_optimizer_running = False

# 初始化优化器
@app.post("/api/init")
async def init_optimizer(seed: int = Query(42, description="Random seed for reproducibility")):
    """初始化优化器"""
    global global_optimizer, global_optimizer_running
    try:
        # 创建新的优化器实例，重置所有状态
        global_optimizer = TraceAwareKGOptimizer(output_dir=OUTPUT_DIR, fig_dir=FIGURE_DIR, seed=seed)
        global_optimizer_running = False
        return {
            "success": True,
            "message": "Optimizer initialized successfully",
            "param_names": global_optimizer.param_names,
            "bounds": global_optimizer.bounds.cpu().numpy().tolist(),
            "phase": global_optimizer.phase
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize optimizer: {str(e)}")

# 生成初始样本
@app.post("/api/generate_initial_samples")
async def generate_initial_samples(n_init: int = Query(5, description="Number of initial samples")):
    """生成初始样本"""
    global global_optimizer
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        initial_samples = global_optimizer.generate_initial_samples(n_init)
        return {
            "success": True,
            "samples": initial_samples.cpu().numpy().tolist(),
            "param_names": global_optimizer.param_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate initial samples: {str(e)}")

# 定义请求体模型
from pydantic import BaseModel

class OptimizeRequest(BaseModel):
    n_iter: int = 5
    simulation_flag: bool = True

# 运行完整优化迭代
@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    """运行完整优化过程"""
    global global_optimizer, global_optimizer_running
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    # 检查优化器是否正在运行
    if global_optimizer_running:
        raise HTTPException(status_code=400, detail="Optimizer is already running")
    
    try:
        # 设置运行状态锁
        global_optimizer_running = True
        
        # 运行优化
        global_optimizer.optimize(n_iter=request.n_iter, simulation_flag=request.simulation_flag)
        
        # 获取结果
        pareto_x, pareto_y = global_optimizer.get_pareto_front()
        
        # 生成最终的可视化结果
        try:
            # 获取generate_visualizations.py的路径
            viz_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemiConductor_3obj', 'generate_visualizations.py')
            print(f"【INFO】Running visualization script: {viz_script_path}")
            
            # 运行可视化脚本
            result = subprocess.run(
                [sys.executable, viz_script_path],
                cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemiConductor_3obj'),
                capture_output=True,
                text=True
            )
            
            print(f"【INFO】Visualization script output: {result.stdout}")
            if result.stderr:
                print(f"【WARNING】Visualization script warnings: {result.stderr}")
            
            # 调用plot_pareto_front和plot_hypervolume_convergence生成最终图片
            print("【INFO】Generating final Pareto front and hypervolume convergence plots...")
            global_optimizer.plot_pareto_front()
            global_optimizer.plot_hypervolume_convergence()
            
            print("【INFO】Visualization generation completed successfully")
        except Exception as e:
            print(f"【ERROR】Failed to generate visualizations: {str(e)}")
        
        return {
            "success": True,
            "message": f"Optimization completed with {request.n_iter} iterations",
            "total_samples": global_optimizer.X.shape[0],
            "final_hypervolume": global_optimizer.hypervolume_history[-1] if global_optimizer.hypervolume_history else 0,
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist()
            },
            "iteration_history": global_optimizer.iteration_history,
            "hypervolume_history": global_optimizer.hypervolume_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    finally:
        # 释放运行状态锁
        global_optimizer_running = False

# 获取优化状态
@app.get("/api/status")
async def get_status():
    """获取优化状态"""
    global global_optimizer, global_optimizer_running
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        pareto_x, pareto_y = global_optimizer.get_pareto_front()
        
        # 获取当前实际迭代次数
        current_iteration = global_optimizer.current_iteration
        
        # 获取实验统计信息
        experiment_stats = global_optimizer.get_experiment_stats()
        
        # 转换phase为前端期望的字符串格式
        phase_map = {
            1: "OXIDE ONLY",
            2: "ORGANIC OPTIMIZATION",
            3: "HYBRID GLOBAL SEARCH"
        }
        phase_str = phase_map.get(global_optimizer.phase, f"Phase {global_optimizer.phase}")
        
        # 计算多目标结果
        max_adhesion = float(experiment_stats["objectives"]["adhesion"]["max"]) if experiment_stats["total_experiments"] > 0 else 0.0
        max_uniformity = float(experiment_stats["objectives"]["uniformity"]["max"]) if experiment_stats["total_experiments"] > 0 else 0.0
        max_coverage = float(experiment_stats["objectives"]["coverage"]["max"]) if experiment_stats["total_experiments"] > 0 else 0.0
        hypervolume = float(experiment_stats["hypervolume"]) if experiment_stats["total_experiments"] > 0 else 0.0
        
        return {
            "success": True,
            "is_running": global_optimizer_running,
            "phase": phase_str,
            "iteration": current_iteration,
            "multi_objective_results": {
                "max_adhesion": max_adhesion,
                "max_uniformity": max_uniformity,
                "max_coverage": max_coverage,
                "hypervolume": hypervolume
            },
            "total_samples": global_optimizer.X.shape[0],
            "current_iteration": current_iteration,
            "hypervolume_history": global_optimizer.hypervolume_history,
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist()
            },
            "experiment_stats": experiment_stats,
            "algorithm_info": global_optimizer.get_algorithm_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

# 获取算法策略信息
@app.get("/api/algorithm_strategy")
async def get_algorithm_strategy():
    """获取算法策略信息"""
    global global_optimizer
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        # 构建符合前端期望的算法策略格式
        algorithm_strategy = {
            "phases": [
                {
                    "id": 1,
                    "name": "Marginal Search (Oxide)",
                    "description": "Finding optimal parameters for the pure oxide formula.",
                    "status": "active" if global_optimizer.phase == 1 else "completed" if global_optimizer.phase > 1 else "pending"
                },
                {
                    "id": 2,
                    "name": "Marginal Search (Organic)",
                    "description": "Finding optimal parameters for the pure organic formula.",
                    "status": "active" if global_optimizer.phase == 2 else "completed" if global_optimizer.phase > 2 else "pending"
                },
                {
                    "id": 3,
                    "name": "Hybrid Global Search",
                    "description": "Optimizing both oxide and organic layers individually (marginals) before searching the complex Hybrid space.",
                    "status": "active" if global_optimizer.phase == 3 else "completed" if global_optimizer.phase > 3 else "pending"
                }
            ],
            "current_phase": global_optimizer.phase,
            "algorithm_name": "Trace-Aware Knowledge Gradient (taKG)",
            "acquisition_function": "qLogExpectedHypervolumeImprovement"
        }
        
        return {
            "success": True,
            "algorithm_strategy": algorithm_strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm strategy: {str(e)}")

# 获取热力图数据
@app.get("/api/heatmap_data")
async def get_heatmap_data(
    param1: str = Query("formula", description="First parameter name"),
    param2: str = Query("concentration", description="Second parameter name"),
    n_grid: int = Query(20, description="Grid size for heatmap")
):
    """获取热力图数据"""
    global global_optimizer
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        # Get parameter indices
        if param1 not in global_optimizer.param_names:
            raise HTTPException(status_code=400, detail=f"Parameter {param1} not found")
        param1_idx = global_optimizer.param_names.index(param1)
        
        if param2 not in global_optimizer.param_names:
            raise HTTPException(status_code=400, detail=f"Parameter {param2} not found")
        param2_idx = global_optimizer.param_names.index(param2)
        
        # Generate heatmap data
        heatmap_data = global_optimizer.get_heatmap_data(param1_idx, param2_idx, n_grid)
        
        # 如果还没有实验数据，返回默认的热力图数据
        if heatmap_data is None:
            # 创建默认的热力图数据
            default_grid = torch.linspace(0, 1, n_grid, device=global_optimizer.device)
            default_heatmap = torch.zeros((n_grid, n_grid, 3), device=global_optimizer.device)
            
            heatmap_data = {
                "param1_name": param1,
                "param2_name": param2,
                "param1_grid": default_grid.cpu().numpy().tolist(),
                "param2_grid": default_grid.cpu().numpy().tolist(),
                "mean": default_heatmap.cpu().numpy().tolist(),
                "variance": default_heatmap.cpu().numpy().tolist(),
                "objectives": ["Uniformity", "Coverage", "Adhesion"]
            }
        
        # 构建符合前端期望的热力图格式
        # 处理heatmap_data["mean"]和heatmap_data["variance"]，将numpy数组转换为Python列表
        def extract_objective_data(data, objective_idx):
            """从热力图数据中提取指定目标的数据"""
            result = []
            for row in data:
                row_data = []
                for cell in row:
                    row_data.append(cell[objective_idx])
                result.append(row_data)
            return result
        
        frontend_heatmap = {
            "parameter1": {
                "name": heatmap_data["param1_name"],
                "min": min(heatmap_data["param1_grid"]),
                "max": max(heatmap_data["param1_grid"]),
                "grid": heatmap_data["param1_grid"]
            },
            "parameter2": {
                "name": heatmap_data["param2_name"],
                "min": min(heatmap_data["param2_grid"]),
                "max": max(heatmap_data["param2_grid"]),
                "grid": heatmap_data["param2_grid"]
            },
            "heatmap_data": {
                "uniformity": {
                    "mean": extract_objective_data(heatmap_data["mean"], 0),
                    "variance": extract_objective_data(heatmap_data["variance"], 0)
                },
                "coverage": {
                    "mean": extract_objective_data(heatmap_data["mean"], 1),
                    "variance": extract_objective_data(heatmap_data["variance"], 1)
                },
                "adhesion": {
                    "mean": extract_objective_data(heatmap_data["mean"], 2),
                    "variance": extract_objective_data(heatmap_data["variance"], 2)
                }
            },
            "objectives": heatmap_data["objectives"],
            "current_phase": global_optimizer.phase,
            "phase_description": "Phase 1: Simple systems (only organic or only oxide)" if global_optimizer.phase == 1 else "Phase 2: Complex systems (both organic and oxide)"
        }
        
        return {
            "success": True,
            "heatmap_data": frontend_heatmap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get heatmap data: {str(e)}")

# 获取trace数据
@app.get("/api/trace_data")
async def get_trace_data():
    """获取trace数据"""
    global global_optimizer
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        trace_data = global_optimizer.get_trace_data()
        
        # 构建符合前端期望的trace格式
        frontend_trace = {
            "iteration_history": trace_data["iteration_history"],
            "hypervolume_history": trace_data["hypervolume_history"],
            "experiment_data": trace_data["experiment_data"],
            "pareto_front": trace_data["pareto_front"],
            "current_iteration": len(trace_data["iteration_history"]),
            "total_experiments": len(trace_data["experiment_data"]["X"]),
            "objectives": trace_data["experiment_data"]["objectives"]
        }
        
        return {
            "success": True,
            "trace_data": frontend_trace
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace data: {str(e)}")

# 获取参数空间信息
@app.get("/api/parameter_space")
async def get_parameter_space():
    """获取参数空间信息"""
    global global_optimizer
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    try:
        return {
            "success": True,
            "param_names": global_optimizer.param_names,
            "bounds": global_optimizer.bounds.cpu().numpy().tolist(),
            "steps": global_optimizer.steps.cpu().numpy().tolist(),
            "pH_safety_constraints": global_optimizer.pH_safety_constraints
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get parameter space: {str(e)}")

# 重置优化器
@app.post("/api/reset")
async def reset_optimizer(seed: int = Query(42, description="Random seed for reproducibility")):
    """重置优化器"""
    global global_optimizer
    try:
        global_optimizer = TraceAwareKGOptimizer(output_dir=OUTPUT_DIR, fig_dir=FIGURE_DIR, seed=seed)
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

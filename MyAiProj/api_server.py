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
from botorch.utils.transforms import unnormalize, normalize

# 添加项目路径以便导入模型
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemiConductor_3obj'))

# 导入优化器类
# 添加了SemiConductor_3obj前缀，方便 debug 代码
from SemiConductor_3obj.src.tkg_optimizer import TraceAwareKGOptimizer
from SemiConductor_3obj.config import OUTPUT_DIR, FIGURE_DIR

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

class OptimizeStepRequest(BaseModel):
    simulation_flag: bool = True
    total_iterations: int = 5  # 用于前端显示总迭代次数

class ExperimentResultRequest(BaseModel):
    """用于提交真实实验结果"""
    candidate_id: int  # 候选点的ID或索引
    uniformity: float
    coverage: float
    adhesion: float

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

# 运行单步优化迭代
@app.post("/api/optimize/step")
async def optimize_step(request: OptimizeStepRequest):
    """运行单次优化迭代"""
    global global_optimizer, global_optimizer_running
    
    if global_optimizer is None:
        raise HTTPException(status_code=400, detail="Optimizer not initialized")
    
    if global_optimizer_running:
        raise HTTPException(status_code=400, detail="Optimizer is already running")
    
    try:
        global_optimizer_running = True
        
        # 检查是否需要生成初始样本
        if global_optimizer.X.shape[0] == 0:
            # 生成初始样本
            print("【INFO】Generating initial samples...")
            X_init = global_optimizer.generate_initial_samples()
            print(f"【INFO】Generated {X_init.shape[0]} initial samples for step 0")
            
            # 评估初始样本
            print("=== Initial Experiments ===")
            for candidate in X_init:
                candidate = candidate.unsqueeze(0)
                
                # Use Real API mode: always use simulate_experiment() for debugging
                # In production, replace this with actual experiment measurements
                if request.simulation_flag:
                    y = global_optimizer.simulate_experiment(candidate)
                else:
                    # y = global_optimizer.get_human_input(candidate)
                    # For Use Real API mode, use simulate_experiment() instead of get_human_input()
                    # This allows testing the full API flow without implementing real experiments
                    print(f"【DEBUG】Use Real API mode: using simulate_experiment() for candidate evaluation")
                    y = global_optimizer.simulate_experiment(candidate)
                
                # 处理观测值
                y_processed = global_optimizer._process_observed_values(y)
                
                # 更新数据
                global_optimizer.X = torch.cat([global_optimizer.X, candidate])
                global_optimizer.Y = torch.cat([global_optimizer.Y, y_processed])
                global_optimizer.save_experiment_data(candidate, y)
            
            # 记录初始迭代
            global_optimizer._record_iteration(iteration=0, candidates=X_init)
            global_optimizer.current_iteration = 0
            
            # 计算初始超体积
            hv = global_optimizer._compute_hypervolume()
            print(f"【INFO】Initial hypervolume: {hv:.4f}")
            
            # 获取当前迭代结果
            pareto_x, pareto_y = global_optimizer.get_pareto_front()
            latest_iteration = global_optimizer.iteration_history[-1] if global_optimizer.iteration_history else None
            
            return {
                "success": True,
                "message": "Initial samples generated and evaluated",
                "iteration": 0,
                "total_samples": global_optimizer.X.shape[0],
                "hypervolume": hv,
                "hypervolume_history": global_optimizer.hypervolume_history,
                "pareto_front": {
                    "X": pareto_x.cpu().numpy().tolist(),
                    "Y": pareto_y.cpu().numpy().tolist()
                },
                "iteration_result": latest_iteration if latest_iteration else {
                    "iteration": 0,
                    "candidates": X_init.cpu().numpy().tolist(),
                    "X": global_optimizer.X.cpu().numpy().tolist(),
                    "Y": global_optimizer.Y.cpu().numpy().tolist(),
                    "hypervolume": hv,
                    "phase": global_optimizer.phase
                },
                "phase": global_optimizer.phase
            }
        
        # 执行单步优化迭代
        print(f"\n【INFO】Running optimization step, current iteration: {global_optimizer.current_iteration}")
        
        # 更新迭代计数
        global_optimizer.current_iteration += 1
        print(f"【INFO】Iteration {global_optimizer.current_iteration}, Phase {global_optimizer.phase}")
        
        # 检查是否需要转换到阶段2
        if global_optimizer.phase == 1:
            if global_optimizer.current_iteration > 1 and len(global_optimizer.hypervolume_history) >= 2:
                # 计算超体积改进率
                hv_improvement = (global_optimizer.hypervolume_history[-1] - global_optimizer.hypervolume_history[-2]) / global_optimizer.hypervolume_history[-2]
                # 如果改进率低或达到最大阶段1迭代次数，转换到阶段2
                if hv_improvement < 0.05 or global_optimizer.current_iteration > global_optimizer.phase_1_iterations:
                    global_optimizer.phase = 2
                    print("【INFO】Transitioning to Phase 2: Complex systems enabled")
                    print(f"【INFO】Phase transition based on hypervolume improvement rate: {hv_improvement:.4f}")
        
        # 创建标准边界
        standard_bounds = torch.zeros_like(global_optimizer.bounds, device=global_optimizer.device)
        standard_bounds[1, :] = 1.0
        
        # 初始化模型（使用所有历史数据）
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Initializing model...")
        print(f"【DEBUG】Using historical data: X.shape={global_optimizer.X.shape}, Y.shape={global_optimizer.Y.shape}")
        print(f"【DEBUG】Total historical samples: {global_optimizer.X.shape[0]}")
        mll, model = global_optimizer.initialize_model()
        fit_gpytorch_mll(mll)
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Model fitting completed")
        
        # 生成候选点使用 taKG 采集函数
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Generating acquisition function...")
        acq_func = global_optimizer._compute_trace_aware_knowledge_gradient(model)
        
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Optimizing acquisition function...")
        candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=global_optimizer.batch_size,
            num_restarts=global_optimizer.num_restarts,
            raw_samples=global_optimizer.raw_samples,
            options={"batch_limit": 5, "maxiter": 200, "seed": global_optimizer.seed},
            sequential=True
        )
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Generated {candidates.shape[0]} candidates")
        
        # 添加多样性促进：如果候选点与现有点太相似，添加探索噪声
        if global_optimizer.X.shape[0] > 0:
            with torch.no_grad():
                X_normalized = normalize(global_optimizer.X, global_optimizer.bounds)
                candidates_normalized = normalize(candidates, global_optimizer.bounds)
                distances = torch.cdist(candidates_normalized, X_normalized)
                min_distances = distances.min(dim=1).values
                avg_min_distance = min_distances.mean().item()
                
                if avg_min_distance < 0.1:
                    print(f"【INFO】Candidates are too similar to existing points (avg min distance: {avg_min_distance:.4f}), adding exploration noise")
                    exploration_noise = torch.randn_like(candidates) * 0.05
                    candidates += exploration_noise
        
        # 反归一化并处理候选点
        candidates = unnormalize(candidates, global_optimizer.bounds)
        
        # 离散化
        for j in range(len(global_optimizer.parameters)):
            candidates[:, j] = torch.round(candidates[:, j] / global_optimizer.steps[j]) * global_optimizer.steps[j]
            candidates[:, j] = torch.clamp(candidates[:, j], global_optimizer.bounds[0, j], global_optimizer.bounds[1, j])
        
        # 应用约束
        candidates = global_optimizer._apply_safety_constraints(candidates)
        candidates = global_optimizer._apply_phase_constraints(candidates)
        
        # 评估实验
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Running experiments...")
        if request.simulation_flag:
            y_new = global_optimizer.simulate_experiment(candidates)
        else:
            # Use Real API mode: use simulate_experiment() instead of get_human_input()
            # This allows testing the full API flow without implementing real experiments
            # In production, replace this with actual experiment measurements or database queries
            print(f"【DEBUG】Use Real API mode: using simulate_experiment() for candidate evaluation")
            y_new = global_optimizer.simulate_experiment(candidates)
        
        # 处理观测值
        y_processed = global_optimizer._process_observed_values(y_new)
        
        # 更新数据
        print(f"【DEBUG】Iteration {global_optimizer.current_iteration}: Updating data...")
        global_optimizer.X = torch.cat([global_optimizer.X, candidates])
        global_optimizer.Y = torch.cat([global_optimizer.Y, y_processed])
        global_optimizer.save_experiment_data(candidates, y_new)
        
        # 记录迭代
        global_optimizer._record_iteration(
            iteration=global_optimizer.current_iteration,
            candidates=candidates,
            acquisition_values=acq_values,
        )
        
        # 计算超体积
        hv = global_optimizer._compute_hypervolume()
        print(f"【INFO】Current hypervolume: {hv:.4f}")
        print(f"【INFO】Added {candidates.shape[0]} new samples")
        
        # 获取结果
        pareto_x, pareto_y = global_optimizer.get_pareto_front()
        latest_iteration = global_optimizer.iteration_history[-1] if global_optimizer.iteration_history else None
        
        return {
            "success": True,
            "message": f"Iteration {global_optimizer.current_iteration} completed",
            "iteration": global_optimizer.current_iteration,
            "total_samples": global_optimizer.X.shape[0],
            "hypervolume": hv,
            "current_hypervolume": hv,
            "hypervolume_history": global_optimizer.hypervolume_history,
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist()
            },
            "iteration_result": latest_iteration if latest_iteration else {
                "iteration": global_optimizer.current_iteration,
                "candidates": candidates.cpu().numpy().tolist(),
                "X": global_optimizer.X.cpu().numpy().tolist(),
                "Y": global_optimizer.Y.cpu().numpy().tolist(),
                "hypervolume": hv,
                "phase": global_optimizer.phase
            },
            "phase": global_optimizer.phase
        }
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

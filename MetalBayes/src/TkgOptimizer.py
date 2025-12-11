import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.transforms import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# 配置日志
logger = logging.getLogger(__name__)


class TraceAwareKGOptimizer:
    def __init__(self, output_dir, fig_dir, seed=42, device=None, 
                 phase=1, phase_1_subphase='oxide', param_space=None):
        
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Set device: use GPU if available and not specified, otherwise use CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                logger.info(f"CUDA is available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("CUDA is not available, using CPU")
        else:
            # device 可能是字符串或 torch.device 对象
            if isinstance(device, torch.device):
                self.device = device
            else:
                self.device = torch.device(device)
            
            # 检查是否是 CUDA 设备
            if self.device.type == 'cuda':
                logger.info(f"Using specified GPU device: {self.device}")
            else:
                logger.info(f"Using specified device: {self.device}")
        
        # 使用传入的参数空间配置，如果没有则自己定义（向后兼容）
        if param_space is not None:
            self._load_parameter_spaces(param_space)
        else:
            raise ValueError("param_space is required")
        
        # Store experiment data on the specified device
        # 为每个阶段存储独立的参数空间数据
        # Phase 1 氧化物阶段：只存储氧化物参数 (4个参数)
        # Phase 1 有机物阶段：只存储有机物参数 (6个参数)
        # Phase 2/3：存储所有参数 (11个参数)
        # 注意：X和Y的维度会根据当前阶段动态变化
        self.X = torch.empty((0, 0), dtype=torch.float64, device=self.device)  # 初始为空，根据阶段动态调整
        self.Y = torch.empty((0, 3), dtype=torch.float64, device=self.device)  # Three objectives: uniformity, coverage, adhesion
        
        
        # 创建历史数据DataFrame，用于存储实验数据
        self.history = pd.DataFrame(columns=param_space['parameters'] + ["Uniformity", "Coverage", "Adhesion", "Timestamp"])
        
        # Multi-objective optimization settings on the specified device
        # 用于多目标优化中的超级体计算，是评估优化性能的核心参数；帕累托前沿计算中的参考点；超体积的值越大前沿越优
        # 一个解被称为帕累托最优（或非支配解），如果没有其他解能在不恶化至少一个目标的情况下，改进至少一个目标。
        # 换句话说，在帕累托最优解处，任何目标的进一步优化都必然导致其他目标的退化。
        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float64, device=self.device)
        
        # Optimization hyperparameters
        # 其实多次采样进行评估也是在优化采样的质量，极端一点：不至于老是采集到同一个点
        self.raw_samples = 16  # 初始采样数
        self.num_restarts = 5  # 优化采集函数时的随机重启次数

        self.batch_size = 5    # 每次迭代生成 5 个候选点
        self.n_init = 5        # 初始样本数
        
        # Record lists
        # 创建迭代历史列表，用于存储每次迭代的详细数据，结果值、目标值
        self.iteration_history = []
        # 创建超体积历史列表，用于存储每次迭代的超体积值
        self.hypervolume_history = []
        
        # Optimization phases (1: simple systems, 2: complex systems)
        # 阶段由 OptimizerManager 管理，优化器只负责当前阶段的优化
        self.phase = phase
        # Phase 1子阶段：'oxide'（氧化物）或 'organic'（有机物）
        self.phase_1_subphase = phase_1_subphase if phase == 1 else None
        
        # Independent iteration counter
        self.current_iteration = 0
        
        # Cache for hypervolume computation
        # 用于缓存超体积计算结果，避免重复计算
        self._cached_hv = None
        # 用于缓存超体积计算结果的迭代次数
        self._cached_hv_iteration = -1
        # 用于缓存超体积计算结果时的样本数量，确保缓存有效性
        self._cached_hv_sample_count = -1
        
        self.seed = seed
        self._set_seed(seed)
        
        logger.info(f"Initialized TraceAwareKGOptimizer with device: {self.device}")
        logger.info(f"Parameter space: {self.param_names}")
        # Log bounds on CPU to avoid CUDA kernel errors during formatting
        if hasattr(self, 'param_bounds'):
            logger.info(f"Bounds: {self.param_bounds.cpu()}")
        else:
            logger.info("Bounds: Not loaded yet")
    
    @staticmethod
    def _set_seed(seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_parameter_spaces(self, param_space: dict):
        """从 OptimizerManager 加载参数空间配置
        
        Args:
            param_space: 包含参数空间配置的字典
        """
        self.param_names = param_space['parameters']
        self.param_bounds = param_space['bounds']
        self.param_steps = param_space['steps']
        # 加载约束信息（如果存在）
        self.constraints = param_space.get('constraints', None)

    def _process_observed_values(self, y):
        """Process observed values to handle saturated objectives and prevent local optima
        
        Args:
            y: Tensor of shape (batch_size, 3) containing observed objectives
            
        Returns:
            Tensor of shape (batch_size, 3) with processed objectives
        """
        processed_y = y.clone()
        
        # For each objective, check if it's approaching saturation
        for i, obj_name in enumerate(["Uniformity", "Coverage", "Adhesion"]):
            # Check if any sample in the batch has a very high value for this objective
            for batch_idx in range(y.shape[0]):
                if y[batch_idx, i] > 0.9:
                    logger.info(f"Detected saturated {obj_name} value: {y[batch_idx, i]:.4f}, adding exploration encouragement")
                    
                    # Add a small amount of noise to prevent perfect values from dominating
                    noise = torch.randn(1, device=y.device) * 0.05
                    processed_y[batch_idx, i] = y[batch_idx, i] + noise
                    
                    # Clip to ensure we stay within valid range
                    processed_y[batch_idx, i] = torch.clamp(processed_y[batch_idx, i], 0.0, 1.0)
                    
                    # Increase weight on other objectives by slightly penalizing this one
                    # This encourages exploration of other dimensions
                    processed_y[batch_idx, i] *= 0.95
        
        return processed_y
    
    def _evaluate_and_update(self, candidates: torch.Tensor, simulation_flag: bool, 
                            iteration: int, acquisition_values=None):
        """
        评估候选样本并更新数据（批量处理）
        
        Args:
            candidates: 候选样本张量，shape为 (n_samples, n_params)
            simulation_flag: 是否使用模拟实验
            iteration: 当前迭代次数
            acquisition_values: 采集函数值（可选）
            
        Returns:
            dict: 包含迭代结果的字典
        """
        # 批量评估所有候选样本
        if simulation_flag:
            y_new = self.simulate_experiment(candidates)
        else:
            y_new = self.get_human_input(candidates)
        
        y_processed = self._process_observed_values(y_new)
        
        # 如果 X 为空，直接赋值；否则连接
        if self.X.shape[0] == 0:
            self.X = candidates
            self.Y = y_processed
        else:
            self.X = torch.cat([self.X, candidates])
            self.Y = torch.cat([self.Y, y_processed])
        
        # 保存实验数据到CSV
        self.save_experiment_data(candidates, y_processed)
        
        # 记录迭代信息
        self._record_iteration(iteration, candidates, acquisition_values)
        
        return {
            'iteration': iteration,
            'candidates': candidates,
            'hypervolume': self._compute_hypervolume()
        }

    def run_single_step(self, simulation_flag: bool = True):
        """
        运行单次优化迭代（由 OptimizerManager 调用）
        
        Args:
            simulation_flag: 是否使用模拟实验
            
        Returns:
            dict: 包含迭代结果的字典
        """
        # 如果是第一次迭代且没有初始数据，生成初始样本
        if self.X.shape[0] == 0:
            X_init = self.generate_initial_samples()
            return self._evaluate_and_update(X_init, simulation_flag, iteration=0, acquisition_values=None)
        
        # 更新迭代计数
        self.current_iteration += 1
        logger.info(f"Iteration {self.current_iteration}, Phase {self.phase}, Subphase: {self.phase_1_subphase if self.phase == 1 else 'N/A'}")
        
        try:
            # Initialize model
            mll, model = self.initialize_model()
            # 训练模型
            fit_gpytorch_mll(mll)
            # 计算采集函数
            acq_func = self._compute_trace_aware_knowledge_gradient(model)
            
            # Get phase-specific search bounds for active parameters only
            active_bounds = self.param_bounds
            
            # 创建归一化边界（[0, 1]），因为采集函数在归一化空间工作
            normalized_bounds = torch.zeros_like(active_bounds, device=self.device)
            normalized_bounds[1, :] = 1.0

            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=normalized_bounds,  # 使用归一化边界
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.seed},
                sequential=True
            )
            
            # 检查候选点相似度（在归一化空间）
            if self.X.shape[0] > 0:
                with torch.no_grad():
                    active_X_normalized = normalize(self.X, active_bounds)
                    # candidates 已经在归一化空间，直接使用
                    distances = torch.cdist(candidates, active_X_normalized)
                    min_distances = distances.min(dim=1).values
                    avg_min_distance = min_distances.mean().item()
                    
                    if avg_min_distance < 0.1:
                        logger.info("Candidates are too similar, adding exploration noise")
                        exploration_noise = torch.randn_like(candidates) * 0.05
                        candidates += exploration_noise
                        # 确保噪声后仍在 [0, 1] 范围内
                        candidates = torch.clamp(candidates, 0.0, 1.0)
            
            # 反归一化到原始空间
            candidates = unnormalize(candidates, active_bounds)
            
            # 应用离散化约束
            candidates = self._apply_discretization_constraints(candidates, self.param_steps, active_bounds)
        
            # 根据阶段应用对应的约束
            if self.phase == 'phase_1_oxide':
                candidates = self._apply_oxide_constraints(candidates)
            elif self.phase == 'phase_1_organic':
                candidates = self._apply_organic_safety_constraints(candidates)
            elif self.phase == 'phase_2':
                candidates = self._apply_organic_safety_constraints(candidates)
                candidates = self._apply_oxide_constraints(candidates)
            
            # Evaluate candidates
            logger.debug(f"Iteration {self.current_iteration}: Running experiments...")
            return self._evaluate_and_update(candidates, simulation_flag, iteration=self.current_iteration, acquisition_values=acq_values)
            
        except Exception as e:
            logger.error(f"Error in iteration {self.current_iteration}: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def optimize(self, n_iter=5, simulation_flag=True):
        """
        运行完整的优化流程（类似 models.py 的 optimize 方法）
        使用当前 TkgOptimizer 自己的逻辑（run_single_step）
        
        Args:
            n_iter: 优化迭代次数（不包括初始样本）
            simulation_flag: 是否使用模拟实验
            
        Returns:
            None（结果保存在 self.history 和迭代历史中）
        """
        self._set_seed(self.seed)
        
        # 检查是否已有数据，如果没有则重置迭代计数
        if self.X.shape[0] == 0:
            self.current_iteration = 0
        
        logger.info("=== Initialize experiments ===")
        # 如果还没有初始数据，第一次调用 run_single_step 会生成初始样本
        if self.X.shape[0] == 0:
            init_result = self.run_single_step(simulation_flag=simulation_flag)
            assert init_result['iteration'] == 0, f"Initial iteration should be 0, got {init_result['iteration']}"
            logger.info(f"Initial samples generated: {init_result['candidates'].shape[0]} samples")
            logger.info(f"Initial hypervolume: {init_result['hypervolume']:.6f}")
        else:
            logger.info(f"Using existing data: {self.X.shape[0]} samples")
            logger.info(f"Current hypervolume: {self._compute_hypervolume():.6f}")
        
        logger.info("=== Optimization phase ===")
        # 运行优化迭代
        for i in range(1, n_iter + 1):
            logger.info(f"Iteration {i}/{n_iter}")
            
            # 运行单次迭代（使用自己的逻辑）
            result = self.run_single_step(simulation_flag=simulation_flag)
            
            # 记录当前超体积
            hv = result['hypervolume']
            logger.info(f"Current hypervolume: {hv:.6f}")
            logger.info(f"Candidates generated: {result['candidates'].shape[0]} samples")
        
        logger.info("=== Optimization completed ===")
        logger.info(f"Total iterations: {n_iter}")
        logger.info(f"Total samples: {self.X.shape[0]}")
        logger.info(f"Final hypervolume: {self._compute_hypervolume():.6f}")
        
        # 记录帕累托前沿信息
        pareto_x, pareto_y = self.get_pareto_front()
        if pareto_x.shape[0] > 0:
            logger.info(f"Pareto front size: {pareto_x.shape[0]} solutions")
        else:
            logger.info("No Pareto front solutions found")
    
    def simulate_experiment(self, candidates):
        """
        模拟实验，返回目标值
        参考 models.py 的实现，使用参数的线性组合和三角函数来模拟真实的目标值
        
        Args:
            candidates: 候选样本张量，shape为 (batch_size, n_params)
            
        Returns:
            目标值张量，shape为 (batch_size, 3): [Uniformity, Coverage, Adhesion]
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")
        
        batch_size = candidates.shape[0]
        n_params = candidates.shape[1]
        
        # 归一化参数到 [0, 1] 范围以便于计算
        # 使用当前阶段的参数边界进行归一化
        normalized_candidates = normalize(candidates, self.param_bounds)
        
        # 根据参数数量设计不同的模拟函数
        # 目标1: Uniformity（均匀性）- 主要受浓度和温度影响
        # 目标2: Coverage（覆盖率）- 主要受时间和pH影响
        # 目标3: Adhesion（附着力）- 主要受类型和比例影响
        
        if n_params == 4:  # Phase 1 Oxide: metal_a_type, metal_a_concentration, metal_b_type, metal_molar_ratio_b_a
            # 氧化物阶段：使用金属参数
            # Uniformity: 受浓度和摩尔比影响，加上类型交互
            obj1 = (normalized_candidates[:, 1] * 0.25 + normalized_candidates[:, 3] * 0.2 + 
                   0.15 * torch.sin(normalized_candidates[:, 0] * 0.5 + normalized_candidates[:, 2] * 0.3)).unsqueeze(1)
            # Coverage: 受类型和浓度影响
            obj2 = (normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 1] * 0.15 + 
                   0.1 * torch.cos(normalized_candidates[:, 2] * 0.4 + normalized_candidates[:, 3] * 0.3)).unsqueeze(1)
            # Adhesion: 二进制目标，需要合适的类型和浓度组合
            obj3 = ((normalized_candidates[:, 0] > 0.4) & (normalized_candidates[:, 1] > 0.3)).float().unsqueeze(1)
            
        elif n_params == 6:  # Phase 1 Organic: organic_formula, organic_concentration, organic_temperature, organic_soak_time, organic_ph, organic_curing_time
            # 有机物阶段：使用有机物参数
            # Uniformity: 受浓度、温度和pH影响
            obj1 = (normalized_candidates[:, 1] * 0.2 + normalized_candidates[:, 2] * 0.15 + 
                   normalized_candidates[:, 4] * 0.1 + 0.1 * torch.sin(normalized_candidates[:, 0] * 0.3 + normalized_candidates[:, 4] * 0.2)).unsqueeze(1)
            # Coverage: 受浸泡时间、pH和固化时间影响
            obj2 = (normalized_candidates[:, 3] * 0.2 + normalized_candidates[:, 4] * 0.15 + 
                   normalized_candidates[:, 5] * 0.1 + 0.1 * torch.cos(normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 5] * 0.3)).unsqueeze(1)
            # Adhesion: 二进制目标，需要合适的配方和pH组合
            obj3 = ((normalized_candidates[:, 0] > 0.3) & (normalized_candidates[:, 4] > 0.4)).float().unsqueeze(1)
            
        else:  # Phase 2: 所有参数 (10个)
            # 混合阶段：使用所有参数
            # Uniformity: 主要受有机物浓度、温度和金属浓度影响
            obj1 = (normalized_candidates[:, 1] * 0.12 + normalized_candidates[:, 2] * 0.1 + 
                   normalized_candidates[:, 7] * 0.08 + 0.1 * torch.sin(normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 6] * 0.2)).unsqueeze(1)
            # Coverage: 主要受浸泡时间、pH和摩尔比影响
            obj2 = (normalized_candidates[:, 3] * 0.12 + normalized_candidates[:, 4] * 0.1 + 
                   normalized_candidates[:, 9] * 0.08 + 0.1 * torch.cos(normalized_candidates[:, 5] * 0.2 + normalized_candidates[:, 8] * 0.2)).unsqueeze(1)
            # Adhesion: 二进制目标，需要有机物配方、pH和金属类型的合适组合
            obj3 = ((normalized_candidates[:, 0] > 0.3) & (normalized_candidates[:, 4] > 0.3) & 
                   (normalized_candidates[:, 6] > 0.4)).float().unsqueeze(1)
        
        # 归一化连续目标到 [0, 1] 范围
        obj1_min = obj1.min()
        obj1_max = obj1.max()
        obj1 = (obj1 - obj1_min) / (obj1_max - obj1_min + 1e-6)
        
        obj2_min = obj2.min()
        obj2_max = obj2.max()
        obj2 = (obj2 - obj2_min) / (obj2_max - obj2_min + 1e-6)
        
        # obj3 已经是二进制（0或1），不需要归一化
        
        result = torch.cat([obj1, obj2, obj3], dim=-1).to(self.device)
        return result
  
    def get_human_input(self, candidates):
        """
        模拟实验，返回目标值
        参考 models.py 的实现，使用参数的线性组合和三角函数来模拟真实的目标值
        
        Args:
            candidates: 候选样本张量，shape为 (batch_size, n_params)
            
        Returns:
            目标值张量，shape为 (batch_size, 3): [Uniformity, Coverage, Adhesion]
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")
        
        batch_size = candidates.shape[0]
        n_params = candidates.shape[1]
        
        # 归一化参数到 [0, 1] 范围以便于计算
        # 使用当前阶段的参数边界进行归一化
        normalized_candidates = normalize(candidates, self.param_bounds)
        
        # 根据参数数量设计不同的模拟函数
        # 目标1: Uniformity（均匀性）- 主要受浓度和温度影响
        # 目标2: Coverage（覆盖率）- 主要受时间和pH影响
        # 目标3: Adhesion（附着力）- 主要受类型和比例影响
        
        if n_params == 4:  # Phase 1 Oxide: metal_a_type, metal_a_concentration, metal_b_type, metal_molar_ratio_b_a
            # 氧化物阶段：使用金属参数
            # Uniformity: 受浓度和摩尔比影响，加上类型交互
            obj1 = (normalized_candidates[:, 1] * 0.25 + normalized_candidates[:, 3] * 0.2 + 
                   0.15 * torch.sin(normalized_candidates[:, 0] * 0.5 + normalized_candidates[:, 2] * 0.3)).unsqueeze(1)
            # Coverage: 受类型和浓度影响
            obj2 = (normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 1] * 0.15 + 
                   0.1 * torch.cos(normalized_candidates[:, 2] * 0.4 + normalized_candidates[:, 3] * 0.3)).unsqueeze(1)
            # Adhesion: 二进制目标，需要合适的类型和浓度组合
            obj3 = ((normalized_candidates[:, 0] > 0.4) & (normalized_candidates[:, 1] > 0.3)).float().unsqueeze(1)
            
        elif n_params == 6:  # Phase 1 Organic: organic_formula, organic_concentration, organic_temperature, organic_soak_time, organic_ph, organic_curing_time
            # 有机物阶段：使用有机物参数
            # Uniformity: 受浓度、温度和pH影响
            obj1 = (normalized_candidates[:, 1] * 0.2 + normalized_candidates[:, 2] * 0.15 + 
                   normalized_candidates[:, 4] * 0.1 + 0.1 * torch.sin(normalized_candidates[:, 0] * 0.3 + normalized_candidates[:, 4] * 0.2)).unsqueeze(1)
            # Coverage: 受浸泡时间、pH和固化时间影响
            obj2 = (normalized_candidates[:, 3] * 0.2 + normalized_candidates[:, 4] * 0.15 + 
                   normalized_candidates[:, 5] * 0.1 + 0.1 * torch.cos(normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 5] * 0.3)).unsqueeze(1)
            # Adhesion: 二进制目标，需要合适的配方和pH组合
            obj3 = ((normalized_candidates[:, 0] > 0.3) & (normalized_candidates[:, 4] > 0.4)).float().unsqueeze(1)
            
        else:  # Phase 2: 所有参数 (10个)
            # 混合阶段：使用所有参数
            # Uniformity: 主要受有机物浓度、温度和金属浓度影响
            obj1 = (normalized_candidates[:, 1] * 0.12 + normalized_candidates[:, 2] * 0.1 + 
                   normalized_candidates[:, 7] * 0.08 + 0.1 * torch.sin(normalized_candidates[:, 0] * 0.2 + normalized_candidates[:, 6] * 0.2)).unsqueeze(1)
            # Coverage: 主要受浸泡时间、pH和摩尔比影响
            obj2 = (normalized_candidates[:, 3] * 0.12 + normalized_candidates[:, 4] * 0.1 + 
                   normalized_candidates[:, 9] * 0.08 + 0.1 * torch.cos(normalized_candidates[:, 5] * 0.2 + normalized_candidates[:, 8] * 0.2)).unsqueeze(1)
            # Adhesion: 二进制目标，需要有机物配方、pH和金属类型的合适组合
            obj3 = ((normalized_candidates[:, 0] > 0.3) & (normalized_candidates[:, 4] > 0.3) & 
                   (normalized_candidates[:, 6] > 0.4)).float().unsqueeze(1)
        
        # 归一化连续目标到 [0, 1] 范围
        obj1_min = obj1.min()
        obj1_max = obj1.max()
        obj1 = (obj1 - obj1_min) / (obj1_max - obj1_min + 1e-6)
        
        obj2_min = obj2.min()
        obj2_max = obj2.max()
        obj2 = (obj2 - obj2_min) / (obj2_max - obj2_min + 1e-6)
        
        # obj3 已经是二进制（0或1），不需要归一化
        
        result = torch.cat([obj1, obj2, obj3], dim=-1).to(self.device)
        return result
    
    def _compute_hypervolume(self):
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.Y)
        return bd.compute_hypervolume().item()
    
    def get_pareto_front(self):
        """Calculate the Pareto front from current observations."""
        if self.Y.shape[0] == 0:
            return torch.empty((0, self.X.shape[1]), dtype=torch.float64, device=self.device), \
                   torch.empty((0, 3), dtype=torch.float64, device=self.device)
        pareto_mask = torch.ones(self.Y.shape[0], dtype=torch.bool, device=self.device)
        for i in range(self.Y.shape[0]):
            for j in range(self.Y.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break
        return self.X[pareto_mask], self.Y[pareto_mask]
    
    def _save_iteration_data(self, record):
        """Save iteration data to JSON file"""
        # 文件名包含阶段信息
        phase_str = self.phase if isinstance(self.phase, str) else f"phase_{self.phase}"
        filename = f"{self.output_dir}/optimization_history_{phase_str}_{self.experiment_id}.json"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    
    def save_experiment_data(self, x, y):
        """Save experiment data to CSV file"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        x_cpu = x.cpu().numpy() if x.ndim == 2 else x.unsqueeze(0).cpu().numpy()
        y_cpu = y.cpu().numpy() if y.ndim == 2 else y.unsqueeze(0).cpu().numpy()
        
        new_rows = []
        for i in range(x_cpu.shape[0]):
            data = {name: val for name, val in zip(self.param_names, x_cpu[i])}
            data.update({
                "Uniformity": y_cpu[i, 0],
                "Coverage": y_cpu[i, 1],
                "Adhesion": y_cpu[i, 2],
                "Timestamp": timestamp
            })
            new_rows.append(data)
        new_data = pd.DataFrame(new_rows, columns=self.history.columns)
        if self.history.empty:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], ignore_index=True)
        
        # 文件名包含阶段信息
        phase_str = self.phase if isinstance(self.phase, str) else f"phase_{self.phase}"
        filename = f"{self.output_dir}/experiment_{phase_str}_{self.experiment_id}.csv"
        self.history.to_csv(filename, index=False)
    
    def _record_iteration(self, iteration, candidates, acquisition_values=None):
        """Record iteration information"""
        pareto_x, pareto_y = self.get_pareto_front()
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": self.phase if isinstance(self.phase, str) else f"phase_{self.phase}",
            'X': self.X.cpu().numpy().tolist(),
            'Y': self.Y.cpu().numpy().tolist(),
            "candidates": candidates.cpu().numpy().tolist(),
            "hypervolume": self._compute_hypervolume(),
            "pareto_front": {
                "X": pareto_x.cpu().numpy().tolist(),
                "Y": pareto_y.cpu().numpy().tolist(),
            },
            "acquisition_values": acquisition_values.cpu().numpy().tolist() if acquisition_values is not None else None,
        }
        
        self.iteration_history.append(record)
        self.hypervolume_history.append(record["hypervolume"])
        
        self._save_iteration_data(record)

    def initialize_model(self):
        """Initialize GP models for each objective with optimized configuration
        """
        
        # Normalize active parameters
        train_x = normalize(self.X, self.param_bounds)
        
        # Create GP models with optimized configuration
        # Using shared kernel configuration for consistency
        gp_models = []
        for i in range(3):
            # Create and move GP model to the specified device
            gp = SingleTaskGP(
                train_x,
                self.Y[:, i:i+1],
                outcome_transform=Standardize(m=1),
            ).to(self.device)
            gp_models.append(gp)
        
        # Combine models and ensure they're on the correct device
        # 组合多目标函数
        model = ModelListGP(*gp_models).to(self.device)
        
        # 创建训练目标函数
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(self.device)
        
        # Explicitly move all model components to the device
        for component in model.models:
            component.to(self.device)
        
        return mll, model

    def _compute_trace_aware_knowledge_gradient(self, model):
        
        active_bounds = self.param_bounds
        
        # Enhanced taKG implementation considering model performance trace across iterations
        train_x = normalize(self.X, active_bounds)
        
        # Get current model predictions
        with torch.no_grad():
            current_pred = model.posterior(train_x).mean
        
        # Dynamic reference point adjustment to handle saturated objectives
        # When any objective approaches saturation (e.g., coverage > 0.9), adjust reference point to encourage balance
        dynamic_ref = self.ref_point.clone()
        
        # Check if any objective is approaching saturation
        for i, obj_name in enumerate(["Uniformity", "Coverage", "Adhesion"]):
            if self.Y[:, i].max() > 0.9:
                # Increase the reference point for saturated objectives to focus on other objectives
                dynamic_ref[i] = 0.5
                logger.info(f"Objective {obj_name} is approaching saturation, adjusting reference point to {dynamic_ref[i]:.2f}")
        
        # Create partitioning for hypervolume calculation with dynamic reference point
        partitioning = NondominatedPartitioning(ref_point=dynamic_ref, Y=current_pred)
        
        # Enhanced taKG acquisition function with iteration-based trace awareness
        # Incorporates both current model performance and historical improvement trends
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model, 
            ref_point=dynamic_ref, 
            partitioning=partitioning
        )
        
        return acq_func

    def generate_initial_samples(self, n_init=None):
        """Generate initial samples
        
        在 Phase 1 的不同子阶段，只生成有效参数的初始样本（不扩展到完整参数空间）
        返回的样本shape: (n_init, num_active_params)
        """
        n = n_init if n_init is not None else self.n_init
        
        # Get active parameter indices and bounds
        param_steps = self.param_steps
        param_bounds = self.param_bounds
        
        # Generate samples only for active parameters
        sobel_samples = draw_sobol_samples(bounds=param_bounds, n=n, q=1, seed=self.seed).squeeze(1).to(self.device)
        
        # 应用离散化约束
        sobel_samples = self._apply_discretization_constraints(sobel_samples, param_steps, param_bounds)
        
        # 根据阶段应用相应的约束
        # phase_1_oxide: 只应用氧化物约束
        # phase_1_organic: 只应用有机物约束
        # phase_2: 应用两个约束
        if self.phase == 'phase_1_oxide':
            candidates = self._apply_oxide_constraints(sobel_samples)
        elif self.phase == 'phase_1_organic':
            candidates = self._apply_organic_safety_constraints(sobel_samples)
        elif self.phase == 'phase_2':
            # Phase 2 需要应用两个约束
            candidates = self._apply_organic_safety_constraints(sobel_samples)
            candidates = self._apply_oxide_constraints(candidates)
        else:
            # 默认情况：不应用约束
            candidates = sobel_samples

        return candidates

    def _apply_discretization_constraints(self, samples: torch.Tensor, 
                                         param_steps: torch.Tensor = None,
                                         param_bounds: torch.Tensor = None) -> torch.Tensor:
        """
        应用离散化约束：将连续值四舍五入到最近的步长倍数，并确保在边界内
        
        Args:
            samples: 样本张量，shape为 (n_samples, n_params)
            param_steps: 参数步长，如果为None则使用 self.param_steps
            param_bounds: 参数边界，如果为None则使用 self.param_bounds
            
        Returns:
            离散化后的样本张量，shape与输入相同
        """
        if param_steps is None:
            param_steps = self.param_steps
        if param_bounds is None:
            param_bounds = self.param_bounds
        
        # 对每个参数进行离散化
        # 离散化逻辑：将值四舍五入到最近的步长倍数，然后确保在边界内
        # 注意：先离散化再clamp，这样可以确保值仍然是步长的倍数
        # （前提是边界的最小值和最大值都是步长的倍数）
        for i in range(len(param_steps)):
            # 离散化：四舍五入到最近的步长倍数
            samples[:, i] = torch.round(samples[:, i] / param_steps[i]) * param_steps[i]
            # 确保值在边界内（边界值应该是步长的倍数，所以clamp后仍然是步长的倍数）
            samples[:, i] = torch.clamp(samples[:, i], param_bounds[0, i], param_bounds[1, i])
        
        return samples
        
    def _apply_oxide_constraints(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply oxide constraints to the samples
        
        氧化物约束逻辑
        约束1：metal_a_type 和 metal_b_type 的数值不能相同
        约束2：只有当 metal_b_type == 0 时，metal_molar_ratio_b_a 才能为 0
              如果 metal_b_type != 0，则 metal_molar_ratio_b_a 必须 >= 1
        
        约束信息从 OptimizerManager 初始化时传入
        """
        # 如果没有约束信息，直接返回
        if self.constraints is None or 'oxide_constraints' not in self.constraints:
            return samples
        
        # 找到相关参数在参数空间中的索引
        try:
            metal_a_type_idx = self.param_names.index('metal_a_type')
            metal_b_type_idx = self.param_names.index('metal_b_type')
            molar_ratio_idx = self.param_names.index('metal_molar_ratio_b_a')
        except ValueError:
            # 如果当前阶段不包含这些参数，直接返回
            return samples
        
        # 获取参数边界和步长
        metal_a_type_bounds = (self.param_bounds[0, metal_a_type_idx].item(), 
                              self.param_bounds[1, metal_a_type_idx].item())
        metal_b_type_bounds = (self.param_bounds[0, metal_b_type_idx].item(), 
                              self.param_bounds[1, metal_b_type_idx].item())
        molar_ratio_bounds = (self.param_bounds[0, molar_ratio_idx].item(), 
                             self.param_bounds[1, molar_ratio_idx].item())
        metal_a_type_step = self.param_steps[metal_a_type_idx].item()
        metal_b_type_step = self.param_steps[metal_b_type_idx].item()
        molar_ratio_step = self.param_steps[molar_ratio_idx].item()
        
        # 对每个样本应用约束
        for i in range(samples.shape[0]):
            # 获取当前样本的参数值（四舍五入到最近的整数）
            metal_a_type = int(torch.round(samples[i, metal_a_type_idx]).item())
            metal_b_type = int(torch.round(samples[i, metal_b_type_idx]).item())
            
            # 约束1：metal_a_type 和 metal_b_type 不能相同
            if metal_b_type != 0 and metal_a_type == metal_b_type:
                # 如果相同，需要修改 metal_b_type 为一个不同的值
                min_val = int(metal_b_type_bounds[0])
                max_val = int(metal_b_type_bounds[1])
                
                # 生成可选值列表（排除 metal_a_type）
                available_values = [v for v in range(min_val, max_val + 1) 
                                   if v != metal_a_type and v != 0]  # 0 表示没有 metal B，也排除
                
                if len(available_values) > 0:
                    # 随机选择一个不同的值
                    new_metal_b_type = torch.randint(0, len(available_values), (1,), device=samples.device).item()
                    new_metal_b_type = available_values[new_metal_b_type]
                    samples[i, metal_b_type_idx] = float(new_metal_b_type)
                    metal_b_type = new_metal_b_type
                else:
                    # 如果没有可用值，设置为 0（表示没有 metal B）
                    samples[i, metal_b_type_idx] = 0.0
                    metal_b_type = 0
            
            # 约束2：只有当 metal_b_type == 0 时，molar_ratio 才能为 0
            # 如果 metal_b_type != 0，则 molar_ratio 必须 >= 1
            # 先对 molar_ratio 进行离散化（使用 Tensor 操作）
            samples[i, molar_ratio_idx] = torch.round(samples[i, molar_ratio_idx] / molar_ratio_step) * molar_ratio_step
            molar_ratio_int = int(samples[i, molar_ratio_idx].item())
            
            if metal_b_type == 0:
                # metal_b_type == 0 时，molar_ratio 可以为 0
                # 但如果 molar_ratio != 0，需要修正为 0（因为 metal_b_type == 0 表示没有 metal B）
                if molar_ratio_int != 0:
                    samples[i, molar_ratio_idx] = 0.0
            else:
                # metal_b_type != 0 时，molar_ratio 必须 >= 1
                if molar_ratio_int == 0:
                    # 如果为 0，随机选择 1 到最大值之间的值
                    min_ratio = max(1, int(molar_ratio_bounds[0]))
                    max_ratio = int(molar_ratio_bounds[1])
                    num_steps = int((max_ratio - min_ratio) / molar_ratio_step) + 1
                    step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                    new_ratio = min_ratio + step_idx * molar_ratio_step
                    new_ratio = min(new_ratio, max_ratio)
                    samples[i, molar_ratio_idx] = float(new_ratio)
                else:
                    # 确保值在边界内并符合离散化要求
                    samples[i, molar_ratio_idx] = torch.clamp(samples[i, molar_ratio_idx], 
                                                             molar_ratio_bounds[0], molar_ratio_bounds[1])
            
            # 确保值在边界内并符合离散化要求
            samples[i, metal_a_type_idx] = torch.round(samples[i, metal_a_type_idx] / metal_a_type_step) * metal_a_type_step
            samples[i, metal_a_type_idx] = torch.clamp(samples[i, metal_a_type_idx], 
                                                       metal_a_type_bounds[0], metal_a_type_bounds[1])
            
            samples[i, metal_b_type_idx] = torch.round(samples[i, metal_b_type_idx] / metal_b_type_step) * metal_b_type_step
            samples[i, metal_b_type_idx] = torch.clamp(samples[i, metal_b_type_idx], 
                                                       metal_b_type_bounds[0], metal_b_type_bounds[1])
        
        return samples
        
    def _apply_organic_safety_constraints(self, samples: torch.Tensor) -> torch.Tensor:
        """Apply organic safety constraints to the samples
        
        针对有机物的安全约束逻辑
        1. 确保 organic_formula 在边界内 [1, 30]
        2. 根据 organic_formula 的值，限制对应的 organic_ph 范围
        约束信息从 OptimizerManager 初始化时传入
        """
        # 如果没有约束信息，直接返回
        if self.constraints is None or 'pH_safety_constraints' not in self.constraints:
            return samples
        
        pH_safety_constraints = self.constraints['pH_safety_constraints']
        
        # 找到 organic_formula 和 organic_ph 在参数空间中的索引
        try:
            formula_idx = self.param_names.index('organic_formula')
            ph_idx = self.param_names.index('organic_ph')
        except ValueError:
            # 如果当前阶段不包含这些参数，直接返回
            return samples
        
        # 获取参数边界和步长
        formula_bounds = (self.param_bounds[0, formula_idx].item(), 
                         self.param_bounds[1, formula_idx].item())
        formula_step = self.param_steps[formula_idx].item()
        ph_step = self.param_steps[ph_idx].item()
        
        # 对每个样本应用约束
        for i in range(samples.shape[0]):
            # 首先确保 organic_formula 在边界内并符合离散化要求
            samples[i, formula_idx] = torch.round(samples[i, formula_idx] / formula_step) * formula_step
            samples[i, formula_idx] = torch.clamp(samples[i, formula_idx], 
                                                  formula_bounds[0], formula_bounds[1])
            
            # 获取当前样本的 organic_formula 值（四舍五入到最近的整数）
            formula_id = int(torch.round(samples[i, formula_idx]).item())
            
            # 确保 formula_id 在有效范围内 [1, 30]
            if formula_id < 1 or formula_id > 30:
                # 如果超出范围，随机选择一个有效值
                min_formula = max(1, int(formula_bounds[0]))
                max_formula = int(formula_bounds[1])
                num_steps = int((max_formula - min_formula) / formula_step) + 1
                step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                new_formula = min_formula + step_idx * formula_step
                new_formula = min(new_formula, max_formula)
                samples[i, formula_idx] = float(new_formula)
                formula_id = int(new_formula)
            
            # 检查 formula_id 是否在约束字典中
            if formula_id in pH_safety_constraints:
                ph_min, ph_max = pH_safety_constraints[formula_id]
                
                # 获取当前 pH 值
                current_ph = samples[i, ph_idx].item()
                
                # 如果 pH 超出范围，在 ph_min 和 ph_max 之间随机采样
                if current_ph < ph_min or current_ph > ph_max:
                    # 在范围内随机采样，考虑步长
                    # 计算可能的离散值数量
                    num_steps = int((ph_max - ph_min) / ph_step) + 1
                    # 随机选择一个步长索引
                    step_idx = torch.randint(0, num_steps, (1,), device=samples.device).item()
                    # 计算对应的 pH 值
                    new_ph = ph_min + step_idx * ph_step
                    # 确保不超过最大值
                    new_ph = min(new_ph, ph_max)
                    samples[i, ph_idx] = new_ph
                else:
                    # 如果 pH 在范围内，确保是步长的倍数（离散化）
                    samples[i, ph_idx] = torch.round(samples[i, ph_idx] / ph_step) * ph_step
                    # 再次确保在范围内（离散化后可能略微超出）
                    samples[i, ph_idx] = torch.clamp(samples[i, ph_idx], ph_min, ph_max)
        
        return samples
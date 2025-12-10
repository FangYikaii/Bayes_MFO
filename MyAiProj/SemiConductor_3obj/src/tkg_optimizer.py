import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.models.transforms import Standardize
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from src.utils import convert_to_list


class TraceAwareKGOptimizer:
    """Trace-Aware Knowledge Gradient (taKG) Multi-fidelity Bayesian Optimization"""
    
    def __init__(self, output_dir, fig_dir, seed=42, device=None, 
                 phase=1, phase_1_subphase='oxide', param_space=None):
        """
        Initialize the Trace-Aware KG Optimizer
        
        Args:
            output_dir: Directory to save output files
            fig_dir: Directory to save figures
            seed: Random seed for reproducibility
            device: Torch device to use
            phase: 优化阶段 (1 或 2)
            phase_1_subphase: Phase 1 的子阶段 ('oxide' 或 'organic')，仅在 phase=1 时有效
            param_space: 参数空间配置字典（由 OptimizerManager 提供），如果为 None 则自己定义（向后兼容）
        """
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Set device: use GPU if available and not specified, otherwise use CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                print(f"【INFO】CUDA is available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("【INFO】CUDA is not available, using CPU")
        else:
            # device 可能是字符串或 torch.device 对象
            if isinstance(device, torch.device):
                self.device = device
            else:
                self.device = torch.device(device)
            
            # 检查是否是 CUDA 设备
            if self.device.type == 'cuda':
                print(f"【INFO】Using specified GPU device: {self.device}")
            else:
                print(f"【INFO】Using specified device: {self.device}")
        
        # 使用传入的参数空间配置，如果没有则自己定义（向后兼容）
        if param_space is not None:
            self._load_parameter_spaces(param_space)
        else:
            # 向后兼容：如果没有传入参数空间，则自己定义
            self._define_parameter_spaces()
        
        # Store experiment data on the specified device
        # 为每个阶段存储独立的参数空间数据
        # Phase 1 氧化物阶段：只存储氧化物参数 (5个参数)
        # Phase 1 有机物阶段：只存储有机物参数 (7个参数)
        # Phase 2/3：存储所有参数 (11个参数)
        # 注意：X和Y的维度会根据当前阶段动态变化
        self.X = torch.empty((0, 0), dtype=torch.float64, device=self.device)  # 初始为空，根据阶段动态调整
        self.Y = torch.empty((0, 3), dtype=torch.float64, device=self.device)  # Three objectives: uniformity, coverage, adhesion
        
        # 存储完整参数空间的原始数据（用于记录和保存）
        self.X_full = torch.empty((0, len(self.param_names)), dtype=torch.float64, device=self.device)
        
        # 创建历史数据DataFrame，用于存储实验数据
        self.history = pd.DataFrame(columns=self.param_names + ["Uniformity", "Coverage", "Adhesion", "Timestamp"])
        
        # Multi-objective optimization settings on the specified device
        # 用于多目标优化中的超级体计算，是评估优化性能的核心参数；帕累托前沿计算中的参考点；超体积的值越大前沿越优
        # 一个解被称为帕累托最优（或非支配解），如果没有其他解能在不恶化至少一个目标的情况下，改进至少一个目标。
        # 换句话说，在帕累托最优解处，任何目标的进一步优化都必然导致其他目标的退化。
        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float64, device=self.device)
        
        # Optimization hyperparameters
        # 其实多次采样进行评估也是在优化采样的质量，极端一点：不至于老是采集到同一个点
        self.num_restarts = 5  # 优化采集函数时的随机重启次数
        self.raw_samples = 16  # 初始采样数
        self.batch_size = 5  # 每次迭代同时评估的候选点数量
        self.n_init = 5 # 初始样本数
        
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
        
        print(f"【INFO】Initialized TraceAwareKGOptimizer with device: {self.device}")
        print(f"【INFO】Parameter space: {self.param_names}")
        # Print bounds on CPU to avoid CUDA kernel errors during formatting
        print(f"【INFO】Bounds: {self.bounds.cpu()}")
    
    def _load_parameter_spaces(self, param_space: dict):
        """从 OptimizerManager 加载参数空间配置
        
        Args:
            param_space: 包含参数空间配置的字典
        """
        # 加载完整参数空间信息
        self.param_names = param_space['param_names']
        # 确保 tensor 在正确的 device 上
        self.bounds = param_space['bounds'].to(self.device) if isinstance(param_space['bounds'], torch.Tensor) else param_space['bounds']
        self.steps = param_space['steps'].to(self.device) if isinstance(param_space['steps'], torch.Tensor) else param_space['steps']
        self.parameters = param_space['parameters']
        
        # 加载参数索引信息
        self.organic_param_indices = param_space['organic_param_indices']
        self.metal_param_indices = param_space['metal_param_indices']
        self.all_param_indices = param_space['all_param_indices']
        
        # 加载安全约束
        self.pH_safety_constraints = param_space['pH_safety_constraints']
        
        # 当前阶段的有效参数信息（用于内部使用）
        self._active_param_names = param_space['active_param_names']
        self._active_indices = param_space['active_indices']
        # 确保 tensor 在正确的 device 上
        self._active_bounds = param_space['active_bounds'].to(self.device) if isinstance(param_space['active_bounds'], torch.Tensor) else param_space['active_bounds']
        self._active_steps = param_space['active_steps'].to(self.device) if isinstance(param_space['active_steps'], torch.Tensor) else param_space['active_steps']
        
        print(f"【INFO】Loaded parameter space from OptimizerManager: {len(self.param_names)} total params, {len(self._active_param_names)} active params")
    
    def _define_parameter_spaces(self):
        """Define parameter spaces based on user requirements"""
        # Organic formula parameters
        self.organic_params = {
            'organic_formula': (1, 30, 1),          # 1-30, step 1
            'organic_concentration': (0.1, 5, 0.1),  # 0.1-5%, step 0.1
            'organic_temperature': (25, 40, 5),      # 25-40°C, step 5
            'organic_soak_time': (1, 30, 1),         # 1-30min, step 1
            'organic_ph': (2.0, 14.0, 0.5),          # 2.0-14.0, step 0.5 (with safety constraints)
            'organic_curing_time': (10, 30, 5)       # 10-30min, step 5
        }
        
        # Metal oxide parameters
        self.metal_params = {
            'metal_a_type': (1, 20, 1),      # 1-20, step 1
            'metal_a_concentration': (10, 50, 10),  # 10-50%, step 10
            'metal_b_type': (0, 20, 1),      # 0-20, step 1 (0 means no metal B)
            'metal_molar_ratio_b_a': (1, 10, 1)    # 1-10%, step 1
        }
        
        # Combine all parameters (no longer need experiment_condition as phases are managed separately)
        self.parameters = {**self.organic_params, **self.metal_params}
        self.param_names = list(self.parameters.keys())
        
        # Record parameter indices for phase constraints
        # 记录有机物和氧化物参数的索引范围，用于阶段约束
        num_organic = len(self.organic_params)
        num_metal = len(self.metal_params)
        self.organic_param_indices = list(range(num_organic))  # 索引 0-5
        self.metal_param_indices = list(range(num_organic, num_organic + num_metal))  # 索引 6-9
        
        # 记录完整参数索引
        self.all_param_indices = list(range(len(self.param_names)))
        
        # Create bounds on CPU first for safe printing
        # 创建边界参数矩阵，用于存储参数的上下界
        bounds_cpu = torch.tensor([
            [param[0] for param in self.parameters.values()],
            [param[1] for param in self.parameters.values()]
        ], dtype=torch.float64)
        
        # Create steps on CPU first for safe printing
        # 创建步长参数向量，用于存储参数的步长
        steps_cpu = torch.tensor([param[2] for param in self.parameters.values()])
        
        # Move tensors to the specified device
        # 将边界参数矩阵和步长参数向量移动到指定的设备上
        self.bounds = bounds_cpu.to(self.device)
        self.steps = steps_cpu.to(self.device)
        
        # Safety constraints for pH based on formula ID
        # Create a direct mapping from formula ID to pH range for better performance and readability
        self.pH_safety_constraints = {}
        
        # Epoxy silanes (ID 1-4): pH=4.0-6.0
        for formula_id in range(1, 5):
            self.pH_safety_constraints[formula_id] = (4.0, 6.0)
        
        # Active hydrogen silanes (ID 5-7)
        self.pH_safety_constraints[5] = (7.0, 10.5)  # AEAPTMS
        self.pH_safety_constraints[6] = (7.0, 10.5)  # APTES
        self.pH_safety_constraints[7] = (7.0, 10.5)  # APTMS
        
        # Linear self-assembled molecules (ID 8-13)
        self.pH_safety_constraints[8] = (3.5, 6.5)   # MPTES
        self.pH_safety_constraints[9] = (3.5, 6.5)   # 3-巯丙基三甲氧基硅烷
        self.pH_safety_constraints[10] = (4.5, 6.0)  # 全氟辛基三乙氧基硅烷
        self.pH_safety_constraints[11] = (2.0, 5.0)  # 四乙氧基硅烷(TEOS)
        self.pH_safety_constraints[12] = (4.0, 5.5)  # 甲基三氯硅烷
        self.pH_safety_constraints[13] = (4.0, 6.0)  # 乙烯基三甲氧基硅烷
        
        # Other silane compounds (ID 14-17)
        self.pH_safety_constraints[14] = (3.5, 4.5)  # γ-甲基丙烯酰氧基丙基三甲氧基硅烷
        self.pH_safety_constraints[15] = (6.0, 8.0)  # 3-叔丙基三甲氧基硅烷
        self.pH_safety_constraints[16] = (7.0, 10.0) # 哌嗪基丙基甲基二甲氧基硅烷
        self.pH_safety_constraints[17] = (5.0, 7.0)  # 乙酰氧基丙基三甲氧基硅烷
        
        # Other organic compounds (ID 18-21)
        self.pH_safety_constraints[18] = (2.0, 7.0)  # 二乙基磷酰乙基三乙氧基硅烷
        self.pH_safety_constraints[19] = (3.0, 7.0)  # 月桂酸
        self.pH_safety_constraints[20] = (7.0, 11.0) # 聚乙烯亚胺(PEI)
        self.pH_safety_constraints[21] = (3.0, 7.0)  # 聚丙烯酸(PAA)
        
        # Additional compounds (ID 22-30)
        self.pH_safety_constraints[22] = (4.0, 6.0)  # 双-[3-(三乙氧基硅基)丙基]-四硫化物(BTSPS)
        self.pH_safety_constraints[23] = (8.0, 10.0) # 双(三甲氧基硅基)丙酸
        self.pH_safety_constraints[24] = (4.0, 6.0)  # 1,2-双(三甲氧基硅基)乙烷
        self.pH_safety_constraints[25] = (4.0, 6.0)  # 1,3-双(三甲氧基硅基)丙烷
        self.pH_safety_constraints[26] = (9.0, 11.0) # 1,3-双(3-氨基丙基)-1,1,3,3-四甲基二硅氧烷
        self.pH_safety_constraints[27] = (8.0, 10.0) # 双端环氧丙基甲基硅氧烷(n≈10)
        self.pH_safety_constraints[28] = (6.0, 8.0)  # 双(三甲氧基硅基丙基)硫化物
        self.pH_safety_constraints[29] = (4.0, 6.0)  # 三(三甲氧基甲硅烷基丙基)异氰脲酸酯
        self.pH_safety_constraints[30] = (7.0, 9.0)  # 2,4,6,8-四甲基-2,4,6,8-四(丙基缩水甘油醚)环四硅氧烷
    
    @staticmethod
    def _set_seed(seed):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _apply_safety_constraints(self, candidates):
        """Apply safety constraints to candidates with vectorized operations for improved efficiency"""
        batch_size = candidates.shape[0]
        
        # Vectorized formula ID extraction
        formula_ids = torch.round(candidates[:, 0]).int()
        
        # Vectorized pH safety constraints
        for formula_id in torch.unique(formula_ids):
            mask = formula_ids == formula_id
            if formula_id.item() in self.pH_safety_constraints:
                min_ph, max_ph = self.pH_safety_constraints[formula_id.item()]
                # Clamp pH to safety range
                candidates[mask, 4] = torch.clamp(candidates[mask, 4], min_ph, max_ph)
                # Re-discretize pH to ensure it's on the discrete grid (step 0.5)
                ph_idx = 4
                candidates[mask, ph_idx] = torch.round(candidates[mask, ph_idx] / self.steps[ph_idx]) * self.steps[ph_idx]
                # Re-clamp after discretization to ensure it's still within safety range
                candidates[mask, 4] = torch.clamp(candidates[mask, 4], min_ph, max_ph)
        
        # Vectorized metal oxide constraints
        metal_a_types = torch.round(candidates[:, 6]).int()
        metal_b_types = torch.round(candidates[:, 8]).int()
        
        # Find candidates where metal B is not 0 and metal A and B types are the same
        conflict_mask = (metal_b_types != 0) & (metal_a_types == metal_b_types)
        
        if conflict_mask.any():
            # Adjust metal B type for conflicting candidates
            # Strategy: try +1 first, if that still conflicts, try -1
            # This ensures we avoid conflicts while staying within bounds
            metal_b_conflict = candidates[conflict_mask, 8]
            metal_a_conflict = metal_a_types[conflict_mask]
            
            # Try +1 first
            adjustment_plus = metal_b_conflict + 1
            adjustment_plus = torch.clamp(adjustment_plus, self.bounds[0, 8], self.bounds[1, 8])
            # Check if +1 still conflicts with metal_a_type
            still_conflicts_plus = torch.round(adjustment_plus).int() == metal_a_conflict
            
            # Try -1 if +1 doesn't work
            adjustment_minus = metal_b_conflict - 1
            adjustment_minus = torch.clamp(adjustment_minus, self.bounds[0, 8], self.bounds[1, 8])
            
            # Use +1 if it doesn't conflict, otherwise use -1
            final_adjustment = torch.where(
                still_conflicts_plus,
                adjustment_minus,
                adjustment_plus
            )
            
            # Re-discretize metal_b_type to ensure it's on the discrete grid (step 1)
            metal_b_idx = 8
            final_adjustment = torch.round(final_adjustment / self.steps[metal_b_idx]) * self.steps[metal_b_idx]
            # Re-clamp after discretization
            final_adjustment = torch.clamp(final_adjustment, self.bounds[0, metal_b_idx], self.bounds[1, metal_b_idx])
            
            candidates[conflict_mask, 8] = final_adjustment
        
        return candidates
    
    def _apply_phase_constraints(self, candidates):
        """Apply phase-specific constraints to candidate solutions
        
        Args:
            candidates: Tensor of shape (batch_size, num_params) containing candidate solutions
            
        Returns:
            Tensor of shape (batch_size, num_params) with phase-specific constraints applied
            
        Phase 1 constraints:
            - Phase 1前半段（oxide子阶段）：只探索氧化物参数空间
              - 有机物参数在优化过程中已被排除（通过搜索边界 [0, 0]）
              - 反归一化后，有机物参数会自然成为边界最小值
            - Phase 1后半段（organic子阶段）：只探索有机物参数空间
              - 氧化物参数在优化过程中已被排除（通过搜索边界 [0, 0]）
              - 反归一化后，氧化物参数会自然成为边界最小值
            
        Phase 2/3 constraints:
            - All parameters are allowed (no constraints applied)
        
        注意：不需要的参数在优化过程中已经被完全排除（通过 _get_phase_search_bounds），
        反归一化后它们会自然成为边界最小值，这里不需要再次设置。
        由于现在使用 OptimizerManager 管理不同阶段的优化器，每个阶段都有独立的优化器实例，
        不再需要通过 experiment_condition 来区分阶段。
        """
        # Phase 1/2/3: 所有阶段都不需要额外的约束，因为参数空间已经在 OptimizerManager 中正确配置
        return candidates
    
    def _get_active_param_indices(self):
        """Get active parameter indices for current phase
        
        Returns:
            List of parameter indices that should be used in optimization
        """
        # 如果从 OptimizerManager 加载了参数空间，直接使用预计算的索引
        if hasattr(self, '_active_indices'):
            return self._active_indices
        
        # 向后兼容：如果没有从 Manager 加载，则自己计算
        if self.phase == 1:
            if self.phase_1_subphase == 'oxide':
                # 氧化物阶段：只使用氧化物参数
                return self.metal_param_indices
            else:  # self.phase_1_subphase == 'organic'
                # 有机物阶段：只使用有机物参数
                return self.organic_param_indices
        else:
            # Phase 2/3: 使用所有参数
            return self.all_param_indices
    
    def _get_active_bounds(self):
        """Get bounds for active parameters
        
        Returns:
            Tensor of shape (2, num_active_params) with bounds for active parameters
        """
        # 如果从 OptimizerManager 加载了参数空间，直接使用预计算的边界
        if hasattr(self, '_active_bounds'):
            return self._active_bounds
        
        # 向后兼容：如果没有从 Manager 加载，则自己计算
        active_indices = self._get_active_param_indices()
        return self.bounds[:, active_indices]
    
    def _get_active_steps(self):
        """Get steps for active parameters
        
        Returns:
            Tensor of shape (num_active_params,) with steps for active parameters
        """
        # 如果从 OptimizerManager 加载了参数空间，直接使用预计算的步长
        if hasattr(self, '_active_steps'):
            return self._active_steps
        
        # 向后兼容：如果没有从 Manager 加载，则自己计算
        active_indices = self._get_active_param_indices()
        return self.steps[active_indices]
    
    @property
    def active_param_names(self):
        """获取当前阶段的参数名称
        
        Returns:
            List of parameter names for the current phase
        """
        # 如果从 OptimizerManager 加载了参数空间，直接使用预计算的参数名称
        if hasattr(self, '_active_param_names'):
            return self._active_param_names
        
        # 向后兼容：如果没有从 Manager 加载，则自己计算
        active_indices = self._get_active_param_indices()
        return [self.param_names[i] for i in active_indices]
    
    @property
    def active_bounds(self):
        """获取当前阶段的参数边界
        
        Returns:
            Tensor of shape (2, num_active_params) with bounds for active parameters
        """
        return self._get_active_bounds()
    
    def _get_phase_search_bounds(self):
        """Get search bounds for acquisition function optimization based on current phase
        
        Returns:
            Tensor of shape (2, num_active_params) with search bounds for active parameters only
        """
        active_indices = self._get_active_param_indices()
        
        # Create bounds only for active parameters (normalized to [0, 1])
        active_bounds = torch.zeros((2, len(active_indices)), device=self.device)
        active_bounds[1, :] = 1.0
        
        return active_bounds
    
    def _extend_candidates_to_full_space(self, candidates):
        """Extend candidates from active parameter space to full parameter space
        
        Args:
            candidates: Tensor of shape (batch_size, num_active_params) with active parameters only
            
        Returns:
            Tensor of shape (batch_size, num_all_params) with all parameters
        """
        active_indices = self._get_active_param_indices()
        batch_size = candidates.shape[0]
        
        # Create full candidate tensor with default values
        full_candidates = torch.zeros((batch_size, len(self.param_names)), 
                                     dtype=candidates.dtype, device=self.device)
        
        # Set active parameters
        for i, idx in enumerate(active_indices):
            full_candidates[:, idx] = candidates[:, i]
        
        # Set default values for inactive parameters
        if self.phase == 1:
            if self.phase_1_subphase == 'oxide':
                # 氧化物阶段：有机物参数设为边界最小值
                for idx in self.organic_param_indices:
                    full_candidates[:, idx] = self.bounds[0, idx]
            else:  # self.phase_1_subphase == 'organic'
                # 有机物阶段：氧化物参数设为边界最小值
                for idx in self.metal_param_indices:
                    full_candidates[:, idx] = self.bounds[0, idx]
        
        return full_candidates
    
    def _extract_active_parameters(self, candidates):
        """Extract active parameters from full parameter space
        
        Args:
            candidates: Tensor of shape (batch_size, num_all_params) with all parameters
            
        Returns:
            Tensor of shape (batch_size, num_active_params) with active parameters only
        """
        active_indices = self._get_active_param_indices()
        return candidates[:, active_indices]
    
    def _get_phase_param_count(self):
        """Get the number of parameters for current phase"""
        return len(self._get_active_param_indices())
    
    def _ensure_X_shape(self):
        """Ensure X has the correct shape for current phase
        
        注意：当阶段切换时，由于参数空间不同，X会被重置为空
        只保留X_full用于记录，新阶段的数据会重新填充X
        """
        active_indices = self._get_active_param_indices()
        expected_dim = len(active_indices)
        
        if self.X.shape[0] == 0:
            # Initialize with correct dimension
            self.X = torch.empty((0, expected_dim), dtype=torch.float64, device=self.device)
        elif self.X.shape[1] != expected_dim:
            # Phase changed: 由于参数空间不同，重置X为空
            # 不同阶段的参数空间不兼容，不能直接转换
            # X_full保留所有历史数据用于记录
            self.X = torch.empty((0, expected_dim), dtype=torch.float64, device=self.device)
            self.Y = torch.empty((0, 3), dtype=torch.float64, device=self.device)
            print(f"【INFO】Phase changed: X reset to empty (new dimension: {expected_dim})")
    
    def generate_initial_samples(self, n_init=None):
        """Generate initial samples
        
        在 Phase 1 的不同子阶段，只生成有效参数的初始样本（不扩展到完整参数空间）
        返回的样本shape: (n_init, num_active_params)
        """
        n = n_init if n_init is not None else self.n_init
        
        # Get active parameter indices and bounds
        active_indices = self._get_active_param_indices()
        active_bounds = self._get_active_bounds()
        active_steps = self._get_active_steps()
        
        # Generate samples only for active parameters
        sobel_samples = draw_sobol_samples(bounds=active_bounds, n=n, q=1, seed=self.seed).squeeze(1).to(self.device)

        # Discretization for active parameters
        for i in range(len(active_indices)):
            sobel_samples[:, i] = torch.round(sobel_samples[:, i] / active_steps[i]) * active_steps[i]
            sobel_samples[:, i] = torch.clamp(sobel_samples[:, i], active_bounds[0, i], active_bounds[1, i])

        # Apply phase constraints (mainly sets experiment_condition)
        # Note: 这里需要扩展到完整参数空间来应用约束，然后再提取有效参数
        # Keep active parameters unchanged, only apply constraints to inactive parameters
        full_samples = self._extend_candidates_to_full_space(sobel_samples)
        
        # Apply constraints (these may modify parameters, but we'll restore active parameters afterwards)
        full_samples = self._apply_safety_constraints(full_samples)
        full_samples = self._apply_phase_constraints(full_samples)
        
        # Re-discretize only inactive parameters, keep active parameters unchanged
        active_indices = self._get_active_param_indices()
        inactive_indices = [i for i in range(len(self.parameters)) if i not in active_indices]
        for i in inactive_indices:
            full_samples[:, i] = torch.round(full_samples[:, i] / self.steps[i]) * self.steps[i]
            full_samples[:, i] = torch.clamp(full_samples[:, i], self.bounds[0, i], self.bounds[1, i])
        
        # Restore active parameters to their original values (ensure they weren't changed by constraints)
        for idx_in_active, idx_in_full in enumerate(active_indices):
            full_samples[:, idx_in_full] = sobel_samples[:, idx_in_active]

        # Extract active parameters back (只返回有效参数，应该与原始 sobel_samples 一致)
        active_samples = self._extract_active_parameters(full_samples)
        
        # Also store full samples for record keeping
        # (This will be used when saving to CSV)
        print(f"FXL:【INFO】Generated initial samples: {active_samples}")
        return active_samples
    
    def initialize_model(self):
        """Initialize GP models for each objective with optimized configuration
        
        在 Phase 1 的不同子阶段，X已经只包含有效参数，直接使用即可
        """
        # Ensure X has correct shape for current phase
        self._ensure_X_shape()
        
        # X already contains only active parameters for current phase
        active_indices = self._get_active_param_indices()
        active_bounds = self._get_active_bounds()
        
        # Normalize active parameters
        train_x = normalize(self.X, active_bounds)
        
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
        model = ModelListGP(*gp_models).to(self.device)
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(self.device)
        
        # Explicitly move all model components to the device
        for component in model.models:
            component.to(self.device)
        
        return mll, model
    
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
                    print(f"【INFO】Detected saturated {obj_name} value: {y[batch_idx, i]:.4f}, adding exploration encouragement")
                    
                    # Add a small amount of noise to prevent perfect values from dominating
                    noise = torch.randn(1, device=y.device) * 0.05
                    processed_y[batch_idx, i] = y[batch_idx, i] + noise
                    
                    # Clip to ensure we stay within valid range
                    processed_y[batch_idx, i] = torch.clamp(processed_y[batch_idx, i], 0.0, 1.0)
                    
                    # Increase weight on other objectives by slightly penalizing this one
                    # This encourages exploration of other dimensions
                    processed_y[batch_idx, i] *= 0.95
        
        return processed_y
    
    def _compute_trace_aware_knowledge_gradient(self, model):
        """Compute Trace-Aware Knowledge Gradient acquisition function
        
        X已经只包含有效参数，直接使用即可
        """
        # Ensure X shape is correct
        self._ensure_X_shape()
        
        # X already contains only active parameters for current phase
        active_indices = self._get_active_param_indices()
        active_bounds = self._get_active_bounds()
        
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
                print(f"【INFO】Objective {obj_name} is approaching saturation, adjusting reference point to {dynamic_ref[i]:.2f}")
        
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
    
    def simulate_experiment(self, x):
        """Simulate experiment results
        
        根据输入维度自动判断阶段：
        - shape (batch_size, 5): Phase 1 氧化物阶段
        - shape (batch_size, 7): Phase 1 有机物阶段
        - shape (batch_size, 11): Phase 2/3 完整参数空间
        """
        if len(x.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {x.shape}")
        
        num_params = x.shape[1]
        active_indices = self._get_active_param_indices()
        
        # 根据输入维度判断阶段
        if num_params == 5:
            # Phase 1 氧化物阶段：只包含氧化物参数和condition
            # x的列对应: [metal_a_type, metal_a_concentration, metal_b_type, metal_molar_ratio_b_a, condition]
            metal_a_type = x[:, 0]
            metal_a_concentration = x[:, 1]
            metal_b_type = x[:, 2]
            metal_molar_ratio_b_a = x[:, 3]
            condition = x[:, 4]
            
            # 有机物参数设为默认值（边界最小值）
            organic_formula = torch.full((x.shape[0],), self.bounds[0, 0].item(), device=self.device)
            organic_concentration = torch.full((x.shape[0],), self.bounds[0, 1].item(), device=self.device)
            organic_temperature = torch.full((x.shape[0],), self.bounds[0, 2].item(), device=self.device)
            organic_soak_time = torch.full((x.shape[0],), self.bounds[0, 3].item(), device=self.device)
            organic_ph = torch.full((x.shape[0],), self.bounds[0, 4].item(), device=self.device)
            organic_curing_time = torch.full((x.shape[0],), self.bounds[0, 5].item(), device=self.device)
            
        elif num_params == 7:
            # Phase 1 有机物阶段：只包含有机物参数和condition
            # x的列对应: [organic_formula, organic_concentration, organic_temperature, organic_soak_time, organic_ph, organic_curing_time, condition]
            organic_formula = x[:, 0]
            organic_concentration = x[:, 1]
            organic_temperature = x[:, 2]
            organic_soak_time = x[:, 3]
            organic_ph = x[:, 4]
            organic_curing_time = x[:, 5]
            condition = x[:, 6]
            
            # 氧化物参数设为默认值（边界最小值）
            metal_a_type = torch.full((x.shape[0],), self.bounds[0, 6].item(), device=self.device)
            metal_a_concentration = torch.full((x.shape[0],), self.bounds[0, 7].item(), device=self.device)
            metal_b_type = torch.full((x.shape[0],), self.bounds[0, 8].item(), device=self.device)
            metal_molar_ratio_b_a = torch.full((x.shape[0],), self.bounds[0, 9].item(), device=self.device)
            
        elif num_params == 11:
            # Phase 2/3：完整参数空间
            organic_formula = x[:, 0]
            organic_concentration = x[:, 1]
            organic_temperature = x[:, 2]
            organic_soak_time = x[:, 3]
            organic_ph = x[:, 4]
            organic_curing_time = x[:, 5]
            metal_a_type = x[:, 6]
            metal_a_concentration = x[:, 7]
            metal_b_type = x[:, 8]
            metal_molar_ratio_b_a = x[:, 9]
            condition = x[:, 10]
        else:
            raise ValueError(f"Unexpected number of parameters: {num_params}. Expected 5 (oxide), 7 (organic), or 11 (full)")
        
        # Simulate uniformity (0-1) - Scaled coefficients to avoid saturation
        uniformity = (0.3 + 0.006 * organic_formula + 0.015 * organic_concentration + 0.003 * organic_temperature + 0.003 * organic_soak_time + 0.006 * organic_ph + 0.003 * organic_curing_time)
        uniformity += 0.1 * torch.sin(organic_formula + organic_concentration + organic_temperature)
        
        # Simulate coverage (0-1) - Scaled coefficients to avoid saturation
        coverage = (0.3 + 0.009 * organic_formula + 0.012 * organic_concentration + 0.006 * organic_temperature + 0.006 * organic_soak_time + 0.003 * organic_ph + 0.003 * organic_curing_time)
        coverage += 0.15 * torch.cos(organic_temperature + organic_ph + organic_curing_time)
        
        # Simulate adhesion (0-1) - Scaled coefficients to avoid saturation
        adhesion = (0.3 + 0.006 * organic_formula + 0.009 * organic_concentration + 0.006 * organic_temperature + 0.003 * organic_soak_time + 0.006 * organic_ph + 0.003 * organic_curing_time)
        adhesion += 0.1 * torch.sin(organic_soak_time + organic_ph + organic_curing_time)
        
        # Add metal oxide effects - Reduced impact
        oxide_effect = 0.08 * (metal_a_concentration / 50.0) * (1.0 + 0.5 * (1.0 - torch.abs(metal_a_type - metal_b_type) / 20.0))
        uniformity += oxide_effect * 0.1
        coverage += oxide_effect * 0.15
        adhesion += oxide_effect * 0.2
        
        # Apply condition effects - Reduced bonus
        condition_effect = torch.where(condition == 3, 0.05, 0.0)  # Condition 3 (both) gets a smaller bonus
        uniformity += condition_effect
        coverage += condition_effect
        adhesion += condition_effect
        
        # Normalize to [0, 1]
        uniformity = torch.clamp(uniformity, 0.0, 1.0).unsqueeze(1)
        coverage = torch.clamp(coverage, 0.0, 1.0).unsqueeze(1)
        adhesion = torch.clamp(adhesion, 0.0, 1.0).unsqueeze(1)
        
        # Avoid local optima: if any objective is too high, add some noise
        for i in range(coverage.shape[0]):
            # Add noise if uniformity is too high
            if uniformity[i, 0] > 0.95:
                uniformity[i, 0] -= torch.rand(()) * 0.08
                coverage[i, 0] += torch.rand(()) * 0.04
            
            # Add noise if coverage is too high
            if coverage[i, 0] > 0.95:
                coverage[i, 0] -= torch.rand(()) * 0.08
                adhesion[i, 0] += torch.rand(()) * 0.04
            
            # Add noise if adhesion is too high
            if adhesion[i, 0] > 0.95:
                adhesion[i, 0] -= torch.rand(()) * 0.08
                uniformity[i, 0] += torch.rand(()) * 0.04
        
        return torch.cat([uniformity, coverage, adhesion], dim=-1).to(self.device)
    
    def _compute_hypervolume(self):
        """Compute hypervolume with caching for improved efficiency"""
        current_sample_count = self.Y.shape[0]
        
        # Use cached value if:
        # 1. Cache exists
        # 2. Iteration hasn't changed
        # 3. Sample count hasn't changed (ensures Y hasn't been updated)
        if (self._cached_hv is not None and 
            self._cached_hv_iteration == self.current_iteration and
            self._cached_hv_sample_count == current_sample_count):
            return self._cached_hv
        
        # Check if we have any samples
        if current_sample_count == 0:
            return 0.0
        
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.Y)
        hv = partitioning.compute_hypervolume().item()
        
        # Cache the result for future use
        self._cached_hv = hv
        self._cached_hv_iteration = self.current_iteration
        self._cached_hv_sample_count = current_sample_count
        
        return hv
    
    def _clean_float_for_json(self, value, param_index=None):
        """Clean a float value for JSON serialization based on parameter step size"""
        if param_index is not None and param_index < len(self.steps):
            step = self.steps[param_index].item()
            if step >= 1.0:
                return int(round(value))
            else:
                # Calculate decimal places from step
                step_str = f"{step:.10f}".rstrip('0').rstrip('.')
                if '.' in step_str:
                    decimal_places = len(step_str.split('.')[-1])
                else:
                    decimal_places = 0
                # Round and format
                rounded = round(value, decimal_places)
                return float(f"{rounded:.{decimal_places}f}")
        else:
            # Fallback: round to reasonable precision
            if value.is_integer():
                return int(value)
            return round(value, 10)
    
    def _clean_record_for_json(self, obj, param_index=None, is_param_list=False):
        """Recursively clean record data for JSON serialization
        
        Args:
            obj: The object to clean
            param_index: Current parameter index (for parameter lists)
            is_param_list: Whether the current list is a parameter list
        """
        if isinstance(obj, float):
            return self._clean_float_for_json(obj, param_index if is_param_list else None)
        elif isinstance(obj, (list, tuple)):
            # If this is a parameter list, use index as parameter index
            # Otherwise, don't use parameter index (for Y values, etc.)
            return [self._clean_record_for_json(item, i if is_param_list else None, False) 
                    for i, item in enumerate(obj)]
        elif isinstance(obj, dict):
            cleaned_dict = {}
            for k, v in obj.items():
                # Parameter lists: X, candidates, pareto_front.X
                if k in ['X', 'candidates'] or (k == 'pareto_front' and isinstance(v, dict) and 'X' in v):
                    cleaned_dict[k] = self._clean_record_for_json(v, None, True)
                else:
                    cleaned_dict[k] = self._clean_record_for_json(v, None, False)
            return cleaned_dict
        else:
            return obj
    
    def _save_iteration_data(self, record):
        """Save iteration data"""
        filename = f"{self.output_dir}/tkg_optimization_history_{self.experiment_id}.json"
        
        # Clean the record to ensure proper precision for discrete parameters
        # This ensures values like 0.6000000089 are serialized as 0.6
        cleaned_record = self._clean_record_for_json(record)
        
        with open(filename, 'a') as f:
            json.dump(cleaned_record, f)
            f.write("\n")
    
    def save_experiment_data(self, x, y):
        """Save experiment data to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        # Clean discrete parameter values to remove floating point precision errors
        x_cleaned = self._clean_discrete_values(x)
        
        new_rows = []
        for i in range(x_cpu.shape[0]):
            # Use cleaned values to ensure proper formatting
            data = {name: val for name, val in zip(self.param_names, x_cleaned[i])}
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
        
        filename = f"{self.output_dir}/tkg_experiment_{self.experiment_id}.csv"
        self.history.to_csv(filename, index=False)
    
    def get_pareto_front(self):
        """Calculate Pareto front
        
        返回当前阶段的参数空间的帕累托前沿（使用X，而不是X_full）
        """
        if self.Y.shape[0] == 0:
            # 返回当前阶段参数空间的空tensor
            active_param_count = len(self._get_active_param_indices())
            return torch.empty((0, active_param_count), device=self.device), torch.empty((0, 3), device=self.device)
        
        pareto_mask = torch.ones(self.Y.shape[0], dtype=torch.bool, device=self.device)
        for i in range(self.Y.shape[0]):
            for j in range(self.Y.shape[0]):
                if i != j and torch.all(self.Y[j] >= self.Y[i]) and torch.any(self.Y[j] > self.Y[i]):
                    pareto_mask[i] = False
                    break
        
        # Return current phase parameter space pareto front (using X, not X_full)
        return self.X[pareto_mask], self.Y[pareto_mask]
    
    def _generate_mixed_initial_samples_from_phase1(self, n_samples=5):
        """从Phase 1的结果中选择最优参数并生成混合初始点
        
        注意：由于现在使用 OptimizerManager 管理不同阶段的优化器，每个阶段有独立的优化器实例。
        Phase 2 的优化器在初始化时可能无法直接访问 Phase 1 的数据。
        如果无法获取 Phase 1 的数据，将使用随机采样。
        
        Args:
            n_samples: 要生成的混合样本数量
            
        Returns:
            Tensor of shape (n_samples, num_params) containing mixed candidate solutions
        """
        # 由于现在不再有 experiment_condition 参数，且每个阶段有独立的优化器，
        # Phase 2 的优化器在初始化时可能没有 Phase 1 的数据
        # 如果当前优化器没有足够的历史数据，使用随机采样
        if self.X.shape[0] < 2:
            print("【WARNING】Phase 2 优化器没有 Phase 1 的历史数据，使用随机采样生成混合初始点")
            return self.generate_initial_samples(n_samples)
        
        # 如果有历史数据，尝试从当前数据中生成混合样本
        # 注意：由于 Phase 2 的优化器可能包含 Phase 1 的数据（如果从 Manager 传递），
        # 但更可能的情况是 Phase 2 优化器是全新的，没有 Phase 1 数据
        # 在这种情况下，使用随机采样是更安全的选择
        print("【INFO】使用随机采样生成 Phase 2 混合初始点（Phase 1 数据由 OptimizerManager 管理）")
        return self.generate_initial_samples(n_samples)
    
    def _clean_discrete_values(self, tensor_data):
        """Clean discrete parameter values to remove floating point precision errors
        
        Args:
            tensor_data: Tensor of shape (n_samples, n_params) containing parameter values
            
        Returns:
            List of lists with cleaned values rounded to appropriate precision based on step size
        """
        import numpy as np
        data_np = tensor_data.cpu().numpy()
        cleaned_data = []
        
        for sample in data_np:
            cleaned_sample = []
            for i, value in enumerate(sample):
                step = self.steps[i].item()
                # Round to nearest step to eliminate floating point errors
                # This ensures values like 0.6000000089406967 become exactly 0.6
                cleaned_value = round(value / step) * step
                # Round to appropriate decimal places for display
                if step >= 1.0:
                    cleaned_value = int(round(cleaned_value))  # Convert to int for integer steps
                else:
                    # Count decimal places in step (e.g., 0.1 -> 1 decimal, 0.5 -> 1 decimal)
                    step_str = f"{step:.10f}".rstrip('0').rstrip('.')
                    if '.' in step_str:
                        decimal_places = len(step_str.split('.')[-1])
                    else:
                        decimal_places = 0
                    # Round and format to ensure exact representation
                    cleaned_value = round(cleaned_value, decimal_places)
                    # Convert to string and back to float to eliminate floating point errors
                    # This ensures JSON serialization produces clean values like 0.6 instead of 0.6000000089
                    cleaned_value = float(f"{cleaned_value:.{decimal_places}f}")
                cleaned_sample.append(cleaned_value)
            cleaned_data.append(cleaned_sample)
        
        return cleaned_data
    
    def _record_iteration(self, iteration, candidates, acquisition_values=None):
        """Record iteration data"""
        pareto_x, pareto_y = self.get_pareto_front()
        
        # Clean discrete parameter values to remove floating point precision errors
        X_cleaned = self._clean_discrete_values(self.X)
        candidates_cleaned = self._clean_discrete_values(candidates)
        pareto_x_cleaned = self._clean_discrete_values(pareto_x)
        
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": self.phase,
            "phase_1_subphase": self.phase_1_subphase if self.phase == 1 else None,
            "X": X_cleaned,
            "Y": self.Y.cpu().numpy().tolist(),
            "candidates": candidates_cleaned,
            "hypervolume": self._compute_hypervolume(),
            "pareto_front": {
                "X": pareto_x_cleaned,
                "Y": pareto_y.cpu().numpy().tolist(),
            },
            "acquisition_values": convert_to_list(acquisition_values),
        }
        
        self.iteration_history.append(record)
        self.hypervolume_history.append(record["hypervolume"])
        self._save_iteration_data(record)
    
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
            print(f"【INFO】Generated {X_init.shape[0]} initial samples")
            
            # 评估初始样本
            print("=== Initial Experiments ===")
            for candidate in X_init:
                candidate = candidate.unsqueeze(0)
                
                if simulation_flag:
                    y = self.simulate_experiment(candidate)
                else:
                    y = self.get_human_input(candidate)
                
                y_processed = self._process_observed_values(y)
                self._ensure_X_shape()
                
                self.X = torch.cat([self.X, candidate])
                self.Y = torch.cat([self.Y, y_processed])
                
                full_candidate = self._extend_candidates_to_full_space(candidate)
                self.X_full = torch.cat([self.X_full, full_candidate])
                self.save_experiment_data(full_candidate, y)
            
            self._record_iteration(iteration=0, candidates=X_init)
            self.current_iteration = 0
            return {
                'iteration': 0,
                'candidates': X_init,
                'hypervolume': self._compute_hypervolume()
            }
        
        # 更新迭代计数
        self.current_iteration += 1
        print(f"\n【INFO】Iteration {self.current_iteration}, Phase {self.phase}, Subphase: {self.phase_1_subphase if self.phase == 1 else 'N/A'}")
        
        try:
            # Initialize model
            print(f"【DEBUG】Iteration {self.current_iteration}: Initializing model...")
            mll, model = self.initialize_model()
            fit_gpytorch_mll(mll)
            print(f"【DEBUG】Iteration {self.current_iteration}: Model fitting completed")
            
            # Generate candidates using taKG acquisition function
            print(f"【DEBUG】Iteration {self.current_iteration}: Generating acquisition function...")
            acq_func = self._compute_trace_aware_knowledge_gradient(model)
            
            # Get phase-specific search bounds for active parameters only
            phase_search_bounds = self._get_phase_search_bounds()
            active_indices = self._get_active_param_indices()
            active_bounds = self._get_active_bounds()
            
            print(f"【DEBUG】Iteration {self.current_iteration}: Optimizing acquisition function...")
            print(f"【DEBUG】Active parameters: {[self.param_names[i] for i in active_indices]}")
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=phase_search_bounds,
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.seed},
                sequential=True
            )
            print(f"【DEBUG】Iteration {self.current_iteration}: Generated {candidates.shape[0]} candidates (active params only)")
            
            # Add diversity promotion
            # Note: self.X already contains only active parameters, so we don't need to index it
            if self.X.shape[0] > 0:
                # self.X already has shape (n_samples, num_active_params)
                # No need to use active_indices since X only contains active parameters
                with torch.no_grad():
                    active_X_normalized = normalize(self.X, active_bounds)
                    candidates_normalized = normalize(candidates, active_bounds)
                    distances = torch.cdist(candidates_normalized, active_X_normalized)
                    min_distances = distances.min(dim=1).values
                    avg_min_distance = min_distances.mean().item()
                    
                    if avg_min_distance < 0.1:
                        print(f"【INFO】Candidates are too similar, adding exploration noise")
                        exploration_noise = torch.randn_like(candidates) * 0.05
                        candidates += exploration_noise
            
            # Unnormalize and discretize active parameters
            candidates = unnormalize(candidates, active_bounds)
            active_steps = self._get_active_steps()
            for j in range(len(active_indices)):
                candidates[:, j] = torch.round(candidates[:, j] / active_steps[j]) * active_steps[j]
                candidates[:, j] = torch.clamp(candidates[:, j], active_bounds[0, j], active_bounds[1, j])
            
            # Extend to full parameter space and apply constraints
            # Note: Keep active parameters unchanged, only apply constraints to inactive parameters
            full_candidates = self._extend_candidates_to_full_space(candidates)
            
            # Apply constraints (these may modify parameters, but we'll restore active parameters afterwards)
            full_candidates = self._apply_safety_constraints(full_candidates)
            full_candidates = self._apply_phase_constraints(full_candidates)
            
            # Re-discretize only inactive parameters, keep active parameters unchanged
            inactive_indices = [i for i in range(len(self.parameters)) if i not in active_indices]
            for j in inactive_indices:
                full_candidates[:, j] = torch.round(full_candidates[:, j] / self.steps[j]) * self.steps[j]
                full_candidates[:, j] = torch.clamp(full_candidates[:, j], self.bounds[0, j], self.bounds[1, j])
            
            # Restore active parameters to their original values (ensure they weren't changed by constraints)
            for i, idx in enumerate(active_indices):
                full_candidates[:, idx] = candidates[:, i]
            
            # Extract active parameters before evaluation (should match original candidates)
            active_candidates = self._extract_active_parameters(full_candidates)
            
            # Evaluate candidates
            print(f"【DEBUG】Iteration {self.current_iteration}: Running experiments...")
            if simulation_flag:
                y_new = self.simulate_experiment(active_candidates)
            else:
                y_new = self.get_human_input(active_candidates)
            
            y_processed = self._process_observed_values(y_new)
            self._ensure_X_shape()
            
            # Update data
            print(f"【DEBUG】Iteration {self.current_iteration}: Updating data...")
            self.X = torch.cat([self.X, active_candidates])
            self.Y = torch.cat([self.Y, y_processed])
            self.X_full = torch.cat([self.X_full, full_candidates])
            self.save_experiment_data(full_candidates, y_new)
            
            # Compute hypervolume
            hv = self._compute_hypervolume()
            
            # Record iteration
            self._record_iteration(
                iteration=self.current_iteration,
                candidates=candidates,
                acquisition_values=acq_values,
            )
            
            print(f"【INFO】Current hypervolume: {hv:.4f}")
            print(f"【INFO】Added {candidates.shape[0]} new samples")
            
            return {
                'iteration': self.current_iteration,
                'candidates': candidates,
                'hypervolume': hv
            }
            
        except Exception as e:
            print(f"【ERROR】Error in iteration {self.current_iteration}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def optimize(self, n_iter=5, simulation_flag=True):
        """Run optimization process (保留用于向后兼容)"""
        self._set_seed(self.seed)
        
        # 生成初始样本（如果还没有）
        if self.X.shape[0] == 0:
            X_init = self.generate_initial_samples()
            print(f"【INFO】Generated {X_init.shape[0]} initial samples")
            
            for candidate in X_init:
                candidate = candidate.unsqueeze(0)
                if simulation_flag:
                    y = self.simulate_experiment(candidate)
                else:
                    y = self.get_human_input(candidate)
                
                y_processed = self._process_observed_values(y)
                self._ensure_X_shape()
                self.X = torch.cat([self.X, candidate])
                self.Y = torch.cat([self.Y, y_processed])
                full_candidate = self._extend_candidates_to_full_space(candidate)
                self.X_full = torch.cat([self.X_full, full_candidate])
                self.save_experiment_data(full_candidate, y)
            
            self._record_iteration(iteration=0, candidates=X_init)
        
        # 运行 n_iter 次迭代
        for _ in range(n_iter):
            self.run_single_step(simulation_flag=simulation_flag)
        
        print(f"\n【INFO】Optimization completed. Total samples: {self.X.shape[0]}")
        if len(self.hypervolume_history) > 0:
            print(f"【INFO】Final hypervolume: {self.hypervolume_history[-1]:.4f}")
    
    def get_human_input(self, candidates):
        """Get human input for experiment results
        
        This method should be overridden in subclasses or extended for real experiments.
        It provides a structured way to obtain actual measurements from experiments.
        
        Args:
            candidates: Tensor of shape (batch_size, num_params) containing the candidate solutions
                        Note: candidates may contain only active parameters (5, 7, or 11 params)
            
        Returns:
            Tensor of shape (batch_size, num_objectives) containing the measured objectives
        
        Note: For debugging purposes, this method currently returns deterministic values
        based on simulate_experiment() with a fixed seed. In production, this should be
        replaced with actual experiment measurements or API calls.
        """
        if len(candidates.shape) != 2:
            raise ValueError(f"Candidates must be 2D tensor, got shape {candidates.shape}")
        
        batch_size = candidates.shape[0]
        num_params = candidates.shape[1]
        
        print(f"【DEBUG】get_human_input called for {batch_size} candidates with {num_params} params (using simulated values for debugging)")
        
        # Ensure candidates are on the correct device
        if candidates.device != self.device:
            candidates = candidates.to(self.device)
        
        # ===== DEBUGGING MODE: Return deterministic values =====
        # Option 1: Return completely fixed values (same for all candidates)
        # Uncomment the following lines to return fixed values:
        # fixed_values = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64, device=self.device)
        # return fixed_values.repeat(batch_size, 1)
        
        # Option 2: Return deterministic values based on candidates (same input = same output)
        # This is the current implementation - uses simulate_experiment with fixed seed
        import random
        import numpy as np
        
        # Save random states (handle both CPU and CUDA)
        old_torch_cpu_state = torch.get_rng_state()
        old_np_state = np.random.get_state()
        old_random_state = random.getstate()
        
        # Save CUDA random state if available
        old_torch_cuda_state = None
        if torch.cuda.is_available():
            old_torch_cuda_state = torch.cuda.get_rng_state()
        
        try:
            # Use fixed seed for deterministic results
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            
            # Use simulate_experiment to get deterministic values
            # simulate_experiment can handle different input dimensions (5, 7, or 11 params)
            # This ensures same candidates always return same results
            y = self.simulate_experiment(candidates)
            
            # Ensure output is on correct device and has correct dtype
            if y.device != self.device:
                y = y.to(self.device)
            if y.dtype != torch.float64:
                y = y.to(torch.float64)
            
        except Exception as e:
            print(f"【ERROR】Error in get_human_input: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return default values
            y = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64, device=self.device).repeat(batch_size, 1)
            
        finally:
            # Restore original random states
            try:
                torch.set_rng_state(old_torch_cpu_state)
                np.random.set_state(old_np_state)
                random.setstate(old_random_state)
                if old_torch_cuda_state is not None:
                    torch.cuda.set_rng_state(old_torch_cuda_state)
            except Exception as e:
                print(f"【WARNING】Failed to restore random state: {e}")
        
        print(f"【DEBUG】get_human_input returning simulated values: shape={y.shape}, device={y.device}")
        return y

    def plot_pareto_front(self):
        """Plot Pareto front"""
        objectives = self.Y.cpu().numpy()
        pareto_x, pareto_y = self.get_pareto_front()
        pareto_front = pareto_y.cpu().numpy()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all experiments
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c="gray", alpha=0.4, label="All experiments")
        
        # Plot Pareto front
        if len(pareto_front) > 0:
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                c='red', label="Pareto front"
            )
        
        ax.set_xlabel('Uniformity')
        ax.set_ylabel('Coverage')
        ax.set_zlabel('Adhesion')
        ax.set_title('Pareto Front of Glass Metallization Optimization')
        ax.legend()
        ax.grid(True)
        
        # Save figure
        fig_name = f"{self.fig_dir}/tkg_pareto_front_{self.experiment_id}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"【INFO】Pareto front plot saved to: {fig_name}")
    
    def plot_hypervolume_convergence(self):
        """Plot hypervolume convergence history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(self.hypervolume_history)), self.hypervolume_history, 'b-', marker='o', label='Hypervolume')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Convergence')
        ax.grid(True)
        ax.legend()
        
        # Save figure
        fig_name = f"{self.fig_dir}/tkg_hypervolume_convergence_{self.experiment_id}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"【INFO】Hypervolume convergence plot saved to: {fig_name}")
    
    def _get_phase_description(self):
        """Get standardized phase description"""
        if self.phase == 1:
            subphase_desc = f" ({self.phase_1_subphase} subphase)" if hasattr(self, 'phase_1_subphase') else ""
            return f"Phase 1: Simple systems - exploring {self.phase_1_subphase if hasattr(self, 'phase_1_subphase') else 'oxide/organic'}{subphase_desc}"
        elif self.phase == 2:
            return "Phase 2: Complex systems (both organic and oxide) - exploring mixed parameter space"
        else:
            return f"Unknown phase: {self.phase}"
    
    def get_algorithm_info(self):
        """Get algorithm information"""
        info = {
            "name": "Trace-Aware Knowledge Gradient (taKG)",
            "acquisition_function": "qLogExpectedHypervolumeImprovement",
            "phase": self.phase,
            "phase_description": self._get_phase_description(),
            "hyperparameters": {
                "batch_size": self.batch_size,
                "num_restarts": self.num_restarts,
                "raw_samples": self.raw_samples,
                "n_init": self.n_init
            },
            "optimization_objectives": ["Uniformity", "Coverage", "Adhesion"]
        }
        
        # 添加Phase 1相关的参数信息
        if hasattr(self, 'phase_1_subphase'):
            info["phase_1"] = {
                "subphase": self.phase_1_subphase,
                "subphase_iteration": getattr(self, 'phase_1_subphase_iteration', 0)
            }
        
        return info
    
    def get_experiment_stats(self):
        """Get experiment statistics"""
        pareto_x, pareto_y = self.get_pareto_front()
        return {
            "total_experiments": self.X.shape[0],
            "total_iterations": len(self.iteration_history),
            "pareto_solutions": len(pareto_x),
            "current_phase": self.phase,
            "hypervolume": self._compute_hypervolume() if self.X.shape[0] > 0 else 0,
            "objectives": {
                "uniformity": {
                    "min": self.Y[:, 0].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 0].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 0].mean().item() if self.Y.shape[0] > 0 else 0
                },
                "coverage": {
                    "min": self.Y[:, 1].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 1].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 1].mean().item() if self.Y.shape[0] > 0 else 0
                },
                "adhesion": {
                    "min": self.Y[:, 2].min().item() if self.Y.shape[0] > 0 else 0,
                    "max": self.Y[:, 2].max().item() if self.Y.shape[0] > 0 else 0,
                    "mean": self.Y[:, 2].mean().item() if self.Y.shape[0] > 0 else 0
                }
            }
        }
    
    def get_heatmap_data(self, param1_idx=0, param2_idx=1, n_grid=20):
        """Generate heatmap data for two parameters"""
        if self.X.shape[0] == 0:
            return None
        
        # Create grid for the two parameters
        param1_min, param1_max = self.bounds[0, param1_idx].item(), self.bounds[1, param1_idx].item()
        param2_min, param2_max = self.bounds[0, param2_idx].item(), self.bounds[1, param2_idx].item()
        
        param1_grid = torch.linspace(param1_min, param1_max, n_grid, device=self.device)
        param2_grid = torch.linspace(param2_min, param2_max, n_grid, device=self.device)
        
        # Create meshgrid
        param1_mesh, param2_mesh = torch.meshgrid(param1_grid, param2_grid, indexing='ij')
        
        # Initialize input tensor with default values (mean of each parameter)
        default_params = (self.bounds[0] + self.bounds[1]) / 2
        X_grid = default_params.repeat(n_grid * n_grid, 1)
        
        # Fill in the grid values for the two parameters
        X_grid[:, param1_idx] = param1_mesh.flatten()
        X_grid[:, param2_idx] = param2_mesh.flatten()
        
        # Apply constraints
        X_grid = self._apply_safety_constraints(X_grid)
        
        # Normalize input
        X_normalized = normalize(X_grid, self.bounds)
        
        # Get model predictions
        _, model = self.initialize_model()
        with torch.no_grad():
            posterior = model.posterior(X_normalized)
            mean = posterior.mean
            variance = posterior.variance
        
        # Reshape to grid
        mean_grid = mean.reshape(n_grid, n_grid, 3)
        variance_grid = variance.reshape(n_grid, n_grid, 3)
        
        return {
            "param1_name": self.param_names[param1_idx],
            "param2_name": self.param_names[param2_idx],
            "param1_grid": param1_grid.cpu().numpy().tolist(),
            "param2_grid": param2_grid.cpu().numpy().tolist(),
            "mean": mean_grid.cpu().numpy().tolist(),
            "variance": variance_grid.cpu().numpy().tolist(),
            "objectives": ["Uniformity", "Coverage", "Adhesion"]
        }
    
    def get_trace_data(self):
        """Get trace data for visualization"""
        return {
            "iteration_history": self.iteration_history,
            "hypervolume_history": self.hypervolume_history,
            "experiment_data": {
                "X": self.X.cpu().numpy().tolist(),
                "Y": self.Y.cpu().numpy().tolist(),
                "param_names": self.param_names,
                "objectives": ["Uniformity", "Coverage", "Adhesion"]
            },
            "pareto_front": {
                "X": self.get_pareto_front()[0].cpu().numpy().tolist(),
                "Y": self.get_pareto_front()[1].cpu().numpy().tolist()
            }
        }
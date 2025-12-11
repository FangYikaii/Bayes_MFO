from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import logging
from .TkgOptimizer import TraceAwareKGOptimizer
from .DataVisualizer import DataVisualizer

# 配置日志
logger = logging.getLogger(__name__)

class OptimizerManager:
     # 阶段定义
    PHASE_1_OXIDE = 'phase_1_oxide'  # Phase 1 氧化物阶段
    PHASE_1_ORGANIC = 'phase_1_organic'  # Phase 1 有机物阶段
    PHASE_2 = 'phase_2'  # Phase 2 混合阶段

    def __init__(
        self,
        output_dir: str,
        fig_dir: str,
        seed: int = 42,
        device: Optional[str] = None,
        phase_1_oxide_max_iterations: int = 5,
        phase_1_organic_max_iterations: int = 5
    ):
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.seed = seed
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 阶段配置
        self.phase_1_oxide_max_iterations = phase_1_oxide_max_iterations
        self.phase_1_organic_max_iterations = phase_1_organic_max_iterations
        
        # 当前阶段
        self.current_phase = self.PHASE_1_OXIDE
        
        # 存储各阶段的优化器实例（完全独立）
        self.optimizers: Dict[str, TraceAwareKGOptimizer] = {}
        
        # 阶段迭代计数
        self.phase_iterations: Dict[str, int] = {
            self.PHASE_1_OXIDE: 0,
            self.PHASE_1_ORGANIC: 0,
            self.PHASE_2: 0
        }
        
        # 阶段超体积历史（用于计算改进率）
        self.phase_hypervolume_history: Dict[str, list] = {
            self.PHASE_1_OXIDE: [],
            self.PHASE_1_ORGANIC: [],
            self.PHASE_2: []
        }
        
        # 记录已生成可视化的阶段（避免重复生成）
        self.visualizations_generated: Dict[str, bool] = {
            self.PHASE_1_OXIDE: False,
            self.PHASE_1_ORGANIC: False,
            self.PHASE_2: False
        }

        # 定义参数空间（由 Manager 统一管理）
        self._define_parameter_spaces()

        # 初始化第一个阶段的优化器
        self._initialize_optimizer(self.PHASE_1_OXIDE)

    def _initialize_optimizer(self, phase: str):
        """初始化优化器"""
        if phase not in self.optimizers:
            # 获取阶段对应的参数空间配置（包含约束信息）
            param_space = self._get_phase_parameter_space(phase)
            
            self.optimizers[phase] = TraceAwareKGOptimizer(
                output_dir=self.output_dir,
                fig_dir=self.fig_dir,
                seed=self.seed,
                device=self.device,
                phase=phase,
                param_space=param_space,
                #TODO: 这里需要添加约束信息因为在不同的阶段有不同的约束
            )
        return self.optimizers[phase]

    def _define_parameter_spaces(self):
        
        # ===============================================
        # 有机物和氧化物的参数空间定义
        # ===============================================

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
            'metal_molar_ratio_b_a': (0, 10, 1)    # 0-10, step 1 (0 only when metal_b_type == 0)
        }
        
        # Combine all parameters (no longer need experiment_condition as phases are managed separately)
        self.parameters = {**self.organic_params, **self.metal_params}
        self.param_names = list(self.parameters.keys())
        
        # Record parameter indices for phase constraints
        num_organic = len(self.organic_params)
        num_metal = len(self.metal_params)
        self.organic_param_indices = list(range(num_organic))  # 索引 0-5
        self.metal_param_indices = list(range(num_organic, num_organic + num_metal))  # 索引 6-9
        
        # 记录完整参数索引
        self.all_param_indices = list(range(len(self.param_names)))
        
        # Create bounds and steps on CPU first
        bounds_cpu = torch.tensor([
            [param[0] for param in self.parameters.values()],
            [param[1] for param in self.parameters.values()]
        ], dtype=torch.float64)
        
        steps_cpu = torch.tensor([param[2] for param in self.parameters.values()])
        
        # Move tensors to the specified device
        self.bounds = bounds_cpu.to(self.device)
        self.steps = steps_cpu.to(self.device)

        # ===============================================
        # 有机物约束空间定义
        # ===============================================

        # Safety constraints for pH based on formula ID
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
        
        # ===============================================
        # TODO:氧化物的参数空间约束：金属A的类型、金属B的类型不能一样
        # 这里 metal_a_type 和  metal_b_type 的约束空间需要改动
        # ===============================================

        pass

    def _get_phase_parameter_space(self, phase: str) -> Dict[str, Any]:
        """根据阶段获取对应的参数空间配置
        
        Args:
            phase: 阶段名称
            
        Returns:
            包含参数空间配置的字典
        """
        try:
            if phase == self.PHASE_1_OXIDE:
                # Phase 1 氧化物阶段：只使用氧化物参数
                active_indices = self.metal_param_indices
                active_param_names = [self.param_names[i] for i in active_indices]
                active_bounds = self.bounds[:, active_indices].clone()  # 使用 clone 确保是新的 tensor
                active_steps = self.steps[active_indices].clone()
                
            elif phase == self.PHASE_1_ORGANIC:
                # Phase 1 有机物阶段：只使用有机物参数
                active_indices = self.organic_param_indices
                active_param_names = [self.param_names[i] for i in active_indices]
                active_bounds = self.bounds[:, active_indices].clone()
                active_steps = self.steps[active_indices].clone()
                
            elif phase == self.PHASE_2:
                # Phase 2：使用所有参数
                active_indices = self.all_param_indices
                active_param_names = self.param_names
                active_bounds = self.bounds.clone()
                active_steps = self.steps.clone()
                
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
            # 确保所有 tensor 都在正确的 device 上
            active_bounds = active_bounds.to(self.device)
            active_steps = active_steps.to(self.device)
            
            # 根据阶段添加约束信息（为所有阶段预留约束结构）
            constraints = {}
            
            if phase == self.PHASE_1_OXIDE:
                # Phase 1 氧化物阶段：预留氧化物约束空间
                # TODO: 后续添加氧化物约束，例如：metal_a_type 和 metal_b_type 不能相同
                constraints['oxide_constraints'] = {}  # 预留空间
                
            elif phase == self.PHASE_1_ORGANIC:
                # Phase 1 有机物阶段：添加pH约束
                constraints['pH_safety_constraints'] = self.pH_safety_constraints.copy()
                
            elif phase == self.PHASE_2:
                # Phase 2 混合阶段：添加所有约束
                constraints['pH_safety_constraints'] = self.pH_safety_constraints.copy()
                constraints['oxide_constraints'] = {}  # 预留空间
            
            result = {
                'bounds': active_bounds,  # 对应阶段参数空间的边界
                'steps': active_steps,  # 对应阶段参数空间的步长
                'parameters': active_param_names,  # 对应阶段参数名称
                'constraints': constraints  # 约束信息（所有阶段都有，即使为空）
            }
            
            return result
        except Exception as e:
            logger.error(f"获取阶段 {phase} 的参数空间失败: {str(e)}")
            raise

    def get_current_optimizer(self) -> TraceAwareKGOptimizer:
        """获取当前阶段的优化器"""
        if self.current_phase not in self.optimizers:
            self._initialize_optimizer(self.current_phase)
        return self.optimizers[self.current_phase]
    
    def _should_switch_phase(self) -> bool:
        """
        判断是否应该切换阶段
        
        Returns:
            bool: 如果应该切换阶段返回 True，否则返回 False
        """
        current_iter = self.phase_iterations[self.current_phase]
        
        # Phase 2 是最终阶段，不切换
        if self.current_phase == self.PHASE_2:
            return False
        
        # 检查是否达到最大迭代次数
        if self.current_phase == self.PHASE_1_OXIDE:
            if current_iter >= self.phase_1_oxide_max_iterations:
                return True
        elif self.current_phase == self.PHASE_1_ORGANIC:
            if current_iter >= self.phase_1_organic_max_iterations:
                return True
        
        return False
    
    def _get_best_parameters_from_phase(self, phase: str, n_best: int = 5):
        """
        从指定阶段获取最优参数
        
        Args:
            phase: 阶段名称
            n_best: 返回的最优参数数量
            
        Returns:
            最优参数张量，shape为 (n_best, n_params)
        """
        if phase not in self.optimizers:
            return None
        
        optimizer = self.optimizers[phase]
        if optimizer.X.shape[0] == 0:
            return None
        
        # 获取帕累托前沿
        pareto_x, pareto_y = optimizer.get_pareto_front()
        
        if pareto_x.shape[0] == 0:
            # 如果没有帕累托前沿，选择超体积贡献最大的点
            # 简单选择目标值总和最大的前 n_best 个
            if optimizer.Y.shape[0] == 0:
                return None
            # 计算每个样本的目标值总和
            y_sum = optimizer.Y.sum(dim=1)
            _, top_indices = torch.topk(y_sum, min(n_best, optimizer.Y.shape[0]))
            return optimizer.X[top_indices]
        else:
            # 从帕累托前沿中选择，优先选择目标值总和最大的
            y_sum = pareto_y.sum(dim=1)
            n_select = min(n_best, pareto_x.shape[0])
            _, top_indices = torch.topk(y_sum, n_select)
            return pareto_x[top_indices]
    
    def _combine_phase_parameters(self, oxide_params: torch.Tensor, organic_params: torch.Tensor, n_init: int):
        """
        组合 Phase 1 Oxide 和 Phase 1 Organic 的参数，生成 Phase 2 的初始样本
        
        Args:
            oxide_params: Phase 1 Oxide 的最优参数，shape为 (n_oxide, n_metal_params)
            organic_params: Phase 1 Organic 的最优参数，shape为 (n_organic, n_organic_params)
            n_init: 需要生成的初始样本数量
            
        Returns:
            组合后的参数，shape为 (n_init, n_all_params)
        """
        if oxide_params is None or organic_params is None:
            return None
        
        n_oxide = oxide_params.shape[0]
        n_organic = organic_params.shape[0]
        
        # 生成所有可能的组合
        combined_samples = []
        
        # 生成所有可能的组合（笛卡尔积）
        for i in range(n_oxide):
            for j in range(n_organic):
                # 组合参数：先是有机物参数，然后是金属参数
                combined = torch.cat([organic_params[j], oxide_params[i]], dim=0)
                combined_samples.append(combined)
        
        # 如果组合数量超过 n_init，选择前 n_init 个
        # 如果组合数量不足 n_init，随机重复选择组合直到达到 n_init
        if len(combined_samples) >= n_init:
            result = torch.stack(combined_samples[:n_init]).to(self.device)
        else:
            # 先使用所有组合
            result_list = combined_samples.copy()
            # 随机重复选择直到达到 n_init
            import random
            while len(result_list) < n_init:
                i = random.randint(0, n_oxide - 1)
                j = random.randint(0, n_organic - 1)
                combined = torch.cat([organic_params[j], oxide_params[i]], dim=0)
                result_list.append(combined)
            result = torch.stack(result_list[:n_init]).to(self.device)
        
        # 应用约束和离散化
        # 获取 Phase 2 的参数空间配置
        param_space = self._get_phase_parameter_space(self.PHASE_2)
        
        # 创建临时优化器来应用约束
        from .TkgOptimizer import TraceAwareKGOptimizer
        temp_optimizer = TraceAwareKGOptimizer(
            output_dir=self.output_dir,
            fig_dir=self.fig_dir,
            seed=self.seed,
            device=self.device,
            phase=self.PHASE_2,
            param_space=param_space
        )
        
        # 先应用离散化约束，确保所有参数都在边界内并符合步长要求
        result = temp_optimizer._apply_discretization_constraints(result, param_space['steps'], param_space['bounds'])
        
        # 然后应用阶段特定的约束
        result = temp_optimizer._apply_organic_safety_constraints(result)
        result = temp_optimizer._apply_oxide_constraints(result)
        
        # 再次应用离散化约束，确保约束修正后的值仍然在边界内
        result = temp_optimizer._apply_discretization_constraints(result, param_space['steps'], param_space['bounds'])
        
        return result
    
    def _switch_to_next_phase(self):
        """切换到下一个阶段"""
        if self.current_phase == self.PHASE_1_OXIDE:
            self.current_phase = self.PHASE_1_ORGANIC
            # 确保下一个阶段的优化器已初始化
            self._initialize_optimizer(self.current_phase)
        elif self.current_phase == self.PHASE_1_ORGANIC:
            self.current_phase = self.PHASE_2
            
            # 获取前两个阶段的最优参数
            oxide_best = self._get_best_parameters_from_phase(self.PHASE_1_OXIDE, n_best=5)
            organic_best = self._get_best_parameters_from_phase(self.PHASE_1_ORGANIC, n_best=5)
            
            if oxide_best is not None and organic_best is not None:
                # 初始化优化器
                self._initialize_optimizer(self.PHASE_2)
                optimizer = self.optimizers[self.PHASE_2]
                
                # 获取 Phase 2 优化器的 n_init 参数
                n_init = optimizer.n_init
                
                # 组合参数生成初始样本
                initial_samples = self._combine_phase_parameters(oxide_best, organic_best, n_init)
                
                if initial_samples is not None:
                    # 评估初始样本并更新数据（包括保存实验数据）
                    optimizer._evaluate_and_update(initial_samples, simulation_flag=True, iteration=0, acquisition_values=None)
                else:
                    logger.warning("无法生成组合初始样本，使用默认初始样本生成")
                    self._initialize_optimizer(self.PHASE_2)
            else:
                logger.warning("无法获取前两个阶段的最优参数，使用默认初始样本生成")
                self._initialize_optimizer(self.PHASE_2)
        else:
            raise ValueError(f"未知的阶段: {self.current_phase}")
    
    def run_single_iteration(self, simulation_flag: bool = True) -> Dict[str, Any]:
        """
        运行单次迭代
        
        Returns:
            包含迭代结果的字典，包括：
            - should_switch_phase: 是否应该切换阶段
            - phase: 当前阶段
            - iteration_result: 迭代结果
        """
        optimizer = self.get_current_optimizer()
        
        # 运行单次迭代
        step_result = optimizer.run_single_step(simulation_flag=simulation_flag)
        
        # 判断是否是初始样本生成（iteration == 0）
        is_initial_samples = step_result['iteration'] == 0
        
        # 更新阶段迭代计数（初始样本不算作一次迭代）
        if not is_initial_samples:
            self.phase_iterations[self.current_phase] += 1
        
        # 更新超体积历史
        current_hv = step_result['hypervolume']
        self.phase_hypervolume_history[self.current_phase].append(current_hv)
        
        # 检查是否需要切换阶段（初始样本生成后不检查，因为还没有真正的迭代）
        should_switch = False
        if not is_initial_samples:
            should_switch = self._should_switch_phase()
        
        result = {
            'should_switch_phase': should_switch,
            'phase': self.current_phase,
            'phase_iteration': self.phase_iterations[self.current_phase],
            'hypervolume': current_hv,
            'iteration': step_result['iteration'],
            'candidates': step_result['candidates'],
            'optimizer': optimizer
        }
        
        # 记录当前阶段（在切换前）
        current_phase_before_switch = self.current_phase
        
        # 如果需要切换阶段，执行切换
        if should_switch:
            old_phase = self.current_phase
            
            # 在切换阶段前，为旧阶段生成可视化图表（强制生成，确保使用最新数据）
            try:
                self.generate_visualizations_for_phase(old_phase, force=True)
                self.visualizations_generated[old_phase] = True
            except Exception as e:
                logger.error(f"生成阶段 {old_phase} 的可视化时出错: {str(e)}", exc_info=True)
            
            self._switch_to_next_phase()
            result['new_phase'] = self.current_phase
            result['old_phase'] = old_phase
            logger.info(f"阶段切换: {old_phase} -> {self.current_phase}")
        
        # 每次迭代后，为当前阶段更新可视化图表（每个阶段每次迭代都更新，包括初始样本生成）
        current_optimizer = self.get_current_optimizer()
        
        if current_optimizer.X.shape[0] > 0:
            # 所有阶段每次迭代后都更新可视化图表（包括初始样本生成后）
            try:
                if is_initial_samples:
                    logger.info(f"阶段 {self.current_phase} 初始样本生成后，生成可视化图表")
                else:
                    logger.info(f"阶段 {self.current_phase} 迭代 {self.phase_iterations[self.current_phase]} 后，更新可视化图表")
                
                self.generate_visualizations_for_phase(self.current_phase, force=True)
            except Exception as e:
                logger.error(f"更新阶段 {self.current_phase} 的可视化图表时出错: {str(e)}", exc_info=True)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        optimizer = self.get_current_optimizer()
        return {
            'current_phase': self.current_phase,
            'phase_iteration': self.phase_iterations[self.current_phase],
            'hypervolume': optimizer._compute_hypervolume() if optimizer.X.shape[0] > 0 else 0.0,
            'total_samples': optimizer.X.shape[0],
            'optimizer': optimizer
        }
    
    def update_max_iterations(
        self,
        phase_1_oxide_max_iterations: Optional[int] = None,
        phase_1_organic_max_iterations: Optional[int] = None
    ):
        """更新阶段的最大迭代次数（由前端动态设置）"""
        if phase_1_oxide_max_iterations is not None:
            self.phase_1_oxide_max_iterations = phase_1_oxide_max_iterations
        if phase_1_organic_max_iterations is not None:
            self.phase_1_organic_max_iterations = phase_1_organic_max_iterations
    
    def generate_visualizations_for_phase(self, phase: str, force: bool = False):
        """
        为指定阶段生成所有可视化图表
        
        Args:
            phase: 阶段名称
            force: 是否强制生成（即使已经生成过）
        """
        try:
            # 检查是否已经生成过（除非强制生成）
            if not force and self.visualizations_generated.get(phase, False):
                return
            
            if phase not in self.optimizers:
                logger.warning(f"阶段 {phase} 的优化器不存在，跳过可视化生成")
                return
            
            optimizer = self.optimizers[phase]
            if optimizer.X.shape[0] == 0:
                logger.warning(f"阶段 {phase} 没有数据，跳过可视化生成")
                return
            
            # 确保目录存在
            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir, exist_ok=True)
            
            # 创建可视化器
            visualizer = DataVisualizer(
                output_dir=self.output_dir,
                fig_dir=self.fig_dir
            )
            
            # 从优化器读取数据
            visualizer.read_data_from_optimizer(optimizer, phase)
            
            # 生成所有可视化图表
            visualizer.generate_all_visualizations(phase)
            
            # 标记为已生成
            self.visualizations_generated[phase] = True
            
            # 获取阶段对应的图表目录并验证
            phase_dir_map = {
                'phase_1_oxide': 'phase_1_oxide',
                'phase_1_organic': 'phase_1_organic',
                'phase_2': 'phase_2'
            }
            phase_dir = phase_dir_map.get(phase, phase)
            phase_fig_path = os.path.join(self.fig_dir, phase_dir)
            
            if os.path.exists(phase_fig_path):
                files = os.listdir(phase_fig_path)
                logger.info(f"阶段 {phase} 可视化图表已生成 ({len(files)} 个文件) -> {phase_fig_path}")
            
        except Exception as e:
            logger.error(f"生成阶段 {phase} 的可视化图表失败: {str(e)}", exc_info=True)
            # 不要抛出异常，避免影响主流程


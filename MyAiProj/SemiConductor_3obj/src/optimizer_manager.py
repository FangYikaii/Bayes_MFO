"""
OptimizerManager: 管理不同阶段的优化器实例
每个阶段的优化器完全独立，由 Manager 负责阶段切换逻辑
参数空间由 Manager 统一管理，根据阶段分配给对应的优化器
"""
from typing import Optional, Dict, Any
import torch
from .tkg_optimizer import TraceAwareKGOptimizer


class OptimizerManager:
    """管理不同阶段的优化器实例"""
    
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
        phase_1_organic_max_iterations: int = 5,
        phase_1_improvement_threshold: float = 0.05
    ):
        """
        初始化 OptimizerManager
        
        Args:
            output_dir: 输出目录
            fig_dir: 图片目录
            seed: 随机种子
            device: 设备
            phase_1_oxide_max_iterations: Phase 1 氧化物阶段最大迭代次数
            phase_1_organic_max_iterations: Phase 1 有机物阶段最大迭代次数
            phase_1_improvement_threshold: 改进率阈值（低于此值则切换阶段）
        """
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
        self.phase_1_improvement_threshold = phase_1_improvement_threshold
        
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
        
        # 定义参数空间（由 Manager 统一管理）
        self._define_parameter_spaces()
        
        # 初始化第一个阶段的优化器
        self._initialize_optimizer(self.PHASE_1_OXIDE)
    
    def _define_parameter_spaces(self):
        """定义参数空间（由 Manager 统一管理）"""
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
        
        print(f"【INFO】Parameter space defined in OptimizerManager: {len(self.param_names)} parameters")
    
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
            
            return {
                'param_names': self.param_names,  # 完整参数名称列表（用于记录）
                'active_param_names': active_param_names,  # 当前阶段的有效参数名称
                'active_indices': active_indices,  # 当前阶段的有效参数索引
                'bounds': self.bounds,  # 完整参数空间的边界
                'active_bounds': active_bounds,  # 当前阶段的有效参数边界
                'steps': self.steps,  # 完整参数空间的步长
                'active_steps': active_steps,  # 当前阶段的有效参数步长
                'organic_param_indices': self.organic_param_indices,
                'metal_param_indices': self.metal_param_indices,
                'all_param_indices': self.all_param_indices,
                'pH_safety_constraints': self.pH_safety_constraints,
                'parameters': self.parameters  # 完整参数定义
            }
        except Exception as e:
            print(f"【ERROR】Error in _get_phase_parameter_space for phase {phase}: {str(e)}")
            print(f"【DEBUG】Device: {self.device}")
            print(f"【DEBUG】Bounds shape: {self.bounds.shape if hasattr(self, 'bounds') else 'N/A'}")
            print(f"【DEBUG】Steps shape: {self.steps.shape if hasattr(self, 'steps') else 'N/A'}")
            raise
    
    def _initialize_optimizer(self, phase: str):
        """初始化指定阶段的优化器，并分配对应的参数空间"""
        if phase in self.optimizers:
            return  # 已经初始化过了
        
        try:
            # 获取阶段对应的参数空间配置
            param_space = self._get_phase_parameter_space(phase)
            
            # 根据阶段创建对应的优化器
            if phase == self.PHASE_1_OXIDE:
                # Phase 1 氧化物阶段：只优化氧化物参数
                optimizer = TraceAwareKGOptimizer(
                    output_dir=self.output_dir,
                    fig_dir=self.fig_dir,
                    seed=self.seed,
                    device=self.device,
                    phase=1,
                    phase_1_subphase='oxide',
                    param_space=param_space  # 传递参数空间配置
                )
            elif phase == self.PHASE_1_ORGANIC:
                # Phase 1 有机物阶段：只优化有机物参数
                optimizer = TraceAwareKGOptimizer(
                    output_dir=self.output_dir,
                    fig_dir=self.fig_dir,
                    seed=self.seed,
                    device=self.device,
                    phase=1,
                    phase_1_subphase='organic',
                    param_space=param_space  # 传递参数空间配置
                )
            elif phase == self.PHASE_2:
                # Phase 2：优化所有参数
                optimizer = TraceAwareKGOptimizer(
                    output_dir=self.output_dir,
                    fig_dir=self.fig_dir,
                    seed=self.seed,
                    device=self.device,
                    phase=2,
                    phase_1_subphase=None,
                    param_space=param_space  # 传递参数空间配置
                )
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
            self.optimizers[phase] = optimizer
            print(f"【INFO】Initialized optimizer for phase: {phase} with {len(param_space['active_param_names'])} active parameters")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"【ERROR】Failed to initialize optimizer for phase {phase}: {str(e)}")
            print(f"【ERROR】Traceback: {error_trace}")
            raise
    
    def get_current_optimizer(self) -> TraceAwareKGOptimizer:
        """获取当前阶段的优化器"""
        if self.current_phase not in self.optimizers:
            self._initialize_optimizer(self.current_phase)
        return self.optimizers[self.current_phase]
    
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
        
        # 如果需要切换阶段，执行切换
        if should_switch:
            old_phase = self.current_phase
            print(f"【DEBUG】准备切换阶段: {old_phase} -> ?")
            self._switch_to_next_phase()
            result['new_phase'] = self.current_phase
            result['old_phase'] = old_phase
            print(f"【DEBUG】阶段切换完成: {old_phase} -> {self.current_phase}")
        
        return result
    
    def _should_switch_phase(self) -> bool:
        """检查是否应该切换阶段"""
        current_iter = self.phase_iterations[self.current_phase]
        hv_history = self.phase_hypervolume_history[self.current_phase]
        
        print(f"【DEBUG】_should_switch_phase: current_phase={self.current_phase}, current_iter={current_iter}, hv_history_len={len(hv_history)}")
        
        # 检查是否达到最大迭代次数
        if self.current_phase == self.PHASE_1_OXIDE:
            max_iter = self.phase_1_oxide_max_iterations
            if current_iter >= max_iter:
                print(f"【INFO】Phase 1 Oxide 达到最大迭代次数: {current_iter}/{max_iter}")
                return True
        elif self.current_phase == self.PHASE_1_ORGANIC:
            max_iter = self.phase_1_organic_max_iterations
            if current_iter >= max_iter:
                print(f"【INFO】Phase 1 Organic 达到最大迭代次数: {current_iter}/{max_iter}")
                return True
        
        # 检查改进率（需要至少2次迭代）
        # 注意：初始样本（iteration 0）不算作一次迭代，所以需要至少3个超体积值（初始+2次迭代）
        if len(hv_history) >= 3:
            current_hv = hv_history[-1]
            previous_hv = hv_history[-2]
            if previous_hv > 0:
                improvement_rate = (current_hv - previous_hv) / previous_hv
                print(f"【DEBUG】改进率检查: current_hv={current_hv:.4f}, previous_hv={previous_hv:.4f}, improvement_rate={improvement_rate:.4f}")
                if improvement_rate < self.phase_1_improvement_threshold:
                    print(f"【INFO】{self.current_phase} 改进率低于阈值: {improvement_rate:.4f} < {self.phase_1_improvement_threshold}")
                    return True
        
        return False
    
    def _switch_to_next_phase(self):
        """切换到下一阶段"""
        old_phase = self.current_phase
        print(f"【DEBUG】_switch_to_next_phase: 从 {old_phase} 切换")
        
        if self.current_phase == self.PHASE_1_OXIDE:
            # 从氧化物阶段切换到有机物阶段
            self.current_phase = self.PHASE_1_ORGANIC
            # 重置有机物阶段的迭代计数（确保从0开始）
            self.phase_iterations[self.PHASE_1_ORGANIC] = 0
            self.phase_hypervolume_history[self.PHASE_1_ORGANIC] = []
            self._initialize_optimizer(self.PHASE_1_ORGANIC)
            print(f"【INFO】切换到 Phase 1 Organic (从 {old_phase})")
        elif self.current_phase == self.PHASE_1_ORGANIC:
            # 从有机物阶段切换到 Phase 2
            self.current_phase = self.PHASE_2
            # 重置 Phase 2 的迭代计数（确保从0开始）
            self.phase_iterations[self.PHASE_2] = 0
            self.phase_hypervolume_history[self.PHASE_2] = []
            self._initialize_optimizer(self.PHASE_2)
            print(f"【INFO】切换到 Phase 2 (从 {old_phase})")
        else:
            # Phase 2 是最后阶段，不切换
            print(f"【INFO】已在最后阶段: {self.current_phase}")
    
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
            print(f"【INFO】更新 Phase 1 Oxide 最大迭代次数: {phase_1_oxide_max_iterations}")
        if phase_1_organic_max_iterations is not None:
            self.phase_1_organic_max_iterations = phase_1_organic_max_iterations
            print(f"【INFO】更新 Phase 1 Organic 最大迭代次数: {phase_1_organic_max_iterations}")


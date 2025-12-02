import os
import sys
import json
import sqlite3
import numpy as np
import torch
from datetime import datetime
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

# 添加项目路径以便导入模型
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SemiConductor_3obj'))
from config import OUTPUT_DIR, FIGURE_DIR

class DatabaseManager:
    """数据库交互类，负责所有与SQLite数据库相关的操作"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"【ERROR】Database connection failed: {e}")
            raise
    
    def get_project(self, proj_name):
        """获取项目信息，返回字典格式便于通过列名访问"""
        conn = self._get_connection()
        try:
            # 使用row_factory返回字典格式的数据
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM AlgoProjInfo WHERE ProjName = ?", (proj_name,))
            result = cursor.fetchone()
            # 将sqlite3.Row对象转换为字典
            return dict(result) if result else None
        finally:
            conn.close()
    
    def insert_samples(self, proj_name, iter_id, samples, is_initial=False):
        """插入样本数据"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 获取项目的BatchNum值
            cursor.execute("SELECT BatchNum FROM AlgoProjInfo WHERE ProjName = ?", (proj_name,))
            batch_num_result = cursor.fetchone()
            if not batch_num_result:
                print(f"【ERROR】Failed to get BatchNum for project '{proj_name}'")
                return False
            batch_num = batch_num_result[0]
            
            # 插入样本，为每个样本生成ExpID（整数类型）
            for i, sample in enumerate(samples):
                # 确保i+1不超过batch_num
                exp_id = i + 1 if (i + 1) <= batch_num else batch_num
                cursor.execute('''
                INSERT INTO BayesExperData (
                    ExpID, ProjName, IterId, Formula, Concentration, Temperature, SoakTime, PH, CreateTime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (exp_id, proj_name, iter_id, sample[0], sample[1], sample[2], sample[3], sample[4], create_time))
            
            # 更新AlgoGenId
            cursor.execute('''
            UPDATE AlgoProjInfo 
            SET AlgoGenId = ?, IterId = ?
            WHERE ProjName = ?
            ''', (iter_id, iter_id, proj_name))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to insert samples: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def update_experiment_data(self, proj_name, iter_id, exp_data):
        """更新实验数据"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for data in exp_data:
                cursor.execute('''
                UPDATE BayesExperData 
                SET Coverage = ?, Adhesion = ?, Uniformity = ?, UpdateTime = ?
                WHERE ProjName = ? AND IterId = ? 
                  AND Formula = ? AND Concentration = ? 
                  AND Temperature = ? AND SoakTime = ? AND PH = ?
                ''', (
                    data.get('Coverage'), data.get('Adhesion'), data.get('Uniformity'), update_time,
                    proj_name, iter_id,
                    data.get('Formula'), data.get('Concentration'), 
                    data.get('Temperature'), data.get('SoakTime'), data.get('PH')
                ))
            
            # 更新AlgoRecevId表示该批次参数已回传成功
            cursor.execute('''
            UPDATE AlgoProjInfo 
            SET AlgoRecevId = ?
            WHERE ProjName = ?
            ''', (iter_id, proj_name))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to update experiment data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_experiment_data(self, proj_name):
        """获取项目的所有实验数据"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT Formula, Concentration, Temperature, SoakTime, PH, Coverage, Uniformity, Adhesion 
            FROM BayesExperData 
            WHERE ProjName = ?
            ''', (proj_name,))
            return cursor.fetchall()
        finally:
            conn.close()


class BayesianOptimizationAlgorithm:
    """贝叶斯优化算法类，负责核心的贝叶斯优化逻辑"""
    
    def __init__(self, device='cpu'):
        # 支持手动指定设备，默认为CPU
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"【INFO】Using device: {self.device}")
        
        # Define parameter spaces based on user requirements
        self._define_parameter_spaces()
        
        # 多目标优化设置 - 确保在指定设备上创建
        self.ref_point = torch.tensor([-0.1, -0.1, -0.1], dtype=torch.float64, device=self.device)
        self.num_restarts = 20
        self.raw_samples = 64
        self.n_init = 5
        
        # Optimization phases (1: simple systems, 2: complex systems)
        self.phase = 1
        self.phase_1_iterations = 3  # Number of iterations in phase 1
    
    def _define_parameter_spaces(self):
        """Define parameter spaces based on user requirements"""
        # Organic formula parameters
        self.organic_params = {
            'formula': (1, 30, 1),          # 1-30, step 1
            'concentration': (0.1, 5, 0.1),  # 0.1-5%, step 0.1
            'temperature': (25, 40, 5),      # 25-40°C, step 5
            'soak_time': (1, 30, 1),         # 1-30min, step 1
            'ph': (2.0, 14.0, 0.5),          # 2.0-14.0, step 0.5 (with safety constraints)
            'curing_time': (10, 30, 5)       # 10-30min, step 5
        }
        
        # Metal oxide parameters
        self.oxide_params = {
            'metal_a_type': (1, 20, 1),      # 1-20, step 1
            'metal_a_concentration': (10, 50, 10),  # 10-50%, step 10
            'metal_b_type': (0, 20, 1),      # 0-20, step 1 (0 means no metal B)
            'molar_ratio_b_a': (1, 10, 1)    # 1-10%, step 1
        }
        
        # Experiment condition parameter
        self.condition_params = {
            'experiment_condition': (1, 3, 1)  # 1-3, step 1 (1: organic only, 2: oxide only, 3: both)
        }
        
        # Combine all parameters
        self.parameters = {**self.organic_params, **self.oxide_params, **self.condition_params}
        self.param_names = list(self.parameters.keys())
        
        # Parameter bounds
        self.bounds = torch.tensor([
            [param[0] for param in self.parameters.values()],
            [param[1] for param in self.parameters.values()]
        ], dtype=torch.float64, device=self.device)
        
        # Parameter steps for discretization
        self.steps = torch.tensor([param[2] for param in self.parameters.values()], device=self.device)
        
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
    
    def _apply_safety_constraints(self, candidates):
        """Apply safety constraints to candidates"""
        for i in range(candidates.shape[0]):
            formula_id = int(round(candidates[i, 0].item()))
            
            # Apply pH safety constraints based on formula ID
            if formula_id in self.pH_safety_constraints:
                min_ph, max_ph = self.pH_safety_constraints[formula_id]
                candidates[i, 4] = torch.clamp(candidates[i, 4], min_ph, max_ph)
            
            # Apply metal oxide constraints: metal A and B cannot be the same type
            metal_a_type = int(round(candidates[i, 6].item()))
            metal_b_type = int(round(candidates[i, 8].item()))
            if metal_b_type != 0 and metal_a_type == metal_b_type:
                # If same, adjust metal B type to a different value
                candidates[i, 8] = torch.clamp(
                    candidates[i, 8] + 1 if candidates[i, 8] < self.bounds[1, 8] else candidates[i, 8] - 1,
                    self.bounds[0, 8], 
                    self.bounds[1, 8]
                )
        
        return candidates
    
    def _apply_phase_constraints(self, candidates):
        """Apply phase-specific constraints"""
        if self.phase == 1:
            # Phase 1: only simple systems (condition 1 or 2, not 3)
            for i in range(candidates.shape[0]):
                # Randomly assign to condition 1 or 2
                condition = 1 if torch.rand(1) > 0.5 else 2
                candidates[i, -1] = condition
        
        return candidates
    
    def generate_initial_samples(self, n_samples=None):
        """生成初始样本"""
        n = n_samples if n_samples is not None else self.n_init
        sobel_samples = draw_sobol_samples(bounds=self.bounds, n=n, q=1, seed=42).squeeze(1).to(self.device)
        
        # Discretization and constraints
        for i in range(len(self.parameters)):
            sobel_samples[:, i] = torch.round(sobel_samples[:, i] / self.steps[i]) * self.steps[i]
            sobel_samples[:, i] = torch.clamp(sobel_samples[:, i], self.bounds[0, i], self.bounds[1, i])
        
        # Apply safety and phase constraints
        sobel_samples = self._apply_safety_constraints(sobel_samples)
        sobel_samples = self._apply_phase_constraints(sobel_samples)
        
        return sobel_samples.cpu().numpy()
        
    def generate_new_samples(self, X_tensor, Y_tensor, batch_size):
        """使用taKG贝叶斯优化生成新样本"""
        # 确保输入张量在正确的设备上
        X_tensor = X_tensor.to(self.device)
        Y_tensor = Y_tensor.to(self.device)
        
        train_x = normalize(X_tensor, self.bounds)
        
        # 每个目标对应一个SingleTaskGP
        gp1 = SingleTaskGP(
            train_x,
            Y_tensor[:, 0:1],
            outcome_transform=Standardize(m=1),
        ).to(self.device)  # 确保模型在正确设备上
        
        gp2 = SingleTaskGP(
            train_x,
            Y_tensor[:, 1:2],
            outcome_transform=Standardize(m=1),
        ).to(self.device)  # 确保模型在正确设备上
        
        gp3 = SingleTaskGP(
            train_x,
            Y_tensor[:, 2:3],
            outcome_transform=Standardize(m=1),
        ).to(self.device)  # 确保模型在正确设备上
        
        model = ModelListGP(gp1, gp2, gp3).to(self.device)  # 确保模型在正确设备上
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        # 训练模型
        fit_gpytorch_mll(mll)
        
        # 计算采集函数 - 使用taKG改进的采集函数
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=pred)
        acq_func = qLogExpectedHypervolumeImprovement(model=model, ref_point=self.ref_point, partitioning=partitioning)
        
        # 标准化边界
        standard_bounds = torch.zeros_like(self.bounds, device=self.device)
        standard_bounds[1, :] = 1.0
        
        # 优化采集函数
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": 200, "seed": 42},
            sequential=True
        )
        
        # 反标准化并离散化候选点
        candidates = unnormalize(candidates, self.bounds)
        for j in range(len(self.parameters)):
            candidates[:, j] = torch.round(candidates[:, j] / self.steps[j]) * self.steps[j]
            candidates[:, j] = torch.clamp(candidates[:, j], self.bounds[0, j], self.bounds[1, j])
        
        # Apply safety and phase constraints
        candidates = self._apply_safety_constraints(candidates)
        candidates = self._apply_phase_constraints(candidates)
        
        return candidates.cpu().numpy()
        
    def prepare_training_data(self, experiment_data):
        """准备训练数据"""
        X = []
        Y = []
        for row in experiment_data:
            # For compatibility with existing database structure, we need to handle the new parameters
            # We'll pad with default values for the new parameters not present in the database
            # Original parameters: Formula, Concentration, Temperature, SoakTime, PH
            # New parameters: Formula, Concentration, Temperature, SoakTime, PH, CuringTime, MetalA, MetalAConc, MetalB, MolarRatio, Condition
            
            # Extract original parameters
            formula = row[0]
            concentration = row[1]
            temperature = row[2]
            soak_time = row[3]
            ph = row[4]
            
            # Default values for new parameters
            curing_time = 20  # Default curing time
            metal_a_type = 1  # Default metal A type
            metal_a_concentration = 30  # Default metal A concentration
            metal_b_type = 0  # Default: no metal B
            molar_ratio = 5  # Default molar ratio
            condition = 1  # Default: organic only
            
            # Create full parameter list
            full_params = [formula, concentration, temperature, soak_time, ph, curing_time, 
                          metal_a_type, metal_a_concentration, metal_b_type, molar_ratio, condition]
            
            X.append(full_params)  # 参数
            Y.append(list(row[5:8]))  # 目标值
            
        # 转换为torch张量
        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float64, device=self.device)
        
        return X_tensor, Y_tensor


class BayesOptimizationService:
    """接口业务逻辑类，负责处理具体的业务逻辑和调用其他类"""
    
    def __init__(self, db_path=r"D:\Parameter\Meta\DB\sampleData.db", device='cpu'):
        self.db_manager = DatabaseManager(db_path)
        self.optimizer = BayesianOptimizationAlgorithm(device=device)
    
    def update_experiment_data(self, proj_name, iter_id, exp_data_json):
        """更新实验数据"""
        # 检查项目是否存在
        proj = self.db_manager.get_project(proj_name)
        if not proj:
            print(f"【ERROR】Project '{proj_name}' does not exist")
            return False
        
        # 检查该批次参数是否已生成（AlgoGenId是否大于等于iter_id）
        try:
            algo_gen_id = proj['AlgoGenId']
            algo_recev_id = proj['AlgoRecevId']
        except KeyError as e:
            print(f"【ERROR】Database column not found: {e}")
            return False
        
        if algo_gen_id < iter_id:
            print(f"【ERROR】Parameters for iteration {iter_id} have not been generated yet")
            return False
        
        # 检查该批次数据是否已回传（AlgoRecevId是否小于iter_id）
        if algo_recev_id >= iter_id:
            print(f"【ERROR】Experiment data for iteration {iter_id} has already been received")
            return False
        
        # 解析实验数据
        try:
            # 尝试直接解析JSON
            data_list = json.loads(exp_data_json)
        except json.JSONDecodeError:
            try:
                # 尝试Base64解码（如果是从C#传来的Base64编码数据）
                import base64
                decoded_bytes = base64.b64decode(exp_data_json)
                decoded_json = decoded_bytes.decode('utf-8')
                data_list = json.loads(decoded_json)
            except Exception as e:
                print(f"【ERROR】Invalid experiment data format: {e}")
                return False
        
        # 更新数据库
        success = self.db_manager.update_experiment_data(proj_name, iter_id, data_list)
        if success:
            print(f"【INFO】Experiment data for project '{proj_name}', iteration {iter_id} updated successfully")
        return success
    
    def get_iteration_sample(self, proj_name, iter_id):
        """获取迭代样本，当iter_id=1时生成初始样本"""
        # 检查项目是否存在
        proj = self.db_manager.get_project(proj_name)
        if not proj:
            print(f"【ERROR】Project '{proj_name}' does not exist")
            return False
        
        try:
            # 使用列名获取值，而不是索引
            max_iter = proj['IterNum']
            algo_gen_id = proj['AlgoGenId']
            algo_recev_id = proj['AlgoRecevId']
            batch_size = proj['BatchNum']
        except KeyError as e:
            print(f"【ERROR】Database column not found: {e}")
            return False
        
        # 检查迭代参数是否在范围内
        if iter_id > max_iter:
            print(f"【ERROR】Iteration {iter_id} exceeds maximum iterations {max_iter}")
            return False
        
        # 检查当前批次参数是否未生成
        if algo_gen_id >= iter_id:
            print(f"【ERROR】Parameters for iteration {iter_id} have already been generated")
            return False
        
        # 区分处理初始样本(iter_id=1)和迭代样本(iter_id>1)
        if iter_id == 1:
            # 生成初始样本
            samples = self.optimizer.generate_initial_samples(batch_size)
            print(f"【INFO】Generating initial samples for project '{proj_name}'")
        else:
            # 检查上一批次实验数据是否已回传
            if algo_recev_id != iter_id - 1:
                print(f"【ERROR】Experiment data for iteration {iter_id - 1} not received yet")
                return False
            
            # 从数据库读取历史数据
            experiment_data = self.db_manager.get_experiment_data(proj_name)
            if len(experiment_data) == 0:
                print(f"【ERROR】No experiment data found for project '{proj_name}'")
                return False

            # 准备训练数据
            X_tensor, Y_tensor = self.optimizer.prepare_training_data(experiment_data)
            
            # 生成新的样本
            samples = self.optimizer.generate_new_samples(X_tensor, Y_tensor, batch_size)
            print(f"【INFO】Generating iteration {iter_id} samples for project '{proj_name}'")
        
        # 写入数据库
        success = self.db_manager.insert_samples(proj_name, iter_id, samples, iter_id == 1)
        if success:
            print(f"【INFO】Samples generated for project '{proj_name}', iteration {iter_id}")
        return success


def main():
    """主函数，处理命令行参数并执行相应操作"""
    if len(sys.argv) < 3:
        print("【INFO】Usage: python bayes.py <command> <args> [--device=cpu|cuda]")
        print("【INFO】Commands:")
        print("【INFO】  UpdateExpData <ProjName> <IterId> <ExpDataJson>")
        print("【INFO】  GetIterIdSample <ProjName> <IterId> (use IterId=1 for initial samples)")
        return
    
    command = sys.argv[1]
    
    # 从命令行参数中获取设备选择（可选）
    device = 'cpu'  # 默认使用CPU
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--device='):
            device_arg = arg.split('=')[1].lower()
            if device_arg in ['cpu', 'cuda']:
                device = device_arg
            break
    
    # 创建服务实例，传递设备参数
    service = BayesOptimizationService(device=device)
    
    # 处理命令，需要调整参数索引，因为可能有--device参数
    cmd_args = [arg for arg in sys.argv[2:] if not arg.startswith('--device=')]
    
    if command == "UpdateExpData" and len(cmd_args) >= 3:
        proj_name = cmd_args[0]
        iter_id = int(cmd_args[1])
        exp_data_json = cmd_args[2]
        success = service.update_experiment_data(proj_name, iter_id, exp_data_json)
        print(f"【OUTPUT】 {success} for project '{proj_name}'")
    elif command == "GetIterIdSample" and len(cmd_args) >= 2:
        proj_name = cmd_args[0]
        iter_id = int(cmd_args[1])
        success = service.get_iteration_sample(proj_name, iter_id)
        print(f"【OUTPUT】 {success} for project '{proj_name}'")
    else:
        print("【ERROR】Invalid command or arguments")
        print("【INFO】Usage: python bayes.py <command> <args> [--device=cpu|cuda]")


if __name__ == "__main__":
    main()
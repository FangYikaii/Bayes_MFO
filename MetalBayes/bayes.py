import os
import sys
import json
import sqlite3
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, Any

# 添加项目路径以便导入模型
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from src.OptimizerManager import OptimizerManager
from src.TkgOptimizer import TraceAwareKGOptimizer


class DatabaseManager:
    """数据库交互类，负责所有与SQLite数据库相关的操作"""
    
    # 定义各阶段对应的表名
    PHASE_TABLE_MAP = {
        'phase_1_oxide': 'BayesExperData_Phase1_Oxide',
        'phase_1_organic': 'BayesExperData_Phase1_Organic',
        'phase_2': 'BayesExperData_Phase2'
    }
    
    def __init__(self, db_path):
        self.db_path = db_path
        # 初始化时创建所有阶段的表（如果不存在）
        self._ensure_tables_exist()
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"【ERROR】Database connection failed: {e}")
            raise
    
    def _get_table_name(self, phase):
        """根据阶段获取对应的表名"""
        return self.PHASE_TABLE_MAP.get(phase, None)
    
    def _ensure_tables_exist(self):
        """确保所有阶段的表都存在，如果不存在则创建"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 为每个阶段创建表
            for phase, table_name in self.PHASE_TABLE_MAP.items():
                # 检查表是否存在
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                ''', (table_name,))
                
                if cursor.fetchone() is None:
                    # 表不存在，创建表
                    self._create_table(cursor, table_name, phase)
                    print(f"【INFO】Created table: {table_name}")
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to ensure tables exist: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _create_table(self, cursor, table_name, phase):
        """创建指定阶段的表，每个阶段有不同的表结构对应自己的参数空间
        
        Args:
            cursor: 数据库游标
            table_name: 表名
            phase: 阶段名称
        """
        if phase == OptimizerManager.PHASE_1_OXIDE:
            # Phase 1 Oxide: 只包含金属参数
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    ExpID INTEGER NOT NULL,
                    ProjName VARCHAR(255) NOT NULL,
                    IterId INTEGER NOT NULL,
                    MetalAType INTEGER NOT NULL,
                    MetalAConc REAL NOT NULL,
                    MetalBType INTEGER NOT NULL,
                    MetalMolarRatio INTEGER NOT NULL,
                    Coverage REAL,
                    Uniformity REAL,
                    Adhesion REAL,
                    CreateTime VARCHAR(255),
                    UpdateTime VARCHAR(255),
                    PRIMARY KEY (ExpID, ProjName, IterId)
                )
            ''')
        elif phase == OptimizerManager.PHASE_1_ORGANIC:
            # Phase 1 Organic: 只包含有机物参数
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    ExpID INTEGER NOT NULL,
                    ProjName VARCHAR(255) NOT NULL,
                    IterId INTEGER NOT NULL,
                    Formula INTEGER NOT NULL,
                    Concentration REAL NOT NULL,
                    Temperature INTEGER NOT NULL,
                    SoakTime INTEGER NOT NULL,
                    PH REAL NOT NULL,
                    CuringTime INTEGER NOT NULL,
                    Coverage REAL,
                    Uniformity REAL,
                    Adhesion REAL,
                    CreateTime VARCHAR(255),
                    UpdateTime VARCHAR(255),
                    PRIMARY KEY (ExpID, ProjName, IterId)
                )
            ''')
        elif phase == OptimizerManager.PHASE_2:
            # Phase 2: 包含所有参数
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    ExpID INTEGER NOT NULL,
                    ProjName VARCHAR(255) NOT NULL,
                    IterId INTEGER NOT NULL,
                    Formula INTEGER NOT NULL,
                    Concentration REAL NOT NULL,
                    Temperature INTEGER NOT NULL,
                    SoakTime INTEGER NOT NULL,
                    PH REAL NOT NULL,
                    CuringTime INTEGER NOT NULL,
                    MetalAType INTEGER NOT NULL,
                    MetalAConc REAL NOT NULL,
                    MetalBType INTEGER NOT NULL,
                    MetalMolarRatio INTEGER NOT NULL,
                    Coverage REAL,
                    Uniformity REAL,
                    Adhesion REAL,
                    CreateTime VARCHAR(255),
                    UpdateTime VARCHAR(255),
                    PRIMARY KEY (ExpID, ProjName, IterId)
                )
            ''')
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
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
    
    def insert_samples(self, proj_name, iter_id, samples, phase, is_initial=False):
        """插入样本数据
        
        Args:
            proj_name: 项目名称
            iter_id: 迭代ID
            samples: 样本列表，每个样本是一个参数数组
            phase: 当前阶段 ('phase_1_oxide', 'phase_1_organic', 'phase_2')
            is_initial: 是否为初始样本
        """
        # 获取对应阶段的表名
        table_name = self._get_table_name(phase)
        if not table_name:
            print(f"【ERROR】Unknown phase: {phase}")
            return False
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            create_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 确保表存在
            self._ensure_table_exists(cursor, table_name, phase)
            
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
                
                # 根据阶段构建不同的插入语句，只插入对应阶段的参数
                if phase == OptimizerManager.PHASE_1_OXIDE:
                    # Phase 1 Oxide: 只插入金属参数
                    cursor.execute(f'''
                    INSERT INTO {table_name} (
                        ExpID, ProjName, IterId,
                        MetalAType, MetalAConc, MetalBType, MetalMolarRatio, CreateTime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exp_id, proj_name, iter_id,
                        int(round(sample[0])), float(sample[1]), int(round(sample[2])), int(round(sample[3])),
                        create_time
                    ))
                elif phase == OptimizerManager.PHASE_1_ORGANIC:
                    # Phase 1 Organic: 只插入有机物参数
                    cursor.execute(f'''
                    INSERT INTO {table_name} (
                        ExpID, ProjName, IterId,
                        Formula, Concentration, Temperature, SoakTime, PH, CuringTime, CreateTime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exp_id, proj_name, iter_id,
                        int(round(sample[0])), float(sample[1]), int(round(sample[2])), 
                        int(round(sample[3])), float(sample[4]), int(round(sample[5])),
                        create_time
                    ))
                elif phase == OptimizerManager.PHASE_2:
                    # Phase 2: 插入所有参数
                    cursor.execute(f'''
                    INSERT INTO {table_name} (
                        ExpID, ProjName, IterId,
                        Formula, Concentration, Temperature, SoakTime, PH, CuringTime,
                        MetalAType, MetalAConc, MetalBType, MetalMolarRatio, CreateTime
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        exp_id, proj_name, iter_id,
                        int(round(sample[0])), float(sample[1]), int(round(sample[2])), 
                        int(round(sample[3])), float(sample[4]), int(round(sample[5])),
                        int(round(sample[6])), float(sample[7]), int(round(sample[8])), int(round(sample[9])),
                        create_time
                    ))
            
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
    
    def _ensure_table_exists(self, cursor, table_name, phase):
        """确保指定表存在，如果不存在则创建"""
        cursor.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        ''', (table_name,))
        
        if cursor.fetchone() is None:
            self._create_table(cursor, table_name, phase)
    
    def update_experiment_data(self, proj_name, iter_id, exp_data):
        """更新实验数据"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for data in exp_data:
                # 根据阶段构建更新条件
                phase = data.get('Phase', OptimizerManager.PHASE_2)
                
                # 获取对应阶段的表名
                table_name = self._get_table_name(phase)
                if not table_name:
                    print(f"【ERROR】Unknown phase: {phase}")
                    continue
                
                if phase == OptimizerManager.PHASE_1_OXIDE:
                    # Phase 1 Oxide: 使用金属参数作为匹配条件
                    cursor.execute(f'''
                    UPDATE {table_name} 
                    SET Coverage = ?, Adhesion = ?, Uniformity = ?, UpdateTime = ?
                    WHERE ProjName = ? AND IterId = ?
                      AND MetalAType = ? AND MetalAConc = ? 
                      AND MetalBType = ? AND MetalMolarRatio = ?
                    ''', (
                        data.get('Coverage'), data.get('Adhesion'), data.get('Uniformity'), update_time,
                        proj_name, iter_id,
                        data.get('MetalAType'), data.get('MetalAConc'),
                        data.get('MetalBType'), data.get('MetalMolarRatio')
                    ))
                elif phase == OptimizerManager.PHASE_1_ORGANIC:
                    # Phase 1 Organic: 使用有机物参数作为匹配条件
                    cursor.execute(f'''
                    UPDATE {table_name} 
                    SET Coverage = ?, Adhesion = ?, Uniformity = ?, UpdateTime = ?
                    WHERE ProjName = ? AND IterId = ?
                      AND Formula = ? AND Concentration = ? 
                      AND Temperature = ? AND SoakTime = ? AND PH = ? AND CuringTime = ?
                    ''', (
                        data.get('Coverage'), data.get('Adhesion'), data.get('Uniformity'), update_time,
                        proj_name, iter_id,
                        data.get('Formula'), data.get('Concentration'), 
                        data.get('Temperature'), data.get('SoakTime'), data.get('PH'), data.get('CuringTime')
                    ))
                elif phase == OptimizerManager.PHASE_2:
                    # Phase 2: 使用所有参数作为匹配条件
                    cursor.execute(f'''
                    UPDATE {table_name} 
                    SET Coverage = ?, Adhesion = ?, Uniformity = ?, UpdateTime = ?
                    WHERE ProjName = ? AND IterId = ?
                      AND Formula = ? AND Concentration = ? 
                      AND Temperature = ? AND SoakTime = ? AND PH = ? AND CuringTime = ?
                      AND MetalAType = ? AND MetalAConc = ? 
                      AND MetalBType = ? AND MetalMolarRatio = ?
                    ''', (
                        data.get('Coverage'), data.get('Adhesion'), data.get('Uniformity'), update_time,
                        proj_name, iter_id,
                        data.get('Formula'), data.get('Concentration'), 
                        data.get('Temperature'), data.get('SoakTime'), data.get('PH'), data.get('CuringTime'),
                        data.get('MetalAType'), data.get('MetalAConc'),
                        data.get('MetalBType'), data.get('MetalMolarRatio')
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
    
    def get_experiment_data(self, proj_name, phase=None):
        """获取项目的实验数据
        
        Args:
            proj_name: 项目名称
            phase: 阶段名称，如果为None则获取所有阶段的数据（使用UNION合并）
        
        Returns:
            返回统一格式的数据，包含所有字段（缺失字段用NULL填充）
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if phase:
                # 获取指定阶段的数据
                table_name = self._get_table_name(phase)
                if not table_name:
                    print(f"【ERROR】Unknown phase: {phase}")
                    return []
                
                if phase == OptimizerManager.PHASE_1_OXIDE:
                    # Phase 1 Oxide: 只查询金属参数，有机物参数用NULL填充
                    cursor.execute(f'''
                    SELECT NULL as Formula, NULL as Concentration, NULL as Temperature, 
                           NULL as SoakTime, NULL as PH, NULL as CuringTime,
                           MetalAType, MetalAConc, MetalBType, MetalMolarRatio,
                           Coverage, Uniformity, Adhesion
                    FROM {table_name} 
                    WHERE ProjName = ?
                    ''', (proj_name,))
                elif phase == OptimizerManager.PHASE_1_ORGANIC:
                    # Phase 1 Organic: 只查询有机物参数，金属参数用NULL填充
                    cursor.execute(f'''
                    SELECT Formula, Concentration, Temperature, SoakTime, PH, CuringTime,
                           NULL as MetalAType, NULL as MetalAConc, NULL as MetalBType, NULL as MetalMolarRatio,
                           Coverage, Uniformity, Adhesion
                    FROM {table_name} 
                    WHERE ProjName = ?
                    ''', (proj_name,))
                elif phase == OptimizerManager.PHASE_2:
                    # Phase 2: 查询所有参数
                    cursor.execute(f'''
                    SELECT Formula, Concentration, Temperature, SoakTime, PH, CuringTime,
                           MetalAType, MetalAConc, MetalBType, MetalMolarRatio,
                           Coverage, Uniformity, Adhesion
                    FROM {table_name} 
                    WHERE ProjName = ?
                    ''', (proj_name,))
            else:
                # 获取所有阶段的数据，使用UNION合并，统一字段顺序
                cursor.execute('''
                SELECT NULL as Formula, NULL as Concentration, NULL as Temperature, 
                       NULL as SoakTime, NULL as PH, NULL as CuringTime,
                       MetalAType, MetalAConc, MetalBType, MetalMolarRatio,
                       Coverage, Uniformity, Adhesion
                FROM BayesExperData_Phase1_Oxide
                WHERE ProjName = ?
                UNION ALL
                SELECT Formula, Concentration, Temperature, SoakTime, PH, CuringTime,
                       NULL as MetalAType, NULL as MetalAConc, NULL as MetalBType, NULL as MetalMolarRatio,
                       Coverage, Uniformity, Adhesion
                FROM BayesExperData_Phase1_Organic
                WHERE ProjName = ?
                UNION ALL
                SELECT Formula, Concentration, Temperature, SoakTime, PH, CuringTime,
                       MetalAType, MetalAConc, MetalBType, MetalMolarRatio,
                       Coverage, Uniformity, Adhesion
                FROM BayesExperData_Phase2
                WHERE ProjName = ?
                ''', (proj_name, proj_name, proj_name))
            return cursor.fetchall()
        finally:
            conn.close()


class BayesianOptimizationAlgorithm:
    """贝叶斯优化算法类，负责核心的贝叶斯优化逻辑（适配MetalBayes的多阶段特性）"""
    
    def __init__(self, device='cpu', output_dir=None, fig_dir=None, seed=42,
                 phase_1_oxide_max_iterations=5, phase_1_organic_max_iterations=5):
        # 支持手动指定设备，默认为CPU
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"【INFO】Using device: {self.device}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(project_root, 'data', 'output')
        if fig_dir is None:
            fig_dir = os.path.join(project_root, 'data', 'figures')
        
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)
        
        # 创建 OptimizerManager 实例
        self.manager = OptimizerManager(
            output_dir=output_dir,
            fig_dir=fig_dir,
            seed=seed,
            device=self.device,
            phase_1_oxide_max_iterations=phase_1_oxide_max_iterations,
            phase_1_organic_max_iterations=phase_1_organic_max_iterations
        )
        
        # 当前阶段
        self.current_phase = self.manager.current_phase
    
    def generate_initial_samples(self, phase=None, n_samples=None):
        """生成初始样本
        
        Args:
            phase: 阶段名称，如果为None则使用当前阶段
            n_samples: 样本数量，如果为None则使用优化器的默认值
        """
        if phase is None:
            phase = self.current_phase
        
        # 切换到指定阶段
        if self.manager.current_phase != phase:
            self.manager.current_phase = phase
            self.current_phase = phase
        
        # 获取当前阶段的优化器
        optimizer = self.manager.get_current_optimizer()
        
        # 生成初始样本
        if n_samples is None:
            samples = optimizer.generate_initial_samples()
        else:
            samples = optimizer.generate_initial_samples(n_init=n_samples)
        
        return samples.cpu().numpy()
    
    def generate_new_samples(self, experiment_data, batch_size=None):
        """使用taKG贝叶斯优化生成新样本
        
        Args:
            experiment_data: 实验数据列表（从数据库获取）
            batch_size: 批次大小，如果为None则使用优化器的默认值
        """
        # 获取当前阶段的优化器
        optimizer = self.manager.get_current_optimizer()
        
        if len(experiment_data) == 0:
            print(f"【ERROR】No experiment data found for phase '{self.current_phase}'")
            return None
        
        # 准备训练数据
        X_tensor, Y_tensor = self.prepare_training_data(experiment_data, self.current_phase)
        
        # 更新优化器的数据
        optimizer.X = X_tensor.to(self.device)
        optimizer.Y = Y_tensor.to(self.device)
        
        # 设置批次大小
        if batch_size is not None:
            optimizer.batch_size = batch_size
        
        # 运行单次迭代生成新样本
        result = optimizer.run_single_step(simulation_flag=False)  # 不使用模拟，等待真实数据
        
        return result['candidates'].cpu().numpy()
    
    def prepare_training_data(self, experiment_data, phase):
        """准备训练数据
        
        Args:
            experiment_data: 实验数据列表
            phase: 当前阶段
        """
        X = []
        Y = []
        
        for row in experiment_data:
            # 根据阶段提取参数
            # row 格式（统一格式，缺失字段为NULL）:
            # Formula(0), Concentration(1), Temperature(2), SoakTime(3), PH(4), CuringTime(5),
            # MetalAType(6), MetalAConc(7), MetalBType(8), MetalMolarRatio(9),
            # Coverage(10), Uniformity(11), Adhesion(12)
            if phase == OptimizerManager.PHASE_1_OXIDE:
                # Phase 1 Oxide: 只有金属参数
                x = [row[6], row[7], row[8], row[9]]  # MetalAType, MetalAConc, MetalBType, MetalMolarRatio
            elif phase == OptimizerManager.PHASE_1_ORGANIC:
                # Phase 1 Organic: 只有有机物参数
                x = [row[0], row[1], row[2], row[3], row[4], row[5]]  # Formula, Concentration, Temperature, SoakTime, PH, CuringTime
            elif phase == OptimizerManager.PHASE_2:
                # Phase 2: 所有参数
                x = [row[0], row[1], row[2], row[3], row[4], row[5],  # 有机物参数
                     row[6], row[7], row[8], row[9]]  # 金属参数
            else:
                print(f"【ERROR】Unknown phase: {phase}")
                continue
            
            # 目标值：Uniformity, Coverage, Adhesion (与TkgOptimizer期望的顺序一致)
            y = [row[11], row[10], row[12]]  # Uniformity, Coverage, Adhesion
            
            X.append(x)
            Y.append(y)
        
        # 转换为torch张量
        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float64, device=self.device)
        
        return X_tensor, Y_tensor
    
    def get_current_phase(self):
        """获取当前阶段"""
        return self.manager.current_phase
    
    def switch_phase(self, phase):
        """切换阶段"""
        if phase not in [OptimizerManager.PHASE_1_OXIDE, 
                         OptimizerManager.PHASE_1_ORGANIC, 
                         OptimizerManager.PHASE_2]:
            print(f"【ERROR】Invalid phase: {phase}")
            return False
        
        self.manager.current_phase = phase
        self.current_phase = phase
        return True


class BayesOptimizationService:
    """接口业务逻辑类，负责处理具体的业务逻辑和调用其他类"""
    
    def __init__(self, db_path=None, device='cpu', seed=42, proj_name=None):
        # 固定数据库路径
        if db_path is None:
            db_path = r"D:\Parameter\Meta\DB\sampleData.db"
        
        self.db_manager = DatabaseManager(db_path)
        
        # 如果提供了项目名称，从数据库读取迭代参数
        phase_1_oxide_max_iterations = 5  # 默认值
        phase_1_organic_max_iterations = 5  # 默认值
        total_iterations = None
        
        if proj_name:
            proj = self.db_manager.get_project(proj_name)
            if proj:
                try:
                    total_iterations = proj.get('IterNum')
                    phase_1_oxide_max_iterations = proj.get('Phase1MaxNum', 5)
                    phase_1_organic_max_iterations = proj.get('Phase2MaxNum', 5)
                    print(f"【INFO】Loaded parameters from database for project '{proj_name}':")
                    print(f"【INFO】  Total iterations: {total_iterations}")
                    print(f"【INFO】  Phase 1 Oxide max iterations: {phase_1_oxide_max_iterations}")
                    print(f"【INFO】  Phase 1 Organic max iterations: {phase_1_organic_max_iterations}")
                except Exception as e:
                    print(f"【WARNING】Failed to load parameters from database: {e}, using defaults")
            else:
                print(f"【WARNING】Project '{proj_name}' not found in database, using default parameters")
        
        self.total_iterations = total_iterations
        self.optimizer = BayesianOptimizationAlgorithm(
            device=device,
            seed=seed,
            phase_1_oxide_max_iterations=phase_1_oxide_max_iterations,
            phase_1_organic_max_iterations=phase_1_organic_max_iterations
        )
    
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
    
    def get_iteration_sample(self, proj_name, iter_id, phase=None):
        """获取迭代样本，当iter_id=1时生成初始样本
        
        Args:
            proj_name: 项目名称
            iter_id: 迭代ID
            phase: 阶段名称，如果为None则自动确定
        """
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
            
            # 从数据库读取迭代参数（如果与初始化时不同，需要更新）
            phase_1_oxide_max = proj.get('Phase1MaxNum', 5)
            phase_1_organic_max = proj.get('Phase2MaxNum', 5)
            
            # 如果参数发生变化，更新优化器
            if (self.optimizer.manager.phase_1_oxide_max_iterations != phase_1_oxide_max or
                self.optimizer.manager.phase_1_organic_max_iterations != phase_1_organic_max):
                print(f"【INFO】Updating iteration parameters from database:")
                print(f"【INFO】  Phase 1 Oxide max iterations: {phase_1_oxide_max}")
                print(f"【INFO】  Phase 1 Organic max iterations: {phase_1_organic_max}")
                self.optimizer.manager.phase_1_oxide_max_iterations = phase_1_oxide_max
                self.optimizer.manager.phase_1_organic_max_iterations = phase_1_organic_max
                self.total_iterations = max_iter
        except KeyError as e:
            print(f"【ERROR】Database column not found: {e}")
            return False
        
        # 检查迭代参数是否在范围内
        effective_max_iter = max_iter
        if iter_id > effective_max_iter:
            print(f"【ERROR】Iteration {iter_id} exceeds maximum iterations {effective_max_iter}")
            return False
        
        # 检查当前批次参数是否未生成
        if algo_gen_id >= iter_id:
            print(f"【ERROR】Parameters for iteration {iter_id} have already been generated")
            return False
        
        # 确定阶段
        if phase is None:
            # 根据迭代ID自动确定阶段
            # 假设前 phase_1_oxide_max_iterations 次迭代是 Phase 1 Oxide
            # 接下来 phase_1_organic_max_iterations 次迭代是 Phase 1 Organic
            # 其余是 Phase 2
            phase_1_oxide_max = self.optimizer.manager.phase_1_oxide_max_iterations
            phase_1_organic_max = self.optimizer.manager.phase_1_organic_max_iterations
            
            if iter_id <= phase_1_oxide_max:
                phase = OptimizerManager.PHASE_1_OXIDE
            elif iter_id <= phase_1_oxide_max + phase_1_organic_max:
                phase = OptimizerManager.PHASE_1_ORGANIC
            else:
                phase = OptimizerManager.PHASE_2
        
        # 切换到指定阶段
        self.optimizer.switch_phase(phase)
        
        # 区分处理初始样本(iter_id=1)和迭代样本(iter_id>1)
        if iter_id == 1:
            # 生成初始样本
            samples = self.optimizer.generate_initial_samples(phase=phase, n_samples=batch_size)
            print(f"【INFO】Generating initial samples for project '{proj_name}', phase '{phase}'")
        else:
            # 检查上一批次实验数据是否已回传
            if algo_recev_id != iter_id - 1:
                print(f"【ERROR】Experiment data for iteration {iter_id - 1} not received yet")
                return False
            
            # 从数据库读取历史数据
            experiment_data = self.db_manager.get_experiment_data(proj_name, phase=phase)
            
            if len(experiment_data) == 0:
                print(f"【ERROR】No experiment data found for project '{proj_name}', phase '{phase}'")
                return False
            
            # 生成新的样本
            samples = self.optimizer.generate_new_samples(experiment_data, batch_size=batch_size)
            if samples is None:
                return False
            print(f"【INFO】Generating iteration {iter_id} samples for project '{proj_name}', phase '{phase}'")
        
        # 写入数据库
        success = self.db_manager.insert_samples(proj_name, iter_id, samples, phase, iter_id == 1)
        if success:
            print(f"【INFO】Samples generated for project '{proj_name}', iteration {iter_id}, phase '{phase}'")
        return success


def main():
    """主函数，处理命令行参数并执行相应操作"""
    if len(sys.argv) < 3:
        print("【INFO】Usage: python bayes.py <command> <args> [options]")
        print("【INFO】Commands:")
        print("【INFO】  UpdateExpData <ProjName> <IterId> <ExpDataJson>")
        print("【INFO】  GetIterIdSample <ProjName> <IterId> (use IterId=1 for initial samples)")
        print("【INFO】Options:")
        print("【INFO】  --device=cpu|cuda  : 选择计算设备（默认: cpu）")
        print("【INFO】Note:")
        print("【INFO】  - 迭代参数（总迭代次数、Phase1MaxNum、Phase2MaxNum）将从数据库中读取")
        print("【INFO】  - 阶段（phase）会根据迭代ID自动确定，无需手动指定")
        return
    
    command = sys.argv[1]
    
    # 从命令行参数中解析可选参数
    device = 'cpu'  # 默认使用CPU
    
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--device='):
            device_arg = arg.split('=')[1].lower()
            if device_arg in ['cpu', 'cuda']:
                device = device_arg
    
    # 处理命令，需要过滤掉所有以--开头的参数
    cmd_args = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
    
    if command == "UpdateExpData" and len(cmd_args) >= 3:
        proj_name = cmd_args[0]
        iter_id = int(cmd_args[1])
        exp_data_json = cmd_args[2]
        
        # 创建服务实例，从数据库读取参数
        service = BayesOptimizationService(device=device, proj_name=proj_name)
        
        success = service.update_experiment_data(proj_name, iter_id, exp_data_json)
        print(f"【OUTPUT】 {success} for project '{proj_name}'")
    elif command == "GetIterIdSample" and len(cmd_args) >= 2:
        proj_name = cmd_args[0]
        iter_id = int(cmd_args[1])
        
        # 创建服务实例，从数据库读取参数
        service = BayesOptimizationService(device=device, proj_name=proj_name)
        
        # phase 参数不传递，由方法内部根据迭代ID自动确定
        success = service.get_iteration_sample(proj_name, iter_id, phase=None)
        print(f"【OUTPUT】 {success} for project '{proj_name}'")
    else:
        print("【ERROR】Invalid command or arguments")
        print("【INFO】Usage: python bayes.py <command> <args> [options]")


if __name__ == "__main__":
    main()

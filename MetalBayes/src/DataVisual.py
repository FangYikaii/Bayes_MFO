import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import umap
from sklearn.preprocessing import MinMaxScaler
import itertools
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import sqlite3
import argparse
from typing import Optional


class DataVisualizerDB:
    """数据可视化类，从数据库读取数据并生成可视化图表（适配MetalBayes多阶段特性）"""
    
    def __init__(self, db_path):
        """
        初始化可视化器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.data = None
        self.scaler = MinMaxScaler()
        self.embedding = None  # 整体设计空间的UMAP嵌入
        self.embedding_exp = None  # 实验数据的UMAP嵌入
        self.target_keys = ['Coverage', 'Uniformity', 'Adhesion']  # 目标参数
        
        # 不同阶段的参数键
        self.phase_param_keys = {
            'phase_1_oxide': ['MetalAType', 'MetalAConc', 'MetalBType', 'MetalMolarRatio'],
            'phase_1_organic': ['Formula', 'Concentration', 'Temperature', 'SoakTime', 'PH', 'CuringTime'],
            'phase_2': ['Formula', 'Concentration', 'Temperature', 'SoakTime', 'PH', 'CuringTime',
                       'MetalAType', 'MetalAConc', 'MetalBType', 'MetalMolarRatio']
        }
    
    def determine_phase_from_db(self, proj_name, iter_id):
        """
        根据ProjName和IterId从数据库查询并确定阶段
        
        Args:
            proj_name: 项目名称
            iter_id: 迭代ID
            
        Returns:
            阶段名称 ('phase_1_oxide', 'phase_1_organic', 'phase_2')
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT Phase1MaxNum, Phase2MaxNum, IterNum
            FROM AlgoProjInfo
            WHERE ProjName = ?
            ''', (proj_name,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Project '{proj_name}' not found in AlgoProjInfo table")
            
            phase_1_oxide_max = result[0] if result[0] is not None else 5
            phase_1_organic_max = result[1] if result[1] is not None else 5
            
            # 根据迭代ID确定阶段
            if iter_id <= phase_1_oxide_max:
                phase = 'phase_1_oxide'
            elif iter_id <= phase_1_oxide_max + phase_1_organic_max:
                phase = 'phase_1_organic'
            else:
                phase = 'phase_2'
            
            print(f"【INFO】Determined phase: {phase} for IterId={iter_id} (Phase1MaxNum={phase_1_oxide_max}, Phase2MaxNum={phase_1_organic_max})")
            return phase
        finally:
            conn.close()
        
    def read_data(self, table_name='BayesExperData', proj_name=None, iter_id=None, phase=None):
        """
        从SQLite数据库读取数据，支持按ProjName、IterId和Phase筛选
        
        Args:
            table_name: 表名
            proj_name: 项目名称
            iter_id: 迭代ID（<= iter_id）
            phase: 阶段名称（phase_1_oxide, phase_1_organic, phase_2）
        """
        conn = sqlite3.connect(self.db_path)
        
        # 构建SQL查询基础
        query = f"SELECT * FROM {table_name}"
        params = []
        
        # 添加筛选条件
        conditions = []
        if proj_name is not None:
            conditions.append("ProjName = ?")
            params.append(proj_name)
        if iter_id is not None:
            conditions.append("IterId <= ?")
            params.append(iter_id)
        if phase is not None:
            conditions.append("Phase = ?")
            params.append(phase)
        
        # 如果有条件，添加WHERE子句
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # 执行查询
        self.data = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        print(f"【INFO】Successfully read data, total {len(self.data)} rows")
        if proj_name or iter_id or phase:
            filters = []
            if proj_name: filters.append(f"ProjName='{proj_name}'")
            if iter_id: filters.append(f"IterId<={iter_id}")
            if phase: filters.append(f"Phase='{phase}'")
            print(f"【INFO】Filter conditions: {' AND '.join(filters)}")
        
        # 根据阶段设置参数键
        if phase and phase in self.phase_param_keys:
            self.param_keys = self.phase_param_keys[phase]
        elif not self.data.empty:
            # 如果没有指定阶段，尝试从数据中推断
            # 检查哪些参数列存在且非空
            all_possible_params = self.phase_param_keys['phase_2']
            self.param_keys = [p for p in all_possible_params 
                             if p in self.data.columns and self.data[p].notna().any()]
        else:
            self.param_keys = []
        
        return self.data
    
    def plot_color_mapping(self, color_by, figsize=(15, 6), save_path=None, proj_name=None, iter_id=None):
        """
        生成平行坐标图，使用三次贝塞尔曲线实现平滑过渡，根据color_by参数使用不同的颜色映射
        
        Args:
            color_by: 用于着色的目标参数
            figsize: 图表尺寸
            save_path: 保存路径
            proj_name: 项目名称（用于文件名）
            iter_id: 迭代ID（用于文件名）
        """
        if color_by not in self.data.columns:
            raise KeyError(f"Column missing in data table: {color_by}")

        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.labelsize"] = 16
        plt.rcParams["xtick.labelsize"] = 14
        plt.rcParams["ytick.labelsize"] = 14
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = False
        plt.rcParams['font.weight'] = 'bold'
        
        # 提取所有参数和目标
        all_keys = self.param_keys + self.target_keys
        # 过滤掉不存在的列
        all_keys = [k for k in all_keys if k in self.data.columns]
        dataset = self.data[all_keys]
        ynames = [key for key in all_keys]

        # 数据归一化 - 根据color_by选择不同的列
        metal_data = self.data.copy()
        metal_data['norm'] = MinMaxScaler().fit_transform(
            np.array(metal_data[color_by].values.reshape(-1, 1))
        )

        # 准备数据用于绘图
        ys = dataset.values
        ymins = ys.min(axis=0).astype(np.float64)
        ymaxs = ys.max(axis=0).astype(np.float64)
        epsilon = 1e-8
        dys = np.maximum(ymaxs - ymins, epsilon)
        dys[dys == 0] = 1e-6
        ymins -= dys * 0.05
        ymaxs += dys * 0.05
        dys = ymaxs - ymins

        # 坐标轴对齐与标准化
        zs = np.zeros_like(ys)
        zs[:, 0] = ys[:, 0]
        zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

        # 创建多轴系统
        fig, host = plt.subplots(figsize=figsize)
        axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
        for i, ax in enumerate(axes):
            ax.set_ylim(ymins[i], ymaxs[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=15)
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.tick_params(axis='both', which='minor', labelsize=15)

        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        host.set_xticklabels(ynames, fontsize=15)
        host.tick_params(axis='x', which='major', pad=10)
        host.spines['right'].set_visible(False)
        host.xaxis.tick_top()

        # 根据不同的color_by选择不同的颜色映射
        color_maps = {
            'Coverage': 'viridis',
            'Uniformity': 'plasma',
            'Adhesion': 'inferno'
        }
        cmap_name = color_maps.get(color_by, 'CMRmap_r')
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 256))
        colo = np.clip(np.round(metal_data.norm * 255), 50, 255).astype(int)

        # 贝塞尔曲线绘制
        for j in range(ys.shape[0]):
            verts = list(
                zip(
                    [
                        x
                        for x in np.linspace(
                            0, len(ynames) - 1, len(ynames) * 3 - 2, endpoint=True
                        )
                    ],
                    np.repeat(zs[j, :], 3)[1:-1],
                )
            )

            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = PathPatch(
                path, facecolor='none', lw=3, alpha=0.6, edgecolor=colors[colo[j]]
            )
            host.add_patch(patch)

        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        
    def plot_all_parameters(self, visualization_dir=None):
        """
        为每个目标参数生成单独的可视化
        
        Args:
            visualization_dir: 保存目录
        """
        # 获取proj_name和iter_id
        proj_name = self.data['ProjName'].iloc[0] if 'ProjName' in self.data.columns and not self.data.empty else None
        iter_id = self.data['IterId'].max() if 'IterId' in self.data.columns and not self.data.empty else None
        
        # 如果没有proj_name和iter_id，使用默认值
        if not proj_name: proj_name = "default"
        if not iter_id: iter_id = "0"
        
        # 确保保存目录存在
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
            print(f"【INFO】Created directory: {visualization_dir}")
        
        # 为每个目标参数生成单独的可视化
        for target in self.target_keys:
            # 生成文件名：projname+iterid+target
            filename = f"{proj_name}_{iter_id}_{target}.png"
            # 构建完整保存路径
            save_path = os.path.join(visualization_dir, filename) if visualization_dir else None
            
            # 调用绘图函数
            self.plot_color_mapping(target, figsize=(15, 6), save_path=save_path, 
                                   proj_name=proj_name, iter_id=iter_id)
    
    def _get_colormap(self, target_param):
        """根据目标参数获取对应的颜色映射"""
        color_maps = {
            'Coverage': 'viridis',
            'Uniformity': 'plasma',
            'Adhesion': 'inferno'
        }
        return color_maps.get(target_param, 'viridis')

    def plot_umap(self, target_param, figsize=(12, 12), save_path=None):
        """
        使用UMAP对设计空间和实验数据进行降维可视化
        
        Args:
            target_param: 用于着色的目标参数 (Coverage, Uniformity, Adhesion)
            figsize: 图表尺寸
            save_path: 保存路径
        """
        if target_param not in self.target_keys:
            raise ValueError(f"Target parameter must be one of: {self.target_keys}")
        
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        # 1. 定义参数取值范围并生成设计空间
        # 从数据中提取每个参数的唯一值（排除NULL值）
        param_ranges = []
        for param in self.param_keys:
            if param in self.data.columns:
                unique_vals = sorted(self.data[param].dropna().unique())
                if len(unique_vals) > 0:
                    param_ranges.append(unique_vals)
                else:
                    # 如果该参数全为NULL，跳过
                    continue
        
        if len(param_ranges) == 0:
            raise ValueError("No valid parameters found for UMAP visualization")
        
        # 生成所有可能的参数组合
        lookup_list = list(itertools.product(*param_ranges))
        lookup_init = pd.DataFrame(lookup_list, columns=[p for p in self.param_keys if p in self.data.columns and self.data[p].notna().any()][:len(param_ranges)])
        
        # 2. 数据标准化
        # 只使用非NULL的参数列
        valid_param_keys = [p for p in self.param_keys if p in self.data.columns and self.data[p].notna().any()]
        whole_space_scaled = self.scaler.fit_transform(lookup_init.values)
        experimental_space_scaled = self.scaler.transform(self.data[valid_param_keys].values)
        
        # 3. UMAP降维
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=1, n_jobs=1)
        self.embedding = reducer.fit_transform(whole_space_scaled)
        self.embedding_exp = reducer.transform(experimental_space_scaled)
        
        # 4. 可视化
        plt.figure(figsize=figsize)
        
        # 绘制整个设计空间（灰色点）
        plt.scatter(
            self.embedding[:, 0], self.embedding[:, 1],
            c='lightgray', s=10, alpha=0.5, label='Design Space'
        )
        
        # 获取目标参数值用于着色
        target_values = self.data[target_param].values
        
        # 绘制实验数据点，根据目标参数着色
        scatter = plt.scatter(
            self.embedding_exp[:, 0], self.embedding_exp[:, 1],
            c=target_values, s=50, alpha=0.8,
            cmap=self._get_colormap(target_param),
            label=f'Experimental Data ({target_param})'
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label(target_param, fontsize=14)
        
        # 设置标题和标签
        plt.title(f'UMAP Projection - Colored by {target_param}', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_all_umap(self, visualization_dir=None):
        """为所有目标参数生成UMAP可视化"""
        # 获取proj_name和iter_id
        proj_name = self.data['ProjName'].iloc[0] if 'ProjName' in self.data.columns and not self.data.empty else "default"
        iter_id = self.data['IterId'].max() if 'IterId' in self.data.columns and not self.data.empty else "0"
        
        # 确保保存目录存在
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
            print(f"Created directory: {visualization_dir}")
        
        # 为每个目标参数生成UMAP可视化
        for target in self.target_keys:
            # 生成文件名 - 注意使用Umap而不是_UMAP
            filename = f"{proj_name}_{iter_id}_{target}Umap.png"
            # 构建完整保存路径
            save_path = os.path.join(visualization_dir, filename) if visualization_dir else None
            
            # 调用UMAP绘图函数
            self.plot_umap(target, figsize=(12, 10), save_path=save_path)

    def plot_iteration_curve(self, figsize=(15, 7), save_path=None, deg=2, smooth=80):
        """
        绘制三个目标参数的迭代值曲线，在一个图中体现但有不同的纵坐标量纲
        
        Args:
            figsize: 图表尺寸
            save_path: 保存路径
            deg: 多项式拟合的阶数
            smooth: 平滑参数
        """
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        # 确保数据按ExpID排序
        sorted_data = self.data.sort_values('ExpID').reset_index(drop=True)
        
        # 获取Ntrain值（训练集大小）
        Ntrain = len(sorted_data) // 5  # 假设前半部分为训练集，后半部分为测试集
        
        # non_monotonic_fit函数 - 非单调拟合
        def non_monotonic_fit(y_grid, dd=3, smooth=100):
            N = y_grid.size
            E = np.eye(N)
            D3 = np.diff(E, n=dd, axis=0)
            z = np.linalg.solve(E + smooth * D3.T @ D3, y_grid)
            return z
        
        # 设置图表参数
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["ytick.labelsize"] = 12
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 创建第二个和第三个Y轴
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # 调整第三个Y轴的位置
        ax3.spines["right"].set_position(("axes", 1.2))
        
        # 设置颜色 - 训练集和测试集使用不同的颜色
        train_colors = ['#000000', '#000000', '#000000']  # 训练集颜色：统一为黑色
        test_colors = ['#636efa', '#ef553b', '#00cc96']   # 测试集颜色：紫色、红色、青绿色
        
        # 设置点样式 - 三个目标参数使用不同的样式
        markers = ['o', '^', 's']  # 圆形、三角形、正方形
        
        # 设置趋势线颜色
        trend_colors = ['#636efa', '#ef553b', '#00cc96']   # 测试集颜色：紫色、红色、青绿色
        
        target_params = self.target_keys
        
        # 为每个目标参数绘制曲线
        for i, target in enumerate(target_params):
            # 选择对应的轴
            ax = ax1 if i == 0 else ax2 if i == 1 else ax3
            
            # 获取数据
            x = sorted_data.index.values
            y = sorted_data[target].values
            
            # 分离训练集和测试集
            x_train = x[:Ntrain]
            y_train = y[:Ntrain]
            x_test = x[Ntrain:]
            y_test = y[Ntrain:]
            
            # 绘制训练集数据点
            ax.scatter(
                x_train,
                y_train,
                linestyle='None',
                c=train_colors[i],
                marker=markers[i],
                s=40,
                alpha=0.8,
                label=f'{target} (Train Data)',
                zorder=10
            )
            
            # 绘制测试集数据点
            ax.scatter(
                x_test,
                y_test,
                linestyle='None',
                c=test_colors[i],
                marker=markers[i],
                s=50,
                alpha=0.9,
                label=f'{target} (Test Data)',
                edgecolor='black',
                linewidth=0.5,
                zorder=11
            )
            
            # 绘制趋势线
            try:
                if len(y_test) > 0:
                    trend_test = non_monotonic_fit(y_test, dd=deg, smooth=smooth)
                    ax.plot(
                        x_test,
                        trend_test,
                        color=trend_colors[i],
                        label=f'{target} curve',
                        lw=3,
                        zorder=20
                    )
            except Exception as e:
                print(f"Failed to generate trend line for {target}: {e}")
            
            # 设置轴标签
            ax.set_ylabel(target, color=test_colors[i], fontsize=14)
            ax.tick_params(axis='y', labelcolor=test_colors[i])
        
        # 设置X轴
        ax1.set_xlabel('ExpID', fontsize=14)
        ax1.tick_params(axis='x')
        
        # 添加垂直线分隔训练集和测试集
        args = {'lw': 0.5, 'color': 'k', 'linestyle': '--'}
        ax1.axvline(x=Ntrain - 1 + 0.5, **args)
        ax1.axvline(x=Ntrain - 1, **args)
        
        # 添加训练集和测试集区域标签
        ax1.text(Ntrain // 2, ax1.get_ylim()[1] * 0.9, 'Train Data', ha='center', fontsize=12)
        ax1.text(Ntrain + (len(sorted_data) - Ntrain) // 2, ax1.get_ylim()[1] * 0.9, 'Test Data', ha='center', fontsize=12)
        
        # 设置标题
        plt.title(f'Iteration Curve (polynomial_degree={deg}, smoothing={smooth})', fontsize=16, pad=20)
        
        # 添加网格线
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 添加水平线
        plt.axhline(y=0, linewidth=0.5, color='k', linestyle='--')
        
        # 设置X轴范围
        ax1.set_xlim(xmin=-1, xmax=np.ceil(x[-1]) + 1)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加图例
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        
        ax1.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3,
                  loc='upper left', facecolor='white', ncol=1)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Plot saved to:  {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_all_iteration_curve(self, visualization_dir=None):
        """为所有目标参数生成迭代值曲线可视化"""
        # 获取proj_name和iter_id
        proj_name = self.data['ProjName'].iloc[0] if 'ProjName' in self.data.columns and not self.data.empty else "default"
        iter_id = self.data['IterId'].max() if 'IterId' in self.data.columns and not self.data.empty else "0"
        
        # 确保保存目录存在
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
            print(f"Created directory: {visualization_dir}")
            
        # 调用迭代值曲线绘图函数
        self.plot_iteration_curve(
            save_path=os.path.join(visualization_dir, f"{proj_name}_{iter_id}_IterValue.png") if visualization_dir else None,
            deg=2,
            smooth=80
        )

    def plot_pareto_front(self, figsize=(10, 8), save_path=None):
        """
        绘制Pareto前沿图，显示所有三个目标参数的非支配解
        
        Args:
            figsize: 图表尺寸
            save_path: 保存路径
        """
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        # 获取目标参数数据
        objectives = self.data[self.target_keys].values
        
        # 计算Pareto前沿
        pareto_mask = np.ones(objectives.shape[0], dtype=bool)
        for i in range(objectives.shape[0]):
            for j in range(objectives.shape[0]):
                if i != j and np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    pareto_mask[i] = False
                    break
        
        pareto_front = objectives[pareto_mask]
        
        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制所有实验点
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c="gray", alpha=0.4, label="All experiments")
        
        # 绘制Pareto前沿点
        if len(pareto_front) > 0:
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                c='red', label="Pareto front"
            )
        
        # 设置坐标轴标签和标题
        ax.set_xlabel(self.target_keys[0])
        ax.set_ylabel(self.target_keys[1])
        ax.set_zlabel(self.target_keys[2])
        ax.set_title('Pareto front of formula optimization')
        ax.legend()
        ax.grid(True)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Pareto plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_all_pareto(self, visualization_dir=None):
        """生成Pareto前沿图并保存"""
        # 获取proj_name
        proj_name = self.data['ProjName'].iloc[0] if 'ProjName' in self.data.columns and not self.data.empty else "default"
        iter_id = self.data['IterId'].max() if 'IterId' in self.data.columns and not self.data.empty else "0"

        # 确保保存目录存在
        if visualization_dir and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
            print(f"Created directory: {visualization_dir}")
            
        # 生成文件名 - 按照要求的格式
        filename = f"{proj_name}_{iter_id}_Pareto.png"
        # 构建完整保存路径
        save_path = os.path.join(visualization_dir, filename) if visualization_dir else None
        
        # 调用Pareto绘图函数
        self.plot_pareto_front(figsize=(10, 8), save_path=save_path)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Data Visualization Tool for MetalBayes')
    parser.add_argument('--db_path', type=str, default=r"D:\Parameter\Meta\DB\sampleData.db",
                        help='Path to the SQLite database file')
    parser.add_argument('--proj_name', type=str, required=True,
                        help='Project name to visualize')
    parser.add_argument('--iter_num', type=int, required=True,
                        help='Iteration number to visualize')
    parser.add_argument('--output_dir', type=str, default="./visualization",
                        help='Directory to save visualization images')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 创建可视化器实例
        visualizer = DataVisualizerDB(args.db_path)
        
        # 根据数据库信息自动确定阶段
        phase = visualizer.determine_phase_from_db(args.proj_name, args.iter_num)
        
        # 读取数据
        print(f"【INFO】Reading data for Project: {args.proj_name}, Iteration: {args.iter_num}, Phase: {phase}")
        visualizer.read_data(proj_name=args.proj_name, iter_id=args.iter_num, phase=phase)
        
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"【INFO】Created output directory: {args.output_dir}")
        
        # 生成所有数据可视化
        print("【INFO】Generating all visualizations...")
        
        # 1. 生成平行坐标图
        visualizer.plot_all_parameters(visualization_dir=args.output_dir)
        
        # 2. 生成UMAP可视化
        visualizer.plot_all_umap(visualization_dir=args.output_dir)
        
        # 3. 生成迭代值曲线
        visualizer.plot_all_iteration_curve(visualization_dir=args.output_dir)
        
        # 4. 生成Pareto前沿图
        visualizer.plot_all_pareto(visualization_dir=args.output_dir)
        
        print("【INFO】All visualizations completed successfully!")
        
        # 更新AlogPlotId到数据库，表示可视化已完成
        print(f"【INFO】Updating AlogPlotId in database for project '{args.proj_name}'")
        conn = sqlite3.connect(args.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE AlgoProjInfo 
            SET AlogPlotId = ?
            WHERE ProjName = ?
            ''', (args.iter_num, args.proj_name))
            conn.commit()
            print(f"【INFO】AlogPlotId updated to {args.iter_num} for project '{args.proj_name}'")
        except sqlite3.Error as e:
            print(f"【ERROR】Failed to update AlogPlotId: {e}")
        finally:
            conn.close()
        
    except Exception as e:
        print(f"【ERROR】Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()


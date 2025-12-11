import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器环境中出现问题
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import umap
from sklearn.preprocessing import MinMaxScaler
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import logging
from typing import Optional, List

# 配置日志
logger = logging.getLogger(__name__)


class DataVisualizer:
    """数据可视化类，用于生成优化过程的可视化图表"""
    
    def __init__(self, output_dir: str, fig_dir: str):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录（包含实验数据CSV文件）
            fig_dir: 图表保存目录
        """
        self.output_dir = output_dir
        self.fig_dir = fig_dir
        self.data = None
        self.scaler = MinMaxScaler()
        self.embedding = None  # 整体设计空间的UMAP嵌入
        self.embedding_exp = None  # 实验数据的UMAP嵌入
        self.target_keys = ['Uniformity', 'Coverage', 'Adhesion']  # 目标参数
        
    def read_data_from_csv(self, csv_path: str):
        """
        从CSV文件读取数据
        
        Args:
            csv_path: CSV文件路径
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        
        # 获取参数列（排除目标列和时间戳列）
        exclude_cols = self.target_keys + ['Timestamp']
        self.param_keys = [col for col in self.data.columns if col not in exclude_cols]
        
        return self.data
    
    def read_data_from_optimizer(self, optimizer, phase: str):
        """
        从优化器读取数据
        
        Args:
            optimizer: TraceAwareKGOptimizer 实例
            phase: 阶段名称
        """
        if optimizer.X.shape[0] == 0:
            raise ValueError("Optimizer has no data")
        
        # 将参数数据转换为 DataFrame
        X_cpu = optimizer.X.cpu().numpy()
        Y_cpu = optimizer.Y.cpu().numpy()
        
        data_dict = {}
        for i, param_name in enumerate(optimizer.param_names):
            data_dict[param_name] = X_cpu[:, i]
        
        # 添加目标值
        data_dict['Uniformity'] = Y_cpu[:, 0]
        data_dict['Coverage'] = Y_cpu[:, 1]
        data_dict['Adhesion'] = Y_cpu[:, 2]
        
        self.data = pd.DataFrame(data_dict)
        self.param_keys = optimizer.param_names
        
        return self.data
    
    def _get_phase_fig_dir(self, phase: str) -> str:
        """
        获取阶段对应的图表保存目录
        
        Args:
            phase: 阶段名称
            
        Returns:
            阶段对应的图表目录路径
        """
        phase_dir_map = {
            'phase_1_oxide': 'phase_1_oxide',
            'phase_1_organic': 'phase_1_organic',
            'phase_2': 'phase_2'
        }
        phase_dir = phase_dir_map.get(phase, phase)
        fig_path = os.path.join(self.fig_dir, phase_dir)
        os.makedirs(fig_path, exist_ok=True)
        return fig_path
    
    def plot_color_mapping(self, color_by: str, phase: str, figsize=None, save_path: Optional[str] = None):
        """
        生成平行坐标图，使用三次贝塞尔曲线实现平滑过渡，根据color_by参数使用不同的颜色映射
        
        Args:
            color_by: 用于着色的目标参数（Uniformity, Coverage, Adhesion）
            phase: 阶段名称
            figsize: 图表尺寸，如果为None则根据轴数量自动计算
            save_path: 保存路径，如果为None则自动生成
        """
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        if color_by not in self.target_keys:
            raise KeyError(f"color_by must be one of {self.target_keys}")
        
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
        dataset = self.data[all_keys]
        ynames = [key for key in all_keys]
        
        # 根据轴数量自动计算宽度，确保每个轴有足够的空间
        num_axes = len(ynames)
        if figsize is None:
            # 每个轴至少2.5单位宽度，最小宽度30
            calculated_width = max(30, num_axes * 2.5)
            figsize = (calculated_width, 6)
        
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
            
            # 对于 organic_formula 参数，禁用科学计数法，直接显示数值
            param_name = ynames[i] if i < len(ynames) else ''
            if param_name == 'organic_formula':
                formatter = ScalarFormatter(useOffset=False, useMathText=False)
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
            
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.tick_params(axis='both', which='minor', labelsize=15)
        
        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        # 进一步减小顶部标签字体大小，增加间距，避免重叠
        host.set_xticklabels(ynames, fontsize=9, rotation=0)
        host.tick_params(axis='x', which='major', pad=20)
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
        if save_path is None:
            phase_fig_dir = self._get_phase_fig_dir(phase)
            save_path = os.path.join(phase_fig_dir, f"{phase}_{color_by}_parallel_coords.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_all_parameters(self, phase: str):
        """生成一个平行坐标图，使用Coverage着色（viridis颜色映射，色彩最丰富）"""
        phase_fig_dir = self._get_phase_fig_dir(phase)
        # 使用Coverage，因为它使用viridis颜色映射，色彩最丰富
        filename = f"{phase}_parallel_coords.png"
        save_path = os.path.join(phase_fig_dir, filename)
        self.plot_color_mapping('Coverage', phase, save_path=save_path)
    
    def _get_colormap(self, target_param: str):
        """根据目标参数获取对应的颜色映射"""
        color_maps = {
            'Coverage': 'viridis',
            'Uniformity': 'plasma',
            'Adhesion': 'inferno'
        }
        return color_maps.get(target_param, 'viridis')
    
    def plot_umap(self, target_param: str, phase: str, figsize=(12, 12), save_path: Optional[str] = None):
        """
        使用UMAP对设计空间和实验数据进行降维可视化
        
        Args:
            target_param: 用于着色的目标参数 (Coverage, Uniformity, Adhesion)
            phase: 阶段名称
            figsize: 图表尺寸
            save_path: 保存路径，如果为None则自动生成
        """
        if target_param not in self.target_keys:
            raise ValueError(f"Target parameter must be one of: {self.target_keys}")
        
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        # 1. 定义参数取值范围并生成设计空间
        param_ranges = [sorted(self.data[param].unique()) for param in self.param_keys]
        
        # 生成所有可能的参数组合
        lookup_list = list(itertools.product(*param_ranges))
        lookup_init = pd.DataFrame(lookup_list, columns=self.param_keys)
        
        # 2. 数据标准化
        whole_space_scaled = self.scaler.fit_transform(lookup_init.values)
        experimental_space_scaled = self.scaler.transform(self.data[self.param_keys].values)
        
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
        plt.title(f'UMAP Projection - Colored by {target_param} ({phase})', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存或显示图形
        if save_path is None:
            phase_fig_dir = self._get_phase_fig_dir(phase)
            save_path = os.path.join(phase_fig_dir, f"{phase}_{target_param}_umap.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_all_umap(self, phase: str):
        """为所有目标参数生成UMAP可视化"""
        for target in self.target_keys:
            self.plot_umap(target, phase)
    
    def plot_iteration_curve(self, phase: str, figsize=(15, 7), save_path: Optional[str] = None, deg=2, smooth=80):
        """
        绘制三个目标参数的迭代值曲线，在一个图中体现但有不同的纵坐标量纲
        
        Args:
            phase: 阶段名称
            figsize: 图表尺寸
            save_path: 保存路径，如果为None则自动生成
            deg: 多项式拟合的阶数
            smooth: 平滑参数
        """
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        # 确保数据按索引排序（假设索引代表实验顺序）
        sorted_data = self.data.reset_index(drop=True)
        
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
        
        # 设置颜色
        colors = ['#636efa', '#ef553b', '#00cc96']  # 紫色、红色、青绿色
        
        # 设置点样式
        markers = ['o', '^', 's']  # 圆形、三角形、正方形
        
        target_params = self.target_keys
        
        # 为每个目标参数绘制曲线
        for i, target in enumerate(target_params):
            # 选择对应的轴
            ax = ax1 if i == 0 else ax2 if i == 1 else ax3
            
            # 获取数据
            x = sorted_data.index.values
            y = sorted_data[target].values
            
            # 绘制数据点
            ax.scatter(
                x,
                y,
                linestyle='None',
                c=colors[i],
                marker=markers[i],
                s=50,
                alpha=0.8,
                label=f'{target}',
                zorder=10
            )
            
            # 绘制趋势线
            try:
                if len(y) > 0:
                    trend = non_monotonic_fit(y, dd=deg, smooth=smooth)
                    ax.plot(
                        x,
                        trend,
                        color=colors[i],
                        label=f'{target} curve',
                        lw=3,
                        zorder=20
                    )
            except Exception as e:
                logger.warning(f"生成趋势线失败 ({target}): {str(e)}")
            
            # 设置轴标签
            ax.set_ylabel(target, color=colors[i], fontsize=14)
            ax.tick_params(axis='y', labelcolor=colors[i])
        
        # 设置X轴
        ax1.set_xlabel('Experiment Index', fontsize=14)
        ax1.tick_params(axis='x')
        
        # 设置标题
        plt.title(f'Iteration Curve ({phase}) - polynomial_degree={deg}, smoothing={smooth}', fontsize=16, pad=20)
        
        # 添加网格线
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 添加水平线
        plt.axhline(y=0, linewidth=0.5, color='k', linestyle='--')
        
        # 设置X轴范围
        ax1.set_xlim(xmin=-1, xmax=np.ceil(x[-1]) + 1 if len(x) > 0 else 10)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加图例
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        
        ax1.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3,
                  loc='upper left', facecolor='white', ncol=1)
        
        # 保存或显示图形
        if save_path is None:
            phase_fig_dir = self._get_phase_fig_dir(phase)
            save_path = os.path.join(phase_fig_dir, f"{phase}_iteration_curve.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pareto_front(self, phase: str, figsize=(10, 8), save_path: Optional[str] = None):
        """
        绘制Pareto前沿图，显示所有三个目标参数的非支配解
        
        Args:
            phase: 阶段名称
            figsize: 图表尺寸
            save_path: 保存路径，如果为None则自动生成
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
                   c="gray", alpha=0.4, label="All experiments", s=30)
        
        # 绘制Pareto前沿点
        if len(pareto_front) > 0:
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                c='red', label="Pareto front", s=50
            )
        
        # 设置坐标轴标签和标题
        ax.set_xlabel(self.target_keys[0], fontsize=12)
        ax.set_ylabel(self.target_keys[1], fontsize=12)
        ax.set_zlabel(self.target_keys[2], fontsize=12)
        ax.set_title(f'Pareto front of formula optimization ({phase})', fontsize=14)
        ax.legend()
        ax.grid(True)
        
        # 保存或显示图形
        if save_path is None:
            phase_fig_dir = self._get_phase_fig_dir(phase)
            save_path = os.path.join(phase_fig_dir, f"{phase}_pareto_front.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self, phase: str):
        """
        为指定阶段生成所有可视化图表
        
        Args:
            phase: 阶段名称 (phase_1_oxide, phase_1_organic, phase_2)
        """
        if self.data is None:
            raise ValueError("Please call read_data method first to read data")
        
        success_count = 0
        total_count = 4
        
        # 1. 生成平行坐标图
        try:
            self.plot_all_parameters(phase)
            success_count += 1
        except Exception as e:
            logger.error(f"生成平行坐标图失败: {str(e)}", exc_info=True)
        
        # 2. 生成UMAP可视化
        try:
            self.plot_all_umap(phase)
            success_count += 1
        except Exception as e:
            logger.error(f"生成UMAP可视化失败: {str(e)}", exc_info=True)
        
        # 3. 生成迭代值曲线
        try:
            self.plot_iteration_curve(phase)
            success_count += 1
        except Exception as e:
            logger.error(f"生成迭代值曲线失败: {str(e)}", exc_info=True)
        
        # 4. 生成Pareto前沿图
        try:
            self.plot_pareto_front(phase)
            success_count += 1
        except Exception as e:
            logger.error(f"生成Pareto前沿图失败: {str(e)}", exc_info=True)
        
        if success_count == 0:
            logger.error(f"所有可视化图表生成失败 (阶段: {phase})")
            raise RuntimeError(f"所有可视化图表生成失败 (阶段: {phase})")

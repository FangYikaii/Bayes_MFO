# 标准库导入
import os

# 数据分析库导入
import pandas as pd
import numpy as np

# 可视化库导入
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 机器学习和特征重要性分析库
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import shap

# 降维库导入
import umap

# 配置导入
from config import OUTPUT_DIR, FIGURE_DIR


class OptimizationVisualizer:
    def __init__(self, output_dir, figure_dir):
        """初始化可视化器"""
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.scaler = MinMaxScaler()
        
        # 旧列名到新列名的映射，用于数据格式转换
        self.column_mapping = {
            'formula': 'organic_formula',
            'concentration': 'organic_concentration',
            'temperature': 'organic_temperature',
            'soak_time': 'organic_soak_time',
            'ph': 'organic_ph',
            'curing_time': 'organic_curing_time',
            'molar_ratio_b_a': 'metal_molar_ratio_b_a'
        }
        
        # 所有参数名称列表
        self.param_keys = [
            'organic_formula', 'organic_concentration', 'organic_temperature', 
            'organic_soak_time', 'organic_ph', 'organic_curing_time', 
            'metal_a_type', 'metal_a_concentration', 'metal_b_type', 
            'metal_molar_ratio_b_a', 'experiment_condition'
        ]
        
        # 参数分类：有机物参数和金属氧化物参数
        self.organic_params = [
            'organic_formula', 'organic_concentration', 'organic_temperature', 
            'organic_soak_time', 'organic_ph', 'organic_curing_time', 
            'experiment_condition'
        ]
        
        self.metal_params = [
            'metal_a_type', 'metal_a_concentration', 'metal_b_type', 
            'metal_molar_ratio_b_a'
        ]
        
        # 目标参数列表
        self.target_keys = ['Uniformity', 'Coverage', 'Adhesion']
        
        # 参数名称缩写映射，用于解决可视化中的重叠问题
        self.param_abbrev = {
            'organic_formula': 'Org. Formula',
            'organic_concentration': 'Org. Conc.',
            'organic_temperature': 'Org. Temp.',
            'organic_soak_time': 'Org. Soak',
            'organic_ph': 'Org. pH',
            'organic_curing_time': 'Org. Cure',
            'metal_a_type': 'Metal A Type',
            'metal_a_concentration': 'Metal A Conc.',
            'metal_b_type': 'Metal B Type',
            'metal_molar_ratio_b_a': 'Metal B/A Ratio',
            'experiment_condition': 'Exp. Cond.'
        }
        
        # 确保输出目录存在
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Nature/Science 风格配置 - 加大字号
        plt.rcParams.update({
            # 字体设置 - 加大字号
            'font.family': 'Arial',
            'font.size': 18,
            'axes.titlesize': 18,
            'axes.labelsize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            
            # 线条设置
            'lines.linewidth': 5,
            'lines.markersize': 12,
            
            # 颜色设置 - Nature风格：饱和度较低，色调协调
            'axes.prop_cycle': plt.cycler(color=[
                '#1f77b4',  # 蓝色
                '#ff7f0e',  # 橙色
                '#2ca02c',  # 绿色
                '#d62728',  # 红色
                '#9467bd',  # 紫色
                '#8c564b',  # 棕色
                '#e377c2',  # 粉色
                '#7f7f7f',  # 灰色
                '#bcbd22',  # 黄绿色
                '#17becf'   # 青色
            ]),
            
            # 网格设置
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.5,
            
            # 背景设置
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            
            # 边框设置
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.bottom': True,
            'axes.spines.left': True,
            'axes.linewidth': 1.5
        })
        
        # Nature风格配色方案
        self.nature_colors = {
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f',
            'yellow_green': '#bcbd22',
            'cyan': '#17becf'
        }
        
        # 目标参数对应的Nature风格颜色
        self.target_colors = {
            'Uniformity': self.nature_colors['blue'],
            'Coverage': self.nature_colors['orange'],
            'Adhesion': self.nature_colors['green']
        }
        
        # 参数类型对应的Nature风格颜色
        self.param_type_colors = {
            'organic': self.nature_colors['blue'],
            'oxide': self.nature_colors['orange']
        }
    
    def get_latest_data_file(self):
        """获取最新生成的数据文件"""
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv') and 'tkg_experiment' in f]
        if not files:
            raise FileNotFoundError(f"No experiment data files found in {self.output_dir}")
        
        # 按修改时间排序，获取最新的文件
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)), reverse=True)
        latest_file = files[0]
        return os.path.join(self.output_dir, latest_file)
    
    def read_data(self, file_path=None):
        """读取数据文件，并应用列名映射"""
        if file_path is None:
            file_path = self.get_latest_data_file()
        
        print(f"【INFO】Reading data from {file_path}")
        self.data = pd.read_csv(file_path)
        
        # 应用列名映射，将旧列名转换为新列名
        self.data.rename(columns=self.column_mapping, inplace=True)
        
        # 确保所有新列名都存在，如果不存在则使用旧列名
        for new_col in self.param_keys:
            if new_col not in self.data.columns:
                # 查找对应的旧列名
                old_col = None
                for col, mapped_col in self.column_mapping.items():
                    if mapped_col == new_col:
                        old_col = col
                        break
                if old_col in self.data.columns:
                    self.data[new_col] = self.data[old_col]
        
        print(f"【INFO】Successfully read {len(self.data)} rows")
        print(f"【INFO】Columns after mapping: {list(self.data.columns)}")
        return self.data
    
    def plot_all_bezier_curves(self):
        """为所有目标参数生成贝塞尔曲线可视化"""
        # 获取当前时间戳用于文件名
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        
        # 生成合并的贝塞尔曲线图
        combined_path = os.path.join(self.figure_dir, f"tkg_bezier_combined_{timestamp}.png")
        self.plot_combined_bezier(save_path=combined_path)
    
    def plot_combined_bezier(self, save_path=None):
        """
        将三个目标参数的贝塞尔曲线可视化合并为一张图
        """
        print("【INFO】Generating combined Bezier curve visualizations...")
        
        # 创建一个3x1的子图布局
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        
        # 为每个目标参数生成贝塞尔曲线可视化
        for i, target in enumerate(self.target_keys):
            ax = axes[i]
            
            # 提取所有参数和目标
            all_keys = self.param_keys + self.target_keys
            dataset = self.data[all_keys]
            ynames = [key for key in all_keys]
            
            # 数据归一化
            metal_data = self.data.copy()
            metal_data['norm'] = MinMaxScaler().fit_transform(
                np.array(metal_data[target].values.reshape(-1, 1))
            )
            
            # 准备数据用于绘图
            ys = dataset.values
            ymins = ys.min(axis=0).astype(np.float64)
            ymaxs = ys.max(axis=0).astype(np.float64)
            epsilon = 1e-8  # 设置最小有效阈值
            dys = np.maximum(ymaxs - ymins, epsilon)  # 避免出现零间距
            dys[dys == 0] = 1e-6
            ymins -= dys * 0.05
            ymaxs += dys * 0.05
            dys = ymaxs - ymins
            
            # 坐标轴对齐与标准化
            zs = np.zeros_like(ys)
            zs[:, 0] = ys[:, 0]  # 保持第一列原始值
            zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]  # 其他列缩放到第一列范围
            
            # 创建多轴系统 - Nature风格：简洁，留白合理
            # 使用当前子图作为host
            host = ax
            axes_list = [host] + [host.twinx() for _ in range(ys.shape[1] - 1)]
            for j, axx in enumerate(axes_list):
                axx.set_ylim(ymins[j], ymaxs[j])
                axx.spines['top'].set_visible(True)
                axx.spines['bottom'].set_visible(True)
                axx.spines['left'].set_visible(True)
                axx.spines['right'].set_visible(True)
                axx.spines['top'].set_linewidth(1.5)
                axx.spines['bottom'].set_linewidth(1.5)
                axx.spines['left'].set_linewidth(1.5)
                axx.spines['right'].set_linewidth(1.5)
                axx.tick_params(axis='both', which='major', labelsize=8)
                axx.tick_params(axis='both', which='minor', labelsize=6)
                if axx != host:
                    axx.spines['left'].set_visible(False)
                    axx.yaxis.set_ticks_position('right')
                    axx.spines["right"].set_position(("axes", j / (ys.shape[1] - 1)))
                
                # 关闭所有轴的网格，删除灰色虚线背景
                axx.grid(False)
            
            host.set_xlim(0, ys.shape[1] - 1)
            host.set_xticks(range(ys.shape[1]))
            
            # 简化标签名称，减少重叠
            simplified_ynames = [
                'formula', 'conc', 'temp', 'soak', 'pH', 'cure',
                'metalA', 'concA', 'metalB', 'ratioB', 'cond',
                'uniformity', 'coverage', 'adhesion'
            ]
            
            # 设置x轴标签 - Nature风格：小字体，45度旋转
            host.set_xticklabels(simplified_ynames, fontsize=8, rotation=45, ha='right')
            host.tick_params(axis='x', which='major', pad=6)
            host.xaxis.tick_top()
            
            # 使用Nature风格的颜色映射
            cmap = plt.cm.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, 256))
            colo = np.clip(np.round(metal_data.norm * 255), 0, 255).astype(int)
            
            # 贝塞尔曲线绘制，使用三次贝塞尔曲线替代传统折线，提升曲线平滑度
            for j in range(ys.shape[0]):
                # 创建贝塞尔曲线
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
                    path, facecolor='none', lw=2.5, alpha=0.5, edgecolor=colors[colo[j]]
                )
                host.add_patch(patch)
            
            # 设置标题 - Nature风格：简洁，小字体
            ax.set_title(f'Bezier Curve Plot - Colored by {target}', fontsize=14, pad=10)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Combined Bezier curve plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_heatmap(self, save_path=None):
        """
        生成参数之间以及参数与目标之间的相关性热力图 - Nature/Science风格
        """
        print("【INFO】Generating correlation heatmap...")
        
        # 选择用于相关性分析的列
        correlation_cols = self.param_keys + self.target_keys
        correlation_data = self.data[correlation_cols]
        
        # 计算相关性矩阵
        corr_matrix = correlation_data.corr()
        
        # 可视化热力图 - Nature风格：简洁，低饱和度颜色
        plt.figure(figsize=(15, 12))
        
        # 使用seaborn绘制热力图 - Nature风格：使用RdBu_r colormap，更符合学术期刊风格
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",  # Nature常用的红蓝渐变配色
            square=True,
            cbar_kws={"shrink": .8, "ticks": [-0.6, -0.3, 0, 0.3, 0.6]},
            annot_kws={"size": 9, "weight": "normal"},
            linewidths=0.5,
            vmin=-0.8, vmax=0.8  # 设置颜色范围
        )
        
        # 设置标题和标签 - Nature风格：简洁，小字体
        plt.title("Correlation Heatmap of Parameters and Objectives", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Correlation heatmap saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_sensitivity_analysis(self, target_param, ax=None, color='blue'):
        """
        使用随机森林进行敏感性分析，评估各参数对目标的影响 - Nature/Science风格
        
        Args:
            target_param: 目标参数名称
            ax: Matplotlib轴对象，用于绘制在子图中
            color: 条形图颜色
        """
        # 准备数据
        X = self.data[self.param_keys].values
        y = self.data[target_param].values
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 计算排列重要性
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # 准备重要性数据
        importance_df = pd.DataFrame({
            'param': self.param_keys,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        
        # 绘制条形图 - Nature风格：细边框，低透明度
        ax.barh(importance_df['param'], importance_df['importance'], xerr=importance_df['std'], 
                capsize=4, color=color, alpha=0.7, edgecolor='k', linewidth=1)
        ax.set_xlabel('Permutation Importance', fontsize=12)
        ax.set_ylabel('Parameter', fontsize=12)
        ax.set_title(f'{target_param}', fontsize=13, pad=8)
        
        # Nature风格网格：仅x轴，淡色
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.grid(axis='y', visible=False)
        
        # 设置坐标轴刻度大小
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        return importance_df
     
    def plot_pair_plots(self, param_type='all', save_path=None):
        """
        生成参数与目标之间的配对散点图 - Nature/Science风格
        
        Args:
            param_type: 参数类型，可选值：'all'（全部）, 'organic'（有机物）, 'oxide'（氧化物）
            save_path: 保存路径
        """
        print(f"【INFO】Generating pair plots for {param_type} parameters...")
        
        # 选择对应的参数集
        if param_type == 'organic':
            param_set = self.organic_params
            param_title = 'Organic'
        elif param_type == 'metal':
            param_set = self.metal_params
            param_title = 'Metal'
        else:
            param_set = self.param_keys
            param_title = 'All'
        
        # 计算参数重要性，选择贡献度前五的变量
        print("【INFO】Calculating parameter importance...")
        
        # 准备数据
        X = self.data[param_set].values
        # 使用三个目标的平均值作为综合目标
        y = self.data[self.target_keys].mean(axis=1).values
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 计算排列重要性
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # 获取贡献度前五的变量
        importance_df = pd.DataFrame({
            'param': param_set,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        
        # 按重要性排序，取前5个参数（如果参数少于5个则取全部）
        top_params = importance_df.sort_values('importance', ascending=False).head(5)['param'].tolist()
        print(f"【INFO】Top contributing parameters for {param_type}: {top_params}")
        
        # 选择用于配对图的列：贡献度前五的变量 + 三个目标
        pair_cols = top_params + self.target_keys
        
        # 生成配对图 - Nature风格：小标记，低透明度，简洁配色
        g = sns.pairplot(
            self.data[pair_cols], 
            diag_kind='kde', 
            corner=True,
            plot_kws={
                'alpha': 0.6,  # 更低的透明度
                's': 40,       # 更小的标记
                'edgecolor': 'k', 
                'linewidth': 0.5
            },
            diag_kws={
                'fill': True,
                'alpha': 0.7,  # 更低的透明度
                'linewidth': 1.5
            }
        )
        
        # 为不同类型的图设置不同颜色 - Nature风格：协调的配色方案
        for i, col1 in enumerate(pair_cols):
            for j, col2 in enumerate(pair_cols):
                if i > j:  # 只处理左下角的图
                    ax = g.axes[i, j]
                    if ax is not None:
                        # 根据列类型设置不同颜色
                        if col1 in self.target_keys and col2 in self.target_keys:
                            # 目标-目标图：使用蓝色
                            for scatter in ax.collections:
                                scatter.set_color(self.nature_colors['blue'])
                                scatter.set_alpha(0.6)
                        elif col1 in self.target_keys or col2 in self.target_keys:
                            # 参数-目标图：使用橙色
                            for scatter in ax.collections:
                                scatter.set_color(self.nature_colors['orange'])
                                scatter.set_alpha(0.6)
                        else:
                            # 参数-参数图：使用绿色
                            for scatter in ax.collections:
                                scatter.set_color(self.nature_colors['green'])
                                scatter.set_alpha(0.6)
        
        # 为对角线密度图设置颜色 - Nature风格：低饱和度
        for i, col in enumerate(pair_cols):
            ax = g.axes[i, i]
            if ax is not None:
                # 获取密度图线条
                for line in ax.lines:
                    if col in self.target_keys:
                        # 目标变量对角线密度图：使用对应目标颜色
                        if col == 'Uniformity':
                            line.set_color(self.nature_colors['blue'])
                        elif col == 'Coverage':
                            line.set_color(self.nature_colors['orange'])
                        else:
                            line.set_color(self.nature_colors['green'])
                        line.set_linewidth(1.5)
                    else:
                        # 参数变量对角线密度图：使用紫色
                        line.set_color(self.nature_colors['purple'])
                        line.set_linewidth(1.5)
                # 获取填充区域
                for patch in ax.patches:
                    if col in self.target_keys:
                        if col == 'Uniformity':
                            patch.set_facecolor(self.nature_colors['blue'])
                        elif col == 'Coverage':
                            patch.set_facecolor(self.nature_colors['orange'])
                        else:
                            patch.set_facecolor(self.nature_colors['green'])
                        patch.set_alpha(0.3)
                    else:
                        patch.set_facecolor(self.nature_colors['purple'])
                        patch.set_alpha(0.3)
        
        # 调整标题和标签 - Nature风格：简洁，小字体
        plt.suptitle(f"Pair Plots: Top {param_title} Parameters vs Objectives", y=1.02, fontsize=14)
        
        # 调整子图间距
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】{param_title} pair plots saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_shap_analysis(self, target_param, ax=None):
        """
        使用SHAP进行模型可解释性分析，可视化各参数对目标的影响 - Nature/Science风格
        
        Args:
            target_param: 目标参数名称
            ax: Matplotlib轴对象，用于绘制在子图中
        """
        # 准备数据
        X = self.data[self.param_keys].values
        y = self.data[target_param].values
        
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        # 可视化SHAP汇总图
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        
        # 使用SHAP的bar_plot而不是summary_plot，更适合子图
        # 计算特征重要性
        feature_importance = np.abs(shap_values).mean(0)
        
        # 准备重要性数据
        importance_df = pd.DataFrame({
            'param': self.param_keys,
            'importance': feature_importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # 绘制条形图 - Nature风格：细边框，低透明度
        ax.barh(importance_df['param'], importance_df['importance'], alpha=0.7, 
                edgecolor='k', linewidth=1)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_ylabel('Parameter', fontsize=12)
        ax.set_title(f'{target_param}', fontsize=13, pad=8)
        
        # Nature风格网格：仅x轴，淡色
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.grid(axis='y', visible=False)
        
        # 设置坐标轴刻度大小
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        return importance_df, shap_values, rf
      
    def plot_combined_param_analysis(self, save_path=None):
        """
        将UMAP、Sensitivity和SHAP图合并为一张大图，共九个图，分三行展示
        """
        print("【INFO】Generating combined parameter analysis...")
        
        # 创建一个3x3的子图布局
        fig, axes = plt.subplots(3, 3, figsize=(30, 24))
        
        # 第一行：UMAP图
        print("【INFO】Generating UMAP visualizations for combined analysis...")
        for i, target in enumerate(self.target_keys):
            ax = axes[0, i]
            
            # 数据标准化
            param_data = self.data[self.param_keys].values
            scaled_data = self.scaler.fit_transform(param_data)
            
            # UMAP降维
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=42, n_jobs=1)
            embedding = reducer.fit_transform(scaled_data)
            
            # 获取目标参数值用于着色
            target_values = self.data[target].values
            
            # 使用Nature风格的颜色映射
            cmap_mapping = {
                'Uniformity': 'plasma',    # 紫色系 - 用于均匀性
                'Coverage': 'viridis',     # 绿色系 - 用于覆盖率
                'Adhesion': 'inferno'      # 红色系 - 用于粘附性
            }
            cmap = plt.cm.get_cmap(cmap_mapping.get(target, 'viridis'))
            
            # 绘制UMAP散点图
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=target_values, s=50, alpha=0.7, 
                cmap=cmap, edgecolor='k', linewidth=0.5
            )
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(target, fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            
            # 设置标题和标签
            ax.set_title(f'UMAP - {target}', fontsize=16, pad=10)
            ax.set_xlabel('UMAP 1', fontsize=14)
            ax.set_ylabel('UMAP 2', fontsize=14)
            
            # 调整坐标轴样式
            ax.tick_params(labelsize=12)
            
            # 网格设置
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # 第二行：敏感性分析
        print("【INFO】Generating sensitivity analysis for combined analysis...")
        for i, target in enumerate(self.target_keys):
            ax = axes[1, i]
            
            # 准备数据
            X = self.data[self.param_keys].values
            y = self.data[target].values
            
            # 训练随机森林模型
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # 计算排列重要性
            perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
            
            # 准备重要性数据
            importance_df = pd.DataFrame({
                'param': self.param_keys,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            # 使用缩写参数名称
            importance_df['abbrev_param'] = importance_df['param'].map(self.param_abbrev)
            
            # 绘制条形图 - Nature风格：细边框，低透明度
            if target == 'Uniformity':
                bar_color = self.nature_colors['blue']
            elif target == 'Coverage':
                bar_color = self.nature_colors['orange']
            else:
                bar_color = self.nature_colors['green']
            
            ax.barh(importance_df['abbrev_param'], importance_df['importance'], xerr=importance_df['std'], 
                    capsize=4, color=bar_color, alpha=0.7, edgecolor='k', linewidth=1.5)
            ax.set_xlabel('Permutation Importance', fontsize=14)
            ax.set_ylabel('Parameter', fontsize=14)
            ax.set_title(f'Sensitivity - {target}', fontsize=16, pad=10)
            
            # Nature风格网格：仅x轴，淡色
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            ax.grid(axis='y', visible=False)
            
            # 设置坐标轴刻度大小
            ax.tick_params(axis='y', labelsize=11)  # 调整y轴刻度大小
            ax.tick_params(axis='x', labelsize=12)
            
            # 设置边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        # 第三行：SHAP分析
        print("【INFO】Generating SHAP analysis for combined analysis...")
        for i, target in enumerate(self.target_keys):
            ax = axes[2, i]
            
            # 准备数据
            X = self.data[self.param_keys].values
            y = self.data[target].values
            
            # 训练随机森林模型
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # 初始化SHAP解释器
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
            
            # 计算特征重要性
            feature_importance = np.abs(shap_values).mean(0)
            
            # 准备重要性数据
            importance_df = pd.DataFrame({
                'param': self.param_keys,
                'importance': feature_importance
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            # 使用缩写参数名称
            importance_df['abbrev_param'] = importance_df['param'].map(self.param_abbrev)
            
            # 绘制条形图 - Nature风格：细边框，低透明度
            if target == 'Uniformity':
                bar_color = self.nature_colors['blue']
            elif target == 'Coverage':
                bar_color = self.nature_colors['orange']
            else:
                bar_color = self.nature_colors['green']
            
            ax.barh(importance_df['abbrev_param'], importance_df['importance'], alpha=0.7, 
                    color=bar_color, edgecolor='k', linewidth=1.5)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=14)
            ax.set_ylabel('Parameter', fontsize=14)
            ax.set_title(f'SHAP - {target}', fontsize=16, pad=10)
            
            # Nature风格网格：仅x轴，淡色
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            ax.grid(axis='y', visible=False)
            
            # 设置坐标轴刻度大小
            ax.tick_params(axis='y', labelsize=11)  # 调整y轴刻度大小
            ax.tick_params(axis='x', labelsize=12)
            
            # 设置边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        # 调整布局
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98, hspace=0.3, wspace=0.2)
        plt.suptitle('Parameter Analysis: UMAP, Sensitivity, and SHAP', fontsize=20, y=0.98)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Combined parameter analysis saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_combined_distributions(self, save_path=None):
        """
        将参数分布和目标分布合并为一张图，都用小提琴图表示
        """
        print("【INFO】Generating combined distribution visualizations...")
        
        # 创建一个2x1的子图布局
        fig, axes = plt.subplots(2, 1, figsize=(20, 16))
        
        # 第一行：参数分布（小提琴图）
        ax = axes[0]
        
        # 准备参数数据，使用缩写列名解决重叠问题
        param_data = self.data[self.param_keys].rename(columns=self.param_abbrev)
        
        # 使用小提琴图展示参数分布
        sns.violinplot(data=param_data, inner="quartile", ax=ax, 
                      linewidth=1.5)  # 细边框
        ax.set_title("Parameter Distributions (Violin Plots)", fontsize=18, pad=15)
        ax.set_ylabel("Value", fontsize=16)
        
        # 设置x轴标签旋转角度和水平对齐
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        
        # Nature风格网格：仅y轴，淡色
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', visible=False)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # 第二行：目标分布（小提琴图）
        ax = axes[1]
        
        # 准备目标数据
        target_data = self.data[self.target_keys]
        
        # 使用小提琴图展示目标分布
        sns.violinplot(data=target_data, inner="quartile", ax=ax, 
                      palette=[self.nature_colors['blue'], self.nature_colors['orange'], self.nature_colors['green']],
                      linewidth=1.5)  # 细边框
        ax.set_title("Objective Distributions (Violin Plots)", fontsize=18, pad=15)
        ax.set_ylabel("Value", fontsize=16)
        
        # 设置x轴标签
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        
        # Nature风格网格：仅y轴，淡色
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', visible=False)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # 调整布局，增加子图之间的间距，避免标题重叠
        plt.subplots_adjust(top=0.9, bottom=0.08, left=0.05, right=0.98, hspace=0.45)
        plt.suptitle('Combined Distribution Analysis', fontsize=24, y=0.97)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Combined distributions saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optimization_stages(self, save_path=None):
        """
        生成优化阶段分析可视化，展示从简单到复杂系统的演变
        - 分析不同优化阶段的参数和目标变化
        - 揭示从简单分析到复杂体系分析的过程
        """
        print("【INFO】Generating optimization stages analysis...")
        
        # 检查数据是否包含step信息
        if 'step' not in self.data.columns:
            # 如果没有step列，根据索引创建简单的阶段划分
            # 分为2个阶段：前50%为阶段1（简单分析），后50%为阶段2（复杂体系分析）
            self.data['step'] = pd.qcut(self.data.index, q=2, labels=['step1', 'step2'])
        
        # 可视化参数和目标在不同阶段的变化
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 目标值在不同阶段的变化（箱线图）
        ax = axes[0, 0]
        # 使用melt将数据转换为适合并列箱线图的格式
        melted_data = self.data.melt(id_vars=['step'], value_vars=self.target_keys, 
                                    var_name='Objective', value_name='Value')
        sns.boxplot(data=melted_data, x='step', y='Value', hue='Objective', ax=ax, 
                   palette=[self.nature_colors['blue'], self.nature_colors['orange'], self.nature_colors['green']],
                   linewidth=1.5)
        ax.set_title('Objective Values Across Optimization Stages', fontsize=14, pad=12)
        ax.set_xlabel('Optimization Stage', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        # Nature风格网格：仅y轴，淡色
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', visible=False)
        # 设置坐标轴刻度大小
        ax.tick_params(axis='both', labelsize=10)
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # 简化图例
        ax.legend(title='Objective', fontsize=9)
        
        # 2. 有机物参数在不同阶段的变化
        ax = axes[0, 1]
        organic_data = self.data.melt(id_vars=['step'], value_vars=self.organic_params, 
                                     var_name='Parameter', value_name='Value')
        sns.boxplot(data=organic_data, x='Parameter', y='Value', hue='step', ax=ax, 
                   palette=[self.nature_colors['blue'], self.nature_colors['orange']],
                   linewidth=1.5)
        ax.set_title('Organic Parameters Across Optimization Stages', fontsize=14, pad=12)
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        # 修复set_ticklabels() bug：先获取当前刻度，再设置标签
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9)
        # Nature风格网格：仅y轴，淡色
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', visible=False)
        # 设置坐标轴刻度大小
        ax.tick_params(axis='y', labelsize=10)
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # 简化图例
        ax.legend(title='Stage', fontsize=9)
        
        # 3. 金属氧化物参数在不同阶段的变化
        ax = axes[1, 0]
        metal_data = self.data.melt(id_vars=['step'], value_vars=self.metal_params, 
                                   var_name='Parameter', value_name='Value')
        sns.boxplot(data=metal_data, x='Parameter', y='Value', hue='step', ax=ax, 
                   palette=[self.nature_colors['blue'], self.nature_colors['orange']],
                   linewidth=1.5)
        ax.set_title('Metal Oxide Parameters Across Optimization Stages', fontsize=14, pad=12)
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        # 修复set_ticklabels() bug：先获取当前刻度，再设置标签
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=9)
        # Nature风格网格：仅y轴，淡色
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.grid(axis='x', visible=False)
        # 设置坐标轴刻度大小
        ax.tick_params(axis='y', labelsize=10)
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # 简化图例
        ax.legend(title='Stage', fontsize=9)
        
        # 4. 目标值相关性在不同阶段的变化
        ax = axes[1, 1]
        
        # 计算不同阶段的相关性矩阵
        step1_corr = self.data[self.data['step'] == 'step1'][self.target_keys].corr()
        step2_corr = self.data[self.data['step'] == 'step2'][self.target_keys].corr()
        
        # 创建相关性差异矩阵
        corr_diff = step2_corr - step1_corr
        
        # 可视化相关性差异 - Nature风格：使用RdBu_r colormap
        sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="RdBu_r", square=True, 
                   cbar_kws={"shrink": .8, "ticks": [-0.6, -0.3, 0, 0.3, 0.6]},
                   annot_kws={"size": 9, "weight": "normal"},
                   linewidths=0.5, vmin=-0.8, vmax=0.8, ax=ax)
        ax.set_title('Objective Correlation Change: Step2 - Step1', fontsize=14, pad=12)
        ax.set_xlabel('Objective (Step2)', fontsize=12)
        ax.set_ylabel('Objective (Step1)', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.suptitle('Optimization Stages Analysis: From Simple to Complex System', fontsize=16, y=0.99)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Optimization stages analysis saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_visualizations(self):
        """生成所有可视化结果"""
        print("【INFO】Generating all visualizations...")
        
        # 读取数据
        self.read_data()
        
        # 获取当前时间戳用于文件名
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        
        # 生成参数分析大图（UMAP + Sensitivity + SHAP）
        print("【INFO】Generating parameter analysis大图...")
        param_analysis_path = os.path.join(self.figure_dir, f"tkg_param_analysis_combined_{timestamp}.png")
        self.plot_combined_param_analysis(save_path=param_analysis_path)
        
        # 生成贝塞尔曲线可视化
        print("【INFO】Generating Bezier curve visualizations...")
        self.plot_all_bezier_curves()
        
        # 生成相关性热力图
        print("【INFO】Generating correlation heatmap...")
        corr_heatmap_path = os.path.join(self.figure_dir, f"tkg_correlation_heatmap_{timestamp}.png")
        self.plot_correlation_heatmap(save_path=corr_heatmap_path)
        
        # 生成有机物参数pair plot
        print("【INFO】Generating organic parameters pair plots...")
        organic_pair_plots_path = os.path.join(self.figure_dir, f"tkg_pair_plots_organic_{timestamp}.png")
        self.plot_pair_plots(param_type='organic', save_path=organic_pair_plots_path)
        
        # 生成金属氧化物参数pair plot
        print("【INFO】Generating metal parameters pair plots...")
        metal_pair_plots_path = os.path.join(self.figure_dir, f"tkg_pair_plots_metal_{timestamp}.png")
        self.plot_pair_plots(param_type='metal', save_path=metal_pair_plots_path)
        
        # 生成合并的分布可视化（参数分布和目标分布，都用小提琴图）
        print("【INFO】Generating combined distribution visualizations...")
        combined_dist_path = os.path.join(self.figure_dir, f"tkg_distributions_combined_{timestamp}.png")
        self.plot_combined_distributions(save_path=combined_dist_path)
        
        # 生成优化阶段分析
        print("【INFO】Generating optimization stages analysis...")
        stages_path = os.path.join(self.figure_dir, f"tkg_optimization_stages_{timestamp}.png")
        self.plot_optimization_stages(save_path=stages_path)
        
        print("【INFO】All visualizations completed successfully!")


def main():
    # 创建可视化器实例
    visualizer = OptimizationVisualizer(output_dir=OUTPUT_DIR, figure_dir=FIGURE_DIR)
    
    # 生成所有可视化
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

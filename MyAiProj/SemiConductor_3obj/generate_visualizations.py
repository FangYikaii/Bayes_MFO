import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import MinMaxScaler
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from config import OUTPUT_DIR, FIGURE_DIR


class OptimizationVisualizer:
    def __init__(self, output_dir, figure_dir):
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.scaler = MinMaxScaler()
        self.param_keys = ['formula', 'concentration', 'temperature', 'soak_time', 'ph', 'curing_time', 
                         'metal_a_type', 'metal_a_concentration', 'metal_b_type', 'molar_ratio_b_a', 
                         'experiment_condition']
        self.target_keys = ['Uniformity', 'Coverage', 'Adhesion']
        
        # 确保输出目录存在
        os.makedirs(self.figure_dir, exist_ok=True)
    
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
        """读取数据文件"""
        if file_path is None:
            file_path = self.get_latest_data_file()
        
        print(f"【INFO】Reading data from {file_path}")
        self.data = pd.read_csv(file_path)
        print(f"【INFO】Successfully read {len(self.data)} rows")
        return self.data
    
    def _get_colormap(self, target_param):
        """根据目标参数获取对应的颜色映射"""
        color_maps = {
            'Uniformity': 'plasma',    # 紫色系 - 用于均匀性
            'Coverage': 'viridis',     # 绿色系 - 用于覆盖率
            'Adhesion': 'inferno'      # 红色系 - 用于粘附性
        }
        return color_maps.get(target_param, 'viridis')
    
    def plot_umap(self, target_param, figsize=(12, 12), save_path=None):
        """
        使用UMAP对设计空间和实验数据进行降维可视化
        """
        if target_param not in self.target_keys:
            raise ValueError(f"Target parameter must be one of: {self.target_keys}")
        
        # 数据标准化
        param_data = self.data[self.param_keys].values
        scaled_data = self.scaler.fit_transform(param_data)
        
        # UMAP降维
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(scaled_data)
        
        # 可视化
        plt.figure(figsize=figsize)
        
        # 获取目标参数值用于着色
        target_values = self.data[target_param].values
        
        # 绘制UMAP散点图
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=target_values, s=50, alpha=0.8, 
            cmap=self._get_colormap(target_param)
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label(target_param, fontsize=14)
        
        # 设置标题和标签
        plt.title(f'UMAP Projection - Colored by {target_param}', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】UMAP plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_umap(self):
        """为所有目标参数生成UMAP可视化"""
        # 获取当前时间戳用于文件名
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        
        # 为每个目标参数生成UMAP可视化
        for target in self.target_keys:
            filename = f"tkg_umap_{target}_{timestamp}.png"
            save_path = os.path.join(self.figure_dir, filename)
            self.plot_umap(target, figsize=(12, 10), save_path=save_path)
    
    def plot_bezier_curve(self, color_by, figsize=(15, 6), save_path=None):
        """
        生成平行坐标图，使用三次贝塞尔曲线实现平滑过渡
        """
        if color_by not in self.target_keys:
            raise ValueError(f"Color by parameter must be one of: {self.target_keys}")
        
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
        
        # 数据归一化
        metal_data = self.data.copy()
        metal_data['norm'] = MinMaxScaler().fit_transform(
            np.array(metal_data[color_by].values.reshape(-1, 1))
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
        
        # 创建多轴系统
        fig, host = plt.subplots(figsize=figsize)
        axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
        for i, ax in enumerate(axes):
            ax.set_ylim(ymins[i], ymaxs[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='both', which='minor', labelsize=8)
        
        host.set_xlim(0, ys.shape[1] - 1)
        host.set_xticks(range(ys.shape[1]))
        
        # 简化标签名称，减少重叠
        simplified_ynames = [
            'formula', 'conc', 'temp', 'soak', 'pH', 'cure',
            'metalA', 'concA', 'metalB', 'ratioB', 'cond',
            'uniformity', 'coverage', 'adhesion'
        ]
        
        # 设置x轴标签，减小字体大小，旋转角度，增加间距
        host.set_xticklabels(simplified_ynames, fontsize=8, rotation=45, ha='left')
        host.tick_params(axis='x', which='major', pad=20, labelrotation=45)
        host.spines['right'].set_visible(False)
        host.xaxis.tick_top()
        
        # 根据不同的color_by选择不同的颜色映射
        color_maps = {
            'Uniformity': 'plasma',    # 紫色系 - 用于均匀性
            'Coverage': 'viridis',     # 绿色系 - 用于覆盖率
            'Adhesion': 'inferno'      # 红色系 - 用于粘附性
        }
        # 获取当前字段的颜色映射
        cmap_name = color_maps.get(color_by, 'CMRmap_r')
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, 256))
        colo = np.clip(np.round(metal_data.norm * 255), 50, 255).astype(int)
        
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
                path, facecolor='none', lw=3, alpha=0.6, edgecolor=colors[colo[j]]
            )
            host.add_patch(patch)
        
        # 设置标题
        plt.title(f'Bezier Curve Plot - Colored by {color_by}', fontsize=18, pad=20)
        
        # 保存或显示图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"【OUTPUT】Bezier curve plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_bezier_curves(self):
        """为所有目标参数生成贝塞尔曲线可视化"""
        # 获取当前时间戳用于文件名
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        
        # 为每个目标参数生成贝塞尔曲线可视化
        for target in self.target_keys:
            filename = f"tkg_bezier_{target}_{timestamp}.png"
            save_path = os.path.join(self.figure_dir, filename)
            self.plot_bezier_curve(target, figsize=(15, 6), save_path=save_path)
    
    def generate_all_visualizations(self):
        """生成所有可视化结果"""
        print("【INFO】Generating all visualizations...")
        
        # 读取数据
        self.read_data()
        
        # 生成UMAP可视化
        print("【INFO】Generating UMAP visualizations...")
        self.plot_all_umap()
        
        # 生成贝塞尔曲线可视化
        print("【INFO】Generating Bezier curve visualizations...")
        self.plot_all_bezier_curves()
        
        print("【INFO】All visualizations completed successfully!")


def main():
    # 创建可视化器实例
    visualizer = OptimizationVisualizer(output_dir=OUTPUT_DIR, figure_dir=FIGURE_DIR)
    
    # 生成所有可视化
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

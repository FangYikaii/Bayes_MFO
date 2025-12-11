#!/usr/bin/env python3
"""
测试可视化功能
"""
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from src.DataVisualizer import DataVisualizer
import pandas as pd
import numpy as np

def test_visualization():
    """测试可视化功能"""
    # 设置路径
    output_dir = os.path.join(project_root, 'data', 'output')
    fig_dir = os.path.join(project_root, 'data', 'figures')
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"图表目录: {fig_dir}")
    
    # 创建测试数据
    logger.info("创建测试数据...")
    test_data = {
        'metal_a_type': [1, 2, 3, 4, 5],
        'metal_a_concentration': [10, 20, 30, 40, 50],
        'metal_b_type': [0, 1, 2, 3, 4],
        'metal_molar_ratio_b_a': [1, 2, 3, 4, 5],
        'Uniformity': [0.5, 0.6, 0.7, 0.8, 0.9],
        'Coverage': [0.4, 0.5, 0.6, 0.7, 0.8],
        'Adhesion': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    df = pd.DataFrame(test_data)
    
    # 创建可视化器
    logger.info("创建可视化器...")
    visualizer = DataVisualizer(
        output_dir=output_dir,
        fig_dir=fig_dir
    )
    
    # 设置数据
    visualizer.data = df
    visualizer.param_keys = ['metal_a_type', 'metal_a_concentration', 'metal_b_type', 'metal_molar_ratio_b_a']
    
    # 测试生成可视化
    logger.info("开始生成可视化...")
    try:
        visualizer.generate_all_visualizations('phase_1_oxide')
        logger.info("✓ 可视化生成成功！")
        
        # 检查文件是否存在
        phase_fig_dir = os.path.join(fig_dir, 'phase_1_oxide')
        if os.path.exists(phase_fig_dir):
            files = os.listdir(phase_fig_dir)
            logger.info(f"✓ 图表目录存在: {phase_fig_dir}")
            logger.info(f"✓ 生成的文件数量: {len(files)}")
            logger.info("✓ 文件列表:")
            for f in files:
                file_path = os.path.join(phase_fig_dir, f)
                file_size = os.path.getsize(file_path)
                logger.info(f"  - {f} ({file_size} bytes)")
        else:
            logger.warning(f"✗ 图表目录不存在: {phase_fig_dir}")
    except Exception as e:
        import traceback
        logger.error(f"✗ 可视化生成失败: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_visualization()

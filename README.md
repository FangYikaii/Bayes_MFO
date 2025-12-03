# Trace-Aware Knowledge Gradient (taKG) Multi-fidelity Bayesian Optimization

## 项目概述

这是一个基于迹感知知识梯度（Trace-Aware Knowledge Gradient, taKG）的多目标贝叶斯优化框架，专门设计用于玻璃金属化过程的优化。该框架能够同时优化三个关键目标：均匀性（Uniformity）、覆盖率（Coverage）和附着力（Adhesion）。

## 功能特性

- **迹感知知识梯度（taKG）算法**：结合了模型性能轨迹和超体积改进的先进获取函数
- **多目标优化**：同时优化三个相互冲突的目标，生成帕累托前沿
- **两阶段优化策略**：先探索简单系统，再过渡到复杂系统
- **安全性约束**：考虑pH值等安全约束，确保实验条件的安全性
- **RESTful API接口**：提供了便于集成的API服务
- **可视化支持**：生成帕累托前沿图、超体积收敛图等可视化结果
- **模拟与真实实验支持**：支持模拟实验和真实实验数据输入

## 安装说明

### 前置要求

- Python 3.8+
- PyTorch 1.10+
- BoTorch 0.6+
- FastAPI 0.70+
- Uvicorn 0.15+

### 安装步骤

1. 克隆项目仓库

```bash
git clone https://github.com/your-username/taKG-optimization.git
cd taKG-optimization
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 直接运行优化

使用 `main.py` 脚本直接运行优化：

```bash
python MyAiProj/SemiConductor_3obj/main.py --n_iter 10
```

### 2. 通过API使用

启动API服务器：

```bash
python MyAiProj/api_server.py
```

API端点说明：

- `POST /api/init` - 初始化优化器
- `POST /api/optimize` - 运行完整优化过程
- `GET /api/status` - 获取优化状态

### 3. 生成可视化结果

```bash
python MyAiProj/SemiConductor_3obj/generate_visualizations.py
```

## 项目结构

```
taKG-optimization/
├── MyAiProj/
│   ├── SemiConductor_3obj/
│   │   ├── src/
│   │   │   ├── tkg_optimizer.py    # 核心优化器实现
│   │   │   └── utils.py            # 工具函数
│   │   ├── main.py                 # 主程序入口
│   │   ├── generate_visualizations.py  # 可视化生成
│   │   └── config.py               # 配置文件
│   ├── api_server.py               # API服务器
│   └── requirements.txt            # 依赖列表
├── README.md                       # 项目说明文档
└── LICENSE                         # 许可证
```

## 算法原理

### 迹感知知识梯度（taKG）

迹感知知识梯度是一种改进的知识梯度算法，它考虑了模型性能在不同迭代中的轨迹。该算法结合了：

1. **当前模型性能**：基于当前模型预测的超体积改进
2. **历史改进趋势**：考虑模型在之前迭代中的改进情况
3. **阶段感知策略**：根据当前优化阶段调整探索与利用的平衡

### 两阶段优化策略

- **阶段1**：仅考虑简单系统（条件1或2），快速探索基础参数空间
- **阶段2**：考虑复杂系统（条件3），在基础探索完成后优化更复杂的组合

### 安全性约束

- 基于配方ID的pH值安全范围
- 金属类型约束：金属A和B不能是相同类型
- 实验条件约束：根据优化阶段限制实验条件

## 优化目标

1. **均匀性（Uniformity）**：衡量金属涂层的均匀程度（0-1）
2. **覆盖率（Coverage）**：衡量玻璃表面被金属覆盖的比例（0-1）
3. **附着力（Adhesion）**：衡量金属涂层与玻璃表面的结合强度（0-1）

## 示例结果

### 帕累托前沿

优化完成后，系统会生成帕累托前沿的3D可视化图，展示三个目标之间的权衡关系。

### 超体积收敛

系统会记录每次迭代的超体积值，生成收敛曲线图，展示优化过程的进展。

## 配置说明

主要配置参数位于 `config.py` 文件中：

- `OUTPUT_DIR`：输出文件目录
- `FIGURE_DIR`：可视化结果目录
- 优化超参数：批量大小、初始样本数等

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。


## 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。

---

**版本**：1.0.0
**发布日期**：2025-12-03
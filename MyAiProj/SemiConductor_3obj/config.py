from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
FIGURE_DIR = PROJECT_ROOT / "figures"

# Phase 1 优化参数配置
# Phase 1各子阶段的最大迭代次数
PHASE_1_OXIDE_MAX_ITERATIONS = 5  # 氧化物阶段最大迭代次数
PHASE_1_ORGANIC_MAX_ITERATIONS = 5  # 有机物阶段最大迭代次数

# Phase 1子阶段改进率阈值（低于此值则切换阶段）
# 超体积改进率 = (当前超体积 - 上一次超体积) / 上一次超体积
# 如果改进率低于此阈值，则认为优化收敛，切换到下一阶段
PHASE_1_IMPROVEMENT_THRESHOLD = 0.05  # 默认值：0.05 (5%)
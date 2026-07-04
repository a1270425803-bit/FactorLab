import sys
from pathlib import Path

# 将 src 目录加入 Python 路径，确保测试文件可以导入 src 下的模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 如果还需要项目根目录在路径中，也添加
sys.path.insert(0, str(PROJECT_ROOT))

#!/usr/bin/env python3
"""FactorLab 数据下载脚本

下载 A 股历史行情数据，用于量化因子研究。
数据来源：AKShare（基于东方财富/同花顺）

使用方法：
    python scripts/download_data.py --fetch
    python scripts/download_data.py --check
"""

import argparse
import sys
from pathlib import Path

# 将 src 加入路径以复用 data_fetcher_v2
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from data_fetcher_v2 import fetch_all, check_data as check_data_integrity_v2
except ImportError as e:
    print(f"导入失败: {e}")
    print("提示：请确保已安装依赖 pip install -r requirements.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FactorLab Data Downloader")
    parser.add_argument("--fetch", action="store_true", help="下载全部 A 股数据")
    parser.add_argument("--check", action="store_true", help="检查数据完整性")
    args = parser.parse_args()

    if args.fetch:
        print("开始下载 A 股数据...")
        try:
            fetch_all()
            print("下载完成")
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)
    elif args.check:
        print("检查数据完整性...")
        try:
            check_data_integrity_v2()
        except Exception as e:
            print(f"检查失败: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

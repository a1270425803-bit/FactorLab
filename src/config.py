#!/usr/bin/env python3
"""FactorLab 配置中心 — 股票池、指数成分股、路径、Phase 控制。

用法:
  python config.py            # 打印当前配置
  python config.py --universe # 输出股票池列表
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
DB_PATH = str(PROJECT_ROOT / "db" / "factorlab.db")
DATA_DIR = str(PROJECT_ROOT / "data")
REPORTS_DIR = str(PROJECT_ROOT / "reports")
LOGS_DIR = str(PROJECT_ROOT / "logs")
ARCHIVE_DIR = str(PROJECT_ROOT / "archive")
HISTORY_DIR = str(PROJECT_ROOT / "history")


class UniverseType(Enum):
    INDEX_CONSTITUENTS = "index_constituents"   # Phase 2: 三大指数成分股
    FULL_MARKET = "full_market"                 # Phase 3: 全 A 股
    CUSTOM = "custom"                           # Phase 1b: 手动 10 只


# Phase 2 三大指数 AKShare 代码
INDEX_CODES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
}

# Phase 1b Demo 股票池（向后兼容）
DEMO_STOCKS = [
    "600519", "000858", "002594", "300750", "000333",
    "600036", "000568", "601318", "600900", "601012",
]

# Phase 3 预留：流动性过滤参数
MIN_MARKET_CAP = 0          # 最低市值（0 = 不过滤）
MIN_DAILY_VOLUME = 0        # 最低日均成交量（0 = 不过滤）

# ========== Phase 3b 新增配置 ==========

# 资金与冲击成本
CAPITAL_ASSUMPTION = 50_000_000          # 假设资金规模（RMB），默认 5000 万
IMPACT_COST_COEFFICIENT = 1.50           # 冲击成本系数 150%
                                         # 校准说明：等权持仓（50M/(N×30%) ≈ 11.3万/只），
                                         #   150% 可保证小盘股（日均成交额 3 亿）冲击 0.57% > 0.5%，
                                         #   大盘股（日均成交额 200 亿）冲击 0.0085% < 0.05%
IMPACT_COST_CAP = 0.02                   # 冲击成本上限 2%
IMPACT_COST_USE_MCAP_WEIGHT = False      # False=等权分配; True=市值加权（需额外测试）

# 回测参数（从 Phase 2 移入配置，便于调整）
HOLDING_PERIOD = 5                       # 持有周期（天）
TOP_PCT = 0.3                            # Top 股票比例
RISK_FREE_RATE = 0.02                    # 无风险利率 2%

# 稳健性检验阈值
MONOTONICITY_THRESHOLD = 0.3             # 单调性 Spearman 秩相关阈值
OOS_IC_THRESHOLD = 0.02                  # 样本外 IC 最低要求
IC_DECAY_RATIO_THRESHOLD = 0.5           # T+20 IC / T+5 IC 最低比率
YEARLY_IC_RATIO_THRESHOLD = 0.6          # 分年 IC / 全时段 IC 最低比率

# 样本外分割日期
OOS_TRAIN_START = "2010-01-01"
OOS_TRAIN_END = "2018-12-31"
OOS_TEST_START = "2019-01-01"
OOS_TEST_END = "2025-12-31"
YEARLY_VALIDATION_START_YEAR = 2020      # 分年验证起始年份

# Phase 3f: OOS 分期检查
YEARLY_OOS_MIN_YEARS = 4                  # 测试期(2019-2025)中最少年份通过数
YEARLY_OOS_IC_THRESHOLD = 0.02           # 单年 |IC| 最低要求

# IC 衰减滞后周期
IC_DECAY_PERIODS = [1, 5, 10, 20]       # T+1, T+5, T+10, T+20

# 日均成交额滚动窗口
ADV_WINDOW = 20                          # 20 日滚动均值

# ========== Phase 3c 新增配置 ==========

# 【注意】reports/ 目录会自动创建（html_reporter 中 pathlib.Path.mkdir）
# .gitignore 已包含 reports/，报告文件不纳入版本控制

# volume 单位说明：
# data_fetcher_v2.py 已将 volume 统一标准化为"股"（shares）写入 CSV。
# AKShare stock_zh_a_hist 原生返回"股"，stock_zh_a_daily 返回"手"（已在下载阶段 ×100 转换）。
# backtest.py 和 robustness_checker.py 直接从 CSV 读取，无需单位转换。


@dataclass
class PhaseConfig:
    """当前运行 Phase 配置。"""
    universe_type: UniverseType = UniverseType.INDEX_CONSTITUENTS
    stock_count: int = 0  # 运行时填充
    index_codes: List[str] = field(default_factory=lambda: list(INDEX_CODES.keys()))
    min_market_cap: int = MIN_MARKET_CAP
    min_daily_volume: int = MIN_DAILY_VOLUME
    sandbox_timeout: int = 20  # Phase 3h: 沙箱超时（秒），15→20（降低23% TIMEOUT率）


def get_config() -> PhaseConfig:
    """获取当前 Phase 配置。Phase 2 默认使用三大指数成分股。"""
    return PhaseConfig()


def get_index_universe() -> List[str]:
    """获取三大指数成分股并集（去重）。依赖 AKShare。"""
    import akshare as ak
    import time

    all_stocks = set()
    for code, name in INDEX_CODES.items():
        try:
            df = ak.index_stock_cons(symbol=code)
            stocks = df["品种代码"].tolist() if "品种代码" in df.columns else df.iloc[:, 0].tolist()
            all_stocks.update(str(s) for s in stocks)
            print(f"  {name} ({code}): {len(stocks)} 只")
            time.sleep(0.5)
        except Exception as e:
            print(f"  [警告] 获取 {name} ({code}) 失败: {e}")

    result = sorted(all_stocks)
    print(f"  合并去重后: {len(result)} 只")
    return result


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    cfg = get_config()
    print(f"=== FactorLab Config ===\n")
    print(f"  Universe:       {cfg.universe_type.value}")
    print(f"  Indices:        {cfg.index_codes}")
    print(f"  Min Mkt Cap:    {cfg.min_market_cap}")
    print(f"  Min Volume:     {cfg.min_daily_volume}")
    print(f"  Phase 1b Demo:  {len(DEMO_STOCKS)} 只")

    if "--universe" in sys.argv:
        print(f"\n--- 获取股票池 ---")
        stocks = get_index_universe()
        print(f"\n共 {len(stocks)} 只股票")
        print(f"前 10: {stocks[:10]}")

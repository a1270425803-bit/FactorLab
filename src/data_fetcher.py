#!/usr/bin/env python3
"""数据获取模块 — AKShare 封装。

单一入口：fetch_demo_pool() 获取 10 只股票日线，CSV 落盘到 data/。
接口变动时只需修改此文件。
"""

import os
import time
from pathlib import Path

import akshare as ak
import pandas as pd

# ── 配置 ──────────────────────────────────────────────────
from config import PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
DEMO_STOCKS = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "002594": "比亚迪",
    "300750": "宁德时代",
    "000333": "美的集团",
    "600036": "招商银行",
    "000568": "泸州老窖",
    "601318": "中国平安",
    "600900": "长江电力",
    "601012": "隆基绿能",
}
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_stock_daily(code: str, start: str = "20230101", end: str = "20251231") -> pd.DataFrame:
    """获取单只 A 股日线数据（后复权）。

    Args:
        code: 6 位数字代码，如 "600519"
        start: 起始日期 "YYYYMMDD"
        end: 结束日期 "YYYYMMDD"
    """
    raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if raw.empty:
        raise RuntimeError(f"AKShare 返回空数据: {code}")

    # 统一列名
    col_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    }
    df = raw.rename(columns=col_map)
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])

    keep = ["date", "code", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def fetch_demo_pool(force: bool = False) -> dict[str, pd.DataFrame]:
    """获取全部 10 只 Demo 股票池数据，按股票代码返回 DataFrame 字典。"""
    DATA_DIR.mkdir(exist_ok=True)
    result = {}
    for code, name in DEMO_STOCKS.items():
        csv_path = DATA_DIR / f"{code}.csv"
        if not force and csv_path.exists():
            result[code] = pd.read_csv(csv_path, parse_dates=["date"])
            continue
        print(f"  获取 {code} {name} ...", end=" ", flush=True)
        try:
            df = fetch_stock_daily(code)
            df.to_csv(csv_path, index=False)
            result[code] = df
            print(f"OK ({len(df)} 行)")
        except Exception as e:
            print(f"FAIL: {e}")
            if code in result:
                del result[code]
        time.sleep(0.5)  # 礼貌限速
    return result


def check_data_integrity() -> tuple[bool, list[str]]:
    """启动校验：检查 data/ 下 CSV 的行数、日期完整性、必要字段。

    Returns:
        (是否全部通过, 问题列表)
    """
    issues = []
    if not DATA_DIR.exists():
        return False, [f"数据目录不存在: {DATA_DIR}"]

    for code in DEMO_STOCKS:
        csv_path = DATA_DIR / f"{code}.csv"
        if not csv_path.exists():
            issues.append(f"缺失文件: {code}.csv")
            continue

        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
        except Exception as e:
            issues.append(f"{code}.csv 读取失败: {e}")
            continue

        if len(df) < 50:
            issues.append(f"{code}.csv 行数不足 ({len(df)} < 50)")

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(f"{code}.csv 缺少字段: {col}")

        if "date" in df.columns:
            if df["date"].isna().any():
                issues.append(f"{code}.csv 日期列有空值")
            if df["date"].duplicated().any():
                issues.append(f"{code}.csv 日期列有重复")
            # 日期连续性检查：相邻日期间的交易日间隔不应超过 5 天
            dates_sorted = sorted(df["date"].unique())
            for i in range(1, len(dates_sorted)):
                gap = (dates_sorted[i] - dates_sorted[i - 1]).days
                if gap > 5:
                    issues.append(
                        f"{code}.csv 日期不连续: {dates_sorted[i-1].date()} -> "
                        f"{dates_sorted[i].date()} 间隔 {gap} 天"
                    )
                    break  # 每只股票只报一次

    return len(issues) == 0, issues


def load_pool_data() -> pd.DataFrame:
    """加载全部股票池数据，返回合并后的 DataFrame（date + code 多列）。"""
    frames = []
    for code in DEMO_STOCKS:
        csv_path = DATA_DIR / f"{code}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["date"])
            frames.append(df)
    if not frames:
        raise RuntimeError("data/ 中无有效 CSV，请先运行 python data_fetcher.py --fetch")
    return pd.concat(frames, ignore_index=True)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        # 启动校验模式
        ok, issues = check_data_integrity()
        if ok:
            print("数据完整性校验通过 [OK]")
        else:
            print(f"数据完整性校验发现问题 ({len(issues)} 项):")
            for i in issues:
                print(f"  - {i}")
    elif "--fetch" in sys.argv:
        # 全量拉取模式
        print("拉取 10 只 Demo 股票数据 ...")
        pool = fetch_demo_pool(force=True)
        print(f"\n完成: {len(pool)}/{len(DEMO_STOCKS)} 只股票数据已保存到 data/")
    else:
        # Demo: 拉取 1 只并展示
        print("Demo: 拉取 1 只股票 (600519 贵州茅台) 前 5 行\n")
        df = fetch_stock_daily("600519", start="20240101", end="20240131")
        print(df.head())
        print(f"\n共 {len(df)} 行, 字段: {list(df.columns)}")

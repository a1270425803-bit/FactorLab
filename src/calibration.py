#!/usr/bin/env python3
"""Phase 3g 因子校准基准 — 获取所有评分维度的基线数据（6 因子：4 经典 + 2 AI 入库）。

用法:
  python calibration.py            # 默认：前 500 只股票
  python calibration.py --all      # 全部 1482 只股票（较慢）

目的：在调整任何阈值之前，先看 6 个因子在各个维度的得分。
Phase 3g 更新：校准集从 4 经典因子 → 6 因子（加入 f001/f002 AI 入库因子）。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 加载项目模块
from config import PROJECT_ROOT

# PROJECT_ROOT 已统一

from score import score_factor
from robustness_checker import evaluate as check_robustness
from config import DATA_DIR, OOS_TRAIN_START, OOS_TRAIN_END, OOS_TEST_START, OOS_TEST_END


# ── 经典因子定义 ─────────────────────────────────────────

def factor_20d_momentum(df: pd.DataFrame) -> pd.Series:
    """20 日动量：close / close.shift(20) - 1"""
    return df.groupby(level='code')['close'].pct_change(20)


def factor_60d_reversal(df: pd.DataFrame) -> pd.Series:
    """60 日反转：负的过去 60 日收益率"""
    return -df.groupby(level='code')['close'].pct_change(60)


def factor_amplitude(df: pd.DataFrame) -> pd.Series:
    """振幅因子：(high - low) / close"""
    return (df['high'] - df['low']) / df['close']


def factor_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """量比：volume / volume.rolling(20).mean()"""
    return df['volume'] / df.groupby(level='code')['volume'].rolling(20, min_periods=5).mean().droplevel(0)


# Phase 3g: f001/f002 AI 入库因子（从 SQLite factors 表提取）
def factor_f001(df: pd.DataFrame) -> pd.Series:
    """f001 (反转类, score=2.31): 日内动量×振幅×量比 → 去趋势 → 截面标准化"""
    intraday_ret = (df['close'] - df['open']) / df['open']
    amplitude = (df['high'] - df['low']) / df['open']
    avg_vol_5 = df.groupby(level='code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    vol_ratio = df['volume'] / avg_vol_5
    raw_signal = intraday_ret * amplitude * vol_ratio
    smoothed = df.groupby(level='code')['close'].transform(
        lambda x: x.rolling(5, min_periods=3).mean())
    factor = raw_signal - smoothed
    factor = (factor - factor.groupby(level='date').transform('mean')) / \
             factor.groupby(level='date').transform('std')
    return factor


def factor_f002(df: pd.DataFrame) -> pd.Series:
    """f002 (反转类, score=1.77): 日内动量×振幅×量比 → 双均线去趋势 → 绝对偏离负值"""
    ret = df['close'] / df['open'] - 1
    amp = (df['high'] - df['low']) / df['open']
    vol_ratio = df['volume'] / df.groupby(level='code')['volume'].transform(
        lambda x: x.rolling(20).mean())
    signal = ret * amp * vol_ratio
    signal_detrend = signal - signal.groupby(level='code').transform(
        lambda x: x.rolling(5).mean())
    ma60 = df.groupby(level='code')['close'].transform(
        lambda x: x.rolling(60).mean())
    price_dev = df['close'] / ma60 - 1
    combined = signal_detrend * price_dev
    factor = -combined.abs()
    factor = factor.groupby(level='date').transform(
        lambda x: (x - x.mean()) / x.std())
    return factor


FACTORS = {
    "20日动量": factor_20d_momentum,
    "60日反转": factor_60d_reversal,
    "振幅因子": factor_amplitude,
    "量比因子": factor_volume_ratio,
    "f001(AI入库)": factor_f001,
    "f002(AI入库)": factor_f002,
}


# ── 数据加载 ─────────────────────────────────────────────

def load_data(max_stocks: int = 500) -> pd.DataFrame:
    """加载宽表 CSV，合并为 MultiIndex DataFrame。"""
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if max_stocks:
        csv_files = csv_files[:max_stocks]

    frames = []
    loaded = 0
    for f in csv_files:
        try:
            df = pd.read_csv(
                f,
                parse_dates=['date'],
                index_col='date',
                encoding='utf-8',
            )
            # 统一列名
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl in ('open', 'high', 'low', 'close', 'volume'):
                    col_map[c] = cl
            df = df.rename(columns=col_map)
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in df.columns for c in required):
                continue
            df = df[required]
            code = f.stem
            df['code'] = code
            df = df.reset_index().set_index(['date', 'code'])
            frames.append(df)
            loaded += 1
        except Exception:
            continue

    if not frames:
        raise RuntimeError("未加载到任何有效数据文件")

    print(f"  加载: {loaded} 只股票")
    return pd.concat(frames).sort_index()


def prepare_returns(df: pd.DataFrame, holding_period: int = 5) -> tuple:
    """计算 T+1 和 T+5 未来收益（矩阵形式）。"""
    close = df['close'].unstack('code')

    # T+1 收益
    ret_t1 = close.pct_change(1).shift(-1)

    # T+5 累计收益（对数）
    log_close = np.log(close)
    ret_t5 = log_close.shift(-holding_period) - log_close

    return ret_t1, ret_t5


# ── 主程序 ───────────────────────────────────────────────

def main():
    use_all = "--all" in sys.argv
    max_stocks = 0 if use_all else 500

    print("=" * 70)
    print("  FactorLab Phase 3f — 经典因子校准基准")
    print(f"  股票数: {'全部' if use_all else f'{max_stocks}'}")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载数据...")
    df = load_data(max_stocks=max_stocks)
    n_stocks = df.index.get_level_values('code').nunique()
    date_range = f"{df.index.get_level_values('date').min().date()} ~ {df.index.get_level_values('date').max().date()}"
    print(f"  规模: {len(df)} 行 × {n_stocks} 只股票, {date_range}")

    # 准备收益
    print("\n[2] 计算收益...")
    ret_t1, ret_t5 = prepare_returns(df)

    # 构建 pool（用于 score.py 相关性检查 — 空 dict 表示无已入库因子）
    pool = {}

    # 校准每个因子
    print("\n[3] 校准经典因子...")
    print()

    results = []

    for name, func in FACTORS.items():
        print(f"  >>> {name} <<<")

        # 计算因子值
        try:
            fv_series = func(df)
            fv_wide = fv_series.unstack('code')
        except Exception as e:
            print(f"    ❌ 计算失败: {e}")
            continue

        # 对齐
        common_dates = fv_wide.index.intersection(ret_t5.index)
        common_stocks = fv_wide.columns.intersection(ret_t5.columns)
        fv = fv_wide.loc[common_dates, common_stocks]
        r5 = ret_t5.loc[common_dates, common_stocks]
        r1 = ret_t1.loc[common_dates, common_stocks]

        # score.py 8 维
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score_result = score_factor(fv, r5, pool, r1)

        dims = getattr(score_result, "dimensions", {})
        print(f"    score.py 8维: passed={score_result.passed_threshold}")
        for dname, dinfo in sorted(dims.items()):
            if isinstance(dinfo, dict):
                v = dinfo.get("value", 0)
                p = "✓" if dinfo.get("pass", False) else "✗"
                print(f"      {dname:25s}: {v:>10.4f}  {p}")

        # robustness 4 维
        try:
            close_wide = df['close'].unstack('code')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                robust_result = check_robustness(fv, close_wide)

            print(f"    robustness: robust_core_passed={robust_result.robust_core_passed}")
            print(f"      oos_train_ic:      {robust_result.oos_ic_train:>10.4f}")
            print(f"      oos_test_ic:       {robust_result.oos_ic_test:>10.4f}")
            yp = getattr(robust_result, "oos_yearly_pass_count", 0)
            yi = getattr(robust_result, "oos_yearly_ics", {})
            print(f"      oos_yearly_pass:   {yp}/7  (分期检查)")
            if yi:
                yrs = " ".join(f"{k}:{v:.4f}" for k, v in sorted(yi.items()))
                print(f"      oos_yearly_ics:    {yrs}")
            print(f"      oos_stability:     {'✓' if robust_result.oos_stability_passed else '✗'}")
            print(f"      ic_decay_t1:       {robust_result.ic_decay_t1:>10.4f}")
            print(f"      ic_decay_t5:       {robust_result.ic_decay_t5:>10.4f}")
            print(f"      ic_decay_t20:      {robust_result.ic_decay_t20:>10.4f}")
            print(f"      ic_decay_ratio:    {robust_result.ic_decay_ratio:>10.4f} {'✓' if robust_result.ic_decay_passed else '✗'}")
            print(f"      monotonicity:      {robust_result.monotonicity:>10.4f}")
        except Exception as e:
            print(f"    robustness: ❌ {e}")

        results.append({
            "name": name,
            "score_passed": score_result.passed_threshold,
            "robust_core": robust_result.robust_core_passed if 'robust_result' in dir() else False,
        })
        print()

    # 汇总
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    header = f"{'因子':15s} {'score 8维':>10s} {'robust core':>12s}"
    print(header)
    print("-" * len(header))
    for r in results:
        sp = "✓" if r["score_passed"] else "✗"
        rp = "✓" if r.get("robust_core", False) else "✗"
        print(f"{r['name']:15s} {sp:>10s} {rp:>12s}")

    print()
    score_pass = sum(1 for r in results if r["score_passed"])
    robust_pass = sum(1 for r in results if r.get("robust_core", False))
    print(f"  score.py 通过: {score_pass}/{len(results)}")
    print(f"  robustness 通过: {robust_pass}/{len(results)}")
    print()
    print("  → 这些是经典因子在当前阈值下的基线表现。")
    print("  → 如果经典因子普遍不达标，说明阈值需要校准。")
    print("  → 如果经典因子达标而 AI 因子不达标，说明 AI 生成质量需提升。")


if __name__ == "__main__":
    main()

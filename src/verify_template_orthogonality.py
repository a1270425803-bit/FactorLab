#!/usr/bin/env python3
"""Phase 3g: 5 个正交方向模板的 pairwise Spearman ρ 验证。

取最近 252 个交易日 × 100 只代表性股票，计算每个模板因子的 pairwise ρ。
要求全部 < 0.5，否则合并或重新设计。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import PROJECT_ROOT

# PROJECT_ROOT 已统一
from config import DATA_DIR


# ── 模板定义（最小可行代码）────────────────────────────

def template_t1_pure_momentum(df: pd.DataFrame) -> pd.Series:
    """T1: 纯时序反转 — close/close.shift(60)-1，无截面操作"""
    ret = df.groupby(level='code')['close'].pct_change(60)
    # per-stock z-score for comparability (still temporal, not cross-sectional)
    z = ret.groupby(level='code').transform(
        lambda x: (x - x.rolling(252, min_periods=60).mean()) / x.rolling(252, min_periods=60).std()
    )
    return -z  # reversal


def template_t2_vol_weighted(df: pd.DataFrame) -> pd.Series:
    """T2: 波动率加权信号 — ret/rolling(20).std()，用波动率归一化而非 rank"""
    ret = df.groupby(level='code')['close'].pct_change()
    vol = ret.groupby(level='code').transform(
        lambda x: x.rolling(20, min_periods=10).std())
    signal = ret / vol.replace(0, np.nan)
    # per-stock normalization
    result = signal.groupby(level='code').transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )
    return result


def template_t3_multi_period(df: pd.DataFrame) -> pd.Series:
    """T3: 多周期共振 — 短窗口信号 × 长窗口方向"""
    ret = df.groupby(level='code')['close'].pct_change()
    # short: 5d momentum
    short_ma = ret.rolling(5, min_periods=3).mean()
    # long: 60d trend direction
    long_dir = np.sign(df.groupby(level='code')['close'].pct_change(60))
    # resonance: short signal aligned with long trend
    raw = short_ma * long_dir
    result = raw.groupby(level='code').transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )
    return result


def template_t4_intraday(df: pd.DataFrame) -> pd.Series:
    """T4: 价格路径不规则性 — (close-open)/(high-low)，日内方向效率"""
    body = df['close'] - df['open']
    range_hl = df['high'] - df['low']
    efficiency = body / range_hl.replace(0, np.nan)  # -1 to +1, efficiency of price movement
    # rolling mean of efficiency
    result = efficiency.groupby(level='code').transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    return result


def template_t5_volume_profile(df: pd.DataFrame) -> pd.Series:
    """T5: 成交量特征 — vol/rolling(20).vol.mean() 的标准化偏离"""
    vol = df['volume']
    vol_ma = df.groupby(level='code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    vol_ratio = vol / vol_ma
    # deviation from stock's own volume profile
    result = vol_ratio.groupby(level='code').transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )
    return -result  # high volume → negative (reversal signal)


TEMPLATES = {
    "T1_纯时序反转": template_t1_pure_momentum,
    "T2_波动率加权": template_t2_vol_weighted,
    "T3_多周期共振": template_t3_multi_period,
    "T4_日内路径效率": template_t4_intraday,
    "T5_成交量特征": template_t5_volume_profile,
}


# ── 数据加载 ─────────────────────────────────────────────

def load_subset(n_stocks: int = 100, n_days: int = 252):
    data_dir = Path(DATA_DIR)
    csv_files = sorted(data_dir.glob("*.csv"))[:n_stocks]

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=['date'], index_col='date', encoding='utf-8')
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
            df['code'] = f.stem
            df = df.reset_index().set_index(['date', 'code'])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("No data loaded")

    combined = pd.concat(frames).sort_index()
    # Take last n_days
    all_dates = combined.index.get_level_values('date').unique()
    recent_dates = all_dates[-n_days:]
    combined = combined.loc[combined.index.get_level_values('date').isin(recent_dates)]
    return combined


# ── 主程序 ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 3g — 模板正交性验证 (pairwise Spearman ρ)")
    print("=" * 60)

    print("\n[1] 加载数据...")
    df = load_subset()
    n_stocks = df.index.get_level_values('code').nunique()
    n_days = df.index.get_level_values('date').nunique()
    print(f"  规模: {len(df)} 行 × {n_stocks} 只股票 × {n_days} 天")

    print("\n[2] 计算模板因子值...")
    results = {}
    for name, func in TEMPLATES.items():
        try:
            fv = func(df)
            results[name] = fv.dropna()
            print(f"  {name}: {len(fv.dropna())} 有效值, mean={fv.mean():.4f}, std={fv.std():.4f}")
        except Exception as e:
            print(f"  {name}: ❌ {e}")

    print("\n[3] 计算 pairwise Spearman ρ...")
    template_names = list(results.keys())
    n = len(template_names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
                continue
            # Align on index
            a, b = results[template_names[i]].align(results[template_names[j]], join='inner')
            if len(a) < 100:
                matrix[i][j] = np.nan
                continue
            matrix[i][j] = a.corr(b, method='spearman')

    # ── 输出 ────────────────────────────────────────────
    print()
    print("Pairwise Spearman ρ 矩阵:")
    print(f"{'':20s}", end="")
    for name in template_names:
        print(f"{name[:8]:>10s}", end="")
    print()

    for i, name in enumerate(template_names):
        print(f"{name:20s}", end="")
        for j in range(n):
            if i == j:
                print(f"{'1.000':>10s}", end="")
            else:
                v = matrix[i][j]
                flag = " ✓" if abs(v) < 0.5 else " ✗"
                print(f"{v:>8.4f}{flag}", end="")
        print()

    max_off_diag = np.nanmax(np.abs(matrix - np.eye(n)))
    all_ok = max_off_diag < 0.5
    print(f"\n最大非对角 |ρ|: {max_off_diag:.4f} {'✅ 全部 < 0.5' if all_ok else '❌ 存在 ≥ 0.5 的配对'}")

    # ── 与 f001/f002 的对比 ─────────────────────────────
    print("\n[4] 与入库因子的 ρ...")
    # f001
    def f001_func(df):
        intraday_ret = (df['close'] - df['open']) / df['open']
        amplitude = (df['high'] - df['low']) / df['open']
        avg_vol_5 = df.groupby(level='code')['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
        vol_ratio = df['volume'] / avg_vol_5
        raw_signal = intraday_ret * amplitude * vol_ratio
        smoothed = df.groupby(level='code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
        factor = raw_signal - smoothed
        return (factor - factor.groupby(level='date').transform('mean')) / factor.groupby(level='date').transform('std')

    try:
        f001_fv = f001_func(df).dropna()
        print(f"\n{'模板':20s} {'vs f001 ρ':>10s} {'|ρ|<0.5':>10s}")
        for name, fv in results.items():
            a, b = fv.align(f001_fv, join='inner')
            if len(a) < 100:
                continue
            rho = a.corr(b, method='spearman')
            ok = "✓" if abs(rho) < 0.5 else "✗"
            print(f"{name:20s} {rho:>10.4f} {ok:>10s}")
    except Exception as e:
        print(f"  f001 对比跳过: {e}")

    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ 所有模板配对 |ρ| < 0.5，正交性验证通过")
    else:
        print("  ❌ 存在高相关配对，需合并或重新设计")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

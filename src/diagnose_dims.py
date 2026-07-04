#!/usr/bin/env python3
"""因子维度诊断脚本
分析 dim1(超跌)/dim2(缩量)/dim3(波动收敛) 各自的质量
"""
import pandas as pd
import numpy as np
from scipy import stats
from batch_pipeline import load_df_1800

print("正在加载数据...")
df_1800 = load_df_1800()
close_df = df_1800['close'].unstack('code')
volume_df = df_1800['volume'].unstack('code')

# ---- 计算三个维度的原始值（与 factor_draft.py 完全一致）----
print("计算三个维度...")

ret_20d = close_df.pct_change(20)
vol_5d = close_df.rolling(5).std()
vol_20d = close_df.rolling(20).std()
volume_5d = volume_df.rolling(5).mean()
volume_20d = volume_df.rolling(20).mean()

dim1 = -ret_20d                    # 超跌
dim2 = -(volume_5d / volume_20d)   # 缩量
dim3 = -(vol_5d / vol_20d)         # 波动收敛

# ---- 截面 z-score 标准化（与 factor_draft.py 一致）----
def zscore(x):
    if x.std() == 0 or x.count() < 2:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean()) / x.std()

z1 = dim1.apply(zscore, axis=1)
z2 = dim2.apply(zscore, axis=1)
z3 = dim3.apply(zscore, axis=1)

# ---- 诊断 1：秩自相关（今天排名 vs 昨天排名的相关性）----
print("\n========== 秩自相关 ==========")
for name, z in [("dim1 超跌", z1), ("dim2 缩量", z2), ("dim3 波动收敛", z3)]:
    rank = z.rank(axis=1)
    # 计算每只股票的时序秩自相关，取均值
    ac = rank.corrwith(rank.shift(1), axis=1).mean()
    print(f"{name}: {ac:.4f}")

# ---- 诊断 2：平均 IC（与 T+1 收益的 Rank IC）----
returns_t1 = close_df.pct_change(1).shift(-1)

print("\n========== 平均 IC（T+1） ==========")
for name, z in [("dim1 超跌", z1), ("dim2 缩量", z2), ("dim3 波动收敛", z3)]:
    ic_vals = []
    for date in z.index[:-1]:
        f = z.loc[date].dropna()
        r = returns_t1.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) > 30:
            ic, _ = stats.spearmanr(f[common], r[common])
            ic_vals.append(ic)
    if ic_vals:
        mean_ic = np.mean(ic_vals)
        print(f"{name}: {mean_ic:.4f} (样本数: {len(ic_vals)})")
    else:
        print(f"{name}: 无有效数据")

# ---- 诊断 3：合成后的等权因子表现 ----
print("\n========== 等权合成因子 ==========")
combo = (z1 + z2 + z3) / 3
rank_combo = combo.rank(axis=1)
ac_combo = rank_combo.corrwith(rank_combo.shift(1), axis=1).mean()
print(f"合成秩自相关: {ac_combo:.4f}")

# 合成因子 IC
ic_vals_combo = []
for date in combo.index[:-1]:
    f = combo.loc[date].dropna()
    r = returns_t1.loc[date].dropna()
    common = f.index.intersection(r.index)
    if len(common) > 30:
        ic, _ = stats.spearmanr(f[common], r[common])
        ic_vals_combo.append(ic)
if ic_vals_combo:
    mean_ic_combo = np.mean(ic_vals_combo)
    print(f"合成 IC: {mean_ic_combo:.4f}")

#!/usr/bin/env python3
"""
经典因子对照测试 — 验证当前评分系统是否能识别学术界公认的因子。

用法:
  python test_classic_factors.py

测试因子：
  1. 5日动量 (MOM5) — 学术界公认的动量因子
  2. 20日反转 (REV20) — 经典的短期反转效应
  3. f001 复刻 — 项目唯一入库因子，验证其稳健性

关键：正确计算 T+5 前向收益（与 batch_pipeline 中 P0-3 修复一致）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import PROJECT_ROOT

# ── 1. 加载数据 ────────────────────────────────────────────
print("[1/5] 加载全量数据 ...")
from batch_pipeline import load_df_1800

df_1800 = load_df_1800()
close_df = df_1800['close'].unstack('code')  # dates × stocks
print(f"      数据维度: {close_df.shape[0]} 天 × {close_df.shape[1]} 只股票")

# ── 2. 构造 T+5 前向收益（与 batch_pipeline P0-3 修复一致）────
print("[2/5] 构造 T+5 前向收益 ...")
# returns_t5(t) = close(t+5) / close(t) - 1
returns_t5 = close_df.shift(-5) / close_df - 1
print(f"      returns_t5 形状: {returns_t5.shape}")

# 同时构造 T+1 收益（用于 IC 衰减比）
returns_t1 = close_df.shift(-1) / close_df - 1

# ── 3. 定义经典因子 ─────────────────────────────────────────

FACTORS = {}

# 因子 1: 5日动量（学术界最经典的因子之一）
# 计算公式：过去5日累计收益率
# 经济学直觉：近期表现好的股票短期内继续表现好（动量效应）
def _momentum_5d(df):
    """5日动量因子：close / close.shift(5) - 1"""
    return df.groupby(level='code')['close'].transform(
        lambda x: x / x.shift(5) - 1
    )

FACTORS["MOM5_动量"] = _momentum_5d

# 因子 2: 20日反转（经典的短期反转效应）
# 计算公式：-过去20日累计收益率
# 经济学直觉：过去20日跌得多的股票短期内反弹（均值回复）
def _reversal_20d(df):
    """20日反转因子：-close.pct_change(20)"""
    return -df.groupby(level='code')['close'].transform(
        lambda x: x.pct_change(20)
    )

FACTORS["REV20_反转"] = _reversal_20d

# 因子 3: f001 复刻（项目唯一入库因子）
def _f001(df):
    """f001 复刻：日内方向 × 振幅 × 成交量放大 - 去趋势"""
    intraday_ret = (df['close'] - df['open']) / df['open']
    amplitude = (df['high'] - df['low']) / df['open']
    avg_vol_5 = df.groupby(level='code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    vol_ratio = df['volume'] / avg_vol_5
    raw_signal = intraday_ret * amplitude * vol_ratio
    smoothed = df.groupby(level='code')['close'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    factor = raw_signal - smoothed
    # 截面 z-score
    factor = (factor - factor.groupby(level='date').transform('mean')) / factor.groupby(level='date').transform('std')
    return factor

FACTORS["F001_入库"] = _f001

# 因子 4: 简单 5日收益率（无 groupby，纯截面）
def _simple_ret5(df):
    """简单5日收益率（无分组，直接用 close）"""
    return df['close'].groupby(level='code').transform(lambda x: x.pct_change(5))

FACTORS["RET5_简单"] = _simple_ret5

# 因子 5: 20日波动率（低波动异象）
def _volatility_20d(df):
    """20日波动率（滚动标准差）— 低波动异象"""
    return df.groupby(level='code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

FACTORS["VOL20_波动率"] = _volatility_20d

# 因子 6: 5日成交量均值（流动性因子）
def _volume_ma5(df):
    """5日成交量均值 / 当前成交量 — 流动性异常"""
    vol_ma5 = df.groupby(level='code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    return vol_ma5 / df['volume']

FACTORS["VOLRATIO_量比"] = _volume_ma5

# ── 4. 评分 ─────────────────────────────────────────────────
print("[3/5] 执行 6 个经典因子评分 ...")
from score import score_factor

results = {}

for name, compute_fn in FACTORS.items():
    print(f"\n  ── 测试因子: {name} ──")
    
    try:
        # 计算因子值
        factor_series = compute_fn(df_1800)
        
        # 转换为 DataFrame (dates × stocks)
        fv_df = factor_series.unstack('code')
        
        # 对齐索引
        common_dates = fv_df.index.intersection(returns_t5.index)
        fv_aligned = fv_df.loc[common_dates]
        ret5_aligned = returns_t5.loc[common_dates]
        ret1_aligned = returns_t1.loc[common_dates]
        
        print(f"      对齐后: {fv_aligned.shape[0]} 天 × {fv_aligned.shape[1]} 只")
        print(f"      因子覆盖率: {fv_aligned.notna().sum().sum() / fv_aligned.size:.2%}")
        
        # 评分
        result = score_factor(fv_aligned, ret5_aligned, returns_t1=ret1_aligned)
        results[name] = result
        
        # 打印结果
        print(f"      passed_threshold: {result.passed_threshold}")
        print(f"      total_score:      {result.total_score:.4f}")
        
        dim_order = ["ic", "ir", "coverage", "correlation", "turnover",
                     "directional_accuracy", "rank_autocorr_5d", "ic_decay_ratio"]
        
        for dim in dim_order:
            if dim in result.dimensions:
                d = result.dimensions[dim]
                status = "✅" if d["pass"] else "❌"
                print(f"      {dim:25s}  {d['value']:8.4f}  (阈 {d['threshold']:8.4f})  {status}")
        
        if result.failed_reasons:
            print(f"      失败: {result.failed_reasons[0]}")
            
    except Exception as e:
        print(f"      [错误] 评分失败: {e}")
        import traceback
        traceback.print_exc()

# ── 5. 汇总 ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  经典因子对照测试 — 汇总")
print("=" * 70)

for name, result in results.items():
    status = "✅ PASS" if result.passed_threshold else "❌ FAIL"
    ic = result.dimensions.get("ic", {}).get("value", 0)
    ir = result.dimensions.get("ir", {}).get("value", 0)
    rank_ac = result.dimensions.get("rank_autocorr_5d", {}).get("value", 0)
    dir_acc = result.dimensions.get("directional_accuracy", {}).get("value", 0)
    print(f"  {name:15s}  {status:8s}  IC={ic:7.4f}  IR={ir:7.4f}  "
          f"rank_ac={rank_ac:7.4f}  dir_acc={dir_acc:.4f}")

print("=" * 70)

# 统计
passed = sum(1 for r in results.values() if r.passed_threshold)
print(f"\n  通过数: {passed} / {len(results)}")

if passed == 0:
    print("\n  ⚠️ 所有经典因子均未通过评分！")
    print("  可能原因：")
    print("    1. 评分系统存在 bug（如 returns_t1 方向错误）")
    print("    2. 当前市场环境确实不适合这些因子")
    print("    3. 阈值设置过高（需要与经典文献对比）")
    print("\n  建议：检查 returns_t1 方向是否正确，")
    print("        以及 close_df.shift(-5) / close_df - 1 的 T+5 计算是否正确")
else:
    print(f"\n  ✅ {passed} 个经典因子通过评分，评分系统工作正常")
    print("  如果 AI 生成的因子仍失败，问题确实在 AI 生成质量")

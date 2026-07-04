#!/usr/bin/env python3
"""test_my_factor.py — 绕过 AI，直接测试 factor_draft.py 中的因子代码并输出 8 维评分。

用法:
  python test_my_factor.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import PROJECT_ROOT

# ── 1. 加载因子代码 ────────────────────────────────────────
factor_py_path = PROJECT_ROOT / "factor_draft.py"
if factor_py_path.exists():
    print(f"[1/6] 从 {factor_py_path} 读取 compute_factor ...")
    # 用沙箱安全执行的方式加载函数
    import importlib.util
    spec = importlib.util.spec_from_file_location("factor_draft", factor_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compute_factor = mod.compute_factor
    print("      因子来源: factor_draft.py")
else:
    print("[1/6] factor_draft.py 不存在，使用内嵌连续值因子 ...")
    def compute_factor(df):
        result = []
        for code, group in df.groupby('code'):
            group = group.sort_values('date')
            ret_20d = group['close'].pct_change(20)
            volume_5d = group['volume'].rolling(5).mean()
            volume_20d = group['volume'].rolling(20).mean()
            vol_5d = group['close'].rolling(5).std()
            vol_20d = group['close'].rolling(20).std()
            dim1 = -ret_20d
            dim2 = -(volume_5d / volume_20d)
            dim3 = -(vol_5d / vol_20d)
            factor = (dim1 + dim2 + dim3) / 3
            result.append(pd.DataFrame({'date': group['date'], 'code': code, 'factor': factor}))
        factor_df = pd.concat(result, ignore_index=True)
        return factor_df.set_index(['date', 'code'])['factor']
    print("      因子来源: 内嵌代码（连续值、无截面 z-score）")

# ── 2. 加载数据 ────────────────────────────────────────────
print("[2/6] 加载全量数据 ...")
from batch_pipeline import load_df_1800
df_1800 = load_df_1800()
close_df = df_1800['close'].unstack('code')  # dates x stocks
print(f"      数据维度: {close_df.shape[0]} 天 × {close_df.shape[1]} 只股票")

# ── 3. 沙箱执行 ────────────────────────────────────────────
print("[3/6] 沙箱执行因子代码 ...")
from sandbox import run_sandbox

# 因子代码中使用 df.groupby('code')，需要 code 是列而非 index level
# 传入 reset_index 后的 DataFrame，code 作为普通列
df_for_factor = df_1800.reset_index()

# 构造可执行的代码字符串（调用已加载的 compute_factor）
factor_code = f"""
def compute_factor(df):
    import pandas as pd
    import numpy as np
    return _compute_factor_impl(df)

_compute_factor_impl = {compute_factor.__name__}
"""

# 沙箱已内置 np/pd，因子代码不需要 import（__import__ 被沙箱拦截）
if factor_py_path.exists():
    with open(factor_py_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # 提取 compute_factor 函数（去掉 import 行和 if __name__ 块）
    lines = raw.split('\n')
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            continue
        if line.strip().startswith('if __name__'):
            break
        code_lines.append(line)
    factor_code_str = '\n'.join(code_lines)
else:
    factor_code_str = """
def compute_factor(df):
    result = []
    for code, group in df.groupby('code'):
        group = group.sort_values('date')
        ret_20d = group['close'].pct_change(20)
        volume_5d = group['volume'].rolling(5).mean()
        volume_20d = group['volume'].rolling(20).mean()
        vol_5d = group['close'].rolling(5).std()
        vol_20d = group['close'].rolling(20).std()
        dim1 = -ret_20d
        dim2 = -(volume_5d / volume_20d)
        dim3 = -(vol_5d / vol_20d)
        factor = (dim1 + dim2 + dim3) / 3
        result.append(pd.DataFrame({'date': group['date'], 'code': code, 'factor': factor}))
    factor_df = pd.concat(result, ignore_index=True)
    return factor_df.set_index(['date', 'code'])['factor']
"""

try:
    factor_series = run_sandbox(factor_code_str, df_for_factor, timeout=120)
    print(f"      因子值数量: {len(factor_series):,}")
    print(f"      非空比例:   {factor_series.notna().mean():.2%}")
except Exception as e:
    print(f"      [错误] 沙箱执行失败: {e}")
    sys.exit(1)

# ── 4. 构造 returns_df ─────────────────────────────────────
print("[4/6] 构造 returns_df (T+1) ...")
returns_df = close_df.pct_change().shift(-1)  # dates x stocks

# 还原 MultiIndex（沙箱传入的是 reset_index 后的 df，返回的 Series 丢失了 MultiIndex）
factor_series.index = df_1800.index
fv_df = factor_series.unstack("code")  # dates x stocks
common_dates = fv_df.index.intersection(returns_df.index)
fv_aligned = fv_df.loc[common_dates]
ret_aligned = returns_df.loc[common_dates]

print(f"      对齐后: {fv_aligned.shape[0]} 天 × {fv_aligned.shape[1]} 只")

# ── 5. 评分 ────────────────────────────────────────────────
print("[5/6] 执行 8 维评分 ...")
from score import score_factor

result = score_factor(fv_aligned, ret_aligned)

# ── 6. 输出结果 ────────────────────────────────────────────
print("\n" + "=" * 70)
print("  8 维评分结果")
print("=" * 70)

dim_order = ["ic", "ir", "coverage", "correlation_max", "turnover",
             "directional_accuracy", "rank_autocorr_5d", "ic_decay_ratio"]
dim_labels = {
    "ic": "IC 值", "ir": "IR 值", "coverage": "覆盖率",
    "correlation_max": "相关性", "turnover": "日换手率",
    "directional_accuracy": "方向正确性", "rank_autocorr_5d": "5日秩自相关",
    "ic_decay_ratio": "IC 衰减比",
}

for dim in dim_order:
    if dim in result.dimensions:
        d = result.dimensions[dim]
        status = "✅ PASS" if d["pass"] else "❌ FAIL"
        print(f"  {dim_labels.get(dim, dim):12s}  {d['value']:10.4f}  "
              f"(阈值 {d['threshold']:8.4f})  {status}")

print(f"\n  {'─'*50}")
print(f"  passed_threshold: {result.passed_threshold}")
print(f"  total_score:      {result.total_score:.4f}")
if result.failed_reasons:
    print(f"  failed_reasons:   {', '.join(result.failed_reasons)}")
print("=" * 70)

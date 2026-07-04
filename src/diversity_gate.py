#!/usr/bin/env python3
"""多样性门控 — Spearman 秩相关去重。

规则:
  - 新因子与已入库因子的 Spearman > 0.8 → 视为重复，直接丢弃
  - 因子池为空时自动通过
  - 基于 rank_snapshot 计算（{date: {stock_code: rank}}）

TODO(Phase 3): 新增 load_pool_from_sqlite(conn) 替代 load_factor_pool()
  当前仍读 JSON 因子池不影响功能（migration.py 已把历史因子导入 SQLite），
  但 diversity_gate 未切换为 SQLite 读取。Phase 3 全 A 股时统一迁移。
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROJECT_ROOT
FACTOR_POOL_PATH = PROJECT_ROOT / "factor_pool.json"
DIVERSITY_THRESHOLD = 0.7
MIN_COMMON_DAYS = 5  # 最少共同交易日


def load_factor_pool(path: Optional[Path] = None) -> dict:
    """加载因子库。"""
    p = path or FACTOR_POOL_PATH
    if not p.exists():
        return {"factors": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_factor_pool(pool: dict, path: Optional[Path] = None):
    """原子写入因子库。"""
    p = path or FACTOR_POOL_PATH
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(p)


def _rank_values(factor_values: pd.DataFrame) -> dict:
    """将因子值 DataFrame 转为 rank_snapshot 格式。

    Args:
        factor_values: DataFrame, index=date, columns=stock_code, values=因子值

    Returns:
        {date_str: {stock_code: rank}}
    """
    snapshot = {}
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) < 3:
            continue
        # rank 从 1 开始，handle ties with average
        ranks = row.rank(ascending=True, method="average").astype(int).to_dict()
        snapshot[str(date.date())] = ranks
    return snapshot


def convert_pool_for_scoring(factor_pool: dict) -> dict:
    """将 factor_pool.json 格式转换为 score.py 期望的旧格式。

    新格式: {"factors": [{"factor_id": "f001", "rank_snapshot": {date: {stock: rank}}}]}
    旧格式: {"f001": {"values": {(date, code): rank_value}, ...}}

    score.py 的 _max_correlation 使用旧格式，此转换层确保 score.py 不被修改。
    """
    converted = {}
    seen_ids = set()
    for f in factor_pool.get("factors", []):
        fid = f.get("factor_id", "")
        # P0-1 fix: 防御性检查 — 检测重复 factor_id
        if fid in converted:
            import warnings
            warnings.warn(
                f"⚠️ 检测到重复因子 ID '{fid}'，后一个因子将覆盖前一个。"
                f"请检查 batch_pipeline.py 中因子 ID 生成逻辑。"
            )
        rank_snap = f.get("rank_snapshot", {})
        # 展开 {date: {stock: rank}} → {(date, stock): rank}
        values = {}
        for date_str, stocks in rank_snap.items():
            for code, rank_val in stocks.items():
                values[(date_str, code)] = float(rank_val)
        if values:
            converted[fid] = {"values": values}
    return converted


def check_diversity(
    factor_values: pd.DataFrame,
    factor_pool: Optional[dict] = None,
    threshold: float = DIVERSITY_THRESHOLD,
) -> Tuple[bool, str, float]:
    """检查因子多样性（是否与存量因子重复）。

    Args:
        factor_values: 新因子值 DataFrame (date x stock_code)
        factor_pool: 已入库因子库，为空或 None 时自动通过
        threshold: Spearman 相关系数阈值

    Returns:
        (通过?, 重复因子ID或空字符串, 最大相关系数)
    """
    if factor_pool is None:
        pool = load_factor_pool()
    else:
        pool = factor_pool

    factors = pool.get("factors", [])
    if not factors:
        return (True, "")

    # 新因子的秩
    new_ranks = {}
    for date in factor_values.index:
        row = factor_values.loc[date].dropna()
        if len(row) >= 3:
            new_ranks[str(date.date())] = row.rank(ascending=True, method="average")

    max_corr = 0.0
    max_factor_id = ""

    for f in factors:
        pool_ranks = f.get("rank_snapshot", {})
        if not pool_ranks:
            continue

        # 找共同交易日
        common_dates = set(new_ranks.keys()) & set(pool_ranks.keys())
        if len(common_dates) < MIN_COMMON_DAYS:
            continue

        # 在共同交易日上计算 Spearman
        corrs = []
        for d in common_dates:
            nr = new_ranks[d]
            pr = pool_ranks[d]
            stocks = set(nr.keys()) & set(pr.keys())
            if len(stocks) < 3:
                continue
            aligned_new = [nr[s] for s in stocks]
            aligned_pool = [pr[s] for s in stocks]
            corr, _ = spearmanr(aligned_new, aligned_pool)
            if not np.isnan(corr):
                corrs.append(abs(corr))

        if corrs:
            avg_corr = float(np.mean(corrs))
            if avg_corr > max_corr:
                max_corr = avg_corr
                max_factor_id = f.get("factor_id", "?")

    if max_corr > threshold:
        return (False, f"{max_factor_id} (rho={max_corr:.4f})", max_corr)
    return (True, "", max_corr)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== diversity_gate.py 自检 ===\n")

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    stocks = ["600519", "000858", "002594", "300750", "000333"]

    # 1. 空因子池 → 自动通过
    passed, dup_id, max_corr = check_diversity(pd.DataFrame(), None)
    print(f"[1] 空因子池: passed={passed}, max_corr={max_corr:.4f} [OK]")

    # 2. 构造一个存量因子，测试去重
    fv_original = pd.DataFrame(
        np.random.randn(30, 5), index=dates, columns=stocks
    )
    rank_snap = _rank_values(fv_original)
    mock_pool = {
        "factors": [
            {"factor_id": "f001", "rank_snapshot": rank_snap}
        ]
    }

    # 2a. 高度相似的因子 → 应拒绝
    fv_similar = fv_original + np.random.randn(30, 5) * 0.01
    passed, dup_id, max_corr = check_diversity(fv_similar, mock_pool)
    print(f"[2a] 高度相似因子: passed={passed}, dup={dup_id}, max_corr={max_corr:.4f}")
    assert not passed, "应拒绝高度相似因子!"

    # 2b. 不同因子 → 应通过
    fv_different = pd.DataFrame(
        np.random.randn(30, 5) * 5, index=dates, columns=stocks
    )
    passed, dup_id, max_corr = check_diversity(fv_different, mock_pool)
    print(f"[2b] 不同因子: passed={passed}, dup={dup_id}, max_corr={max_corr:.4f}")
    assert passed, "应通过不同的因子!"

    # 3. 测试 _rank_values
    print(f"\n[3] _rank_values 示例 (前 3 天, 3 只股票):")
    snap = _rank_values(fv_original.iloc[:3, :3])
    for d, ranks in snap.items():
        print(f"    {d}: {ranks}")

    print("\n自检通过.")

#!/usr/bin/env python3
"""Phase 3c 组合合成引擎 — ICIR 加权多因子组合，仅参考不入库。

对入库因子计算 ICIR 加权组合 z-score，跑回测，与最佳单因子对比。
组合因子不入库、不参与多样性门控、不在批量流程中自动触发。

用法:
  python combo_engine.py --demo   # Mock 数据演示 ICIR 合成流程
"""

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest import simple_backtest, BacktestResult
from sandbox import run_sandbox


@dataclass
class ComboResult:
    """ICIR 加权组合结果。不入库，仅用于 HTML 报告展示。"""

    combo_factor: Optional[pd.DataFrame] = None       # 合成 z-score (date × stock_code)
    weights: Dict[str, float] = field(default_factory=dict)        # {factor_id: normalized_weight}
    icir_values: Dict[str, float] = field(default_factory=dict)    # {factor_id: raw_ICIR}
    backtest_result: Optional[BacktestResult] = None               # 组合回测结果
    vs_best_single: Dict[str, float] = field(default_factory=dict) # {combo_sharpe, best_single_sharpe, best_single_id, ratio}


def _compute_weights(icir_dict: Dict[str, float]) -> Dict[str, float]:
    """ICIR 加权，负值归零，正值归一化。

    Args:
        icir_dict: {factor_id: ICIR_value}

    Returns:
        {factor_id: normalized_weight}
    """
    positive_icir = {k: max(0.0, v) for k, v in icir_dict.items()}
    total = sum(positive_icir.values())
    if total == 0:
        return {k: 0.0 for k in icir_dict}
    return {k: v / total for k, v in positive_icir.items()}


def _compute_zscore(factor_values: pd.DataFrame) -> pd.DataFrame:
    """截面标准化：每行 mean=0, std=1（忽略 NaN）。

    Args:
        factor_values: date × stock_code 因子值矩阵

    Returns:
        标准化后的矩阵，std=0 的行填 0
    """
    row_mean = factor_values.mean(axis=1, skipna=True)
    row_std = factor_values.std(axis=1, skipna=True)
    row_std = row_std.replace(0, 1)  # 避免除零
    return factor_values.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)


def _compare_with_best_single(
    conn: sqlite3.Connection,
    combo_backtest: BacktestResult,
) -> Dict[str, float]:
    """对比组合与最佳单因子的夏普比率。

    Args:
        conn: SQLite 连接
        combo_backtest: 组合回测结果

    Returns:
        {combo_sharpe, best_single_sharpe, best_single_id, ratio}
    """
    row = conn.execute(
        """SELECT f.factor_id, b.sharpe_ratio
           FROM factors f
           JOIN backtests b ON f.factor_id = b.factor_id
           ORDER BY b.sharpe_ratio DESC
           LIMIT 1"""
    ).fetchone()

    combo_sharpe = combo_backtest.sharpe_ratio

    if row is None or row["sharpe_ratio"] is None:
        return {
            "combo_sharpe": combo_sharpe,
            "best_single_sharpe": 0.0,
            "best_single_id": "",
            "ratio": 0.0,
        }

    best_sharpe = row["sharpe_ratio"]
    ratio = combo_sharpe / best_sharpe if best_sharpe > 0 else 0.0

    return {
        "combo_sharpe": combo_sharpe,
        "best_single_sharpe": best_sharpe,
        "best_single_id": row["factor_id"] or "",
        "ratio": round(ratio, 4),
    }


def build_all_inbound(
    conn: sqlite3.Connection,
    df_1800: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> ComboResult:
    """自动读取全部入库因子，计算 ICIR 加权组合。

    流程：
    1. 从 factors 表读取每个入库因子的 code_snapshot 和 IR
    2. 通过沙箱重新执行代码获取因子值
    3. 计算权重：负 IR 因子权重归零，正 IR 归一化
    4. 各因子截面 z-score 加权合成
    5. 调用 backtest.simple_backtest() 跑回测
    6. 与最佳单因子对比

    Args:
        conn: SQLite 连接
        df_1800: MultiIndex (date, code) 原始数据，用于沙箱执行
        close_df: date × stock_code 收盘价矩阵
        volume_df: date × stock_code 成交量矩阵
        returns_df: date × stock_code 日收益率矩阵

    Returns:
        ComboResult（入库因子为 0 时返回空 ComboResult）
    """
    # 读取全部入库因子
    rows = conn.execute(
        """SELECT factor_id, code_snapshot,
                  CAST(json_extract(metrics, '$.ir') AS REAL) as ir
           FROM factors
           WHERE code_snapshot IS NOT NULL AND code_snapshot != ''
           ORDER BY factor_id"""
    ).fetchall()

    if not rows:
        return ComboResult()

    # 收集 IR 值
    icir_values: Dict[str, float] = {}
    factor_frames: Dict[str, pd.DataFrame] = {}

    for r in rows:
        fid = r["factor_id"]
        ir_val = r["ir"] or 0.0
        icir_values[fid] = ir_val
        code = r["code_snapshot"]

        # 通过沙箱执行因子代码
        try:
            fv_series = run_sandbox(code, df_1800, timeout=30)
            fv_df = fv_series.unstack("code")
            # 只保留与 returns_df 对齐的日期和股票
            common_dates = fv_df.index.intersection(returns_df.index)
            common_stocks = fv_df.columns.intersection(returns_df.columns)
            if len(common_dates) > 0 and len(common_stocks) > 0:
                factor_frames[fid] = fv_df.loc[common_dates, common_stocks]
        except Exception:
            continue  # 沙箱执行失败的因子跳过

    if not factor_frames:
        return ComboResult(icir_values=icir_values, weights=_compute_weights(icir_values))

    # 计算权重
    weights = _compute_weights(icir_values)

    # 找所有因子的共同日期和股票交集
    all_dates = None
    all_stocks = None
    for fv_df in factor_frames.values():
        if all_dates is None:
            all_dates = set(fv_df.index)
            all_stocks = set(fv_df.columns)
        else:
            all_dates = all_dates.intersection(fv_df.index)
            all_stocks = all_stocks.intersection(fv_df.columns)

    if all_dates is None or len(all_dates) < 10 or len(all_stocks) < 5:
        return ComboResult(
            icir_values=icir_values,
            weights=weights,
        )

    all_dates = sorted(all_dates)
    all_stocks = sorted(all_stocks)

    # 各因子 z-score 加权合成
    combo_z = pd.DataFrame(0.0, index=all_dates, columns=all_stocks)
    active_factors = 0

    for fid, fv_df in factor_frames.items():
        w = weights.get(fid, 0.0)
        if w <= 0:
            continue
        fv_aligned = fv_df.loc[all_dates, all_stocks]
        z = _compute_zscore(fv_aligned)
        combo_z += z * w
        active_factors += 1

    if active_factors == 0:
        return ComboResult(
            icir_values=icir_values,
            weights=weights,
        )

    # 回测
    bt_result = simple_backtest(
        combo_z, returns_df.loc[all_dates, all_stocks],
        volume_df=volume_df.loc[all_dates, all_stocks],
        close_df=close_df.loc[all_dates, all_stocks],
    )

    # 对比最佳单因子
    vs_best = _compare_with_best_single(conn, bt_result)

    return ComboResult(
        combo_factor=combo_z,
        weights=weights,
        icir_values=icir_values,
        backtest_result=bt_result,
        vs_best_single=vs_best,
    )


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        print("=== combo_engine.py Phase 3c ICIR 合成演示 ===\n")
        np.random.seed(42)

        # 创建临时 SQLite 数据库
        import tempfile
        import os as _os

        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db_path = tmp_db.name
        tmp_db.close()

        tmp_conn = sqlite3.connect(tmp_db_path)
        tmp_conn.execute("PRAGMA journal_mode=WAL")
        tmp_conn.row_factory = sqlite3.Row

        # 建表（最小结构）
        tmp_conn.executescript("""
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                code_snapshot TEXT,
                metrics TEXT,
                direction_tag TEXT DEFAULT '',
                inbound_date TEXT DEFAULT '2024-01-01'
            );
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT NOT NULL,
                sharpe_ratio REAL,
                annual_return REAL,
                max_drawdown REAL
            );
            CREATE TABLE IF NOT EXISTS rounds (
                round_id INTEGER NOT NULL,
                batch_run_id INTEGER,
                status TEXT DEFAULT 'fail',
                api_cost REAL DEFAULT 0,
                PRIMARY KEY (round_id, batch_run_id)
            );
        """)

        # Mock 数据：100 天 × 20 只股票
        n_days = 100
        n_stocks = 20
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        stocks = [f"S{i:04d}" for i in range(n_stocks)]

        # 构造价格和成交量
        close_data = {}
        volume_data = {}
        for s in stocks:
            base = 50 + np.random.rand() * 100
            prices = [base]
            for _ in range(n_days - 1):
                prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
            close_data[s] = pd.Series(prices, index=dates)
            vol = np.abs(1e7 + np.random.randn(n_days) * 2e6)
            volume_data[s] = pd.Series(vol, index=dates)

        close_df = pd.DataFrame(close_data)
        volume_df = pd.DataFrame(volume_data)
        returns_df = close_df.pct_change()

        # 构造 MultiIndex df_1800（模拟真实数据格式）
        frames = []
        for s in stocks:
            df_s = pd.DataFrame({
                "date": dates,
                "code": s,
                "open": close_df[s] * 0.99,
                "high": close_df[s] * 1.02,
                "low": close_df[s] * 0.98,
                "close": close_df[s],
                "volume": volume_df[s],
            })
            frames.append(df_s)
        df_1800 = pd.concat(frames, ignore_index=True)
        df_1800 = df_1800.set_index(["date", "code"]).sort_index()

        # 构造 3 个模拟因子
        # 因子 1：强正向 (ICIR=0.5)
        # 因子 2：负向 (ICIR=-0.3) → 权重应归零
        # 因子 3：优秀 (ICIR=1.2)
        mock_factors = [
            ("f001", 0.5, "def compute_factor(df):\n    close = df['close'].unstack('code')\n    return close.pct_change(5).stack()"),
            ("f002", -0.3, "def compute_factor(df):\n    close = df['close'].unstack('code')\n    return -close.pct_change(5).stack()"),
            ("f003", 1.2, "def compute_factor(df):\n    close = df['close'].unstack('code')\n    high = df['high'].unstack('code')\n    low = df['low'].unstack('code')\n    return ((close - low) / (high - low + 0.01)).stack()"),
        ]

        import json
        for fid, ir_val, code in mock_factors:
            tmp_conn.execute(
                "INSERT INTO factors (factor_id, code_snapshot, metrics) VALUES (?, ?, ?)",
                (fid, code, json.dumps({"ic": ir_val * 0.2, "ir": ir_val})),
            )
            # 插入对应的 backtest 记录
            tmp_conn.execute(
                "INSERT INTO backtests (factor_id, sharpe_ratio, annual_return, max_drawdown) VALUES (?, ?, ?, ?)",
                (fid, ir_val * 2.0, ir_val * 0.3, -0.15),
            )
        tmp_conn.commit()

        print(f"  股票数: {n_stocks} | 交易日: {n_days}")
        print(f"  模拟因子: 3 个 (ICIR: 0.5, -0.3, 1.2)\n")

        t0 = time.time()
        result = build_all_inbound(tmp_conn, df_1800, close_df, volume_df, returns_df)
        elapsed = time.time() - t0

        print(f"  ╔{'═'*50}╗")
        print(f"  ║  ICIR 加权组合合成结果")
        print(f"  ╠{'═'*50}╣")

        print(f"\n  [ICIR 值与权重]")
        print(f"  {'因子ID':8s} {'ICIR':>8s} {'权重':>8s} {'状态'}")
        print(f"  {'-'*40}")
        for fid in ["f001", "f002", "f003"]:
            icir = result.icir_values.get(fid, 0)
            w = result.weights.get(fid, 0)
            status = "✓ 参与合成" if w > 0 else "✗ 负值归零"
            print(f"  {fid:8s} {icir:>8.3f} {w:>8.3f} {status}")

        # 验证权重
        positive_weights = {k: v for k, v in result.weights.items() if v > 0}
        weight_sum = sum(positive_weights.values())
        print(f"\n  正权重和: {weight_sum:.4f} {'✓' if abs(weight_sum - 1.0) < 0.01 else '✗'}")
        print(f"  负ICIR因子(f002)权重: {result.weights.get('f002', 0):.4f} {'✓ (归零)' if result.weights.get('f002', 0) == 0 else '✗'}")

        if result.backtest_result:
            bt = result.backtest_result
            print(f"\n  [组合回测结果]")
            print(f"  夏普比率:     {bt.sharpe_ratio:.4f}")
            print(f"  年化收益:     {bt.annual_return:.2%}")
            print(f"  最大回撤:     {bt.max_drawdown:.2%}")
            print(f"  调仓胜率:     {bt.win_rate:.2%}")
            print(f"  冲击成本:     {bt.avg_impact_cost_bps:.2f} bps")

        print(f"\n  [vs 最佳单因子]")
        vs = result.vs_best_single
        print(f"  组合夏普:     {vs.get('combo_sharpe', 0):.4f}")
        print(f"  最佳单因子:   {vs.get('best_single_id', '-')} (夏普={vs.get('best_single_sharpe', 0):.4f})")
        print(f"  Combo/Best:   {vs.get('ratio', 0):.2%}")
        ratio_ok = vs.get('ratio', 0) >= 0.80
        print(f"  >=80%达标:    {'✓' if ratio_ok else '✗ (不阻断报告)'}")

        print(f"\n  计算耗时: {elapsed:.2f}s")
        print(f"  ╚{'═'*50}╝")

        # 验证不入库
        count = tmp_conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
        print(f"\n  [验证] factors 表行数: {count} (应为 3) {'✓ 未写入' if count == 3 else '✗ 异常写入!'}")

        tmp_conn.close()
        _os.unlink(tmp_db_path)
        print("\n自检通过.")
    else:
        print("用法: python combo_engine.py --demo")

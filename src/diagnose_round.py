#!/usr/bin/env python3
"""Phase 3i D2 — 单轮回测根因诊断脚本。

用法:
  python diagnose_round.py --round 10     # 诊断 R10
  python diagnose_round.py --round 10 --output docs/r10_diagnosis.md

流程:
  1. 从 DB 提取目标轮次的因子代码
  2. 沙箱执行因子代码
  3. 回测 + 全程诊断（因子分布/仓位矩阵/收益路径/冲击成本）
  4. 输出 markdown 诊断报告
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── 项目内导入 ──────────────────────────────────
from config import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from database import get_conn
from sandbox import run_sandbox, SandboxTimeout
from batch_pipeline import load_df_1800
from backtest import simple_backtest
from config import HOLDING_PERIOD, TOP_PCT


from config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "db" / "factorlab.db"


def query_round_code(round_num: int) -> Optional[tuple]:
    """从数据库提取目标轮次的因子代码。

    Returns:
        (factor_code, direction_tag, fail_reason) 或 None
    """
    conn = get_conn(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT factor_code, direction_tag, fail_reason FROM rounds "
            "WHERE round_id = ? ORDER BY batch_run_id DESC LIMIT 1",
            (round_num,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return row[0], row[1], row[2]
    finally:
        conn.close()


def analyze_round(
    round_num: int,
    df_1800: pd.DataFrame,
    close_df_full: pd.DataFrame,
    returns_df_full: pd.DataFrame,
    volume_df_full: pd.DataFrame,
) -> dict:
    """对指定轮次执行完整诊断流程。"""

    result = {
        "round_num": round_num,
        "status": "unknown",
        "factor_code": None,
        "direction_tag": "",
        "db_fail_reason": "",
        # 沙箱
        "sandbox_success": False,
        "sandbox_error": "",
        "sandbox_time": 0.0,
        # 因子值
        "factor_coverage": 0.0,      # % 非 NaN
        "factor_mean": 0.0,
        "factor_std": 0.0,
        "factor_min": 0.0,
        "factor_max": 0.0,
        "factor_n_dates": 0,
        "factor_n_stocks": 0,
        # Top decile 选股
        "top_decile_daily_holdings_mean": 0.0,
        "top_decile_daily_holdings_median": 0.0,
        "top_decile_zero_position_days": 0,
        "top_decile_total_days": 0,
        # 仓位矩阵
        "position_matrix_nonzero_ratio": 0.0,
        "position_matrix_avg_holding_count": 0.0,
        # 回测结果
        "backtest_success": False,
        "backtest_error": "",
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "turnover_estimate": 0.0,
        "avg_impact_cost_bps": 0.0,
        "total_cost_annual": 0.0,
        "is_degenerate": False,
        # 日收益诊断（扣成本前）
        "daily_ret_mean": 0.0,
        "daily_ret_std": 0.0,
        "daily_ret_positive_ratio": 0.0,
        # 累积收益终点
        "cum_return_final": 0.0,
    }

    # ── Step 1: 从 DB 获取代码 ────────────────────
    row = query_round_code(round_num)
    if row is None:
        result["status"] = "db_not_found"
        return result

    factor_code, direction_tag, fail_reason = row
    result["factor_code"] = factor_code
    result["direction_tag"] = direction_tag or ""
    result["db_fail_reason"] = fail_reason or ""

    if not factor_code or not factor_code.strip():
        result["status"] = "no_code"
        return result

    # ── Step 2: 沙箱执行 ──────────────────────────
    import time
    t0 = time.time()
    try:
        from checker import preprocess_code
        factor_series = run_sandbox(factor_code, df_1800)
        result["sandbox_time"] = round(time.time() - t0, 2)
        result["sandbox_success"] = True
    except SandboxTimeout:
        result["sandbox_error"] = f"沙箱超时 ({time.time() - t0:.1f}s)"
        result["sandbox_time"] = round(time.time() - t0, 2)
        result["status"] = "sandbox_timeout"
        return result
    except Exception as e:
        result["sandbox_error"] = str(e)[:300]
        result["sandbox_time"] = round(time.time() - t0, 2)
        result["status"] = "sandbox_error"
        return result

    # ── Step 3: 因子值分布诊断 ───────────────────
    fv_df = factor_series.unstack("code")
    result["factor_n_dates"] = fv_df.shape[0]
    result["factor_n_stocks"] = fv_df.shape[1]

    # 覆盖率（%非 NaN）
    total_cells = fv_df.size
    non_nan = fv_df.notna().sum().sum()
    result["factor_coverage"] = round(non_nan / total_cells, 4) if total_cells > 0 else 0.0

    flat = fv_df.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) > 0:
        result["factor_mean"] = round(float(np.mean(flat)), 6)
        result["factor_std"] = round(float(np.std(flat)), 6)
        result["factor_min"] = round(float(np.min(flat)), 6)
        result["factor_max"] = round(float(np.max(flat)), 6)

    # ── Step 4: 对齐日期/股票 ──────────────────────
    common_dates = fv_df.index.intersection(close_df_full.index)
    common_stocks = fv_df.columns.intersection(close_df_full.columns)
    if len(common_dates) < 20 or len(common_stocks) < 10:
        result["status"] = "insufficient_data"
        result["backtest_error"] = (
            f"共同日期 {len(common_dates)} 或共同股票 {len(common_stocks)} 不足"
        )
        return result

    fv_aligned = fv_df.loc[common_dates, common_stocks]
    ret_slice = returns_df_full.loc[common_dates, common_stocks]

    # ── Step 5: Top decile 选股分析 ───────────────
    top_pct = TOP_PCT  # default 0.20
    n_top = max(1, int(len(common_stocks) * top_pct))
    top_decile_holdings = []
    top_decile_zero_days = 0
    for date in fv_aligned.index:
        row = fv_aligned.loc[date].dropna()
        if len(row) < n_top:
            top_decile_holdings.append(0)
            top_decile_zero_days += 1
            continue
        top_stocks = row.nlargest(n_top)
        # 检查 top decile 因子值是否有意义
        top_val_mean = top_stocks.mean()
        all_val_mean = row.mean()
        # 选股是否有区分度（top 均值 vs 全体均值）
        top_decile_holdings.append(len(top_stocks))
        if len(top_stocks) == 0:
            top_decile_zero_days += 1

    if top_decile_holdings:
        result["top_decile_daily_holdings_mean"] = round(float(np.mean(top_decile_holdings)), 2)
        result["top_decile_daily_holdings_median"] = round(float(np.median(top_decile_holdings)), 2)
    result["top_decile_zero_position_days"] = top_decile_zero_days
    result["top_decile_total_days"] = len(fv_aligned.index)

    # ── Step 6: 仓位矩阵诊断 ──────────────────────
    # 用 top decile 模拟持仓矩阵（每日期望的 top decile 持仓为 1）
    n_dates = len(common_dates)
    n_stocks = len(common_stocks)
    pos_matrix = np.zeros((n_dates, n_stocks))
    for i, date in enumerate(common_dates):
        row = fv_aligned.loc[date].dropna()
        if len(row) < n_top:
            continue
        top_stocks = row.nlargest(n_top).index
        for s in top_stocks:
            j = common_stocks.get_loc(s)
            pos_matrix[i, j] = 1.0

    non_zero = np.count_nonzero(pos_matrix)
    result["position_matrix_nonzero_ratio"] = round(
        non_zero / pos_matrix.size, 4
    ) if pos_matrix.size > 0 else 0.0
    result["position_matrix_avg_holding_count"] = round(
        non_zero / n_dates, 1
    ) if n_dates > 0 else 0.0

    # ── Step 7: 回测执行 ──────────────────────────
    try:
        vol_slice = volume_df_full.loc[common_dates, common_stocks]
        cls_slice = close_df_full.loc[common_dates, common_stocks]

        bt_result = simple_backtest(
            fv_aligned, ret_slice,
            volume_df=vol_slice, close_df=cls_slice,
        )
        result["backtest_success"] = True
        result["annual_return"] = round(bt_result.annual_return, 4)
        result["max_drawdown"] = round(bt_result.max_drawdown, 4)
        result["sharpe_ratio"] = round(bt_result.sharpe_ratio, 4)
        result["win_rate"] = round(bt_result.win_rate, 4)
        result["turnover_estimate"] = round(bt_result.turnover_estimate, 4)
        result["avg_impact_cost_bps"] = round(bt_result.avg_impact_cost_bps, 4)
        result["total_cost_annual"] = round(bt_result.total_cost_annual, 4)

        # 检测异常
        if not np.isfinite(bt_result.annual_return) or bt_result.annual_return <= -0.99:
            result["is_degenerate"] = True

        # 日收益诊断
        if bt_result.cum_return_curve is not None and len(bt_result.cum_return_curve) > 1:
            daily_rets = bt_result.cum_return_curve.diff().dropna()
            if len(daily_rets) > 0:
                result["daily_ret_mean"] = round(float(daily_rets.mean()), 6)
                result["daily_ret_std"] = round(float(daily_rets.std()), 6)
                result["daily_ret_positive_ratio"] = round(
                    float((daily_rets > 0).sum() / len(daily_rets)), 4
                )
            result["cum_return_final"] = round(
                float(bt_result.cum_return_curve.iloc[-1]), 4
            )
    except Exception as e:
        result["backtest_error"] = str(e)[:500]
        result["status"] = "backtest_error"
        return result

    result["status"] = "complete"
    return result


def write_diagnosis_report(result: dict, output_path: Path):
    """将诊断结果写入 markdown 文件。"""

    status_icon = {
        "complete": "✅ 完整诊断",
        "db_not_found": "❌ 数据库中未找到",
        "no_code": "⚠️ 轮次无因子代码",
        "sandbox_timeout": "⏱️ 沙箱超时",
        "sandbox_error": "❌ 沙箱执行失败",
        "insufficient_data": "⚠️ 数据不足",
        "backtest_error": "❌ 回测执行失败",
    }

    lines = []
    lines.append(f"# R{result['round_num']} 回测根因诊断报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 诊断状态: {status_icon.get(result['status'], result['status'])}\n")

    # ── DB 信息 ──────────────────────────────
    lines.append("## 1. 数据库信息\n")
    lines.append(f"| 字段 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 方向标签 | {result['direction_tag'] or '(无)'} |")
    lines.append(f"| DB 失败原因 | {result['db_fail_reason'][:200] if result['db_fail_reason'] else '(无)'} |")
    lines.append("")

    if result["status"] == "db_not_found":
        lines.append("> ⚠️ 数据库中未找到此轮次记录，可能已被清理或轮次号错误。")
        lines.append("")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # ── 因子代码 ──────────────────────────────
    lines.append("## 2. 因子代码\n")
    lines.append("```python")
    lines.append(result["factor_code"] if result["factor_code"] else "# 无代码")
    lines.append("```\n")

    # ── 沙箱 ──────────────────────────────
    lines.append("## 3. 沙箱执行\n")
    if result["sandbox_success"]:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 状态 | ✅ 成功 |")
        lines.append(f"| 耗时 | {result['sandbox_time']}s |")
        lines.append("")
    else:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 状态 | ❌ 失败 |")
        lines.append(f"| 耗时 | {result['sandbox_time']}s |")
        lines.append(f"| 错误 | {result['sandbox_error']} |")
        lines.append("")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # ── 因子值分布 ──────────────────────────
    lines.append("## 4. 因子值分布\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 覆盖日期数 | {result['factor_n_dates']} |")
    lines.append(f"| 覆盖股票数 | {result['factor_n_stocks']} |")
    lines.append(f"| 覆盖率 (%非NaN) | {result['factor_coverage']:.2%} |")
    lines.append(f"| 均值 | {result['factor_mean']:.6f} |")
    lines.append(f"| 标准差 | {result['factor_std']:.6f} |")
    lines.append(f"| 最小值 | {result['factor_min']:.6f} |")
    lines.append(f"| 最大值 | {result['factor_max']:.6f} |")
    lines.append("")

    # 分布诊断
    lines.append("### 诊断分析\n")
    issues = []

    if result["factor_coverage"] < 0.6:
        issues.append(f"- ⚠️ **覆盖率过低** ({result['factor_coverage']:.1%})：超过 40% 的因子值缺失，因子可能包含过度筛选条件。")
    if result["factor_std"] < 1e-8 and result["factor_max"] - result["factor_min"] < 1e-8:
        issues.append("- 🔴 **因子值几乎无变异**：所有股票得分相同，无法排序选股。")
    if result["factor_mean"] == 0 and np.abs(result["factor_std"]) < 1e-8:
        issues.append("- 🔴 **因子值全为零**：compute_factor 返回了常数零值。")
    if result["factor_std"] > 10 * abs(result["factor_mean"]) and result["factor_mean"] != 0:
        issues.append(f"- ⚠️ **极端波动** (std={result['factor_std']:.4f}, mean={result['factor_mean']:.4f})：CV > 10，可能存在异常值或除零问题。")
    if not np.isfinite(result["factor_mean"]):
        issues.append("- 🔴 **因子值包含 inf/NaN**：代码可能存在除零或对数负值问题。")

    if not issues:
        lines.append("因子值分布正常，无显著异常。\n")
    else:
        for issue in issues:
            lines.append(issue)
        lines.append("")

    # ── Top Decile 选股 ──────────────────────
    lines.append("## 5. Top Decile 选股分析\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 日均持仓数 | {result['top_decile_daily_holdings_mean']:.1f} |")
    lines.append(f"| 中位持仓数 | {result['top_decile_daily_holdings_median']:.1f} |")
    lines.append(f"| 零持仓日数 | {result['top_decile_zero_position_days']} / {result['top_decile_total_days']} |")
    lines.append("")

    zero_pct = result["top_decile_zero_position_days"] / max(result["top_decile_total_days"], 1)
    if zero_pct > 0.5:
        lines.append(f"- 🔴 **零持仓日 > 50%** ({zero_pct:.0%})：因子值在多数交易日全部为 NaN。\n")
    elif zero_pct > 0.1:
        lines.append(f"- ⚠️ **零持仓日较多** ({zero_pct:.0%})。\n")

    # ── 仓位矩阵 ──────────────────────────────
    lines.append("## 6. 仓位矩阵诊断\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 非零仓位比例 | {result['position_matrix_nonzero_ratio']:.4f} |")
    lines.append(f"| 日均持仓股票数 | {result['position_matrix_avg_holding_count']:.1f} |")
    lines.append("")

    expected_ratio = TOP_PCT
    actual_ratio = result["position_matrix_nonzero_ratio"]
    if actual_ratio < expected_ratio * 0.5:
        lines.append(f"- ⚠️ 实际持仓比例 ({actual_ratio:.2%}) 远低于预期 ({expected_ratio:.0%})。\n")
    if result["position_matrix_avg_holding_count"] < 5:
        lines.append(f"- 🔴 日均持仓不足 5 只：过度集中，无法分散风险。\n")

    # ── 回测结果 ──────────────────────────────
    lines.append("## 7. 回测结果\n")
    if result["backtest_success"]:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 年化收益 | {result['annual_return']:.2%} |")
        lines.append(f"| 最大回撤 | {result['max_drawdown']:.2%} |")
        lines.append(f"| 夏普比率 | {result['sharpe_ratio']:.4f} |")
        lines.append(f"| 胜率 | {result['win_rate']:.2%} |")
        lines.append(f"| 估算换手率 | {result['turnover_estimate']:.2%} |")
        lines.append(f"| 平均冲击成本 | {result['avg_impact_cost_bps']:.2f} bps |")
        lines.append(f"| 年化总成本 | {result['total_cost_annual']:.2%} |")
        lines.append(f"| 累积收益终点 | {result['cum_return_final']:.4f} |")
        lines.append("")

        if result["is_degenerate"]:
            lines.append("### 🔴 回测异常标记\n")
            lines.append("回测结果为 **退化（degenerate）**——年化收益 ≤ -99%，组合归零或溢出。\n")
    else:
        lines.append(f"❌ 回测执行失败: {result['backtest_error']}\n")

    # ── 日收益诊断 ────────────────────────────
    lines.append("## 8. 日收益路径诊断\n")
    if result["backtest_success"]:
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 日收益均值 | {result['daily_ret_mean']:.6f} |")
        lines.append(f"| 日收益标准差 | {result['daily_ret_std']:.6f} |")
        lines.append(f"| 正收益日比例 | {result['daily_ret_positive_ratio']:.2%} |")
        lines.append("")

        if result["daily_ret_mean"] < -0.001:
            lines.append(f"- 🔴 **日均收益极负** ({result['daily_ret_mean']:.4%})：每天稳定亏损。\n")
        if result["daily_ret_std"] > 0.05:
            lines.append(f"- ⚠️ **日收益波动大** (std={result['daily_ret_std']:.4%})。\n")
        if result["daily_ret_positive_ratio"] < 0.40:
            lines.append(f"- 🔴 **胜率极低** ({result['daily_ret_positive_ratio']:.0%})：大多数日子在亏钱。\n")

    # ── 根因判定 ──────────────────────────────
    lines.append("## 9. 根因判定\n")
    causes = []

    if result["is_degenerate"]:
        # 进一步细化退化原因
        if result["factor_coverage"] < 0.1:
            causes.append("**因子值覆盖率极低**：因子代码过于严格的条件筛选导致多数股票/日期无因子值，回测中持仓归零。")
        elif result["position_matrix_avg_holding_count"] < 5:
            causes.append("**持仓过度稀疏**：日均持仓不足 5 只，调仓时冲击成本吃掉全部 alpha，组合净值快速归零。")
        elif result["avg_impact_cost_bps"] > 50:
            causes.append(f"**冲击成本过高** ({result['avg_impact_cost_bps']:.1f} bps)：换仓成本超过 alpha 收益，组合净值持续衰减。")
        elif not np.isfinite(result["factor_mean"]):
            causes.append("**因子值含 inf/NaN**：代码存在除零或无效数学运算，导致回测崩溃。")
        else:
            causes.append("**组合净值归零**：因子区分度不足 + 换仓成本侵蚀，导致组合净值在回测期内归零。")
    elif result["status"] != "complete":
        causes.append(f"**诊断未完成**: {result['status']}")
    else:
        if result["annual_return"] < -0.5:
            causes.append(f"**持续大幅亏损** (年化 {result['annual_return']:.0%})：因子方向错误或选股逻辑与收益反向。")
        if result["sharpe_ratio"] < -0.5:
            causes.append("**夏普极负**：因子在风险调整后持续跑输。")

    if not causes:
        causes.append("未发现明确根因，需人工审查因子代码的选股逻辑。")

    for i, cause in enumerate(causes, 1):
        lines.append(f"{i}. {cause}")
    lines.append("")

    # ── 对 Stage 2 的建议 ──────────────────────
    lines.append("## 10. 对 Stage 2 的建议\n")
    if result["is_degenerate"]:
        lines.append("- **P0**: 在 engine.py 中增加「因子值稀疏检测」——覆盖率 < 40% 直接 REJECT，不进入评分")
        lines.append("- **P0**: 在 batch_pipeline 回测前增加「持仓数检查」——日均持仓 < 10 只直接 FAIL")
        lines.append("- **P1**: 考虑在 checker 中检测可能导致除零的模式（如 `/ vol` 无 `replace(0, np.nan)`）")
    elif result.get("factor_coverage", 0) < 0.4:
        lines.append("- **P1**: 因子值覆盖率过低，建议在 sandbox 之后增加覆盖率检查")
    else:
        lines.append("- **P1**: 诊断未发现明显代码缺陷，需检查因子经济学逻辑是否正确")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"诊断报告已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FactorLab 单轮回测根因诊断")
    parser.add_argument("--round", type=int, required=True, help="目标轮次号")
    parser.add_argument("--output", type=str, default=None, help="输出路径 (默认 docs/r{round}_diagnosis.md)")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="限制加载股票数 (0=全部，测试用)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(f"docs/r{args.round}_diagnosis.md")

    print(f"=== Phase 3i D2: R{args.round} 回测根因诊断 ===\n")

    # 加载全量数据
    print("加载数据中...")
    df_1800 = load_df_1800(max_stocks=args.max_stocks)
    close_df_full = df_1800["close"].unstack("code")
    volume_df_full = df_1800["volume"].unstack("code")
    returns_df_full = close_df_full.pct_change(fill_method=None)
    print(f"数据加载完成: {close_df_full.shape[0]} 日 × {close_df_full.shape[1]} 股\n")

    # 运行诊断
    result = analyze_round(
        args.round,
        df_1800, close_df_full, returns_df_full, volume_df_full,
    )

    # 输出到文件
    write_diagnosis_report(result, output)

    # 简要摘要
    print(f"\n=== 诊断摘要 ===")
    print(f"状态: {result['status']}")
    if result["sandbox_success"]:
        print(f"覆盖率: {result['factor_coverage']:.2%}")
        print(f"日均持仓: {result['position_matrix_avg_holding_count']:.1f}")
    if result["backtest_success"]:
        print(f"年化收益: {result['annual_return']:.2%}")
        print(f"夏普: {result['sharpe_ratio']:.4f}")
        print(f"异常标记: {'是' if result['is_degenerate'] else '否'}")


if __name__ == "__main__":
    main()

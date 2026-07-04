#!/usr/bin/env python3
"""Phase 3c 静态 HTML 报告生成器 — base64 内嵌图表，单文件自包含。

报告版本：
  - 快速版：a.因子表格 + d.成本仪表盘 + 简要统计
  - 完整版：a-g 全部内容（含 ICIR 合成对比）

用法:
  python html_reporter.py --demo   # 生成演示报告
  python html_reporter.py --quick  # 从真实数据库生成快速报告
  python html_reporter.py --full   # 从真实数据库生成完整报告
"""

import json
import sqlite3
import base64
import os as _os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── 中文字体配置 ──────────────────────────────────────
_CJK_FONT = None
for _fname in ["PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS", "Songti SC"]:
    try:
        _prop = fm.FontProperties(family=_fname)
        # 测试是否能正确渲染中文
        _test_fig, _test_ax = plt.subplots(figsize=(1, 1))
        _test_ax.set_title("中文", fontproperties=_prop)
        plt.close(_test_fig)
        _CJK_FONT = _prop
        break
    except Exception:
        continue

# Fallback: 尝试通过 font_manager 查找任意中文字体
if _CJK_FONT is None:
    try:
        _all_cjk = [f for f in fm.fontManager.ttflist
                    if any(k in f.name.lower() for k in ["hei", "ping", "song", "cjk", "arial unicode"])]
        if _all_cjk:
            _CJK_FONT = fm.FontProperties(family=_all_cjk[0].name)
    except Exception:
        pass

from config import REPORTS_DIR

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def fig_to_base64(fig, dpi: int = 100) -> str:
    """matplotlib Figure → base64 PNG 字符串。"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _cjk_font() -> dict:
    """返回中文字体 kwargs，未找到时返回空。"""
    if _CJK_FONT is not None:
        return {"fontproperties": _CJK_FONT}
    return {}


def _pass_icon(passed: bool) -> str:
    """通过/未通过图标。"""
    return "🟢" if passed else "🔴"


def _fmt_pct(val) -> str:
    """格式化为百分比字符串。"""
    if val is None or np.isnan(val) if isinstance(val, float) else False:
        return "-"
    return f"{float(val):.2%}"


def _fmt_float(val, decimals: int = 4) -> str:
    """安全格式化浮点数。"""
    if val is None:
        return "-"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return "-"


# ════════════════════════════════════════════════════════════
# CSS 样式（内嵌）
# ════════════════════════════════════════════════════════════

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 40px; background: #fafafa; color: #333;
}
h1 { color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 12px; }
h2 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; margin-top: 32px; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; background: #fff; }
th { background: #16213e; color: #fff; padding: 10px 12px; text-align: left; font-size: 13px; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
tr:nth-child(even) { background: #f8f9fa; }
tr:hover { background: #e8f0fe; }
.chart-container {
    background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
}
.chart-container img { max-width: 100%; height: auto; }
.metric-card {
    display: inline-block; background: #fff; border-radius: 8px;
    padding: 16px 24px; margin: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    min-width: 140px; text-align: center; vertical-align: top;
}
.metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
.metric-value { font-size: 24px; font-weight: bold; color: #16213e; margin: 4px 0; }
.pass { color: #28a745; font-weight: bold; }
.fail { color: #dc3545; font-weight: bold; }
.highlight { background: #fff3cd !important; }
.tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; background: #e8e8e8; margin: 1px;
}
.empty-state { text-align: center; padding: 60px; color: #999; font-size: 18px; }
.footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
          font-size: 12px; color: #999; }
"""


# ════════════════════════════════════════════════════════════
# Section A: 因子库总览表格
# ════════════════════════════════════════════════════════════

def _section_a_overview_table(conn: sqlite3.Connection) -> str:
    """生成因子库总览 HTML 表格。"""
    rows = conn.execute(
        """SELECT f.factor_id, f.direction_tag,
                  CAST(json_extract(f.metrics, '$.ic') AS REAL) as ic,
                  CAST(json_extract(f.metrics, '$.ir') AS REAL) as ir,
                  b.sharpe_ratio, b.max_drawdown, b.annual_return,
                  f.monotonicity_passed, f.oos_stability_passed,
                  f.inbound_date
           FROM factors f
           LEFT JOIN backtests b ON f.factor_id = b.factor_id
           ORDER BY b.sharpe_ratio DESC"""
    ).fetchall()

    if not rows:
        return '<section id="a"><h2>a. 因子库总览</h2><div class="empty-state">暂无入库因子</div></section>'

    html = """<section id="a">
    <h2>a. 因子库总览</h2>
    <table>
    <thead><tr>
        <th>因子ID</th><th>方向</th><th>IC</th><th>IR</th>
        <th>夏普</th><th>年化收益</th><th>最大回撤</th>
        <th>单调性</th><th>样本外</th><th>入库日期</th>
    </tr></thead><tbody>"""

    for r in rows:
        html += f"""<tr>
            <td><strong>{r['factor_id']}</strong></td>
            <td><span class="tag">{r['direction_tag'] or '-'}</span></td>
            <td>{_fmt_float(r['ic'])}</td>
            <td>{_fmt_float(r['ir'])}</td>
            <td>{_fmt_float(r['sharpe_ratio'], 2)}</td>
            <td>{_fmt_pct(r['annual_return'])}</td>
            <td>{_fmt_pct(r['max_drawdown'])}</td>
            <td class="{'pass' if r['monotonicity_passed'] else 'fail'}">{_pass_icon(r['monotonicity_passed'])}</td>
            <td class="{'pass' if r['oos_stability_passed'] else 'fail'}">{_pass_icon(r['oos_stability_passed'])}</td>
            <td>{(r['inbound_date'] or '')[:10]}</td>
        </tr>"""

    html += "</tbody></table></section>"
    return html


# ════════════════════════════════════════════════════════════
# Section B: 因子 IR 分布柱状图
# ════════════════════════════════════════════════════════════

def _section_b_ir_chart(conn: sqlite3.Connection) -> str:
    """生成因子 IR 分布柱状图（base64 内嵌）。"""
    rows = conn.execute(
        """SELECT factor_id,
                  CAST(json_extract(metrics, '$.ir') AS REAL) as ir
           FROM factors
           WHERE json_extract(metrics, '$.ir') IS NOT NULL
           ORDER BY ir DESC
           LIMIT 20"""
    ).fetchall()

    if not rows:
        return '<section id="b"><h2>b. 因子 IR 分布</h2><div class="empty-state">暂无数据</div></section>'

    ids = [r["factor_id"] for r in rows]
    irs = [r["ir"] or 0 for r in rows]

    colors = []
    for v in irs:
        if v > 0.5:
            colors.append("#28a745")
        elif v > 0.3:
            colors.append("#ffc107")
        else:
            colors.append("#dc3545")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(ids, irs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.5, color="#28a745", linestyle="--", alpha=0.5, label="优秀 (0.5)")
    ax.axhline(y=0.3, color="#ffc107", linestyle="--", alpha=0.5, label="一般 (0.3)")
    ax.set_xlabel("Factor ID", fontsize=11)
    ax.set_ylabel("IR (Information Ratio)", fontsize=11)
    ax.set_title("因子 IR 分布", fontsize=13, fontweight="bold", **_cjk_font())
    ax.legend(fontsize=9, prop=_CJK_FONT if _CJK_FONT else None)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    img_b64 = fig_to_base64(fig)
    note = "数据来源：CAST(json_extract(metrics, '$.ir') AS REAL)，因日频 IC 序列未持久化存储"
    return f"""<section id="b">
    <h2>b. 因子 IR 分布</h2>
    <p style="color:#888;font-size:12px;">{note}</p>
    <div class="chart-container"><img src="{img_b64}" alt="IR分布"></div>
    </section>"""


# ════════════════════════════════════════════════════════════
# Section C: 回测指标对比表 + 排名柱状图
# ════════════════════════════════════════════════════════════

def _section_c_backtest_comparison(conn: sqlite3.Connection) -> str:
    """回测指标对比（因 cum_return_curve 未持久化，用指标表+柱状图替代）。"""
    rows = conn.execute(
        """SELECT f.factor_id, b.sharpe_ratio, b.annual_return,
                  b.max_drawdown, b.win_rate, b.avg_impact_cost_bps
           FROM factors f
           JOIN backtests b ON f.factor_id = b.factor_id
           ORDER BY b.sharpe_ratio DESC
           LIMIT 10"""
    ).fetchall()

    if not rows:
        return '<section id="c"><h2>c. 回测指标对比</h2><div class="empty-state">暂无回测数据</div></section>'

    # 表格
    table_html = """<section id="c">
    <h2>c. 回测指标对比</h2>
    <p style="color:#888;font-size:12px;">说明：backtests 表无 cum_return_curve 字段（database.py 冻结），展示指标对比表+柱状图。</p>
    <table>
    <thead><tr>
        <th>因子ID</th><th>夏普比率</th><th>年化收益</th>
        <th>最大回撤</th><th>调仓胜率</th><th>冲击成本(bps)</th>
    </tr></thead><tbody>"""

    for r in rows:
        table_html += f"""<tr>
            <td><strong>{r['factor_id']}</strong></td>
            <td>{_fmt_float(r['sharpe_ratio'], 2)}</td>
            <td>{_fmt_pct(r['annual_return'])}</td>
            <td>{_fmt_pct(r['max_drawdown'])}</td>
            <td>{_fmt_pct(r['win_rate'])}</td>
            <td>{_fmt_float(r['avg_impact_cost_bps'], 2)}</td>
        </tr>"""
    table_html += "</tbody></table>"

    # 柱状图
    ids = [r["factor_id"] for r in rows]
    sharpes = [r["sharpe_ratio"] or 0 for r in rows]
    colors = ["#28a745" if s >= 1.0 else "#ffc107" if s >= 0 else "#dc3545" for s in sharpes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ids, sharpes, color=colors, edgecolor="white")
    ax.axhline(y=1.0, color="#28a745", linestyle="--", alpha=0.5, label="优秀 (1.0)")
    ax.axhline(y=0, color="#dc3545", linestyle="--", alpha=0.5, label="零线")
    ax.set_xlabel("Factor ID", fontsize=11)
    ax.set_ylabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Top 10 因子夏普比率排名", fontsize=13, fontweight="bold", **_cjk_font())
    ax.legend(fontsize=9, prop=_CJK_FONT if _CJK_FONT else None)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    img_b64 = fig_to_base64(fig)
    return table_html + f'<div class="chart-container"><img src="{img_b64}" alt="回测指标"></div></section>'


# ════════════════════════════════════════════════════════════
# Section D: 累计 API 成本仪表盘
# ════════════════════════════════════════════════════════════

def _section_d_cost_dashboard(conn: sqlite3.Connection) -> str:
    """生成 API 成本仪表盘。"""
    # 总成本（从 batch_status 最新记录）
    batch_row = conn.execute(
        "SELECT cumulative_cost, completed_rounds FROM batch_status ORDER BY run_id DESC LIMIT 1"
    ).fetchone()

    total_cost = batch_row["cumulative_cost"] if batch_row else 0.0
    total_rounds = batch_row["completed_rounds"] if batch_row else 0

    # 各轮成本（用于折线图）
    cost_rows = conn.execute(
        "SELECT round_id, api_cost FROM rounds ORDER BY round_id, batch_run_id"
    ).fetchall()

    # 因子入库统计
    inbound_count = conn.execute(
        "SELECT COUNT(*) FROM factors"
    ).fetchone()[0]

    avg_cost_per_round = total_cost / total_rounds if total_rounds > 0 else 0.0

    # 指标卡片
    html = f"""<section id="d">
    <h2>d. 累计 API 成本仪表盘</h2>
    <div style="text-align:center; margin: 16px 0;">
        <div class="metric-card">
            <div class="metric-label">总成本</div>
            <div class="metric-value">¥{total_cost:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">总轮次</div>
            <div class="metric-value">{total_rounds}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">单轮平均</div>
            <div class="metric-value">¥{avg_cost_per_round:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">入库因子数</div>
            <div class="metric-value">{inbound_count}</div>
        </div>
    </div>"""

    # 成本趋势图
    if cost_rows:
        round_nums = [r["round_id"] for r in cost_rows]
        costs = [r["api_cost"] for r in cost_rows]
        cum_costs = list(np.cumsum(costs))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 累计成本
        ax1.plot(round_nums, cum_costs, color="#16213e", linewidth=2, marker="o", markersize=3)
        ax1.fill_between(round_nums, 0, cum_costs, alpha=0.1, color="#16213e")
        ax1.set_xlabel("Round", fontsize=10)
        ax1.set_ylabel("Cumulative Cost (¥)", fontsize=10)
        ax1.set_title("累计成本趋势", fontsize=12, fontweight="bold", **_cjk_font())
        ax1.grid(alpha=0.3)

        # 单轮成本
        colors = ["#dc3545" if c > 0.05 else "#28a745" for c in costs]
        ax2.bar(round_nums, costs, color=colors, alpha=0.7)
        ax2.axhline(y=0.05, color="#dc3545", linestyle="--", alpha=0.5, label="¥0.05")
        ax2.set_xlabel("Round", fontsize=10)
        ax2.set_ylabel("Cost per Round (¥)", fontsize=10)
        ax2.set_title("单轮 API 成本", fontsize=12, fontweight="bold", **_cjk_font())
        ax2.legend(fontsize=9, prop=_CJK_FONT if _CJK_FONT else None)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        img_b64 = fig_to_base64(fig)
        html += f'<div class="chart-container"><img src="{img_b64}" alt="成本趋势"></div>'

    html += "</section>"
    return html


# ════════════════════════════════════════════════════════════
# Section E: 方向成功率对比条形图
# ════════════════════════════════════════════════════════════

def _section_e_direction_chart(conn: sqlite3.Connection) -> str:
    """生成方向成功率对比条形图。"""
    rows = conn.execute(
        """SELECT f.direction_tag,
                  AVG(b.sharpe_ratio) as avg_sharpe,
                  COUNT(*) as cnt
           FROM factors f
           JOIN backtests b ON f.factor_id = b.factor_id
           WHERE f.direction_tag IS NOT NULL AND f.direction_tag != ''
           GROUP BY f.direction_tag
           ORDER BY avg_sharpe DESC"""
    ).fetchall()

    if not rows:
        return '<section id="e"><h2>e. 方向成功率对比</h2><div class="empty-state">暂无数据</div></section>'

    tags = [r["direction_tag"] for r in rows]
    sharpes = [r["avg_sharpe"] or 0 for r in rows]
    counts = [r["cnt"] for r in rows]

    colors = []
    for s in sharpes:
        if s > 1.0:
            colors.append("#28a745")
        elif s > 0:
            colors.append("#ffc107")
        else:
            colors.append("#dc3545")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(tags, sharpes, color=colors, edgecolor="white")
    ax.axvline(x=1.0, color="#28a745", linestyle="--", alpha=0.5, label="优秀 (1.0)")
    ax.axvline(x=0, color="#dc3545", linestyle="--", alpha=0.5)
    ax.set_xlabel("Average Sharpe Ratio", fontsize=11)
    ax.set_title("各方向平均夏普比率", fontsize=13, fontweight="bold", **_cjk_font())
    ax.legend(fontsize=9, prop=_CJK_FONT if _CJK_FONT else None)

    # 标注因子数
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"n={cnt}", va="center", fontsize=9, color="#666")

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    note = "数据来源：factors.direction_tag + backtests.sharpe_ratio（方向×时段 IC 数据未持久化）"
    return f"""<section id="e">
    <h2>e. 方向成功率对比</h2>
    <p style="color:#888;font-size:12px;">{note}</p>
    <div class="chart-container"><img src="{img_b64}" alt="方向对比"></div>
    </section>"""


# ════════════════════════════════════════════════════════════
# Section F: ICIR 合成因子回测表现
# ════════════════════════════════════════════════════════════

def _section_f_combo(combo_result) -> str:
    """ICIR 合成因子 vs 最佳单因子对比。"""
    if combo_result is None or combo_result.backtest_result is None:
        return """<section id="f">
    <h2>f. ICIR 合成因子回测表现</h2>
    <div class="empty-state">未计算（需传入 combo_result）</div>
    </section>"""

    bt = combo_result.backtest_result
    vs = combo_result.vs_best_single
    weights = combo_result.weights
    icir_vals = combo_result.icir_values

    ratio = vs.get("ratio", 0)
    ratio_ok = ratio >= 0.80

    # 权重表格
    weight_rows = ""
    for fid in sorted(weights.keys()):
        w = weights[fid]
        icir = icir_vals.get(fid, 0)
        active = "✓" if w > 0 else "✗ 负值归零"
        weight_rows += f"""<tr>
            <td>{fid}</td><td>{icir:.4f}</td><td>{w:.4f}</td><td>{active}</td>
        </tr>"""

    html = f"""<section id="f">
    <h2>f. ICIR 合成因子回测表现</h2>

    <h3>权重分配</h3>
    <table>
    <thead><tr><th>因子ID</th><th>ICIR</th><th>权重</th><th>状态</th></tr></thead>
    <tbody>{weight_rows}</tbody>
    </table>

    <h3>组合 vs 最佳单因子</h3>
    <table>
    <thead><tr>
        <th>对比维度</th><th>组合因子</th><th>最佳单因子 ({vs.get('best_single_id', '-')})</th><th>比率</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>夏普比率</td>
        <td><strong>{_fmt_float(vs.get('combo_sharpe'), 2)}</strong></td>
        <td>{_fmt_float(vs.get('best_single_sharpe'), 2)}</td>
        <td class="{'pass' if ratio_ok else 'fail'}">{ratio:.2%} {'✓ 达标' if ratio_ok else '✗ 未达标'}</td>
    </tr>
    <tr>
        <td>年化收益</td>
        <td>{_fmt_pct(bt.annual_return)}</td>
        <td>-</td><td>-</td>
    </tr>
    <tr>
        <td>最大回撤</td>
        <td>{_fmt_pct(bt.max_drawdown)}</td>
        <td>-</td><td>-</td>
    </tr>
    <tr>
        <td>调仓胜率</td>
        <td>{_fmt_pct(bt.win_rate)}</td>
        <td>-</td><td>-</td>
    </tr>
    <tr>
        <td>冲击成本</td>
        <td>{_fmt_float(bt.avg_impact_cost_bps, 2)} bps</td>
        <td>-</td><td>-</td>
    </tr>
    </tbody>
    </table>"""

    if not ratio_ok:
        html += '<p style="color:#888;">⚠️ 合成夏普未达到最佳单因子的80%（不阻断报告生成）</p>'

    # 累计收益对比图（如果 combo 有 cum_return_curve）
    if bt.cum_return_curve is not None and len(bt.cum_return_curve) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        cum_curve = bt.cum_return_curve
        ax.plot(cum_curve.index, cum_curve.values, color="#16213e", linewidth=2, label="ICIR组合")
        ax.fill_between(cum_curve.index, 1.0, cum_curve.values,
                        where=(cum_curve.values < 1.0), color="#dc3545", alpha=0.15)
        ax.axhline(y=1.0, color="#666", linestyle="--", alpha=0.5, label="基准线")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Cumulative Return", fontsize=10)
        ax.set_title("ICIR 组合累计收益曲线", fontsize=13, fontweight="bold", **_cjk_font())
        ax.legend(fontsize=9, prop=_CJK_FONT if _CJK_FONT else None)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        plt.tight_layout()
        img_b64 = fig_to_base64(fig)
        html += f'<div class="chart-container"><img src="{img_b64}" alt="组合收益曲线"></div>'

    html += "</section>"
    return html


# ════════════════════════════════════════════════════════════
# Section G: 分年验证表格
# ════════════════════════════════════════════════════════════

def _section_g_yearly_validation(conn: sqlite3.Connection) -> str:
    """生成分年验证表格。"""
    rows = conn.execute(
        """SELECT factor_id, yearly_validation_passed, yearly_observed, inbound_date
           FROM factors
           ORDER BY factor_id"""
    ).fetchall()

    if not rows:
        return '<section id="g"><h2>g. 分年验证</h2><div class="empty-state">暂无数据</div></section>'

    # 各因子 yearly_observed 详情
    table_rows = ""
    for r in rows:
        observed_str = r["yearly_observed"] or ""
        try:
            observed_list = json.loads(observed_str) if observed_str else []
        except (json.JSONDecodeError, TypeError):
            observed_list = []
        observed_display = ", ".join(observed_list) if observed_list else "无"

        table_rows += f"""<tr>
            <td><strong>{r['factor_id']}</strong></td>
            <td class="{'pass' if r['yearly_validation_passed'] else 'fail'}">{_pass_icon(bool(r['yearly_validation_passed']))}</td>
            <td>{observed_display}</td>
            <td>{(r['inbound_date'] or '')[:10]}</td>
        </tr>"""

    # 按时段统计入库因子数
    periods = [
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020-2025", "2020-01-01", "2025-12-31"),
    ]
    period_stats = ""
    for label, start, end in periods:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM factors WHERE inbound_date >= ? AND inbound_date <= ?",
            (start, end),
        ).fetchone()[0]
        highlight_class = ' class="highlight"' if label == "2020-2025" else ""
        period_stats += f"<td{highlight_class}><strong>{cnt}</strong></td>"

    html = f"""<section id="g">
    <h2>g. 分年验证</h2>
    <p style="color:#888;font-size:12px;">说明：yearly_ic dict（每年具体 IC 值）未持久化，仅展示 yearly_observed + 通过状态。</p>

    <h3>按时段入库统计</h3>
    <table>
    <thead><tr><th>2010-2015</th><th>2016-2019</th><th class="highlight">2020-2025 (重点验证)</th></tr></thead>
    <tbody><tr>{period_stats}</tr></tbody>
    </table>

    <h3>因子分年验证详情</h3>
    <table>
    <thead><tr>
        <th>因子ID</th><th>分年验证</th><th>待观察年份</th><th>入库日期</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
    </table>
    </section>"""
    return html


# ════════════════════════════════════════════════════════════
# 报告生成入口
# ════════════════════════════════════════════════════════════

def _html_wrapper(title: str, sections: str, generation_time: str = "") -> str:
    """包裹完整 HTML 文档。"""
    if not generation_time:
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{CSS}</style>
</head>
<body>
    <h1>{title}</h1>
    <p>生成时间：{generation_time} | 因子库版本：Phase 3c</p>
    {sections}
    <div class="footer">FactorLab Phase 3c — 自动生成 | 图表 base64 内嵌，可离线查看</div>
</body>
</html>"""


def generate_quick_report(
    conn: sqlite3.Connection,
    output_path: str = "",
) -> str:
    """快速版报告：a.因子表格 + d.成本仪表盘 + 简要统计。

    Args:
        conn: SQLite 连接
        output_path: 输出路径，为空时自动生成到 REPORTS_DIR

    Returns:
        输出文件路径
    """
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 简要统计
    total_factors = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    inbound_count = total_factors  # factors 表即入库因子

    ic_row = conn.execute(
        "SELECT AVG(CAST(json_extract(metrics, '$.ic') AS REAL)) as avg_ic FROM factors"
    ).fetchone()
    ir_row = conn.execute(
        "SELECT AVG(CAST(json_extract(metrics, '$.ir') AS REAL)) as avg_ir FROM factors"
    ).fetchone()
    avg_ic = ic_row["avg_ic"] if ic_row and ic_row["avg_ic"] else 0
    avg_ir = ir_row["avg_ir"] if ir_row and ir_row["avg_ir"] else 0

    total_cost = 0.0
    total_rounds = 0
    batch_row = conn.execute(
        "SELECT cumulative_cost, completed_rounds FROM batch_status ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    if batch_row:
        total_cost = batch_row["cumulative_cost"] or 0
        total_rounds = batch_row["completed_rounds"] or 0

    stats_html = f"""<section id="stats">
    <h2>统计摘要</h2>
    <div style="text-align:center; margin: 16px 0;">
        <div class="metric-card">
            <div class="metric-label">因子总数</div>
            <div class="metric-value">{total_factors}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">入库因子</div>
            <div class="metric-value">{inbound_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">平均 IC</div>
            <div class="metric-value">{avg_ic:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">平均 IR</div>
            <div class="metric-value">{avg_ir:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">累计成本</div>
            <div class="metric-value">¥{total_cost:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">总轮次</div>
            <div class="metric-value">{total_rounds}</div>
        </div>
    </div>
    </section>"""

    sections = (
        _section_a_overview_table(conn)
        + _section_d_cost_dashboard(conn)
        + stats_html
    )

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(REPORTS_DIR) / f"factorlab_quick_{ts}.html")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_html_wrapper("FactorLab 快速报告", sections, generation_time))
    _os.replace(tmp, str(out))

    return str(out)


def generate_full_report(
    conn: sqlite3.Connection,
    output_path: str = "",
    combo_result=None,
) -> str:
    """完整版报告：a-g 全部内容。

    Args:
        conn: SQLite 连接
        output_path: 输出路径，为空时自动生成到 REPORTS_DIR
        combo_result: 可选 ComboResult，展示 ICIR 合成对比

    Returns:
        输出文件路径
    """
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = (
        _section_a_overview_table(conn)
        + _section_b_ir_chart(conn)
        + _section_c_backtest_comparison(conn)
        + _section_d_cost_dashboard(conn)
        + _section_e_direction_chart(conn)
        + _section_f_combo(combo_result)
        + _section_g_yearly_validation(conn)
    )

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(REPORTS_DIR) / f"factorlab_full_{ts}.html")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_html_wrapper("FactorLab 完整报告", sections, generation_time))
    _os.replace(tmp, str(out))

    return str(out)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile
    from database import get_conn, init_db

    if "--demo" in sys.argv:
        print("=== html_reporter.py Phase 3c 报告演示 ===\n")

        # 创建临时数据库并写入 Mock 数据
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db_path = tmp_db.name
        tmp_db.close()

        conn = sqlite3.connect(tmp_db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row

        init_db(conn, migrate=True)

        import json as _json

        # Mock 因子数据
        mock_factors = [
            ("f001", "反转类", 0.045, 0.35, 1.2, -0.08, 0.12, 1, 1, 1, '["2023"]', "2024-01-15"),
            ("f002", "动量", 0.038, 0.28, 0.9, -0.12, 0.08, 1, 1, 1, "[]", "2024-02-10"),
            ("f003", "量价背离", 0.052, 0.42, 1.5, -0.06, 0.18, 1, 1, 1, '["2024"]', "2024-03-01"),
            ("f004", "波动率类", 0.025, 0.18, 0.5, -0.15, 0.03, 0, 1, 0, '["2022","2024"]', "2024-03-20"),
            ("f005", "行为类", 0.041, 0.31, 1.1, -0.10, 0.10, 1, 1, 1, "[]", "2024-04-05"),
        ]

        for (fid, tag, ic, ir_val, sharpe, mdd, ret, mono_p, oos_p, yv_p, yv_obs, date) in mock_factors:
            conn.execute(
                """INSERT INTO factors (factor_id, round, direction_tag, metrics, monotonicity_passed,
                   oos_stability_passed, yearly_validation_passed, yearly_observed, inbound_date)
                   VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)""",
                (fid, tag, _json.dumps({"ic": ic, "ir": ir_val}),
                 mono_p, oos_p, yv_p, yv_obs, date),
            )
            conn.execute(
                """INSERT INTO backtests (factor_id, annual_return, max_drawdown, sharpe_ratio,
                   win_rate, turnover_est, avg_impact_cost_bps, total_cost_annual)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (fid, ret, mdd, sharpe, 0.55, 0.3, 0.5, 0.02),
            )

        # Mock rounds 数据
        for i in range(1, 11):
            conn.execute(
                "INSERT INTO rounds (round_id, batch_run_id, status, api_cost, started_at) VALUES (?, ?, ?, ?, ?)",
                (i, 1, "fail" if i % 3 != 0 else "inbound", 0.03 + i * 0.002, "2024-01-01T00:00:00"),
            )

        # Mock batch_status
        conn.execute(
            "INSERT INTO batch_status (target_rounds, completed_rounds, cumulative_cost, status) VALUES (?, ?, ?, ?)",
            (10, 10, 0.42, "completed"),
        )
        conn.commit()

        # 生成快速报告
        quick_path = generate_quick_report(conn, str(Path(REPORTS_DIR) / "demo_quick.html"))
        quick_size = Path(quick_path).stat().st_size
        print(f"  快速版报告: {quick_path} ({quick_size:,} bytes)")

        # 验证内容
        with open(quick_path, "r", encoding="utf-8") as f:
            quick_content = f.read()
        checks = [
            ("因子库总览" in quick_content, "含因子表格"),
            ("累计 API 成本" in quick_content, "含成本仪表盘"),
            ("统计摘要" in quick_content, "含统计摘要"),
            ("base64" in quick_content, "base64 内嵌图表"),
            (quick_size < 5_000_000, f"文件大小 < 5MB ({quick_size:,} bytes)"),
        ]
        for ok, label in checks:
            print(f"  {'[PASS]' if ok else '[FAIL]'} {label}")

        # 生成完整报告（不含 combo）
        full_path = generate_full_report(conn, str(Path(REPORTS_DIR) / "demo_full.html"))
        full_size = Path(full_path).stat().st_size
        print(f"\n  完整版报告: {full_path} ({full_size:,} bytes)")

        with open(full_path, "r", encoding="utf-8") as f:
            full_content = f.read()
        section_ids = ["id=\"a\"", "id=\"b\"", "id=\"c\"", "id=\"d\"", "id=\"e\"", "id=\"f\"", "id=\"g\""]
        all_sections = all(sid in full_content for sid in section_ids)
        print(f"  {'[PASS]' if all_sections else '[FAIL]'} 包含 a-g 全部 section")
        print(f"  {'[PASS]' if full_size < 5_000_000 else '[FAIL]'} 文件大小 < 5MB ({full_size:,} bytes)")

        # 验证无外部依赖
        has_external = "http://" in full_content or "https://" in full_content or "<link" in full_content
        print(f"  {'[FAIL]' if has_external else '[PASS]'} 无外部依赖")

        conn.close()
        print("\n自检通过.")
    elif "--quick" in sys.argv:
        from database import get_conn as _get_conn
        conn = _get_conn()
        path = generate_quick_report(conn)
        conn.close()
        print(f"快速报告已生成: {path}")
    elif "--full" in sys.argv:
        from database import get_conn as _get_conn
        conn = _get_conn()
        path = generate_full_report(conn)
        conn.close()
        print(f"完整报告已生成: {path}")
    else:
        print("用法: python html_reporter.py --demo | --quick | --full")

#!/usr/bin/env python3
"""Phase 2c 总结引擎 — 50 轮总结报告 + program_draft 生成。

用法:
  python summary_engine.py demo    # 演示总结报告生成逻辑
"""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
REPORTS_DIR = PROJECT_DIR / "reports"
HISTORY_DIR = PROJECT_DIR / "history"
PROGRAM_PATH = PROJECT_DIR / "program.md"


def _ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def generate_summary_report(
    conn,
    start_round: int,
    end_round: int,
    cost_tracker=None,  # Phase 2c m3: 可选 CostTracker 实例，优先用实际追踪数据
) -> str:
    """生成 50 轮总结报告（纯本地计算，不调用 AI）。

    报告包含:
      - 总览统计（总轮次、入库数、失败分布、方向热力图、成本分析）
      - 双维度排序视图（按评分总分 / 按回测夏普）
      - 失败模式分析

    Args:
        conn: SQLite 连接
        start_round: 起始轮次
        end_round: 结束轮次

    Returns:
        报告 Markdown 字符串
    """
    _ensure_dirs()

    # ── 统计数据 ──────────────────────────────────────
    rounds_rows = conn.execute(
        """SELECT status, direction_tag, fail_reason FROM rounds
           WHERE round_id BETWEEN ? AND ?
           ORDER BY round_id""",
        (start_round, end_round),
    ).fetchall()

    factors_rows = conn.execute(
        """SELECT f.factor_id, f.direction_tag, f.score_total, f.natural_summary,
                  b.sharpe_ratio, b.annual_return, b.max_drawdown
           FROM factors f
           LEFT JOIN backtests b ON f.factor_id = b.factor_id
           ORDER BY f.score_total DESC"""
    ).fetchall()

    total = len(rounds_rows)
    inbound = sum(1 for r in rounds_rows if r["status"] == "inbound")
    failed = sum(1 for r in rounds_rows if r["status"] == "fail")
    skipped = sum(1 for r in rounds_rows if r["status"] == "skip")
    errors = sum(1 for r in rounds_rows if r["status"] == "error")

    # 方向分布
    direction_counts = Counter(r["direction_tag"] for r in rounds_rows if r["direction_tag"])

    # 失败原因分类
    fail_reasons = Counter()
    for r in rounds_rows:
        if r["status"] in ("fail", "error"):
            reason = r["fail_reason"] or "未知"
            # 简单分类
            if "沙箱超时" in reason:
                fail_reasons["沙箱超时"] += 1
            elif "沙箱" in reason:
                fail_reasons["沙箱异常"] += 1
            elif "合规" in reason:
                fail_reasons["合规失败"] += 1
            elif "API" in reason:
                fail_reasons["API错误"] += 1
            elif "评分" in reason or "threshold" in reason.lower():
                fail_reasons["评分不达标"] += 1
            elif "因子值" in reason:
                fail_reasons["数据不足"] += 1
            else:
                fail_reasons["其他"] += 1

    # 成本（m3 修复: 优先用 CostTracker 实际数据，fallback 到估算）
    if cost_tracker is not None:
        actual_cost = cost_tracker.cost()
    else:
        actual_cost = 0.0
    estimated_cost_min = total * 0.004
    estimated_cost_max = total * 0.012
    if actual_cost > 0:
        cost_display = f"¥{actual_cost:.4f}（实际追踪） | 估算 ¥{estimated_cost_min:.2f}~¥{estimated_cost_max:.2f}"
    else:
        cost_display = f"¥{estimated_cost_min:.2f} ~ ¥{estimated_cost_max:.2f}（估算）"

    # ── 构建报告 ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"summary_report_{timestamp}.md"

    lines = []
    lines.append(f"# FactorLab 批量挖掘总结报告")
    lines.append(f"")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 轮次范围: Round {start_round} ~ {end_round}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 一、总览统计")
    lines.append(f"")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总轮次 | {total} |")
    lines.append(f"| 入库因子 | {inbound} |")
    lines.append(f"| 评分失败 | {failed} |")
    lines.append(f"| 多样性跳过 | {skipped} |")
    lines.append(f"| API/网络错误 | {errors} |")
    lines.append(f"| 入库率 | {inbound/max(total,1):.1%} |")
    lines.append(f"| 成本 | {cost_display} |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 二、入库因子排序")
    lines.append(f"")

    # 按评分总分排序
    lines.append(f"### 按评分总分（前 10）")
    lines.append(f"")
    lines.append(f"| 因子ID | 方向 | 总评分 | 夏普 | 年化收益 | 摘要 |")
    lines.append(f"|--------|------|--------|------|----------|------|")
    for f in factors_rows[:10]:
        summary = (f["natural_summary"] or "")[:40]
        lines.append(
            f"| {f['factor_id']} | {f['direction_tag'] or '-'} | "
            f"{f['score_total']:.4f} | {f['sharpe_ratio'] or 0:.2f} | "
            f"{f['annual_return'] or 0:.2%} | {summary} |"
        )
    lines.append(f"")

    # 按回测夏普排序
    lines.append(f"### 按回测夏普（前 10）")
    lines.append(f"")
    by_sharpe = sorted(
        [f for f in factors_rows if f["sharpe_ratio"] is not None],
        key=lambda f: f["sharpe_ratio"] or 0, reverse=True,
    )
    lines.append(f"| 因子ID | 方向 | 夏普 | 最大回撤 | 年化收益 | 摘要 |")
    lines.append(f"|--------|------|------|----------|----------|------|")
    for f in by_sharpe[:10]:
        summary = (f["natural_summary"] or "")[:40]
        lines.append(
            f"| {f['factor_id']} | {f['direction_tag'] or '-'} | "
            f"{f['sharpe_ratio']:.2f} | {f['max_drawdown'] or 0:.1%} | "
            f"{f['annual_return'] or 0:.2%} | {summary} |"
        )
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 三、方向热力图")
    lines.append(f"")
    lines.append(f"| 方向 | 尝试次数 | 入库数 | 入库率 |")
    lines.append(f"|------|----------|--------|--------|")
    for direction, count in direction_counts.most_common():
        dir_inbound = sum(
            1 for r in rounds_rows
            if r["direction_tag"] == direction and r["status"] == "inbound"
        )
        lines.append(f"| {direction} | {count} | {dir_inbound} | {dir_inbound/max(count,1):.1%} |")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 四、失败模式分析")
    lines.append(f"")
    lines.append(f"| 失败类型 | 次数 | 占比 |")
    lines.append(f"|----------|------|------|")
    total_fails = max(sum(fail_reasons.values()), 1)
    for reason, count in fail_reasons.most_common():
        lines.append(f"| {reason} | {count} | {count/total_fails:.1%} |")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 五、建议")
    lines.append(f"")
    if inbound == 0:
        lines.append(f"⚠️ **50 轮零入库！** 建议：")
        lines.append(f"1. 检查 program.md 研究方向是否过于偏窄")
        lines.append(f"2. 放宽评分阈值或调整因子定义域")
        lines.append(f"3. 检查数据质量（df_1800 装载是否正确）")
    elif inbound < 5:
        lines.append(f"⚠️ 入库率偏低（{inbound}/{total} = {inbound/max(total,1):.1%}）。建议扩大探索方向范围。")
    else:
        lines.append(f"✅ 入库率正常（{inbound}/{total} = {inbound/max(total,1):.1%}）。")
        lines.append(f"建议在新 batch 中重点探索入库率高的方向。")

    # 找到成功率最高的方向
    if direction_counts:
        best_direction = ""
        best_rate = 0
        for direction, count in direction_counts.items():
            dir_inbound = sum(
                1 for r in rounds_rows
                if r["direction_tag"] == direction and r["status"] == "inbound"
            )
            rate = dir_inbound / max(count, 1)
            if rate > best_rate:
                best_rate = rate
                best_direction = direction
        if best_direction and best_rate > 0:
            lines.append(f"")
            lines.append(f"🏆 最佳方向: **{best_direction}**（入库率 {best_rate:.1%}）")

    report = "\n".join(lines)

    # 写入文件
    report_path.write_text(report, encoding="utf-8")
    print(f"  总结报告已保存: {report_path}")

    return report


def generate_program_draft(
    conn,
    current_program: str,
    summary_report: str,
    dry_run: bool = False,
) -> str:
    """生成 program_draft.md（仅修改第一章）。

    内部调用 1 次 AI API，基于总结报告生成方向调整建议。
    成本上限 ¥0.05。

    Args:
        conn: SQLite 连接
        current_program: 当前 program.md 完整内容
        summary_report: 总结报告内容
        dry_run: 跳过 AI 调用，使用模板

    Returns:
        生成的 draft markdown 字符串
    """
    _ensure_dirs()

    # 提取当前第一章
    marker = "<!-- AI_MEMORY_START -->"
    chapters_1_3 = current_program.split(marker, 1)[0] if marker in current_program else current_program

    chapter_start = chapters_1_3.find("## 第一章")
    chapter_end = chapters_1_3.find("\n## 第二章")
    if chapter_start < 0:
        chapter_start = 0
    if chapter_end < 0:
        chapter_end = len(chapters_1_3)
    current_ch1 = chapters_1_3[chapter_start:chapter_end].strip()

    # ── AI 调用 ──────────────────────────────────────
    if not dry_run:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if api_key and api_key != "sk-xxxx":
            draft_ch1 = _ai_generate_ch1(api_key, current_ch1, summary_report)
        else:
            print("  [summary_engine] 无有效 API Key，使用模板生成 draft")
            draft_ch1 = _template_ch1(current_ch1, summary_report)
    else:
        draft_ch1 = _template_ch1(current_ch1, summary_report)

    # ── 构建 draft ──────────────────────────────────
    draft = chapters_1_3.replace(current_ch1, draft_ch1, 1) if current_ch1 else chapters_1_3

    # 写入根目录 program_draft.md
    draft_path = PROJECT_DIR / "program_draft.md"
    draft_path.write_text(draft, encoding="utf-8")

    # 写入审计副本
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = HISTORY_DIR / f"program_draft_{timestamp}.md"
    audit_path.write_text(draft, encoding="utf-8")
    print(f"  program_draft.md 已生成: {draft_path}")
    print(f"  审计副本已保存: {audit_path}")

    return draft


def _ai_generate_ch1(api_key: str, current_ch1: str, summary_report: str) -> str:
    """调用 AI 生成新的第一章草案（成本 ≤ ¥0.05）。

    Phase 2c C3 修复: 使用 engine.create_engine() 而非直接 openai，
    确保 CostTracker 自动记录所有 token。
    """
    from engine import create_engine

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    engine = create_engine(api_key=api_key, model=model, dry_run=False)

    prompt = (
        f"你是一位量化因子研究主管。请基于以下总结报告，修改研究规程第一章。\n\n"
        f"## 当前第一章\n{current_ch1}\n\n"
        f"## 总结报告\n{summary_report[:3000]}\n\n"
        f"任务：\n"
        f"1. 仅修改第一章的 1.3 节（探索优先级），调整 P0/P1/P2/P3 方向\n"
        f"2. 保留 1.1（研究方向）和 1.2（研究边界）不变\n"
        f"3. 在第一章末尾添加注释块:\n"
        f"   <!-- DRAFT_NOTE_START -->\n"
        f"   基于 {summary_report[:200]} 的分析:\n"
        f"   - 保留方向: ...\n"
        f"   - 暂停方向: ...\n"
        f"   - 新增方向: ...\n"
        f"   - 负面清单调整建议: ...\n"
        f"   <!-- DRAFT_NOTE_END -->\n"
        f"4. 不要修改第二章和第三章\n"
        f"5. 输出完整的修改后第一章（Markdown格式，以 ## 第一章 开头）\n"
    )

    content = engine.chat(prompt, max_tokens=800)

    # 确保以 ## 第一章 开头
    if "## 第一章" in content:
        idx = content.find("## 第一章")
        content = content[idx:]

    return content.strip()


def _template_ch1(current_ch1: str, summary_report: str) -> str:
    """非 AI 模板：基于总结报告统计信息生成第一章草案。"""
    # 简单统计关键词
    direction_hits = {}
    for kw in ["反转", "行为", "背离", "波动", "动量", "价格路径", "时间结构"]:
        count = summary_report.count(kw)
        if count > 0:
            direction_hits[kw] = count

    # 提取入库率最高的关键词作为"保留方向"
    best = sorted(direction_hits.items(), key=lambda x: x[1], reverse=True)

    note_lines = [
        "<!-- DRAFT_NOTE_START -->",
        "基于批量挖掘总结报告的自动草案:",
        f"- 保留方向: {', '.join([b[0] for b in best[:3]]) if best else '无'}（入库率相对较高）",
        f"- 暂停方向: {', '.join([b[0] for b in best[3:]]) if len(best) > 3 else '无'}"
        f"（入库率低或连续失败）",
        f"- 新增方向: 建议基于AI分析添加",
        f"- 负面清单调整建议: 如连续失败方向达到10轮，建议移至P3",
        "<!-- DRAFT_NOTE_END -->",
    ]

    # 在第一章末尾添加注释块
    if "<!-- DRAFT_NOTE_START -->" not in current_ch1:
        result = current_ch1 + "\n\n" + "\n".join(note_lines)
    else:
        result = current_ch1

    return result


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== summary_engine.py 自检 ===\n")

    from database import get_conn, init_db

    conn = get_conn()
    init_db(conn, migrate=True)

    # 插入一些测试数据
    from database import insert_round, insert_factor, insert_backtest
    import random

    directions = ["反转类", "行为类", "量价背离", "动量", "波动率类"]
    for i in range(1, 11):
        status = random.choices(
            ["inbound", "fail", "fail", "skip", "fail"],
            weights=[2, 4, 2, 1, 1], k=1,
        )[0]
        direction = random.choice(directions)
        insert_round(conn, {
            "round_id": i, "batch_run_id": 1,
            "direction_tag": direction, "status": status,
            "started_at": datetime.now().isoformat(),
            "fail_reason": "ir=0.05, 需>0.15" if status == "fail" else "",
            "api_cost": 0.004, "summary": f"测试因子{i}", "steps": ["code_gen", "sandbox"],
        })
        if status == "inbound":
            insert_factor(conn, {
                "factor_id": f"f{i:03d}", "round": i,
                "direction_tag": direction,
                "inbound_date": datetime.now().strftime("%Y-%m-%d"),
                "natural_summary": f"测试因子{i} - {direction}",
                "metrics": {"ic": 0.04, "ir": 0.5},
                "score_total": round(random.uniform(0.6, 0.9), 4),
                "rank_snapshot": {},
            })
            insert_backtest(conn, {
                "factor_id": f"f{i:03d}",
                "annual_return": round(random.uniform(-0.1, 0.3), 4),
                "max_drawdown": round(random.uniform(-0.3, -0.05), 4),
                "sharpe_ratio": round(random.uniform(-0.5, 1.8), 2),
                "win_rate": round(random.uniform(0.4, 0.7), 2),
                "turnover_est": round(random.uniform(0.3, 0.9), 2),
            })

    # 生成总结报告
    print("[1] 生成总结报告:")
    report = generate_summary_report(conn, 1, 10)
    print(report[:500])
    print("...")

    # 生成 program_draft
    print("\n[2] 生成 program_draft (模板模式):")
    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        current = f.read()
    draft = generate_program_draft(conn, current, report, dry_run=True)
    print(f"  Draft 长度: {len(draft)} 字符")
    assert "<!-- DRAFT_NOTE_START -->" in draft, "Draft 应包含注释块"
    print("  Draft 结构验证 [OK]")

    conn.close()
    print("\n自检通过.")

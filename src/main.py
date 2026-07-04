#!/usr/bin/env python3
"""FactorLab CLI 主入口 — 8 项菜单 + 启动校验 + 批量模式参数解析 + 熔断机制。

Phase 3c 升级:
  - 新增菜单 [6] 生成 HTML 报告（快速版/完整版）
  - 新增菜单 [7] 因子检索（自然语言/结构化过滤）
  - 退出顺延为 [8]

用法:
  python main.py                        # 交互模式
  python main.py --batch 50             # 启动 50 轮全自动
  python main.py --batch 50 --resume    # 续跑
  python main.py --batch 50 --dry-run   # 模拟模式
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # 必须在任何 os.getenv 之前执行，确保 .env 配置生效

from data_fetcher import check_data_integrity, DEMO_STOCKS
from data_fetcher_v2 import check_data as check_data_integrity_v2
# _load_downloaded 已重构为 _load_set，通过 data_fetcher_v2.DOWNLOADED_FILE 读取
from data_fetcher_v2 import _load_set as _load_downloaded_v2_helper
from data_fetcher_v2 import DOWNLOADED_FILE as _DV2_DOWNLOADED_FILE
from engine import CostTracker
from pipeline import run_pipeline, load_factor_pool

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
FACTOR_POOL_PATH = PROJECT_ROOT / "factor_pool.json"


def _check_git() -> bool:
    """检测当前目录是否已初始化 Git 仓库。"""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(PROJECT_DIR),
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _print_header():
    print("\n" + "=" * 50)
    print("  FactorLab Phase 3c — AI 辅助 A 股量化因子挖掘")
    print("=" * 50)


def _print_status(pool_size: int, max_round: int):
    """打印当前状态：因子池数量、轮次、累计成本。"""
    tracker = CostTracker()
    if (PROJECT_DIR / "data" / "downloaded.txt").exists():
        n_stocks = len(_load_downloaded_v2_helper(_DV2_DOWNLOADED_FILE))
    else:
        n_stocks = len(DEMO_STOCKS)
    print(f"\n  因子池: {pool_size} 个 | 已完成轮次: {max_round} | 累计 API 成本: {tracker.cost():.4f}")
    print(f"  数据状态: {n_stocks} 只股票 | 沙箱超时: 30s | 因子行数上限: 20")


def cmd_view_pool():
    """查看因子池（Phase 2c: 支持 SQLite 查询子菜单）。"""
    # 优先从 SQLite 读取
    try:
        from database import get_conn, query_factors, count_factors
        conn = get_conn()
        total = count_factors(conn)
        if total == 0:
            print("\n  (SQLite) 因子库为空，还没有入库的因子。")
            conn.close()
            return

        print(f"\n  (SQLite) 因子库 ({total} 个因子):")
        print(f"  查询子菜单:")
        print(f"    a — 全部（按评分排序，最近 10 个）")
        print(f"    d — 按方向筛选")
        print(f"    i — 按 IC 降序")
        print(f"    s — 按回测夏普降序")

        choice = input("\n  请选择 (a/d/i/s): ").strip().lower()

        sort_by = "score_total"
        direction_tag = ""
        if choice == "i":
            sort_by = "CAST(json_extract(metrics, '$.ic') AS REAL)"
        elif choice == "s":
            rows = conn.execute(
                """SELECT f.*, b.sharpe_ratio FROM factors f
                   LEFT JOIN backtests b ON f.factor_id = b.factor_id
                   ORDER BY b.sharpe_ratio DESC LIMIT 10"""
            ).fetchall()
            print(f"\n  {'ID':6s} {'方向':10s} {'评分':8s} {'夏普':8s} {'摘要'}")
            print(f"  {'-'*70}")
            for r in rows:
                summary = (dict(r).get("natural_summary", "") or "")[:40]
                print(f"  {r['factor_id']:6s} {(r['direction_tag'] or '-'):10s} "
                      f"{r['score_total'] or 0:8.4f} {r['sharpe_ratio'] or 0:8.2f} {summary}")
            conn.close()
            return
        elif choice == "d":
            direction_tag = input("  输入方向标签（如: 反转类、行为类、动量）: ").strip()

        rows = query_factors(conn, sort_by=sort_by, limit=10, direction_tag=direction_tag)
        print(f"\n  {'ID':6s} {'方向':10s} {'评分':8s} {'日期':12s} {'摘要'}")
        print(f"  {'-'*70}")
        for r in rows:
            summary = (r.get("natural_summary", "") or "")[:40]
            print(f"  {r['factor_id']:6s} {(r.get('direction_tag') or '-'):10s} "
                  f"{r.get('score_total', 0):8.4f} "
                  f"{(r.get('inbound_date') or '')[:10]:12s} {summary}")
        conn.close()
    except Exception as e:
        # 降级到 JSON
        print(f"\n  (SQLite 不可用: {e})")
        pool = load_factor_pool()
        factors = pool.get("factors", [])
        if not factors:
            print("\n  因子池为空，还没有入库的因子。")
            return
        print(f"\n  因子池 ({len(factors)} 个因子):")
        print(f"  {'ID':6s} {'轮次':6s} {'日期':12s} {'评分':8s} {'摘要'}")
        print(f"  {'-'*70}")
        for f in factors[-10:]:
            print(f"  {f['factor_id']:6s} {str(f['round']):6s} {f['inbound_date']:12s} "
                  f"{f.get('score_total', 0):8.4f} {f.get('natural_summary', '')[:40]}")


def cmd_view_cost():
    """查看累计 API 成本与轮次统计。"""
    tracker = CostTracker()
    print(f"\n  累计 API 成本:")
    print(f"    Input tokens:  {tracker.input_tokens:,}")
    print(f"    Output tokens: {tracker.output_tokens:,}")
    print(f"    估算总成本:    {tracker.cost():.4f}")
    print(f"    (DeepSeek: ¥1/百万 input, ¥2/百万 output)")

    # Phase 2c: 额外显示 SQLite 轮次统计
    try:
        from database import get_conn, get_latest_batch, get_batch_round_summary
        conn = get_conn()
        latest = get_latest_batch(conn)
        if latest:
            summary = get_batch_round_summary(conn, latest["run_id"])
            print(f"\n  最新批量运行 (run_id={latest['run_id']}):")
            print(f"    目标轮次: {latest['target_rounds']}")
            print(f"    已完成:   {latest['completed_rounds']}")
            print(f"    状态:     {latest['status']}")
            print(f"    入库: {summary['inbound']} | 失败: {summary['fail']} | "
                  f"跳过: {summary['skip']} | Error: {summary['error']}")
        conn.close()
    except Exception:
        pass


def cmd_generate_report():
    """生成 HTML 报告（Phase 3c 新增）— 子菜单选择快速版/完整版。"""
    print(f"\n  {'─'*40}")
    print(f"  HTML 报告生成:")
    print(f"    1 — 生成快速版报告（因子库摘要）")
    print(f"    2 — 生成完整版报告（含全部图表 + ICIR 合成）")
    print(f"    0 — 返回上级")

    try:
        choice = input("\n  请选择 (0-2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if choice not in ("1", "2"):
        return

    try:
        from database import get_conn
        from html_reporter import generate_quick_report, generate_full_report
        from config import REPORTS_DIR

        conn = get_conn()

        if choice == "1":
            print("\n  正在生成快速版报告...")
            path = generate_quick_report(conn)
            print(f"  ✅ 快速报告已保存: {path}")
        elif choice == "2":
            print("\n  正在生成完整版报告（含 ICIR 合成，可能需要数据加载）...")
            # 尝试计算 combo_result
            combo_result = None
            try:
                from combo_engine import build_all_inbound
                from batch_pipeline import load_df_1800

                print("  加载数据矩阵...")
                df_1800 = load_df_1800(max_stocks=0)  # 全部股票
                close_df = df_1800["close"].unstack("code")
                volume_df = df_1800["volume"].unstack("code")
                returns_df = close_df.pct_change(fill_method=None)

                print("  计算 ICIR 组合...")
                combo_result = build_all_inbound(conn, df_1800, close_df, volume_df, returns_df)
                if combo_result.backtest_result:
                    print(f"  ICIR 组合夏普: {combo_result.backtest_result.sharpe_ratio:.4f}")
            except Exception as e:
                print(f"  [提示] ICIR 组合计算跳过: {e}")

            path = generate_full_report(conn, combo_result=combo_result)
            print(f"  ✅ 完整报告已保存: {path}")

        conn.close()
    except Exception as e:
        print(f"  ❌ 报告生成失败: {e}")


def cmd_factor_search():
    """因子检索（Phase 3c 新增）— 子菜单选择检索方式。"""
    try:
        from database import get_conn
        from nl_query import interactive_query

        conn = get_conn()
        try:
            interactive_query(conn)
        finally:
            conn.close()
    except Exception as e:
        print(f"  ❌ 因子检索失败: {e}")


def cmd_adopt_draft():
    """采纳/拒绝 AI 方向建议（program_draft.md）。"""
    draft_path = PROJECT_DIR / "program_draft.md"
    if not draft_path.exists():
        print("\n  暂无 AI 方向建议（program_draft.md 不存在）。")
        print("  完成批量运行后将自动生成。")
        return

    from program_updater import show_diff_and_confirm, apply_adopt, apply_reject, apply_edit

    # 读取当前和草案第一章
    with open(PROJECT_DIR / "program.md", "r", encoding="utf-8") as f:
        current = f.read()
    with open(draft_path, "r", encoding="utf-8") as f:
        draft = f.read()

    marker = "<!-- AI_MEMORY_START -->"
    current_ch1 = current.split(marker, 1)[0] if marker in current else current
    draft_ch1 = draft.split(marker, 1)[0] if marker in draft else draft

    # 提取第一章
    for label, text in [("current", current_ch1), ("draft", draft_ch1)]:
        ch1_start = text.find("## 第一章")
        ch1_end = text.find("\n## 第二章")
        if ch1_start >= 0 and ch1_end >= 0:
            if label == "current":
                current_ch1 = text[ch1_start:ch1_end]
            else:
                draft_ch1 = text[ch1_start:ch1_end]

    choice = show_diff_and_confirm(current_ch1, draft_ch1)

    if choice == "adopt":
        apply_adopt()
    elif choice == "reject":
        apply_reject()
    elif choice == "edit":
        apply_edit()


def cmd_batch_mode(dry_run: bool = False):
    """交互式批量模式：菜单选 2 后，输入轮数 N。"""
    try:
        n = input("\n  请输入目标轮数 (如 50): ").strip()
        batch_size = int(n)
        if batch_size < 1:
            print("  轮数必须 > 0")
            return
    except ValueError:
        print("  无效数字")
        return

    resume_choice = input("  是否从上次中断续跑？(y/N): ").strip().lower()
    resume = resume_choice == "y"

    print(f"\n  >>> 启动批量模式: {batch_size} 轮 {'(续跑)' if resume else ''} {'(dry-run)' if dry_run else ''}...")
    print(f"  [提示] 全自动运行，日志是主输出。按 Ctrl+C 可提前终止。")

    from batch_pipeline import run_batch as batch_run
    result = batch_run(batch_size=batch_size, resume=resume, dry_run=dry_run)

    print(f"\n{'='*60}")
    print(f"  批量运行完成!")
    print(f"    总轮次:   {result.total_rounds}")
    print(f"    入库:     {result.inbound_count}")
    print(f"    失败:     {result.fail_count}")
    print(f"    跳过:     {result.skip_count}")
    print(f"    终止原因: {result.termination_reason}")
    print(f"    累计成本: ¥{result.cumulative_cost:.4f}")
    print(f"{'='*60}")

    # 如果正常完成，询问是否生成总结
    if result.termination_reason == "normal" and not dry_run:
        print(f"\n  50 轮已完成，是否生成总结报告和方向建议？")
        gen = input("  生成 program_draft.md？(Y/n): ").strip().lower()
        if gen != "n":
            from summary_engine import generate_summary_report, generate_program_draft
            from database import get_conn

            conn = get_conn()
            report = generate_summary_report(conn, 1, batch_size)
            with open(PROJECT_DIR / "program.md", "r", encoding="utf-8") as f:
                current_prog = f.read()
            generate_program_draft(conn, current_prog, report, dry_run=dry_run)
            conn.close()

            print(f"\n  已生成 program_draft.md。请通过菜单 [5] 审核采纳。")


def _show_unfinished_batch():
    """启动时展示未完成的批量任务状态。"""
    try:
        from database import get_conn, get_latest_batch, get_batch_round_summary
        conn = get_conn()
        latest = get_latest_batch(conn)
        if latest and latest["status"] in ("running", "paused"):
            summary = get_batch_round_summary(conn, latest["run_id"])
            print(f"\n  ⚠️  发现未完成的批量任务:")
            print(f"      Run ID: {latest['run_id']}")
            print(f"      进度:   {latest['completed_rounds']}/{latest['target_rounds']}")
            print(f"      状态:   {latest['status']}")
            print(f"      入库: {summary['inbound']} | 失败: {summary['fail']} | "
                  f"跳过: {summary['skip']}")
            print(f"      使用 --batch N --resume 续跑")
        conn.close()
    except Exception:
        pass


def main(dry_run: bool = False, batch_size: int = 0, resume: bool = False):
    """CLI 主入口。

    Args:
        dry_run: Mock API 模式
        batch_size: > 0 时直接启动批量模式（非交互）
        resume: 续跑（仅 batch 模式）
    """
    _print_header()

    # ── Phase 3b 数据库迁移 ────────────────────────
    try:
        from database import get_conn, migrate_v2_to_v3b
        conn = get_conn()
        migrate_v2_to_v3b(conn)
        conn.close()
    except Exception:
        pass  # 数据库未就绪时静默降级

    # ── score.py MD5 启动阻断 ────────────────────────
    from score import verify_md5 as verify_score_md5
    ok, actual = verify_score_md5()
    if not ok:
        print(f"[致命错误] score.py 文件完整性校验失败！MD5={actual[:16]}...")
        print("系统拒绝启动。如需修改评分标准，请人工编辑后运行：python score.py --update-md5")
        sys.exit(1)

    # Git 状态检测
    git_ok = _check_git()
    if not git_ok:
        print("\n  [提示] Git 未初始化，已降级为文件备份。建议运行 git init 启用版本控制。")

    # Phase 2 启动校验
    if (PROJECT_DIR / "data" / "downloaded.txt").exists():
        n_downloaded = len(_load_downloaded_v2_helper(_DV2_DOWNLOADED_FILE))
        data_ok, issues = check_data_integrity_v2()
        print(f"\n  Phase 2 数据模式: {n_downloaded} 只股票")
        if not data_ok:
            print(f"  [警告] 数据问题 ({len(issues)} 项):")
            for i in issues[:5]:
                print(f"    - {i}")
            if not dry_run and batch_size == 0:
                proceed = input("  是否仍要继续？(y/N): ").strip().lower()
                if proceed != "y":
                    return
    else:
        data_ok, issues = check_data_integrity()
        if not data_ok:
            print(f"\n  [警告] 数据不完整 ({len(issues)} 个问题)。请运行: python data_fetcher.py --fetch")
            if not dry_run and batch_size == 0:
                proceed = input("  是否仍要继续？(y/N): ").strip().lower()
                if proceed != "y":
                    return

    # ── 非交互模式: --batch N ────────────────────────
    if batch_size > 0:
        from batch_pipeline import run_batch as batch_run
        print(f"\n  >>> 批量模式启动: {batch_size} 轮 {'(续跑)' if resume else ''} "
              f"{'(dry-run)' if dry_run else ''}...")
        print(f"  [提示] 全自动运行，日志是主输出。Ctrl+C 可提前终止。\n")

        result = batch_run(batch_size=batch_size, resume=resume, dry_run=dry_run)

        print(f"\n{'='*60}")
        print(f"  批量运行完成!")
        print(f"    总轮次:   {result.total_rounds}")
        print(f"    入库:     {result.inbound_count}")
        print(f"    失败:     {result.fail_count}")
        print(f"    跳过:     {result.skip_count}")
        print(f"    终止原因: {result.termination_reason}")
        print(f"    累计成本: ¥{result.cumulative_cost:.4f}")
        print(f"{'='*60}")

        # 正常完成 → 生成总结
        if result.termination_reason == "normal" and not dry_run:
            print(f"\n  生成总结报告...")
            from summary_engine import generate_summary_report, generate_program_draft
            from database import get_conn
            conn = get_conn()
            report = generate_summary_report(conn, 1, batch_size)
            with open(PROJECT_DIR / "program.md", "r", encoding="utf-8") as f:
                current_prog = f.read()
            generate_program_draft(conn, current_prog, report, dry_run=dry_run)
            conn.close()
            print(f"\n  ✅ program_draft.md 已生成。启动交互模式审核：python main.py → 菜单 [5]")
        return

    # ── 交互模式 ──────────────────────────────────────
    _show_unfinished_batch()

    round_num = 1
    consecutive_failures = 0

    while True:
        pool = load_factor_pool()
        pool_size = len(pool.get("factors", []))
        max_round = max((f["round"] for f in pool["factors"]), default=0)

        _print_status(pool_size, max_round)
        print(f"\n  菜单:")
        print(f"    1 — 开始新一轮（单轮，含用户确认）")
        print(f"    2 — 批量模式（输入轮数 N，全自动）")
        print(f"    3 — 查看因子库（查询子菜单）")
        print(f"    4 — 查看累计成本与轮次统计")
        print(f"    5 — 审核 AI 方向建议（program_draft.md）")
        print(f"    6 — 生成 HTML 报告（快速版/完整版）")
        print(f"    7 — 因子检索（自然语言/结构化过滤）")
        print(f"    8 — 退出")

        choice = input("\n  请选择 (1-8): ").strip()

        if choice == "1":
            print(f"\n  >>> 开始 Round {round_num} {'(dry-run)' if dry_run else ''}...")
            result = run_pipeline(round_num=round_num, dry_run=dry_run)

            if result.passed:
                print(f"\n  [OK] Round {round_num} 完成: 因子 {result.factor_id} 已入库")
                consecutive_failures = 0
                round_num += 1
            elif "用户拒绝" in (result.fail_reason or ""):
                print(f"\n  [SKIP] Round {round_num}: 用户跳过")
            elif "3 次代码生成均未通过" in (result.fail_reason or ""):
                print(f"\n  [ABORT] Round {round_num}: 3 次代码生成均失败")
                consecutive_failures += 1
                round_num += 1
            else:
                print(f"\n  [FAIL] Round {round_num}: {result.fail_reason}")
                consecutive_failures += 1
                round_num += 1

            if consecutive_failures >= 3:
                print("\n" + "!" * 50)
                print("[熔断] 连续 3 轮生成不合格，系统自动暂停。")
                print("建议：检查 program.md 第一章研究方向是否过偏，")
                print("       或调整评分阈值。修改后重新启动系统。")
                print("!" * 50 + "\n")
                break

        elif choice == "2":
            cmd_batch_mode(dry_run=dry_run)

        elif choice == "3":
            cmd_view_pool()

        elif choice == "4":
            cmd_view_cost()

        elif choice == "5":
            cmd_adopt_draft()

        elif choice == "6":
            cmd_generate_report()

        elif choice == "7":
            cmd_factor_search()

        elif choice == "8":
            print("\n  退出 FactorLab。")
            break

        else:
            print("\n  无效选择，请输入 1-8。")


if __name__ == "__main__":
    # ── 参数解析 ──────────────────────────────────────
    dry = "--dry-run" in sys.argv
    batch = 0
    do_resume = "--resume" in sys.argv

    # 解析 --batch N
    for i, arg in enumerate(sys.argv):
        if arg == "--batch" and i + 1 < len(sys.argv):
            try:
                batch = int(sys.argv[i + 1])
            except ValueError:
                print(f"错误: --batch 需要数字参数，如 --batch 50")
                sys.exit(1)

    if dry:
        print("[模式] Dry-run — 使用 Mock API，不消耗额度")

    main(dry_run=dry, batch_size=batch, resume=do_resume)

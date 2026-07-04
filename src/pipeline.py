#!/usr/bin/env python3
"""自闭环流程编排 — 九步串联 + Git 自动提交 + 失败熔断。

流程:
  1. 读取规程 → 2. 生成因子 → 3. 合规检查 → 4. 摘要翻译
  → 5. 用户确认 → 6. 沙箱执行 → 7. 评分+门控 → 8. 入库/回滚+报告+记忆
  → 9. Git 提交
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from checker import check_compliance
from data_fetcher import check_data_integrity, DEMO_STOCKS
from batch_pipeline import load_df_1800
from diversity_gate import (
    check_diversity, load_factor_pool, save_factor_pool,
    _rank_values, convert_pool_for_scoring,
)
from engine import create_engine, MockEngine, _build_system_prompt
from memory_manager import append_memory, _compute_chapters_md5
from sandbox import run_sandbox, SandboxTimeout
from backtest import simple_backtest
from score import score_factor, verify_md5 as verify_score_md5

MAX_CODE_ATTEMPTS = 3
# 注: 跨轮熔断逻辑已迁移至 main.py, 此处仅保留单轮内代码生成重试上限
from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
HISTORY_DIR = PROJECT_ROOT / "history"


@dataclass
class PipelineResult:
    """单轮 pipeline 执行结果。"""

    round_num: int
    passed: bool
    factor_id: str = ""
    factor_code: str = ""
    summary: str = ""
    score_result: Optional[Any] = None
    bt_result: Optional[Any] = None  # Phase 2b: 回测结果（参考，不过滤）
    fail_reason: str = ""
    report: str = ""
    api_cost: float = 0.0
    steps_completed: list[str] = field(default_factory=list)


def _git_commit(files: list[str], message: str) -> bool:
    """Git 自动提交。失败返回 False。"""
    try:
        subprocess.run(
            ["git", "add"] + files,
            cwd=str(PROJECT_DIR),
            check=True,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_DIR),
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_rollback() -> bool:
    """回滚 factor_draft.py 到 Git HEAD 状态。失败返回 False。"""
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", "factor_draft.py"],
            cwd=str(PROJECT_DIR),
            check=True,
            capture_output=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _history_backup(files: list[str], round_num: int):
    """Git 未初始化时的降级备份方案。"""
    backup_dir = HISTORY_DIR / f"round_{round_num:03d}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = PROJECT_DIR / f
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / src.name))


def _check_startup() -> bool:
    """启动自检：score.py MD5 + 数据完整性。"""
    ok = True

    # score.py MD5 — C2 修复: 从警告升级为致命错误
    md5_ok, actual = verify_score_md5()
    if not md5_ok:
        print(f"[致命错误] score.py MD5 校验失败！MD5={actual[:16]}...", file=sys.stderr)
        print("系统拒绝启动。如需修改评分标准，请人工编辑后运行：python score.py --update-md5", file=sys.stderr)
        sys.exit(1)

    # 数据完整性
    data_ok, issues = check_data_integrity()
    if not data_ok:
        print(f"[启动检查] 数据完整性存在问题 ({len(issues)} 项):", file=sys.stderr)
        for i in issues[:5]:
            print(f"  - {i}", file=sys.stderr)
        ok = False

    return ok


def run_pipeline(round_num: int, dry_run: bool = False) -> PipelineResult:
    """执行一轮完整的因子挖掘流程。

    Args:
        round_num: 当前轮次编号
        dry_run: 是否使用 MockEngine（不调用真实 API）

    Returns:
        PipelineResult 包含本轮全部信息
    """
    result = PipelineResult(round_num=round_num, passed=False)
    chapters_md5 = _compute_chapters_md5()

    # ── 1. 读取规程 ──────────────────────────────────────
    system_prompt = _build_system_prompt()
    result.steps_completed.append("load_program")

    # ── 2-3. 生成因子 + 合规检查（最多 3 次） ─────────────
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    engine = create_engine(api_key=api_key, model=model, dry_run=dry_run)

    factor_code = ""
    compliance_level = "ERROR"
    compliance_reason = ""

    for attempt in range(1, MAX_CODE_ATTEMPTS + 1):
        try:
            factor_code = engine.generate_code(system_prompt)
        except Exception as e:
            compliance_reason = f"API 调用失败: {e}"
            continue

        if not factor_code or not factor_code.strip():
            compliance_reason = "API 返回空代码"
            continue

        compliance_level, compliance_reason = check_compliance(factor_code)

        if compliance_level == "PASS":
            result.steps_completed.append(f"generate_code(attempt={attempt})")
            result.steps_completed.append("compliance_check")
            break
        elif compliance_level == "WARNING":
            result.steps_completed.append(f"generate_code(attempt={attempt}, warning)")
            result.steps_completed.append("compliance_check")
            break
        # ERROR → 继续下一轮重试
    else:
        # 3 次全部失败
        result.fail_reason = f"3 次代码生成均未通过合规检查。最后错误: {compliance_reason}"
        result.steps_completed.append("generate_code(all_failed)")
        result.api_cost = engine.cost_tracker.cost()
        _git_rollback()
        print(f"\n  [X] 本轮放弃: {result.fail_reason}")
        print(f"  本轮 API 成本: 0.0000 (未调用)")
        return result

    result.factor_code = factor_code

    # ── 4. 摘要翻译 ──────────────────────────────────────
    try:
        summary = engine.generate_summary(factor_code)
        result.summary = summary[:300]
    except Exception as e:
        summary = f"(摘要生成失败: {e})"
        result.summary = summary
    result.steps_completed.append("summary")

    # ── 5. 用户确认 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  因子摘要: {result.summary}")
    print(f"  {'─'*56}")
    code_preview = "\n".join(factor_code.split("\n")[:10])
    print(f"  代码预览 (前 10 行):\n{code_preview}")
    print(f"{'='*60}")

    if not dry_run:
        user_input = input("  确认执行此因子？(Y/n): ").strip().lower()
        if user_input == "n":
            result.fail_reason = "用户拒绝执行"
            result.steps_completed.append("user_rejected")
            result.api_cost = engine.cost_tracker.cost()
            _git_rollback()
            print(f"  本轮 API 成本: {engine.cost_tracker.cost():.4f} | 累计: {engine.cost_tracker.cost():.4f}")
            return result
    else:
        print("  [Dry-run] 自动确认 Y")
    result.steps_completed.append("user_confirmed")

    # ── 6. 沙箱执行 ──────────────────────────────────────
    try:
        df = load_df_1800()  # 全量股票池（MultiIndex date, code）
        factor_series = run_sandbox(factor_code, df, timeout=30)
        result.steps_completed.append("sandbox")
    except SandboxTimeout:
        result.fail_reason = "沙箱执行超时（30 秒）"
        result.steps_completed.append("sandbox_timeout")
        result.api_cost = engine.cost_tracker.cost()
        _git_rollback()
        return result
    except Exception as e:
        result.fail_reason = f"沙箱执行异常: {e}"
        result.steps_completed.append("sandbox_error")
        result.api_cost = engine.cost_tracker.cost()
        _git_rollback()
        return result

    # ── 7. 评分 + 多样性门控 ─────────────────────────────
    # 准备收益率数据（T+5 版本: 同时计算 T+1 和 T+5 收益）
    fv_df = factor_series.unstack("code")  # dates x stocks
    close_df = df["close"].unstack("code")
    ret_t1_df = close_df.shift(-1) / close_df - 1    # T+1 收益
    ret_t5_df = close_df.shift(-5) / close_df - 1    # T+5 累计收益

    common_dates = fv_df.index.intersection(ret_t5_df.index)
    fv_aligned = fv_df.loc[common_dates]
    ret_t5_aligned = ret_t5_df.loc[common_dates]
    ret_t1_aligned = ret_t1_df.loc[common_dates]

    if fv_aligned.dropna(how="all").empty:
        result.fail_reason = "因子值全为空（数据不足）"
        result.steps_completed.append("score_no_data")
        result.api_cost = engine.cost_tracker.cost()
        _git_rollback()
        return result

    # 加载因子池
    factor_pool = load_factor_pool()

    # 评分（T+5 版本: 传入 T+5 收益 + 可选 T+1 收益用于 IC 衰减比）
    score_pool = convert_pool_for_scoring(factor_pool) if factor_pool else None
    score_result = score_factor(fv_aligned, ret_t5_aligned, score_pool, returns_t1=ret_t1_aligned)
    result.score_result = score_result

    # 多样性门控
    gate_passed, gate_dup_id, _ = check_diversity(fv_aligned, factor_pool)

    passed = score_result.passed_threshold and gate_passed
    result.passed = passed

    if not passed:
        reasons = list(score_result.failed_reasons)
        if not gate_passed:
            reasons.append(f"多样性门控: 与 {gate_dup_id} 重复")
        result.fail_reason = "; ".join(reasons)
    result.steps_completed.append("score_and_gate")

    # Phase 2b: 回测（独立于评分，结果仅参考不过滤）
    try:
        bt_result = simple_backtest(fv_aligned, ret_t1_aligned)
        result.bt_result = bt_result
        result.steps_completed.append("backtest")
    except Exception:
        result.steps_completed.append("backtest_skipped")

    # ── 8. 入库/回滚 + 报告 + 记忆 ────────────────────────
    if passed:
        # 入库
        factor_id = f"f{len(factor_pool.get('factors', [])) + 1:03d}"
        rank_snap = _rank_values(fv_aligned)

        new_entry = {
            "factor_id": factor_id,
            "round": round_num,
            "inbound_date": datetime.now().strftime("%Y-%m-%d"),
            "code_snapshot": factor_code,
            "natural_summary": result.summary,
            "metrics": {
                dim: d["value"]
                for dim, d in score_result.dimensions.items()
            },
            "score_total": score_result.total_score,
            "rank_snapshot": rank_snap,
        }
        factor_pool["factors"].append(new_entry)
        save_factor_pool(factor_pool)
        result.factor_id = factor_id

        # 写入 factor_draft.py
        draft_code = (
            "import numpy as np\n"
            "import pandas as pd\n\n\n"
            f"{factor_code}\n"
        )
        with open(PROJECT_DIR / "factor_draft.py", "w", encoding="utf-8") as f:
            f.write(draft_code)

        result.steps_completed.append("inbound")
    else:
        result.steps_completed.append("discard")
        _git_rollback()

    # 生成报告（API 调用 #3）
    round_info = {
        "round": round_num,
        "passed": passed,
        "summary": result.summary,
        "score": round(score_result.total_score, 4) if score_result else "N/A",
        "reason": result.fail_reason,
    }
    try:
        report = engine.generate_report(round_info)
        result.report = report
    except Exception as e:
        result.report = f"(报告生成失败: {e})"
    result.steps_completed.append("report")

    # 追加记忆
    next_suggestion = ""
    if not passed:
        # 尝试从报告中提取建议
        if "建议" in result.report:
            next_suggestion = "请参考报告中的建议调整研究方向。"

    memory_ok = append_memory(
        round_num=round_num,
        factor_summary=result.summary,
        passed=passed,
        fail_reason=result.fail_reason,
        next_suggestion=next_suggestion,
        expected_md5=chapters_md5,
    )
    if memory_ok:
        result.steps_completed.append("memory")
    else:
        result.steps_completed.append("memory_failed")

    # 成本
    result.api_cost = engine.cost_tracker.cost()
    cumulative = engine.cost_tracker.cost()
    print(f"\n  本轮 API 成本: {result.api_cost:.4f} | 累计: {cumulative:.4f}")

    # Phase 2b: 回测参考指标（仅展示，不影响入库决策）
    if result.bt_result:
        bt = result.bt_result
        print(f"  回测参考: 夏普={bt.sharpe_ratio:.2f} | 回撤={bt.max_drawdown:.1%} | 年化={bt.annual_return:.1%} | 胜率={bt.win_rate:.0%}")

    # ── 9. Git 自动提交 ──────────────────────────────────
    commit_msg = f"Round {round_num}: {result.summary[:60]}"
    commit_ok = _git_commit(["factor_draft.py", "program.md", "factor_pool.json"], commit_msg)
    if commit_ok:
        result.steps_completed.append("git_commit")
    else:
        _history_backup(["factor_draft.py", "program.md", "factor_pool.json"], round_num)
        result.steps_completed.append("history_backup")

    return result


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== pipeline.py 自检 (Dry-run) ===\n")

    # 启动检查
    _check_startup()

    # Dry-run 模式跑一轮
    print("\n--- Running Round 1 (dry-run) ---\n")
    result = run_pipeline(round_num=1, dry_run=True)

    print(f"\n{'='*60}")
    print(f"  Pipeline Result:")
    print(f"    round:          {result.round_num}")
    print(f"    passed:         {result.passed}")
    print(f"    factor_id:      {result.factor_id or 'N/A'}")
    print(f"    fail_reason:    {result.fail_reason or 'N/A'}")
    print(f"    api_cost:       {result.api_cost:.4f}")
    print(f"    steps:          {result.steps_completed}")
    print(f"{'='*60}")

    if result.passed:
        # 验证 factor_pool.json 有内容
        pool = load_factor_pool()
        assert len(pool["factors"]) > 0, "因子应已入库!"
        print(f"\n因子池数量: {len(pool['factors'])}")
        print(f"最新因子: {pool['factors'][-1]['factor_id']}")

    print("\n自检通过.")

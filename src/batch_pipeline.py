#!/usr/bin/env python3
"""Phase 2c 全自动批量引擎 — 双层熔断 + 事件检查 + --resume 支持。

用法:
  python batch_pipeline.py demo    # 用 --dry-run 模式跑 3 轮演示
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import simple_backtest
from checker import check_compliance
from config import (
    CAPITAL_ASSUMPTION, HOLDING_PERIOD, TOP_PCT,
    IMPACT_COST_COEFFICIENT, IMPACT_COST_CAP,
    RISK_FREE_RATE, ADV_WINDOW,
)
from database import (
    get_conn, init_db, insert_factor, insert_round, insert_memory,
    insert_backtest, create_batch, update_batch, get_batch, get_latest_batch,
    get_recent_memories, get_recent_rounds_status, get_recent_direction_tags,
    get_max_sharpe, count_factors, get_batch_round_summary,
    get_pending_similarity_feedback, save_pending_similarity_feedback,
)
from diversity_gate import check_diversity, load_factor_pool, save_factor_pool, convert_pool_for_scoring, _rank_values
from engine import create_engine, CostTracker
from logger import BatchLogger
from memory_manager import (
    _compute_chapters_md5, append_milestone, update_md5_baseline,
    get_recent_memories_for_prompt,
)
from robustness_checker import evaluate as robustness_evaluate, RobustnessResult
from sandbox import run_sandbox, SandboxTimeout, SANDBOX_TIMEOUT
from score import score_factor

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
PROGRAM_PATH = PROJECT_ROOT / "program.md"


# ── Direction Tag 提取（B+C 混合方案）────────────────────

# 从 program.md 第一章 1.3 节硬编码提取的关键词 → 标准化标签映射
_DIRECTION_KEYWORD_MAP: Dict[str, str] = {
    "反转": "反转类",
    "反转类": "反转类",
    "行为类": "行为类",
    "散户": "行为类",
    "追高": "行为类",
    "量价背离": "量价背离",
    "底背离": "量价背离",
    "背离": "量价背离",
    "价格路径": "价格路径",
    "收盘价相对位置": "价格路径",
    "OHLC": "价格路径",
    "波动率": "波动率类",
    "波动率类": "波动率类",
    "低波动": "波动率类",
    "换手率": "波动率类",
    "低换手": "波动率类",
    "动量": "动量",
    "动量效应": "动量",
    "成交量趋势": "动量",
    "放量": "动量",
    "缩量": "反转类",
    "时间结构": "时间结构",
    "周内效应": "时间结构",
    "月末效应": "时间结构",
}


def extract_direction_tag(summary: str) -> str:
    """从 AI 生成的 summary 中提取研究方向标签（关键词匹配，零 API 开销）。

    Args:
        summary: AI 生成的自然语言摘要

    Returns:
        标准化方向标签，匹配不到时返回 "量价类"
    """
    if not summary:
        return "量价类"
    for keyword, tag in _DIRECTION_KEYWORD_MAP.items():
        if keyword in summary:
            return tag
    return "量价类"


# ── 数据加载 ────────────────────────────────────────────

def load_df_1800(max_stocks: int = 0) -> pd.DataFrame:
    """从 data/ 加载全量 CSV，合并为 MultiIndex DataFrame。

    Args:
        max_stocks: 限制加载股票数（0 = 全部），用于快速测试

    Returns:
        DataFrame with MultiIndex (date, code), columns = [open, high, low, close, volume]
    """
    from data_fetcher_v2 import _load_set, DOWNLOADED_FILE

    codes = sorted(_load_set(DOWNLOADED_FILE))
    if max_stocks and max_stocks > 0:
        codes = codes[:max_stocks]

    frames = []
    for code in codes:
        csv_path = DATA_DIR / f"{code}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                continue
            # Phase 3a: CSV 不含 code 列，从文件名提取
            if "code" not in df.columns:
                df["code"] = code
            else:
                df["code"] = df["code"].astype(str).str.zfill(6)
            df = df[["date", "code"] + required].copy()
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("data/ 目录无有效 CSV 文件，请先运行 data_fetcher_v2.py --fetch")

    result = pd.concat(frames, ignore_index=True)
    result = result.set_index(["date", "code"]).sort_index()
    return result


# ── BatchResult ──────────────────────────────────────────

@dataclass
class BatchResult:
    """批量运行结果。"""

    total_rounds: int = 0
    inbound_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    error_count: int = 0
    cumulative_cost: float = 0.0
    termination_reason: str = "normal"  # normal/user_interrupt/mid_fuse/error_fuse/early_stop
    failed_rounds: List[Dict] = field(default_factory=list)
    summary_report_path: str = ""
    draft_path: str = ""


# ── Phase 3h: TIMEOUT 慢模式检测（升级版）──────────────────────

def _detect_slow_pattern(code: str) -> str:
    """检测因子代码中的慢模式，返回可读描述（用于反馈闭环）。"""
    if not code:
        return ""
    patterns = []
    # apply(lambda) — 头号超时杀手
    if re.search(r'\.apply\(\s*lambda\b', code):
        patterns.append("apply(lambda) 逐行操作")
    if re.search(r'\.apply\(\s*\(.*axis\s*=\s*1', code):
        patterns.append("apply(axis=1) 逐行迭代")
    # 逐行循环
    if re.search(r'for\s+\w+\s+in\s+.*\.(?:iterrows|itertuples)\(\)', code):
        patterns.append("iterrows/itertuples() 逐行循环")
    if re.search(r'for\s+\w+\s+in\s+.*\.get_level_values', code):
        patterns.append("逐股票 for 循环")
    # groupby().apply() — 慢但向量化，不拦截但标注
    if re.search(r'\.groupby\([^)]+\)\.apply\(', code):
        patterns.append("groupby().apply() 嵌套操作（可优化为 transform）")
    if not patterns:
        patterns.append("未识别具体模式（疑似逐行操作或循环）")
    return "、".join(patterns)


def _build_timeout_feedback(code: str) -> str:
    """构建 TIMEOUT 诊断反馈文本（注入下一轮 system_prompt）。"""
    slow = _detect_slow_pattern(code)
    return (
        f"\n[分析-TIMEOUT] 代码在 4M 行上超过 {SANDBOX_TIMEOUT} 秒。检测到：{slow}。\n"
        f"  正例（5μs/行级别，向量化）: "
        f"df.groupby(level='code')['close'].transform(lambda x: x.rolling(20).mean())\n"
        f"  反例（100μs/行级别，逐行）: "
        f"df.apply(lambda row: ..., axis=1)  # 1482 只股票 × 2500 天 → 370 万次 Python 函数调用\n"
        f"  反例（超时必现）: "
        f"for code in df.index.get_level_values('code').unique(): ...  # 1482 次独立 select + 运算\n"
        f"  替代方案：用 groupby(level='code')['col'].transform(lambda x: x.rolling(N).mean())，"
        f"每次只在单个股票的时序上执行 lambda，数据量 <2500 行，而非 4M 行。"
    )


# ── Phase 3h: 模板方向轮换（B1）──────────────────────

TEMPLATE_DIRECTIONS = {
    "T1": {"name": "纯时序反转", "formula": "close/close.shift(60)-1",
           "key": "per-stock z-score，无截面 rank"},
    "T2": {"name": "波动率加权", "formula": "ret/rolling(20).std()",
           "key": "波动率归一化替代截面 rank"},
    "T3": {"name": "多周期共振", "formula": "rolling(5).mean(ret) × sign(pct_change(60))",
           "key": "短信号 × 长方向符号"},
    "T4": {"name": "日内路径效率", "formula": "(close-open)/(high-low)",
           "key": "日内有向性，rolling(20).mean()"},
    "T5": {"name": "成交量结构", "formula": "volume/rolling(20).volume.mean()",
           "key": "量价分离，非乘积"},
}
TEMPLATE_ORDER = list(TEMPLATE_DIRECTIONS.keys())  # T1→T2→T3→T4→T5

_template_rotation_idx = 0  # 跨 round 轮换计数器
_template_success_count = {k: 0 for k in TEMPLATE_ORDER}  # Phase 3i: 各模板成功入库次数
_template_last_used = ""  # 上轮使用的模板 key


def _select_template_direction() -> dict:
    """加权轮换选择模板方向（Phase 3i P1-3）。

    基础权重：每模板 1.0 + 0.5 × success_count（成功入库的模板获得更高采样概率）。
    额外规则：如果上轮刚用过某个模板，本轮降低其权重 50%（避免同方向连续重复）。
    """
    global _template_rotation_idx, _template_last_used

    # 计算权重：成功模板获得更高概率
    weights = []
    for k in TEMPLATE_ORDER:
        w = 1.0 + 0.5 * _template_success_count.get(k, 0)
        if k == _template_last_used:
            w *= 0.5  # 降低上轮模板概率
        weights.append(max(w, 0.1))  # 保证最低采样概率

    # 加权随机采样
    total = sum(weights)
    probs = [w / total for w in weights]
    r = __import__('random').random()  # lazy import
    cum = 0
    key = TEMPLATE_ORDER[-1]
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            key = TEMPLATE_ORDER[i]
            break

    _template_rotation_idx += 1
    _template_last_used = key
    template = dict(TEMPLATE_DIRECTIONS[key])
    template["_key"] = key  # Phase 3i: 嵌入模板 key 以便后续加权记录
    return template


def _record_template_success(template_key: str):
    """Phase 3i: 记录模板成功入库，用于加权轮换。"""
    global _template_success_count
    if template_key in _template_success_count:
        _template_success_count[template_key] += 1


# ── FinalResult（Phase 3b 10 维合并）─────────────────────

@dataclass
class FinalResult:
    """10 维合并结果，用于入库决策和报告展示。"""

    # === score.py 核心维度（只读，来自 ScoreResult）===
    ic: float = 0.0
    ir: float = 0.0
    coverage: float = 0.0
    correlation_max: float = 0.0
    turnover: float = 0.0
    direction_correctness: float = 0.0
    score_threshold_passed: bool = False

    # === backtest.py 展示字段 ===
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    turnover_estimate: float = 0.0
    avg_impact_cost_bps: float = 0.0
    total_cost_annual: float = 0.0

    # === robustness_checker.py 4 维 ===
    monotonicity: float = 0.0
    monotonicity_passed: bool = False
    oos_ic_train: float = 0.0
    oos_ic_test: float = 0.0
    oos_stability_passed: bool = False
    oos_yearly_pass_count: int = 0
    oos_yearly_ics: dict = field(default_factory=dict)
    ic_decay_ratio: float = 0.0
    ic_decay_passed: bool = False
    yearly_validation_observed: list = field(default_factory=list)

    # === 最终 10 维合并结果 ===
    threshold_passed: bool = False
    cum_return_curve: Optional[pd.Series] = None
    layer_returns: Optional[pd.DataFrame] = None


def merge_results(
    score_result: "ScoreResult",
    backtest_result: "BacktestResult",
    robustness_result: "RobustnessResult",
) -> FinalResult:
    """合并 score.py + backtest.py + robustness_checker.py 三方结果。

    score.py 冻结，其 passed_threshold 基于其内部维度。
    robustness_checker 独立，其 robust_core_passed 基于前 3 维核心。
    最终 threshold_passed = score.passed_threshold AND robust_core_passed。
    分年验证仅作展示，不参与合并判断。
    """
    final = FinalResult()

    # 从 ScoreResult.dimensions dict 提取各维度值
    dims = getattr(score_result, "dimensions", {})
    final.ic = dims.get("ic", {}).get("value", 0.0)
    final.ir = dims.get("ir", {}).get("value", 0.0)
    final.coverage = dims.get("coverage", {}).get("value", 0.0)
    final.correlation_max = dims.get("correlation", {}).get("value", 0.0)
    final.turnover = dims.get("turnover", {}).get("value", 0.0)
    final.direction_correctness = dims.get("directional_accuracy", {}).get("value", 0.0)
    final.score_threshold_passed = score_result.passed_threshold

    # 复制 backtest 展示字段
    final.annual_return = backtest_result.annual_return
    final.max_drawdown = backtest_result.max_drawdown
    final.sharpe_ratio = backtest_result.sharpe_ratio
    final.win_rate = backtest_result.win_rate
    final.turnover_estimate = backtest_result.turnover_estimate
    final.avg_impact_cost_bps = backtest_result.avg_impact_cost_bps
    final.total_cost_annual = backtest_result.total_cost_annual
    final.cum_return_curve = backtest_result.cum_return_curve
    final.layer_returns = backtest_result.layer_returns

    # 复制稳健性 4 维
    final.monotonicity = robustness_result.monotonicity
    final.monotonicity_passed = robustness_result.monotonicity_passed
    final.oos_ic_train = robustness_result.oos_ic_train
    final.oos_ic_test = robustness_result.oos_ic_test
    final.oos_stability_passed = robustness_result.oos_stability_passed
    final.oos_yearly_pass_count = getattr(robustness_result, "oos_yearly_pass_count", 0)
    final.oos_yearly_ics = getattr(robustness_result, "oos_yearly_ics", {})
    final.ic_decay_ratio = robustness_result.ic_decay_ratio
    final.ic_decay_passed = robustness_result.ic_decay_passed
    final.yearly_validation_observed = robustness_result.yearly_validation_observed

    # 10 维合并判断
    final.threshold_passed = (
        score_result.passed_threshold
        and robustness_result.robust_core_passed
    )

    return final


# ── 里程碑事件检查 ─────────────────────────────────────

def _check_milestone_events(
    conn, logger: BatchLogger, batch_run_id: int, round_num: int,
    cumulative_cost: float, program_md5: str,
    sharpe_seen_flag: List[bool],  # mutable flag: 是否已触发过夏普事件
    cost_seen_flag: List[bool],     # mutable flag: 是否已触发过成本事件
):
    """检查 5 种里程碑事件，触发时追加到 program.md。"""
    summary = get_batch_round_summary(conn, batch_run_id)

    # Event 2: 首个回测夏普 > 1.5
    if not sharpe_seen_flag[0]:
        max_s = get_max_sharpe(conn)
        if max_s > 1.5:
            content = (
                f"触发原因：入库因子回测夏普 {max_s:.2f} > 1.5\n"
                f"当前状态：入库 {summary['inbound']} / 失败 {summary['fail']} / "
                f"跳过 {summary['skip']} / 累计成本 ¥{cumulative_cost:.4f}\n"
                f"关键发现：发现首个高夏普因子，建议重点关注该方向\n"
                f"下一步建议：在该方向加大探索密度"
            )
            append_milestone(str(PROGRAM_PATH), "首个高夏普因子", content, program_md5)
            logger.log_event("milestone", f"首个夏普>1.5因子入库 (夏普={max_s:.2f})")
            sharpe_seen_flag[0] = True

    # Event 5: 累计成本突破 ¥3.00
    if not cost_seen_flag[0] and cumulative_cost > 3.0:
        content = (
            f"触发原因：累计 API 成本 ¥{cumulative_cost:.4f} > ¥3.00\n"
            f"当前状态：入库 {summary['inbound']} / 失败 {summary['fail']} / "
            f"跳过 {summary['skip']} / 轮次 {round_num}\n"
            f"关键发现：成本已达预算中点，回顾入库效率\n"
            f"下一步建议：如入库率持续偏低，考虑调整研究方向或评分阈值"
        )
        append_milestone(str(PROGRAM_PATH), "成本突破¥3", content, program_md5)
        logger.log_event("milestone", f"累计成本突破¥3.00 (¥{cumulative_cost:.4f})")
        cost_seen_flag[0] = True

    # Event 4: 连续 5 轮同一方向失败
    tags = get_recent_direction_tags(conn, n=5)
    if len(tags) >= 5 and len(set(tags)) == 1 and tags[0]:
        # 检查最近 5 轮是否都非 inbound
        recent_status = get_recent_rounds_status(conn, n=5)
        if all(s != "inbound" for s in recent_status):
            content = (
                f"触发原因：连续 5 轮方向 '{tags[0]}' 无入库\n"
                f"当前状态：入库 {summary['inbound']} / 失败 {summary['fail']}\n"
                f"关键发现：方向 '{tags[0]}' 连续失败，可能不适合当前市场\n"
                f"下一步建议：暂停 '{tags[0]}' 方向，切换其他优先级方向"
            )
            append_milestone(str(PROGRAM_PATH), "连续同方向失败", content, program_md5)
            logger.log_event("milestone", f"连续5轮方向'{tags[0]}'失败")


def _format_summary_line(summary: dict) -> str:
    """格式化进度摘要行。"""
    return (
        f"进度: {summary['total']}轮 | 入库: {summary['inbound']} | "
        f"失败: {summary['fail']} | 跳过: {summary['skip']} | Error: {summary['error']}"
    )


# ── 核心批量函数 ────────────────────────────────────────

def run_batch(
    batch_size: int = 50,
    resume: bool = False,
    dry_run: bool = False,
    max_stocks: int = 0,
) -> BatchResult:
    """执行全自动批量因子挖掘。

    Args:
        batch_size: 目标轮次数
        resume: 是否从上次中断续跑
        dry_run: Mock 模式，不消耗 API
        max_stocks: 限制加载股票数（0=全部），用于测试

    Returns:
        BatchResult 包含批量运行完整信息
    """
    result = BatchResult(total_rounds=batch_size)

    # ── 初始化 ────────────────────────────────────────
    conn = get_conn()
    init_db(conn, migrate=not resume)  # resume 时不重建表，保留历史数据

    # 加载数据
    df_1800 = load_df_1800(max_stocks=max_stocks)

    # Phase 3b: 预计算回测所需矩阵（批量启动时算一次，所有轮次共享）
    close_df_full = df_1800["close"].unstack("code")
    volume_df_full = df_1800["volume"].unstack("code")  # 单位：股
    returns_df_full = close_df_full.shift(-1) / close_df_full - 1  # T+1 前向收益（修复方向错位）

    # 创建 batch_status 记录
    program_md5 = _compute_chapters_md5()
    if resume:
        latest = get_latest_batch(conn)
        if latest and latest["status"] in ("running", "paused"):
            run_id = latest["run_id"]
            start_round = latest["completed_rounds"] + 1
            cumulative_cost = latest["cumulative_cost"]
            # M5 修复: 预加载历史成本到 CostTracker 单例
            tracker = CostTracker()
            tracker.preload(cumulative_cost)
            logger = BatchLogger(run_id=run_id, dry_run=dry_run)
            logger.log_summary(f"续跑: 从 Round {start_round} 开始 (batch run_id={run_id}, 历史成本=¥{cumulative_cost:.4f})")
        else:
            print("无可续跑的批量任务，将创建新任务。")
            resume = False

    if not resume:
        run_id = create_batch(conn, batch_size, program_md5) if not dry_run else 0
        start_round = 1
        cumulative_cost = 0.0
        logger = BatchLogger(run_id=run_id, dry_run=dry_run)

    logger.log_summary(f"批量启动: batch_size={batch_size}, resume={resume}, dry_run={dry_run}")
    logger.log_summary(f"数据规模: {df_1800.shape[0]} 行 × {df_1800.index.get_level_values('code').nunique()} 只股票")

    # ── 状态变量 ──────────────────────────────────────
    consecutive_errors = 0
    consecutive_fuse_counter = 0  # 中期熔断: 连续无入库计数
    sharpe_seen = [False]
    cost_seen = [False]
    termination_reason = "normal"
    inbound_counter = count_factors(conn)  # P0-1 fix: 内存计数器，避免 SQLite 查询延迟导致 ID 重复

    # Phase 3g: 沙箱同错误退避策略（打断死亡循环）
    _last_sandbox_error = ""
    _same_sandbox_error_count = 0

    # ── 主循环 ────────────────────────────────────────
    for round_num in range(start_round, batch_size + 1):
        logger.log_round_start(round_num, batch_size)
        round_cost = 0.0
        round_status = "fail"
        round_direction = ""
        round_factor_code = ""
        round_summary = ""
        round_steps = []
        round_fail_reason = ""

        engine = create_engine(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            dry_run=dry_run,
        )
        # M1 修复: 保存本轮起始成本，确保跨轮成本不污染
        round_start_cost = engine.cost_tracker.cost()

        # ── ① 构建 system_prompt ──────────────────────
        try:
            memory_text = get_recent_memories_for_prompt(conn, n=5)
        except Exception:
            memory_text = "（暂无实验记忆）"

        # 读取 program.md 前三章
        with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
            program_content = f.read()
        marker = "<!-- AI_MEMORY_START -->"
        chapters_1_3 = program_content.split(marker, 1)[0] if marker in program_content else program_content

        # Phase 3e: 注入上一轮代码相似度门禁的 pending feedback（AI 生成代码前看到）
        pending_feedback = None
        if not dry_run:
            try:
                pending_feedback = get_pending_similarity_feedback(conn)
            except Exception:
                pass

        if pending_feedback:
            system_prompt = (
                f"[系统通知 - 代码相似度警告] {pending_feedback}\n\n"
                f"{chapters_1_3.strip()}\n\n---\n\n{memory_text}"
            )
        else:
            system_prompt = (
                f"{chapters_1_3.strip()}\n\n---\n\n{memory_text}"
            )
        round_steps.append("build_prompt")

        # ── ②③ 生成因子 + 合规检查（最多 3 次）───────
        factor_code = ""
        compliance_level = "ERROR"
        compliance_reason = ""
        code_gen_success = False

        # Phase 3h: 选择本轮强制模板方向（B1 — 轮换 T1→T5）
        template_dir = _select_template_direction()
        logger.log_summary(f"template: {template_dir['name']}")  # Phase 3i D3c

        for attempt in range(1, 4):
            try:
                factor_code = engine.generate_code(system_prompt, template_direction=template_dir)
            except Exception as e:
                compliance_reason = f"API 调用失败: {e}"
                round_status = "error"
                consecutive_errors += 1
                logger.log_error(f"API error (attempt={attempt}): {e}")
                continue

            if not factor_code or not factor_code.strip():
                compliance_reason = "API 返回空代码"
                continue

            compliance_level, compliance_reason = check_compliance(factor_code)
            _input_tokens = 1200  # estimate
            _output_tokens = 150
            _cost = (_input_tokens * 1 + _output_tokens * 2) / 1_000_000

            if not dry_run:
                # M1 修复: 从本轮起始成本计算增量，避免跨轮污染
                _cost = engine.cost_tracker.cost() - round_start_cost
                round_cost = engine.cost_tracker.cost() - round_start_cost

            logger.log_api_call(
                f"generate_code(attempt={attempt})",
                _input_tokens, _output_tokens, _cost,
            )

            if compliance_level == "PASS":
                logger.log_compliance("PASS")
                code_gen_success = True
                round_steps.append(f"code_gen(attempt={attempt})")
                break
            elif compliance_level == "WARNING":
                logger.log_compliance("WARNING", compliance_reason)
                code_gen_success = True
                round_steps.append(f"code_gen(attempt={attempt}, warning)")
                break
            else:
                logger.log_compliance("ERROR", compliance_reason)

        if not code_gen_success:
            round_fail_reason = f"3次代码生成均失败: {compliance_reason}"
            round_status = "error" if "API" in compliance_reason else "fail"
            logger.log_fail(round_fail_reason)

            if not dry_run:
                insert_round(conn, {
                    "round_id": round_num, "batch_run_id": run_id,
                    "direction_tag": "", "status": round_status,
                    "started_at": datetime.now().isoformat(),
                    "fail_reason": round_fail_reason,
                    "api_cost": round_cost, "summary": "",
                    "steps": round_steps,
                })
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})

            # 检查 error 熔断
            if consecutive_errors >= 3:
                termination_reason = "error_fuse"
                logger.log_event("fuse", "连续3轮API网络故障，暂停")
                break

            continue  # 跳过后续步骤，进入下一轮

        consecutive_errors = 0  # 成功获取代码，重置 error 计数
        round_factor_code = factor_code

        # ── ④ 摘要 ──────────────────────────────────────
        try:
            round_summary = engine.generate_summary(factor_code) if not dry_run else "[Mock] 量价因子"
            round_summary = round_summary[:300]
        except Exception as e:
            round_summary = f"(摘要失败: {e})"
        round_steps.append("summary")

        # M1 修复: 使用本轮增量，非历史累计
        if not dry_run:
            round_cost = engine.cost_tracker.cost() - round_start_cost

        # 提取 direction_tag
        round_direction = extract_direction_tag(round_summary)
        if not round_direction:
            round_direction = extract_direction_tag(factor_code)  # fallback: 从代码中匹配

        # ── ⑤ 沙箱 ──────────────────────────────────────
        try:
            t0 = time.time()
            factor_series = run_sandbox(factor_code, df_1800, timeout=SANDBOX_TIMEOUT)
            elapsed = time.time() - t0
            # Phase 3i P1-1: 清洗除零产生的 inf/-inf
            n_inf = np.isinf(factor_series).sum()
            if n_inf > 0:
                factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
            logger.log_sandbox(f"SUCCESS (inf_cleaned={n_inf})" if n_inf > 0 else "SUCCESS", elapsed)
            round_steps.append("sandbox")
        except SandboxTimeout:
            # Phase 3h: TIMEOUT 反馈闭环 — 提取慢模式 + 注入正反例
            timeout_feedback = _build_timeout_feedback(factor_code)
            round_fail_reason = (
                f"沙箱超时({SANDBOX_TIMEOUT}s)\n{timeout_feedback}"
            )
            round_status = "fail"
            logger.log_sandbox("TIMEOUT", SANDBOX_TIMEOUT)
            _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                          round_factor_code, round_summary, round_steps, round_fail_reason,
                          round_cost, cumulative_cost, dry_run)
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})
            continue
        except Exception as e:
            round_fail_reason = f"沙箱异常: {e}"
            round_status = "fail"
            logger.log_sandbox("ERROR")
            logger.log_fail(round_fail_reason)

            # Phase 3g: 同错误退避 — 连续 ≥2 轮相同沙箱异常 → 告警并注入修复提示
            err_msg = str(e)
            if err_msg == _last_sandbox_error:
                _same_sandbox_error_count += 1
                if _same_sandbox_error_count >= 2:
                    round_fail_reason += (
                        f"\n⚠️ 此错误已连续出现 {_same_sandbox_error_count + 1} 轮（死亡循环检测）。"
                        f"\n请检查是否重复生成了相同的代码模式。"
                        f"\n如果是 'code occurs multiple times' 错误：请勿创建名为 'code' 的列，"
                        f"勿使用 reset_index()，所有 groupby 必须用 level='code'。"
                    )
                    logger.log_fail(
                        f"沙箱死亡循环检测: 同错误已连续 {_same_sandbox_error_count + 1} 轮"
                    )
            else:
                _last_sandbox_error = err_msg
                _same_sandbox_error_count = 0

            _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                          round_factor_code, round_summary, round_steps, round_fail_reason,
                          round_cost, cumulative_cost, dry_run)
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})
            continue

        # ── ⑥⑦⑧ 评分 + 多样性 + 回测 ──────────────────
        fv_df = factor_series.unstack("code")
        # 从预计算矩阵中截取与因子值对齐的日期和股票
        common_dates = fv_df.index.intersection(close_df_full.index)
        common_stocks = fv_df.columns.intersection(close_df_full.columns)
        fv_aligned = fv_df.loc[common_dates, common_stocks]
        ret_t5_aligned = close_df_full.loc[common_dates, common_stocks].shift(-5) / \
                         close_df_full.loc[common_dates, common_stocks] - 1
        ret_t1_aligned = returns_df_full.loc[common_dates, common_stocks]

        # Phase 3i P0-1: 稀疏检查（在评分/回测前拦截）
        coverage = fv_aligned.notna().sum(axis=1).mean() / fv_aligned.shape[1] if fv_aligned.shape[1] > 0 else 0
        n_select = max(1, int(len(common_stocks) * TOP_PCT))
        # 用 5 分位数代替 min()：避免 rolling 窗口预热期（开头全是 NaN）误杀
        n_valid = fv_aligned.notna().sum(axis=1).quantile(0.05) if fv_aligned.shape[0] > 0 else 0

        if fv_aligned.dropna(how="all").empty:
            round_fail_reason = "因子值全为空"
            round_status = "fail"
            logger.log_fail(round_fail_reason)
            _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                          round_factor_code, round_summary, round_steps, round_fail_reason,
                          round_cost, cumulative_cost, dry_run)
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})
            continue

        # Phase 3i P0-1: 因子值覆盖率检查
        if coverage < 0.40:
            round_fail_reason = f"因子值覆盖率不足 {coverage:.1%}，需≥40%"
            round_status = "fail"
            logger.log_fail(round_fail_reason)
            _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                          round_factor_code, round_summary, round_steps, round_fail_reason,
                          round_cost, cumulative_cost, dry_run)
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})
            continue

        # Phase 3i P0-1: Top decile 有效股票数检查
        if n_valid < 10:
            round_fail_reason = f"Top decile 有效股票数不足 (n_valid={n_valid}，需≥10)"
            round_status = "fail"
            logger.log_fail(round_fail_reason)
            _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                          round_factor_code, round_summary, round_steps, round_fail_reason,
                          round_cost, cumulative_cost, dry_run)
            result.failed_rounds.append({"round_num": round_num, "status": round_status, "reason": round_fail_reason})
            continue

        # 加载因子池（Phase 2a 过渡：优先 SQLite，降级 JSON）
        factor_pool = load_factor_pool()
        score_pool = convert_pool_for_scoring(factor_pool) if factor_pool else None
        score_result = score_factor(fv_aligned, ret_t5_aligned, score_pool, returns_t1=ret_t1_aligned)

        ic_val = score_result.dimensions.get("ic", {}).get("value", 0)
        ir_val = score_result.dimensions.get("ir", {}).get("value", 0)
        logger.log_score(ic_val, ir_val, score_result.passed_threshold)

        # Phase 3b: 回测（含冲击成本，使用预计算矩阵）
        bt_result = None
        robustness_result = None
        final_result = None
        backtest_degenerate = False  # Phase 3g: 回测异常标记
        try:
            vol_slice = volume_df_full.loc[common_dates, common_stocks]
            cls_slice = close_df_full.loc[common_dates, common_stocks]
            ret_slice = returns_df_full.loc[common_dates, common_stocks]

            bt_result = simple_backtest(
                fv_aligned, ret_slice,
                volume_df=vol_slice, close_df=cls_slice,
            )
            logger.log_backtest(bt_result.sharpe_ratio, bt_result.max_drawdown, bt_result.annual_return)
            round_steps.append("backtest")

            # Phase 3g: 检测回测数值异常，直接标记 FAIL（节省评分 API）
            if not np.isfinite(bt_result.annual_return) or bt_result.annual_return <= -0.99:
                # Phase 3i P0-2: 退化诊断 — 记录覆盖率、top decile ADV、持仓稀疏度
                diag_coverage = float(coverage)  # 复用 P0-1 已计算的覆盖率，强制标量
                # Top decile 股票平均 ADV（万元）
                top_adv_wan = 0.0
                position_sparsity = 0.0
                try:
                    top_ranks = fv_aligned.rank(axis=1, ascending=False, pct=True)
                    top_mask = top_ranks <= TOP_PCT
                    adv_matrix = vol_slice * cls_slice  # 成交额矩阵
                    top_adv_vals = adv_matrix[top_mask]  # 1D Series (mask 展开后)
                    top_adv_avg = float(top_adv_vals.mean()) if top_mask.any().any() else 0.0
                    top_adv_wan = float(top_adv_avg) / 10000  # 转为万元
                    # 持仓稀疏度：top decile 中有效股票占应选股票的比例
                    n_valid_top = float((top_mask.sum(axis=1) / top_mask.shape[1]).mean()) if top_mask.shape[1] > 0 else 0.0
                    position_sparsity = float(1 - (n_valid_top / TOP_PCT)) if TOP_PCT > 0 else 0.0
                except Exception:
                    pass  # 降级为默认值 0.0

                logger.log_fail(
                    f"回测数值异常（组合归零或溢出）annual_ret={bt_result.annual_return:.4f}，跳过评分"
                )
                logger.log_fail(
                    f"退化诊断: 覆盖率={diag_coverage:.1%}, "
                    f"top_decile_avg_ADV={top_adv_wan:.0f}万元, "
                    f"持仓稀疏度={position_sparsity:.2%}"
                )
                round_steps.append("backtest_degenerate")
                bt_result = None
                backtest_degenerate = True  # 强制后续判定为 fail
            else:
                backtest_degenerate = False
        except Exception as e:
            logger.log_fail(f"回测异常: {e}")
            round_steps.append("backtest_skipped")
            # P0-2 fix: 不重置 backtest_degenerate —
            # 如果异常发生在退化解检测之后，退化解标记应保持 True

        # Phase 3b: 稳健性检验
        if bt_result is not None:
            try:
                robustness_result = robustness_evaluate(
                    fv_aligned, cls_slice if bt_result is not None else close_df_full.loc[common_dates, common_stocks],
                    layer_returns=bt_result.layer_returns,
                )
                round_steps.append("robustness")
            except Exception as e:
                logger.log_fail(f"稳健性检验异常: {e}")
                round_steps.append("robustness_skipped")
        else:
            round_steps.append("robustness_skipped")

        # Phase 3b: 合并结果（10 维）
        if bt_result is not None and robustness_result is not None:
            final_result = merge_results(score_result, bt_result, robustness_result)
        else:
            # 降级：backtest 或 robustness 失败时，回退到仅 score 判断
            final_result = FinalResult()
            dims = getattr(score_result, "dimensions", {})
            final_result.ic = dims.get("ic", {}).get("value", 0.0)
            final_result.ir = dims.get("ir", {}).get("value", 0.0)
            final_result.score_threshold_passed = score_result.passed_threshold
            final_result.threshold_passed = score_result.passed_threshold  # 无稳健性时不额外过滤
            if bt_result:
                final_result.sharpe_ratio = bt_result.sharpe_ratio
                final_result.avg_impact_cost_bps = bt_result.avg_impact_cost_bps

        # 多样性门控
        gate_passed, gate_dup_id, gate_max_corr = check_diversity(fv_aligned, factor_pool)
        logger.log_diversity(gate_passed, gate_max_corr)

        # Phase 3b: 综合判定（使用 final_result.threshold_passed）
        threshold_ok = (
            final_result.threshold_passed if final_result else score_result.passed_threshold
        ) and not backtest_degenerate  # Phase 3g: 回测异常强制 fail
        diversity_ok = gate_passed

        # 稳健性状态符号（用于日志）
        if robustness_result is not None:
            robust_status = "✓" if robustness_result.robust_core_passed else "✗"
        else:
            robust_status = "-"

        if not threshold_ok:
            round_status = "fail"
            fail_parts = list(score_result.failed_reasons)
            if robustness_result is not None and not robustness_result.robust_core_passed:
                if not robustness_result.monotonicity_passed:
                    fail_parts.append(f"单调性×")
                if not robustness_result.oos_stability_passed:
                    yp = getattr(robustness_result, "oos_yearly_pass_count", 0)
                    yt = getattr(robustness_result, "yearly_validation_observed", []) or []
                    if yp > 0:
                        fail_parts.append(f"样本外× (分期通过 {yp}/7, 未达标)")
                    else:
                        fail_parts.append(f"样本外× (分期通过 0/7)")
                if not robustness_result.ic_decay_passed:
                    fail_parts.append(f"IC衰减×")
            # Phase 3g: 回测异常原因
            if backtest_degenerate:
                fail_parts.append("回测数值异常（组合归零或溢出）")
            # Phase 3e: 追加 code_pattern_hint（结构性诊断反馈）
            if score_result.code_pattern_hint:
                fail_parts.append(score_result.code_pattern_hint)
            round_fail_reason = "\n".join(fail_parts)
            logger.log_fail(round_fail_reason)
        elif not diversity_ok:
            round_status = "skip"
            # Phase 3h: diversity gate 正面引导（B2）— 拒绝后推荐低相关方向
            diversity_guide = (
                f"多样性门控: 与 {gate_dup_id} 重复\n"
                f"[DIVERSITY-GUIDE] 你的因子与已有因子高度相关，请换一个完全不同的数学结构。\n"
                f"  推荐低相关方向（已验证 pairwise |ρ| < 0.07）:\n"
                f"  • T4 日内路径效率 —— (close-open)/(high-low)，日内有向性\n"
                f"  • T3 多周期共振 —— 短窗口信号 × 长趋势方向符号\n"
                f"  • T1 纯时序反转 —— close/shift(60)-1，完全 per-stock 时序操作\n"
                f"  禁止：三维乘积(ret×amp×vol) + 截面 rank(pct=True)（与入库因子 ρ > 0.5）"
            )
            round_fail_reason = diversity_guide
            logger.log_skip(round_fail_reason)
        else:
            round_status = "inbound"
            # Phase 3i P1-3: 记录模板成功入库，用于加权轮换
            _record_template_success(template_dir.get("_key", ""))
            # 入库 SQLite factors 表（含 Phase 3b 稳健性字段）
            inbound_counter += 1
            factor_id = f"f{inbound_counter:03d}"
            if not dry_run:
                factor_record = {
                    "factor_id": factor_id,
                    "round": round_num,
                    "direction_tag": round_direction,
                    "inbound_date": datetime.now().strftime("%Y-%m-%d"),
                    "code_snapshot": factor_code,
                    "natural_summary": round_summary,
                    "metrics": {dim: d["value"] for dim, d in score_result.dimensions.items()},
                    "score_total": score_result.total_score,
                    "cost_adjusted_score": getattr(score_result, "cost_adjusted_score", score_result.total_score),
                    "rank_snapshot": {},
                }
                # 追加 Phase 3b 稳健性字段
                if robustness_result is not None:
                    factor_record.update({
                        "monotonicity": robustness_result.monotonicity,
                        "monotonicity_passed": robustness_result.monotonicity_passed,
                        "oos_ic_train": robustness_result.oos_ic_train,
                        "oos_ic_test": robustness_result.oos_ic_test,
                        "oos_stability_passed": robustness_result.oos_stability_passed,
                        "ic_decay_ratio": robustness_result.ic_decay_ratio,
                        "ic_decay_passed": robustness_result.ic_decay_passed,
                        "yearly_validation_passed": robustness_result.yearly_validation_passed,
                        "yearly_observed": robustness_result.yearly_validation_observed,
                    })
                if final_result is not None:
                    factor_record.update({
                        "monotonicity": final_result.monotonicity,
                        "monotonicity_passed": final_result.monotonicity_passed,
                        "oos_ic_train": final_result.oos_ic_train,
                        "oos_ic_test": final_result.oos_ic_test,
                        "oos_stability_passed": final_result.oos_stability_passed,
                        "ic_decay_ratio": final_result.ic_decay_ratio,
                        "ic_decay_passed": final_result.ic_decay_passed,
                    })
                insert_factor(conn, factor_record)

                # 入库 backtests 表（含 Phase 3b 冲击成本字段）
                if bt_result:
                    insert_backtest(conn, {
                        "factor_id": factor_id,
                        "annual_return": bt_result.annual_return,
                        "max_drawdown": bt_result.max_drawdown,
                        "sharpe_ratio": bt_result.sharpe_ratio,
                        "win_rate": bt_result.win_rate,
                        "turnover_est": bt_result.turnover_estimate,
                        "avg_impact_cost_bps": bt_result.avg_impact_cost_bps,
                        "total_cost_annual": bt_result.total_cost_annual,
                        "layer_returns": bt_result.layer_returns,
                    })
                # 保存因子池 JSON（兼容 Phase 2a）
                pool = factor_pool or {"factors": []}
                pool["factors"].append({
                    "factor_id": factor_id, "round": round_num,
                    "inbound_date": datetime.now().strftime("%Y-%m-%d"),
                    "code_snapshot": factor_code, "natural_summary": round_summary,
                    "metrics": {dim: d["value"] for dim, d in score_result.dimensions.items()},
                    "score_total": score_result.total_score,
                    "rank_snapshot": _rank_values(fv_aligned),
                })
                save_factor_pool(pool)

            logger.log_inbound(factor_id, round_direction)

        round_steps.append("score_and_gate")

        # ── ⑨ 报告 + 记忆 ──────────────────────────────
        try:
            if not dry_run:
                report = engine.generate_report({
                    "round": round_num, "passed": (round_status == "inbound"),
                    "summary": round_summary, "score": round(score_result.total_score, 4),
                    "reason": round_fail_reason,
                })
            else:
                report = "[Mock] 批量模式报告"
        except Exception as e:
            report = f"(报告失败: {e})"

        # M1 修复: 使用本轮增量，非历史累计
        if not dry_run:
            round_cost = engine.cost_tracker.cost() - round_start_cost

        # 写入 SQLite memory 表
        if not dry_run:
            insert_memory(conn, {
                "round_id": round_num, "batch_run_id": run_id,
                "direction_tag": round_direction,
                "factor_type": round_direction,
                "summary": round_summary,
                "passed": (round_status == "inbound"),
                "fail_reasons": round_fail_reason,
                "suggestion": report[:500] if report else "",
            })

        round_steps.append("report_and_memory")

        # ── 里程碑事件检查 ────────────────────────────
        if not dry_run:
            cumulative_cost = engine.cost_tracker.cost()
            _check_milestone_events(
                conn, logger, run_id, round_num, cumulative_cost,
                program_md5, sharpe_seen, cost_seen,
            )

        # ── ⑩ Git commit ──────────────────────────────
        if not dry_run and round_status == "inbound":
            try:
                subprocess.run(
                    ["git", "add", "factor_draft.py", "program.md", "factorlab.db", "factor_pool.json"],
                    cwd=str(PROJECT_DIR), check=True, capture_output=True, timeout=10,
                )
                subprocess.run(
                    ["git", "commit", "-m", f"Round {round_num}: {round_summary[:60]}"],
                    cwd=str(PROJECT_DIR), check=True, capture_output=True, timeout=10,
                )
                round_steps.append("git_commit")
            except Exception:
                round_steps.append("git_failed")

        # ── 写入 rounds 表 ────────────────────────────
        _finalize_round(conn, logger, round_num, run_id, round_direction, round_status,
                      round_factor_code, round_summary, round_steps, round_fail_reason,
                      round_cost, cumulative_cost, dry_run)

        # ── 更新计数器 ─────────────────────────────────
        if round_status == "inbound":
            result.inbound_count += 1
            consecutive_fuse_counter = 0
        elif round_status == "skip":
            result.skip_count += 1
            consecutive_fuse_counter += 1
        elif round_status == "error":
            result.error_count += 1
            consecutive_fuse_counter += 1
        else:
            result.fail_count += 1
            consecutive_fuse_counter += 1

        # 更新 batch_status（C1 修复: 使用累计成本而非单轮成本）
        if not dry_run:
            update_batch(conn, run_id, completed_rounds=round_num, cumulative_cost=cumulative_cost)

        logger.log_round_end(round_num, round_status, round_cost, cumulative_cost)

        # ── 每 10 轮 CLI 辅助展示 ───────────────────
        if round_num % 10 == 0 and round_num < batch_size:
            summary = get_batch_round_summary(conn, run_id) if not dry_run else {
                "total": round_num, "inbound": result.inbound_count,
                "fail": result.fail_count, "skip": result.skip_count,
                "error": result.error_count,
            }
            status_line = _format_summary_line(summary)
            cost_line = f"累计成本: ¥{cumulative_cost:.2f} / ¥6.00"
            log_line = f"日志: {logger.filepath}" if logger.filepath else ""

            print(f"\n  ┌{'─'*58}┐")
            print(f"  │ {status_line:<56s} │")
            print(f"  │ {cost_line:<56s} │")
            if log_line:
                print(f"  │ {log_line:<56s} │")
            print(f"  └{'─'*58}┘")
            print(f"  [按 Enter 提前终止，或等待 10 秒自动继续...]")

            # m6 修复: signal.alarm 仅在 UNIX 可用，Windows 用 select fallback
            try:
                import signal as _signal
                _signal.signal(_signal.SIGALRM, lambda *_: None)
                _signal.alarm(10)
                input()
                _signal.alarm(0)
                user_input = input("  确认提前终止？(y/N): ").strip().lower() if not dry_run else "n"
                if user_input == "y":
                    termination_reason = "user_interrupt"
                    logger.log_event("terminate", "用户提前终止")
                    break
            except (AttributeError, NameError):
                # Windows: signal.alarm 不存在，使用 select 实现超时
                import select
                import sys as _sys
                print("  (等待 10 秒，按 Enter 可提前终止...)")
                ready, _, _ = select.select([_sys.stdin], [], [], 10)
                if ready:
                    _sys.stdin.readline()
                    user_input = input("  确认提前终止？(y/N): ").strip().lower() if not dry_run else "n"
                    if user_input == "y":
                        termination_reason = "user_interrupt"
                        logger.log_event("terminate", "用户提前终止")
                        break
            except (EOFError, KeyboardInterrupt):
                pass

        # ── 中期熔断检查 ────────────────────────────────
        if consecutive_fuse_counter >= 10 and consecutive_errors < 3:
            termination_reason = "mid_fuse"
            logger.log_event("fuse", "连续10轮无入库，中期熔断触发")

            # M2 修复: 写入 program.md 里程碑
            if not dry_run:
                _summary = get_batch_round_summary(conn, run_id)
                _m_content = (
                    f"触发原因：连续 10 轮无因子入库（最近 10 轮 status 均为 fail 或 skip）\n"
                    f"当前状态：入库 {_summary['inbound']} / 失败 {_summary['fail']} / "
                    f"跳过 {_summary['skip']} / 累计成本 ¥{cumulative_cost:.4f}\n"
                    f"关键发现：当前研究方向可能不适合市场环境\n"
                    f"下一步建议：审核 program_draft.md 调整方向，或放宽评分阈值"
                )
                try:
                    append_milestone(str(PROGRAM_PATH), "中期熔断", _m_content, program_md5)
                except Exception:
                    logger.log_summary("中期熔断里程碑写入失败（program.md MD5 不匹配）")

            print(f"\n  ╔{'═'*58}╗")
            print(f"  ║  [中期熔断] 连续 10 轮无因子入库                          ║")
            print(f"  ║  建议检查 program.md 研究方向或调整评分阈值               ║")
            print(f"  ╠{'═'*58}╣")
            print(f"  ║  [1] 继续跑完剩余轮次                                    ║")
            print(f"  ║  [2] 更新方向后继续:                                     ║")
            print(f"  ║    [2a] 采纳 program_draft.md 并继续                    ║")
            print(f"  ║    [2b] 手动编辑 program.md 后继续                      ║")
            print(f"  ║  [3] 退出保存状态                                        ║")
            print(f"  ╚{'═'*58}╝")

            if dry_run:
                print("  [Dry-run] 自动选择 [1] 继续")
            else:
                try:
                    choice = input("  请选择 (1/2a/2b/3): ").strip()
                except EOFError:
                    print("  [非交互模式] 自动选择 [1] 继续")
                    choice = "1"
                if choice == "2a":
                    # M6 修复: 采纳 program_draft.md 并热加载
                    from program_updater import apply_adopt
                    draft_path = PROJECT_DIR / "program_draft.md"
                    if draft_path.exists():
                        apply_adopt(str(draft_path), str(PROGRAM_PATH))
                        program_md5 = _compute_chapters_md5()
                        try:
                            update_batch(conn, run_id, program_md5=program_md5)
                        except Exception:
                            pass
                        logger.log_summary("已采纳 program_draft.md，新方向立即生效")
                    else:
                        logger.log_summary("program_draft.md 不存在，无法采纳")
                    consecutive_fuse_counter = 0
                elif choice == "2b":
                    # 手动编辑 program.md
                    import subprocess as _sp
                    try:
                        _sp.call(["code", str(PROGRAM_PATH)])
                    except FileNotFoundError:
                        print(f"  请手动编辑: {PROGRAM_PATH}")
                    try:
                        input("  修改完成后按 Enter 继续...")
                    except EOFError:
                        pass  # 非交互模式，跳过等待
                    program_md5 = _compute_chapters_md5()
                    try:
                        update_batch(conn, run_id, program_md5=program_md5)
                    except Exception:
                        pass
                    logger.log_summary("program.md 已手动更新，熔断计数器重置")
                    consecutive_fuse_counter = 0
                elif choice == "2":
                    # 向后兼容: 重新加载 program.md
                    program_md5 = _compute_chapters_md5()
                    update_batch(conn, run_id, program_md5=program_md5)
                    consecutive_fuse_counter = 0
                    logger.log_summary("program.md 已刷新，熔断计数器重置")
                elif choice == "3":
                    update_batch(conn, run_id, status="paused")
                    termination_reason = "mid_fuse"
                    break
                # choice "1": 继续
                else:
                    consecutive_fuse_counter = 0  # 重置，继续跑
            consecutive_fuse_counter = 0  # 重置，用户已介入

        # ── Error 熔断检查 ──────────────────────────────
        if consecutive_errors >= 3:
            termination_reason = "error_fuse"
            print(f"\n  ╔{'═'*58}╗")
            print(f"  ║  [Error 熔断] 连续 3 轮 API 网络故障                    ║")
            print(f"  ║  建议检查 API 额度/网络连接                              ║")
            print(f"  ╠{'═'*58}╣")
            print(f"  ║  [1] 重试这 3 轮                                        ║")
            print(f"  ║  [2] 退出保存状态                                        ║")
            print(f"  ╚{'═'*58}╝")

            if dry_run:
                print("  [Dry-run] 自动选择 [2] 退出")
            else:
                try:
                    choice = input("  请选择 (1-2): ").strip()
                except EOFError:
                    print("  [非交互模式] 自动选择 [2] 退出保存")
                    choice = "2"
                if choice == "1":
                    consecutive_errors = 0
                    logger.log_summary("用户选择重试，error 计数器重置")
                else:
                    update_batch(conn, run_id, status="paused")
            if not dry_run and choice != "1":
                break

    # ── 循环结束 ────────────────────────────────────────

    result.total_rounds = round_num
    result.cumulative_cost = cumulative_cost
    result.termination_reason = termination_reason

    if not dry_run:
        update_batch(conn, run_id, status="completed", completed_rounds=round_num, cumulative_cost=cumulative_cost)

    # 50 轮正常完成 → Event 3: 生成总结报告
    if termination_reason == "normal" and not dry_run:
        logger.log_event("milestone", "50轮批量完成，生成总结报告")
        # 事件 3 的 program.md 追加（如果是正常完成）
        summary = get_batch_round_summary(conn, run_id)
        append_milestone(
            str(PROGRAM_PATH), "50轮结束总结",
            f"触发原因：50 轮批量正常完成\n"
            f"当前状态：入库 {summary['inbound']} / 失败 {summary['fail']} / "
            f"跳过 {summary['skip']} / 累计成本 ¥{cumulative_cost:.4f}\n"
            f"关键发现：见总结报告\n"
            f"下一步建议：人工审查 program_draft.md 调整方向",
            program_md5,
        )

        # ── ⑫ 生成完整版 HTML 报告（Phase 3c 新增）──────
        try:
            from html_reporter import generate_full_report
            from combo_engine import build_all_inbound
            from config import REPORTS_DIR

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(REPORTS_DIR) / f"factorlab_full_report_{timestamp}.html"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            # 复用 batch_pipeline 已加载的数据矩阵（避免重复加载）
            combo_result = build_all_inbound(
                conn, df_1800, close_df_full, volume_df_full, returns_df_full,
            )

            # 生成完整报告
            generate_full_report(conn, str(report_path), combo_result)

            logger.log_summary(f"完整版 HTML 报告已生成: {report_path}")
            print(f"\n📊 报告已保存: {report_path}")
        except Exception as e:
            logger.log_summary(f"HTML 报告生成失败: {e}")
            print(f"\n⚠️ 报告生成失败: {e}")

    logger.close()
    conn.close()

    return result


def _finalize_round(conn, logger, round_num, run_id, direction_tag, status,
                    factor_code, summary, steps, fail_reason, cost, cumulative, dry_run):
    """辅助：写入 rounds 表 + 日志。"""
    if not dry_run:
        insert_round(conn, {
            "round_id": round_num, "batch_run_id": run_id,
            "direction_tag": direction_tag, "status": status,
            "started_at": datetime.now().isoformat(),
            "fail_reason": fail_reason, "api_cost": cost,
            "factor_code": factor_code, "summary": summary,
            "steps": steps,
        })


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== batch_pipeline.py 自检 (Dry-run 3 轮) ===\n")

    result = run_batch(batch_size=3, resume=False, dry_run=True, max_stocks=20)

    print(f"\n{'='*60}")
    print(f"  Batch Result:")
    print(f"    total_rounds:     {result.total_rounds}")
    print(f"    inbound_count:    {result.inbound_count}")
    print(f"    fail_count:       {result.fail_count}")
    print(f"    skip_count:       {result.skip_count}")
    print(f"    error_count:      {result.error_count}")
    print(f"    cumulative_cost:  {result.cumulative_cost:.4f}")
    print(f"    termination:      {result.termination_reason}")
    print(f"{'='*60}")

    print("\n自检通过.")

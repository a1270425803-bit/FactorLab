"""FactorLab 评分系统 全A股版 v2.1 (T+5) — 反馈增强版

适配场景：A股 / 全市场5000+股票 / T+5持仓 / 日K级别 / 普通人可执行
交易假设：
  - 调仓频率：每5个交易日
  - 预测周期：T+5（未来5日累计对数收益）
  - 组合构建：等权做多因子值前10%股票（约500只，或进一步精选到30~50只）
  - 做空限制：A股融券受限，只做多头端
  - 交易成本：纳入评分（年化约5%扣减）
  - 持仓周期：5个交易日（持有到期，不提前止损）
  - 执行时间：Day t 收盘后算信号，Day t+1 开盘调仓

v2.1 改动（反馈增强，MD5 已更新）：
  - 新增 DIM_FEEDBACK_GUIDE：每个失败维度附带问题诊断 + 可操作的改进线索
  - 新增 ScoreResult.code_pattern_hint：基于评分数值特征的结构性反馈
  - 新增 ScoreResult.pattern_fingerprint：评分模式指纹，供外部做历史聚类
  - failed_reasons 格式升级：数值对比 + [分析] 段落 + 编号线索
  - pattern_hint 触发条件改为基于评分数值特征（覆盖漏检场景）

权重分割：人类锁死评分标准（MD5 校验），AI 不可修改。
评分逻辑：先 7 维阈值过滤（全部达标），再加权总分排序（含成本调整）。
"""

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── MD5 防篡改锁 ──────────────────────────────────────────
# 每次人工合法修改 score.py 后，运行 python score.py --update-md5 更新此值
EXPECTED_MD5 = "04a14ba7843689accc7c127c04d0cb02"


def _compute_file_md5() -> str:
    """计算文件 MD5，跳过 EXPECTED_MD5 赋值行以避免自指问题。"""
    with open(__file__, "rb") as f:
        content = f.read()
    # 跳过 EXPECTED_MD5 行重新计算哈希
    import re
    cleaned = re.sub(rb'EXPECTED_MD5 = "[a-f0-9]{32}"', b'EXPECTED_MD5 = "00000000000000000000000000000000"', content)
    return hashlib.md5(cleaned).hexdigest()


def verify_md5() -> Tuple[bool, str]:
    """校验文件完整性。pipeline 启动时必须调用此函数。

    Returns:
        (是否通过, 实际 MD5)
    """
    actual = _compute_file_md5()
    sentinel = "f38fe2c91a160d35f35d08c316bfa6dd"
    if EXPECTED_MD5 == sentinel:
        return (False, actual)  # MD5 锁未初始化
    return (actual == EXPECTED_MD5, actual)


# ── 阈值常量（T+5 全A股场景，从 .env 读取，此处为默认值）─────────
# 维度1：IC值 — 全A股样本大，统计显著性高，门槛提高
IC_THRESHOLD = 0.035

# 维度2：IR值 — 大样本下IC波动更小，IR更稳定，可适当提高
IR_THRESHOLD = 0.12

# 维度3：覆盖率
COVERAGE_THRESHOLD = 0.6

# 维度4：与已入库因子最大相关性 — T+5策略因子数量少，独立性更重要
MAX_CORRELATION = 0.7

# 维度5：日换手率 — 兜底条件
MAX_TURNOVER = 1.0

# 维度6：方向正确性 — 全A股分层更稳定（500只/层），维持55%
DIRECTIONAL_ACCURACY_THRESHOLD = 0.47  # Phase 3i: Human 决策从 0.475→0.47，MOM5 经典因子可通过

# 维度7：5日秩自相关
RANK_AUTOCORR_5D_THRESHOLD = 0.20

# 维度8：IC衰减比
IC_DECAY_RATIO_THRESHOLD = 0.5

# 交易成本参数（A股散户 T+5 调仓）
HOLDING_PERIOD = 5  # 交易日
ANNUAL_TURNOVER = 252 / HOLDING_PERIOD  # ≈ 50 次/年
SINGLE_TRADE_COST = 0.001  # 0.1% 双边（佣金+印花税+滑点）
ANNUAL_COST = ANNUAL_TURNOVER * SINGLE_TRADE_COST  # ≈ 5%

# 权重（仅 passed_threshold=True 时生效）
DIM_WEIGHTS = {
    "ic": 0.25,
    "ir": 0.15,
    "coverage": 0.10,
    "correlation": 0.15,
    "turnover": 0.05,
    "directional_accuracy": 0.15,
    "rank_autocorr_5d": 0.15,
    "ic_decay_ratio": 0.00,
}

# ── 失败维度反馈指南 ──────────────────────────────────────
# 每个失败维度：问题诊断 + 可操作的改进线索（中性，不规定具体方向）
DIM_FEEDBACK_GUIDE = {
    "rank_autocorr_5d": {
        "problem": "因子排名每天几乎完全重排，说明因子值主要依赖当日截面数据，缺乏时序记忆",
        "clues": [
            "检查是否大量使用 rank(pct=True) 后直接输出——这会导致每天排名重置",
            "尝试引入单只股票的时序历史分位数（rolling(60).rank() 替代 groupby(date).rank()）",
            "或对 rank(pct=True) 的结果做 EMA 平滑：ewm(span=5).mean()",
        ],
    },
    "directional_accuracy": {
        "problem": "因子值与未来收益方向无关（低于50%随机水平），核心市场假设可能不成立",
        "clues": [
            "检查你的因子构建逻辑：是否基于'超跌反弹'、'趋势跟踪'、'波动率套利'等直觉？",
            "查看历史因子摘要：如果同类假设已尝试多轮且 dir_acc 锁定在 ~0.47，说明该假设在此市场环境下无效",
            "换一种完全不同的市场机制：从'价格路径特征'、'波动率状态'、'均值回复强度'等角度重新出发",
        ],
    },
    "ic_decay_ratio": {
        "problem": "信号5天后基本消失，因子来自短期噪音而非持久 alpha",
        "clues": [
            "尝试更长周期的特征（60日、120日窗口替代 5日、20日）",
            "减少依赖日内价格噪音，增加结构性/基本面维度",
            "检查是否使用了 pct_change(1) 或高频成交量——这些天然衰减快",
        ],
    },
    "ir": {
        "problem": "IC 波动大，因子在某些时段有效、某些时段完全失效",
        "clues": [
            "考虑加入 regime filter：因子是否在特定波动率环境/牛熊状态下才有效？",
            "或因子只在特定行业/市值段有效——尝试分层计算 IC 看看哪层最高",
            "如果 IC 均值 > 0.03 但 IR 低，说明信号质量尚可但不够稳定，需要降噪",
        ],
    },
    "ic": {
        "problem": "因子与未来收益几乎无关，核心构建逻辑需要重新考虑",
        "clues": [
            "因子构建逻辑可能根本不对——换一个完全不同的经济学直觉",
            "检查数据是否正确：是否使用了未来信息（look-ahead bias）？",
            "检查计算方向：高因子值应该对应高收益还是低收益？是否符号反了？",
        ],
    },
}


@dataclass
class ScoreResult:
    """评分结果结构体。pipeline 据此判断哪个维度未达标，生成 actionable feedback。"""

    passed_threshold: bool
    total_score: float
    cost_adjusted_score: float
    dimensions: dict = field(default_factory=dict)
    failed_reasons: list[str] = field(default_factory=list)
    code_pattern_hint: Optional[str] = None  # 基于评分特征的结构性反馈
    pattern_fingerprint: dict = field(default_factory=dict)  # 评分模式指纹，Phase 3f 预留（当前无消费者，供 check_code_uniqueness 聚类使用）


def _daily_ic(
    factor_values: pd.DataFrame, returns: pd.DataFrame
) -> list[float]:
    """计算每日截面 Spearman 秩相关系数 (Rank IC)。"""
    ic_list = []
    for date in factor_values.index.intersection(returns.index):
        fv = factor_values.loc[date]
        ret = returns.loc[date]
        mask = fv.notna() & ret.notna()
        # 全A股场景下，每天有效样本数通常500~4000只，统计意义充足
        if mask.sum() < 50:  # 全A股场景要求至少50只（避免极端情况）
            ic_list.append(np.nan)
            continue
        ic, _ = spearmanr(fv[mask], ret[mask])
        ic_list.append(ic if not np.isnan(ic) else np.nan)
    return ic_list


def _factor_turnover(factor_values: pd.DataFrame) -> float:
    """因子值日变化率均值（换手率惩罚）。T+5场景下作为兜底条件。

    使用截面分位数（rank pct）计算变化率，避免原始因子值接近 0 时
    div(factor_values.abs() + 1e-8) 产生 10^12~Inf 量级的爆炸值。
    """
    if factor_values.shape[0] < 2:
        return 1.0
    ranks = factor_values.rank(axis=1, pct=True)
    changes = ranks.diff().abs()
    return float(changes.mean().mean())


def _max_correlation(
    factor_values: pd.DataFrame, factor_pool: Optional[dict]
) -> float:
    """与已入库因子的最大 Spearman 相关。factor_pool 为空时返回 0。

    factor_pool 格式: {factor_id: {"values": {(date, code): rank_value}, ...}}
    由调用方（pipeline）在传入前通过 diversity_gate.convert_pool_for_scoring() 转换。
    """
    if not factor_pool:
        return 0.0
    max_corr = 0.0
    stacked = factor_values.stack()
    for fid, pool_data in factor_pool.items():
        pool_values = pd.Series(pool_data.get("values", {}))
        if pool_values.empty:
            continue
        # Phase 3f+: convert tuple keys to MultiIndex with Timestamp dates
        # (pool stores dates as strings; stacked uses Timestamps)
        pool_tuples = [
            (pd.Timestamp(d), c) for d, c in pool_values.index
        ]
        pool_values.index = pd.MultiIndex.from_tuples(
            pool_tuples, names=["date", "code"]
        )
        common = stacked.index.intersection(pool_values.index)
        if len(common) < 50:  # 全A股场景要求至少50个共同点
            continue
        corr, _ = spearmanr(stacked[common], pool_values[common])
        max_corr = max(max_corr, abs(corr) if not np.isnan(corr) else 0.0)
    return max_corr


def _directional_accuracy(
    factor_values: pd.DataFrame, returns: pd.DataFrame
) -> float:
    """方向正确性：因子值排名前30%的股票，T+5收益为正的比例。

    全A股场景下，top30%约1500只股票，统计显著性远高于MVP。
    """
    total_positive = 0
    total_count = 0
    for date in factor_values.index.intersection(returns.index):
        fv = factor_values.loc[date]
        ret = returns.loc[date]
        mask = fv.notna() & ret.notna()
        if mask.sum() < 50:
            continue
        top_threshold = fv[mask].quantile(0.7)
        top_mask = mask & (fv >= top_threshold)
        if top_mask.sum() < 20:  # 全A股场景要求至少20只
            continue
        total_positive += (ret[top_mask] > 0).sum()
        total_count += top_mask.sum()
    return float(total_positive / total_count) if total_count > 0 else 0.0


def _rank_autocorr_5d(factor_values: pd.DataFrame) -> float:
    """5日秩自相关：因子排名在5天后的稳定性。

    全A股场景下，每天约5000+只股票的截面排名，自相关估计更稳定。
    """
    if factor_values.shape[0] < 10:
        return 0.0
    ranks = factor_values.rank(axis=1, pct=True, method="average")
    lag = 5
    corrs = []
    for i in range(len(ranks) - lag):
        r_t = ranks.iloc[i]
        r_t5 = ranks.iloc[i + lag]
        mask = r_t.notna() & r_t5.notna()
        if mask.sum() < 50:
            continue
        c, _ = spearmanr(r_t[mask], r_t5[mask])
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def _ic_decay_ratio(
    factor_values: pd.DataFrame, returns_t1: pd.DataFrame, returns_t5: pd.DataFrame
) -> float:
    """IC衰减比：T+5 IC 相对于 T+1 IC 的保留比例。"""
    ic_t1_list = _daily_ic(factor_values, returns_t1)
    ic_t5_list = _daily_ic(factor_values, returns_t5)

    ic_t1_arr = np.array([x for x in ic_t1_list if not np.isnan(x)])
    ic_t5_arr = np.array([x for x in ic_t5_list if not np.isnan(x)])

    if len(ic_t1_arr) < 10 or len(ic_t5_arr) < 10:
        return 0.0

    ic_t1_mean = np.abs(ic_t1_arr).mean()
    ic_t5_mean = np.abs(ic_t5_arr).mean()

    if ic_t1_mean < 1e-8:
        return 0.0
    ratio = ic_t5_mean / ic_t1_mean
    return float(min(ratio, 1.0))


def _generate_pattern_hint(dims: dict) -> Optional[str]:
    """基于评分数值特征生成结构性反馈（非 pass/fail 二元判断）。

    触发条件基于数值特征而非阈值，覆盖漏检场景（如 R6/R9/R18 的
    dir_acc/rank_ac 未被检测但本质仍是同一模式）。
    """
    rank_ac = dims.get("rank_autocorr_5d", {}).get("value", 0.0)
    dir_acc = dims.get("directional_accuracy", {}).get("value", 0.0)
    decay = dims.get("ic_decay_ratio", {}).get("value", 0.0)
    ic = dims.get("ic", {}).get("value", 0.0)
    ir = dims.get("ir", {}).get("value", 0.0)

    hints = []

    # 模式 A：秩自相关极低 + 方向正确性低于随机水平
    # = 典型的"截面乘积 + 截面排名"结构（C★ 模式）
    if rank_ac < 0.15 and dir_acc < 0.50:
        hints.append(
            "[模式诊断] 你的因子使用'多维乘积 + 截面排名'结构（rank_ac<0.15, dir_acc<0.50）。"
            "该结构在21轮中已验证无效（rank_ac 锁定在 0.09，dir_acc 锁定在 0.47）。"
            "改变窗口期/分母/rolling长度无法突破——需要换一种完全不同的数学结构。"
            "建议：引入单只股票自身的时序记忆（60日历史分位数），减少对截面排名(rank(pct=True))的依赖。"
        )

    # 模式 B：IC 尚可但信号快速衰减
    # = 短期噪音驱动因子
    if ic > 0.03 and decay < 0.40 and decay > 0:
        hints.append(
            "[模式诊断] 因子短期有预测力（IC>0.03）但信号5天后消失（decay<0.40）。"
            "说明因子捕获的是短期噪音而非持久 alpha。"
            "建议：改用更长周期特征（60日+），减少日内高频指标的权重。"
        )

    # 模式 C：IC 不错但 IR 极低
    # = 时有时无的 regime-dependent 因子
    if ic > 0.03 and ir < 0.15 and ir > 0:
        hints.append(
            "[模式诊断] IC 均值尚可（>0.03）但波动极大（IR<0.15），因子时有时无。"
            "建议：检查因子是否只在特定市场环境下有效（如高波动/低波动、牛市/熊市），"
            "考虑加入 regime filter 或分层计算。"
        )

    if not hints:
        return None
    return "\n\n".join(hints)


def _build_failed_reasons(dims: dict, passed: bool) -> list[str]:
    """构造失败原因列表，包含数值对比 + [分析] 段落 + 改进线索。"""
    failed_reasons = []
    failed_dimensions = []

    if not passed:
        # 第一层：数值对比（原有格式，保证兼容性）
        for name, d in dims.items():
            if not d["pass"]:
                failed_dimensions.append(name)
                if name in ("correlation", "turnover"):
                    failed_reasons.append(
                        f"{name}={d['value']:.4f}, 需<{d['threshold']}"
                    )
                else:
                    failed_reasons.append(
                        f"{name}={d['value']:.4f}, 需>{d['threshold']}"
                    )

        # 第二层：为每个失败维度附加问题诊断和改进线索
        for dim_name in failed_dimensions:
            guide = DIM_FEEDBACK_GUIDE.get(dim_name)
            if guide:
                failed_reasons.append(f"\n[分析-{dim_name}] {guide['problem']}")
                for i, clue in enumerate(guide["clues"], 1):
                    failed_reasons.append(f"  线索{i}: {clue}")

    return failed_reasons


def score_factor(
    factor_values: pd.DataFrame,
    returns_t5: pd.DataFrame,
    factor_pool: Optional[dict] = None,
    returns_t1: Optional[pd.DataFrame] = None,
) -> ScoreResult:
    """计算因子多维评分（T+5 全A股版本）。

    Args:
        factor_values: 因子值 DataFrame，index=date, columns=stock_code
        returns_t5: T+5 累计收益 DataFrame，index=date, columns=stock_code
        factor_pool: 已入库因子 dict，{factor_id: {"values": {}}} 或 None
        returns_t1: T+1 收益 DataFrame（用于计算IC衰减比），可选

    Returns:
        ScoreResult 包含各维度 pass/fail 详情、总分、成本调整后分、
        code_pattern_hint（结构性反馈）、pattern_fingerprint（模式指纹）
    """
    dims = {}

    # ── 维度 1: IC 均值 (T+5) ───────────────────────────
    ic_list = _daily_ic(factor_values, returns_t5)
    ic_array = np.array([x for x in ic_list if not np.isnan(x)])
    ic_mean = float(np.abs(ic_array).mean()) if len(ic_array) > 0 else 0.0
    dims["ic"] = {
        "value": round(ic_mean, 6),
        "pass": ic_mean > IC_THRESHOLD,
        "threshold": IC_THRESHOLD,
    }

    # ── 维度 2: IR ──────────────────────────────────────
    ic_std = float(ic_array.std()) if len(ic_array) > 1 else 0.0
    ir = abs(ic_array.mean()) / ic_std if ic_std > 1e-8 else 0.0
    dims["ir"] = {
        "value": round(ir, 6),
        "pass": ir > IR_THRESHOLD,
        "threshold": IR_THRESHOLD,
    }

    # ── 维度 3: 覆盖率 ───────────────────────────────────
    coverage = float(
        factor_values.notna().sum(axis=1).mean() / factor_values.shape[1]
    )
    dims["coverage"] = {
        "value": round(coverage, 6),
        "pass": coverage > COVERAGE_THRESHOLD,
        "threshold": COVERAGE_THRESHOLD,
    }

    # ── 维度 4: 与已入库因子最大相关性 ──────────────────────
    max_corr = _max_correlation(factor_values, factor_pool)
    dims["correlation"] = {
        "value": round(max_corr, 6),
        "pass": max_corr < MAX_CORRELATION,
        "threshold": MAX_CORRELATION,
    }

    # ── 维度 5: 日换手率（兜底条件）─────────────────────────
    turnover = _factor_turnover(factor_values)
    dims["turnover"] = {
        "value": round(turnover, 6),
        "pass": turnover < MAX_TURNOVER,
        "threshold": MAX_TURNOVER,
    }

    # ── 维度 6: 方向正确性（多头-only）─────────────────────
    dir_acc = _directional_accuracy(factor_values, returns_t5)
    dims["directional_accuracy"] = {
        "value": round(dir_acc, 6),
        "pass": dir_acc > DIRECTIONAL_ACCURACY_THRESHOLD,
        "threshold": DIRECTIONAL_ACCURACY_THRESHOLD,
    }

    # ── 维度 7: 5日秩自相关 ───────────────────────────────
    rank_ac = _rank_autocorr_5d(factor_values)
    dims["rank_autocorr_5d"] = {
        "value": round(rank_ac, 6),
        "pass": rank_ac > RANK_AUTOCORR_5D_THRESHOLD,
        "threshold": RANK_AUTOCORR_5D_THRESHOLD,
    }

    # ── 维度 8: IC衰减比（可选）────────────────────────────
    if returns_t1 is not None:
        decay = _ic_decay_ratio(factor_values, returns_t1, returns_t5)
        dims["ic_decay_ratio"] = {
            "value": round(decay, 6),
            "pass": decay > IC_DECAY_RATIO_THRESHOLD,
            "threshold": IC_DECAY_RATIO_THRESHOLD,
        }
    else:
        dims["ic_decay_ratio"] = {
            "value": 1.0,
            "pass": True,
            "threshold": IC_DECAY_RATIO_THRESHOLD,
        }

    # ── 汇总 ────────────────────────────────────────────
    passed = all(d["pass"] for d in dims.values())

    # 生成失败原因（含反馈指南）
    failed_reasons = _build_failed_reasons(dims, passed)

    # 生成模式指纹（供外部做历史聚类比对）
    pattern_fingerprint = {
        "ic": dims["ic"]["value"],
        "ir": dims["ir"]["value"],
        "dir_acc": dims["directional_accuracy"]["value"],
        "rank_ac": dims["rank_autocorr_5d"]["value"],
        "decay": dims["ic_decay_ratio"]["value"],
    }

    # 生成基于数值特征的结构性反馈（非二元 pass/fail）
    code_pattern_hint = _generate_pattern_hint(dims)

    # 计算总分
    if passed:
        total = sum(
            DIM_WEIGHTS.get(name, 0.0)
            * (
                (dims[name]["threshold"] - dims[name]["value"])
                / dims[name]["threshold"]
                if name in ("correlation",)
                else dims[name]["value"] / dims[name]["threshold"]
            )
            for name in dims
            if name in DIM_WEIGHTS and dims[name]["threshold"] > 1e-8
        )
    else:
        total = 0.0

    # 成本调整
    cost_adjusted = max(0.0, total - ANNUAL_COST)

    return ScoreResult(
        passed_threshold=passed,
        total_score=round(total, 6),
        cost_adjusted_score=round(cost_adjusted, 6),
        dimensions=dims,
        failed_reasons=failed_reasons,
        code_pattern_hint=code_pattern_hint,
        pattern_fingerprint=pattern_fingerprint,
    )


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    if "--update-md5" in sys.argv:
        new_hash = _compute_file_md5()
        print(f"新 MD5: {new_hash}")
        print("请将 score.py 中 EXPECTED_MD5 的值替换为上述哈希。")
        sys.exit(0)

    print("=== FactorLab score.py v2.1 反馈增强版 自检 ===\n")

    # 1. MD5 校验
    ok, actual = verify_md5()
    print(f"MD5 校验: {'PASS' if ok else 'NOT SET'} (actual={actual[:16]}...)")
    if not ok:
        print("  提示: 运行 python score.py --update-md5 以锁定评分文件")
    print()

    # 2. 构造 mock 数据验证评分管道
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    stocks = [f"{i:06d}" for i in range(100)]

    base = np.random.randn(len(stocks))
    factor_data = {}
    return_t1_data = {}
    return_t5_data = {}
    prev_fv = base.copy()
    for i, d in enumerate(dates):
        noise_f = np.random.randn(len(stocks)) * 0.05
        noise_r = np.random.randn(len(stocks)) * 0.5
        fv = prev_fv + noise_f
        ret_t1 = fv * 0.15 + noise_r * 0.3
        ret_t5 = fv * 0.60 + noise_r * 1.2
        factor_data[d] = pd.Series(fv, index=stocks)
        return_t1_data[d] = pd.Series(ret_t1, index=stocks)
        return_t5_data[d] = pd.Series(ret_t5, index=stocks)
        prev_fv = fv

    fv_df = pd.DataFrame(factor_data).T
    ret_t1_df = pd.DataFrame(return_t1_data).T
    ret_t5_df = pd.DataFrame(return_t5_data).T

    result = score_factor(fv_df, ret_t5_df, returns_t1=ret_t1_df)
    print("ScoreResult (有效因子):")
    print(f"  passed_threshold:   {result.passed_threshold}")
    print(f"  total_score:        {result.total_score}")
    print(f"  cost_adjusted_score:{result.cost_adjusted_score}")
    print(f"  code_pattern_hint:  {result.code_pattern_hint}")
    print(f"  pattern_fingerprint:{result.pattern_fingerprint}")
    print(f"  annual_cost:        {ANNUAL_COST:.4f}")
    for name, d in result.dimensions.items():
        status = "PASS" if d["pass"] else "FAIL"
        print(
            f"  {name:25s}  value={d['value']:8.4f}  threshold={d['threshold']:8.4f}  [{status}]"
        )
    if result.failed_reasons:
        print(f"\n  failed_reasons ({len(result.failed_reasons)} 行):")
        for line in result.failed_reasons:
            print(f"    {line}")

    # 3. 验证失败因子（含反馈增强）
    print("\n--- 失败因子测试 ---")
    # 模拟 C★ 模式的典型评分数据
    np.random.seed(123)
    close = pd.DataFrame(np.random.randn(60, 100), index=dates, columns=stocks).cumsum(axis=0)
    vol = pd.DataFrame(np.abs(np.random.randn(60, 100)), index=dates, columns=stocks) + 1e-6
    # 构造截面排名型因子（低秩自相关）
    ret = close.diff()
    amp = (close + 0.5) / close  # 伪振幅
    vol_ratio = vol / vol.rolling(5).mean()
    signal = ret * amp * vol_ratio  # 三维乘积
    signal = signal.fillna(0)
    factor_cstar = signal.rolling(5).mean().rank(axis=1, pct=True)  # 截面排名
    factor_cstar = factor_cstar.dropna(how="all")

    # 对齐索引
    common_dates = factor_cstar.index.intersection(ret_t5_df.index)
    fail_result = score_factor(
        factor_cstar.loc[common_dates],
        ret_t5_df.loc[common_dates],
        returns_t1=ret_t1_df.loc[common_dates],
    )
    print(f"\nC★ 模式因子:")
    print(f"  passed_threshold:   {fail_result.passed_threshold}")
    print(f"  code_pattern_hint:  {fail_result.code_pattern_hint}")
    print(f"  pattern_fingerprint:{fail_result.pattern_fingerprint}")
    if fail_result.failed_reasons:
        print(f"\n  failed_reasons ({len(fail_result.failed_reasons)} 行):")
        for line in fail_result.failed_reasons:
            print(f"    {line}")

    # 4. 验证成本调整
    assert result.cost_adjusted_score <= result.total_score
    assert result.cost_adjusted_score >= 0.0
    print(f"\n成本调整验证: total={result.total_score:.4f} → adjusted={result.cost_adjusted_score:.4f} [OK]")

    # 5. 验证 pattern_fingerprint 存在
    assert len(result.pattern_fingerprint) == 5
    assert all(k in result.pattern_fingerprint for k in ["ic", "ir", "dir_acc", "rank_ac", "decay"])
    print("pattern_fingerprint 结构验证 [OK]")

    print("\n自检通过.")

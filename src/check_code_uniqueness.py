#!/usr/bin/env python3
"""代码级相似度门禁 — AST 结构签名 + 数学语义标签双重检测。

Phase 3e 新增模块，在因子提交沙箱前检测与历史轮次的代码结构相似度。
与 diversity_gate.py 互补：diversity_gate 检测因子值相关性，本模块检测代码结构相似度。

用法:
  from check_code_uniqueness import check_code_uniqueness
  is_unique, feedback, metadata = check_code_uniqueness(new_code, conn)

红线:
  - 不修改 diversity_gate.py（因子值相关性检测）
  - 不引入 ast 以外的第三方依赖（python-Levenshtein 可选，不可用时回退 difflib）
  - 不做 continue 跳过——只负责检测和生成 feedback，由 batch_pipeline 决定如何使用
"""

import ast
import difflib
import functools
import json
import re
import sqlite3
from typing import Optional, Tuple


# ── 可选依赖：python-Levenshtein 加速，不可用时回退纯 Python ──
try:
    import Levenshtein as _lev

    def _edit_distance(s1: str, s2: str) -> int:
        return _lev.distance(s1, s2)

    _LEV_AVAILABLE = True
except ImportError:
    _LEV_AVAILABLE = False

    def _edit_distance(s1: str, s2: str) -> int:
        """纯 Python Wagner-Fischer 编辑距离（回退）。"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[-1]


# ════════════════════════════════════════════════════════════
# 1. AST 紧凑结构签名（替代 verbose ast.dump，快 100x）
# ════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=256)
def _normalize_ast(code: str) -> str:
    """将代码解析为紧凑的结构签名字符串（~150-400 字符，非完整 AST dump）。

    签名 = 有序的 "操作序列"，只保留函数/方法调用名和关键 AST 节点类型，
    忽略变量名和常量值。例如：
    "Assign|Sub:close|Call:pct_change|Call:groupby|Sub:volume|Call:pct_change|..."

    结果被 _sequence_similarity() 高效比较。Levenshtein 在 200 字符上 O(40k) vs 3000 字符上 O(9M)。
    """
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        cleaned = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    ops: list[str] = []

    class _SigExtractor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                ops.append(f"C:{node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                ops.append(f"C:{node.func.attr}")
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp):
            om = {ast.Mult: "*", ast.Div: "/", ast.Add: "+",
                  ast.Sub: "-", ast.Pow: "**", ast.Mod: "%"}
            ops.append(f"Op:{om.get(type(node.op), '?')}")
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript):
            if isinstance(node.slice, ast.Constant):
                ops.append(f"Sub:{node.slice.value}")
            elif isinstance(node.slice, ast.Str):  # Py<3.8
                ops.append(f"Sub:{node.slice.s}")
            else:
                ops.append("Sub:[]")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            ops.append(f"Attr:{node.attr}")
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            ops.append("=")
            self.generic_visit(node)

        def visit_Return(self, node: ast.Return):
            ops.append("ret")

        def visit_Compare(self, node: ast.Compare):
            ops.append("Cmp")
            self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                ops.append("Neg")
            self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda):
            ops.append("Lambda")

    _SigExtractor().visit(tree)
    return "|".join(ops)


# ════════════════════════════════════════════════════════════
# 2. 数学语义标签提取
# ════════════════════════════════════════════════════════════

_PD_REF = r"\w+(?:\['\w+'\])?(?:\.\w+)*"  # Pandas 变量引用：标识符 + 可选下标

_SEMANTIC_TAG_PATTERNS: dict[str, re.Pattern] = {
    "multivariate_product": re.compile(
        _PD_REF + r"\s*\*\s*" + _PD_REF + r"\s*\*\s*" + _PD_REF, re.MULTILINE
    ),
    "rolling_mean": re.compile(
        r"\.rolling\(\s*\d+\s*\)\s*\.\s*mean\s*\(\)", re.MULTILINE
    ),
    "rolling_any": re.compile(
        r"\.rolling\(\s*\d+\s*\)", re.MULTILINE
    ),
    "cross_sectional_rank": re.compile(
        r"\.rank\(\s*(?:pct\s*=\s*True|axis\s*=\s*1)", re.MULTILINE
    ),
    "groupby_pct_change": re.compile(
        r"\.groupby\s*\(.+\)\s*\[.+\]\s*\.\s*pct_change\s*\(\)", re.MULTILINE
    ),
    "groupby_shift": re.compile(
        r"\.groupby\s*\(.+\)\s*\[.+\]\s*\.\s*shift\s*\(", re.MULTILINE
    ),
    "unstack_reshape": re.compile(
        r"\.unstack\s*\(.+\)", re.MULTILINE
    ),
    "stack_reshape": re.compile(
        r"\.stack\s*\(\)", re.MULTILINE
    ),
    "ewm_smooth": re.compile(
        r"\.ewm\s*\(\s*span\s*=\s*\d+\s*\)\s*\.\s*mean\s*\(\)", re.MULTILINE
    ),
    "zscore_normalize": re.compile(
        r"\(\s*\w+\s*-\s*\w+\.mean\s*\(\s*\)\s*\)\s*/\s*\w+\.std\s*\(\s*\)", re.MULTILINE
    ),
    "three_input_product": re.compile(
        r"(?:ret|returns|r)\s*\*\s*(?:amp|amplitude|am|vol_ratio|vr|vol_chg)",
        re.MULTILINE | re.IGNORECASE,
    ),
    "close_pct_change": re.compile(
        r"\[.close.\]\s*\.\s*pct_change\s*\(", re.MULTILINE
    ),
    "close_div_open": re.compile(
        r"\[.close.\]\s*/\s*\[.open.\]\s*-\s*1", re.MULTILINE
    ),
    "high_minus_low_div_close": re.compile(
        r"\(\s*\[.high.\]\s*-\s*\[.low.\]\s*\)\s*/\s*\[.close.\]", re.MULTILINE
    ),
    "vol_div_vol_shift": re.compile(
        r"\[.volume.\]\s*/\s*.*\[.volume.\].*\.shift", re.MULTILINE
    ),
}


def _extract_semantic_tags(code: str) -> set[str]:
    """提取代码的数学语义标签（补充 AST 结构签名的不足）。

    检测代码中是否包含特定的数学运算模式（无论具体 API 写法）。
    例如 groupby().pct_change() 和 unstack().shift().stack()
    在数学语义上等价但 AST 结构不同。
    """
    tags: set[str] = set()
    for tag, pattern in _SEMANTIC_TAG_PATTERNS.items():
        if pattern.search(code):
            tags.add(tag)
    return tags


# ════════════════════════════════════════════════════════════
# 3. 相似度计算
# ════════════════════════════════════════════════════════════

def _sequence_similarity(s1: str, s2: str) -> float:
    """归一化编辑距离相似度，范围 [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = _edit_distance(s1, s2)
    return 1.0 - dist / max_len


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard 相似度，范围 [0, 1]."""
    if not set1 and not set2:
        return 1.0
    u = len(set1 | set2)
    return len(set1 & set2) / u if u > 0 else 0.0


def _compute_similarity(new_code: str, hist_code: str) -> dict:
    """计算两段代码的双重相似度。

    Returns:
        {"ast_sim": float, "semantic_sim": float, "combined": float}
        combined = 0.6 * ast_sim + 0.4 * semantic_sim
    """
    new_norm = _normalize_ast(new_code)
    hist_norm = _normalize_ast(hist_code)
    ast_sim = _sequence_similarity(new_norm, hist_norm)

    new_tags = _extract_semantic_tags(new_code)
    hist_tags = _extract_semantic_tags(hist_code)
    semantic_sim = _jaccard_similarity(new_tags, hist_tags)

    combined = 0.6 * ast_sim + 0.4 * semantic_sim
    return {"ast_sim": ast_sim, "semantic_sim": semantic_sim, "combined": combined}


# ════════════════════════════════════════════════════════════
# 4. 死锁检测
# ════════════════════════════════════════════════════════════

def _detect_homogeneous_deadlock(
    conn: sqlite3.Connection,
    lookback_rounds: int,
) -> Tuple[bool, list, Optional[str]]:
    """检测最近 N 轮是否全部是同一模式（死锁风险）。

    判定：最近 N 轮中的所有两两组合，>80% 的 AST 结构签名相似度 > 0.7。
    """
    rows = conn.execute(
        """SELECT round_id, factor_code FROM rounds
           WHERE status != 'error' AND factor_code IS NOT NULL AND factor_code != ''
           ORDER BY round_id DESC LIMIT ?""",
        (lookback_rounds,),
    ).fetchall()

    if len(rows) < 3:
        return (False, [], None)

    codes = [(r["round_id"], r["factor_code"]) for r in rows]
    # 使用双重相似度（AST + semantic）进行死锁判定
    pairs_checked = 0
    similar_pairs = 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            pairs_checked += 1
            sim = _compute_similarity(codes[i][1], codes[j][1])
            if sim["combined"] > 0.70:
                similar_pairs += 1

    if pairs_checked == 0:
        return (False, [], None)

    ratio = similar_pairs / pairs_checked
    if ratio < 0.80:
        return (False, [], None)

    round_ids = [r[0] for r in codes]
    feedback = (
        f"[系统警告] 过去 {len(round_ids)} 轮全部是同一公式的微调变体 "
        f"(内部两两相似度比例 {ratio:.0%})。"
        "你的公式空间已穷尽。继续调整窗口/分母/rolling 长度不会有任何效果。"
        "请彻底换一种数学结构——不要保留任何之前的运算步骤。"
        "建议参考 program.md 中的 P0a★（价格路径不规则性）或 "
        "P0b★（波动率状态转换）方向。"
    )
    return (True, round_ids, feedback)


# ════════════════════════════════════════════════════════════
# 5. 失败摘要提取
# ════════════════════════════════════════════════════════════

def _extract_failed_summary(fail_reason: Optional[str]) -> str:
    """从 rounds.fail_reason 中提取简洁的失败维度摘要。"""
    if not fail_reason:
        return "无详细评分数据"
    # 尝试 JSON
    try:
        data = json.loads(fail_reason)
        if isinstance(data, dict) and "dimensions" in data:
            failed = [
                f"{k}={v['value']:.3f}" if isinstance(v, dict) else f"{k}={v}"
                for k, v in data["dimensions"].items()
                if isinstance(v, dict) and not v.get("pass", True)
            ]
            return ", ".join(failed) if failed else "全部通过"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    # 纯文本：提取 key=value
    m = re.findall(r"(\w+)=([\d.-]+)", fail_reason)
    if m:
        return ", ".join(f"{k}={v}" for k, v in m[:5])
    return fail_reason[:120]


# ════════════════════════════════════════════════════════════
# 6. Feedback 构造
# ════════════════════════════════════════════════════════════

def _build_feedback_text(
    matched_rounds: list,
    max_similarity: float,
    history_failures: dict,
    is_deadlock: bool,
    deadlock_feedback: Optional[str] = None,
) -> str:
    """构造注入下一轮 system_prompt 的反馈文本。"""
    lines = []

    if is_deadlock and deadlock_feedback:
        lines.append(deadlock_feedback)
        lines.append("")
        lines.append("这些因子的评分结果：")
        for rid in sorted(history_failures.keys()):
            lines.append(f"  R{rid}: {history_failures[rid]}")
        lines.append("")
        lines.append(
            "建议：换一种完全不同的数学结构，不要保留 ret×amp×vol "
            "的三维乘积模式。优先尝试 program.md 中 P0a★ 或 P0b★ 方向。"
        )
    elif matched_rounds:
        rid_str = "/".join(f"R{r}" for r in sorted(matched_rounds))
        lines.append(
            f"[代码重复警告] 你的代码与 {rid_str} "
            f"高度相似（综合相似度 {max_similarity:.0%}）。"
        )
        lines.append("")
        lines.append("这些因子的评分结果：")
        for rid in sorted(history_failures.keys()):
            if rid in matched_rounds:
                lines.append(f"  R{rid}: {history_failures.get(rid, '无数据')}")
        lines.append("")
        lines.append("建议：换一种完全不同的数学结构，不要微调参数。")

    return "\n".join(lines) if len(lines) > 1 else ""


# ════════════════════════════════════════════════════════════
# 7. 主函数
# ════════════════════════════════════════════════════════════

def check_code_uniqueness(
    new_code: str,
    conn: sqlite3.Connection,
    similarity_threshold: float = 0.80,
    lookback_rounds: int = 10,
) -> Tuple[bool, Optional[str], dict]:
    """检查新因子代码是否与最近历史因子代码过于相似。

    结合 AST 结构签名和数学语义标签进行双重检测。

    Args:
        new_code: AI 生成的因子代码字符串（沙箱执行前）
        conn: SQLite 数据库连接
        similarity_threshold: 综合相似度阈值，超过即判定为重复
        lookback_rounds: 回溯检查最近 N 轮

    Returns:
        (is_unique, feedback_for_next_round, metadata)
        - is_unique: True=通过，False=与历史因子过于相似
        - feedback_for_next_round: 注入下一轮 system_prompt 的反馈文本
        - metadata: {"similarity_scores": {...}, "max_similarity": float,
                      "matched_rounds": [...], "all_same_pattern": bool}
    """
    metadata: dict = {
        "similarity_scores": {},
        "max_similarity": 0.0,
        "matched_rounds": [],
        "all_same_pattern": False,
        "similarity_method": "ast_sig+semantic" + ("+lev" if _LEV_AVAILABLE else "+purepy"),
    }

    if not new_code or not new_code.strip():
        return (True, None, metadata)

    # 快速路径：语法错误 → 让沙箱去报错
    try:
        ast.parse(new_code.strip())
    except SyntaxError:
        return (True, None, metadata)

    # 查询最近 lookback_rounds 轮（排除 error）
    rows = conn.execute(
        """SELECT round_id, factor_code, fail_reason FROM rounds
           WHERE status != 'error' AND factor_code IS NOT NULL AND factor_code != ''
           ORDER BY round_id DESC LIMIT ?""",
        (lookback_rounds,),
    ).fetchall()

    if len(rows) < 3:
        return (True, None, metadata)

    # 计算与每个历史轮的相似度
    history_failures: dict = {}
    max_similarity = 0.0
    matched_rounds: list = []

    for row in rows:
        rid = row["round_id"]
        hist_code = row["factor_code"]
        fail_reason = row["fail_reason"]

        if not hist_code or not hist_code.strip():
            continue

        sim = _compute_similarity(new_code, hist_code)
        metadata["similarity_scores"][rid] = {
            "combined": round(sim["combined"], 4),
            "ast_sim": round(sim["ast_sim"], 4),
            "semantic_sim": round(sim["semantic_sim"], 4),
        }

        history_failures[rid] = _extract_failed_summary(fail_reason)

        if sim["combined"] > max_similarity:
            max_similarity = sim["combined"]

        if sim["combined"] > similarity_threshold:
            matched_rounds.append(rid)

    metadata["max_similarity"] = round(max_similarity, 4)
    metadata["matched_rounds"] = sorted(matched_rounds)

    # 死锁检测
    is_deadlock, _, deadlock_feedback = _detect_homogeneous_deadlock(
        conn, lookback_rounds
    )
    metadata["all_same_pattern"] = is_deadlock

    # 如果新代码与任何历史轮次都不相似 → 通过（即使历史本身同质化）
    if not matched_rounds:
        return (True, None, metadata)

    feedback = _build_feedback_text(
        matched_rounds=matched_rounds,
        max_similarity=max_similarity,
        history_failures=history_failures,
        is_deadlock=is_deadlock,
        deadlock_feedback=deadlock_feedback,
    )

    return (False, feedback if feedback else None, metadata)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== check_code_uniqueness.py 自检 ===\n")
    print(f"  Levenshtein: {'可用' if _LEV_AVAILABLE else '不可用 (纯 Python 回退)'}")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE rounds (
            round_id INTEGER NOT NULL, batch_run_id INTEGER,
            status TEXT DEFAULT 'fail', fail_reason TEXT, factor_code TEXT,
            PRIMARY KEY (round_id, batch_run_id)
        )
    """)

    C1 = '''def compute_factor(df):
    df = df.copy()
    df['ret'] = df.groupby(level='code')['close'].pct_change()
    df['amp'] = (df['high'] - df['low']) / df['close']
    df['vr'] = df.groupby(level='code')['volume'].pct_change()
    df['s'] = df['ret'] * df['amp'] * df['vr']
    factor = df.groupby(level='code')['s'].rolling(5,min_periods=3).mean()
    factor = factor.groupby(level='date').rank(pct=True)
    return factor'''

    C2 = '''def compute_factor(df):
    df = df.copy()
    df['returns'] = df.groupby(level='code')['close'].pct_change()
    df['amplitude'] = (df['high'] - df['low']) / df['close']
    df['vol_chg'] = df.groupby(level='code')['volume'].pct_change()
    df['sig'] = df['returns'] * df['amplitude'] * df['vol_chg']
    factor = df.groupby(level='code')['sig'].rolling(3,min_periods=3).mean()
    factor = factor.groupby(level='date').rank(pct=True)
    return factor'''

    C3 = '''def compute_factor(df):
    cw = df['close'].unstack(level='code')
    vw = df['volume'].unstack(level='code')
    ret = cw / cw.shift(1) - 1
    amp = (df['high'].unstack(level='code') - df['low'].unstack(level='code')) / cw
    vr = vw / vw.shift(1) - 1
    signal = ret * amp * vr
    factor = signal.rolling(5,min_periods=3).mean().rank(axis=1,pct=True).stack()
    return factor'''

    HURST = '''def compute_factor(df):
    close = df['close']
    ma60 = close.groupby(level='code').rolling(60).mean()
    std60 = close.groupby(level='code').rolling(60).std()
    dim1 = abs(close - ma60) / (std60 + 1e-8)
    ma20 = close.groupby(level='code').rolling(20).mean()
    dim2 = abs(close - ma20) / (close.groupby(level='code').rolling(20).std() + 1e-8)
    dim1 = dim1.groupby(level='date').rank(pct=True)
    dim2 = dim2.groupby(level='date').rank(pct=True)
    factor = (dim1 + dim2) / 2
    return factor'''

    for rid, code in [(1, C1), (2, C1), (3, C2), (4, C1)]:
        conn.execute(
            "INSERT INTO rounds VALUES(?,1,'fail',?,?)",
            (rid, "dir_acc=0.4691,需>0.48; rank_ac=0.0897,需>0.2", code),
        )

    import time

    # ── 测试 1: 相同结构 → 应检测出高相似度 ──
    print("[1] 相同结构（改窗口 5→3）:")
    t0 = time.time()
    is_unique, feedback, meta = check_code_uniqueness(C2, conn)
    t = time.time() - t0
    print(f"    is_unique={is_unique} (预期 False), max_sim={meta['max_similarity']:.4f}, "
          f"matched={meta['matched_rounds']}, time={t:.2f}s")
    assert not is_unique, "应检测出高相似度!"
    assert meta["max_similarity"] > 0.80, f"相似度应>0.80, 实际={meta['max_similarity']:.4f}"
    print("    [PASS]")

    # ── 测试 2: 完全不同结构 → 应通过 ──
    print("[2] 完全不同结构（趋势偏离度）:")
    t0 = time.time()
    is_unique, feedback, meta = check_code_uniqueness(HURST, conn)
    t = time.time() - t0
    print(f"    is_unique={is_unique} (预期 True), max_sim={meta['max_similarity']:.4f}, time={t:.2f}s")
    assert is_unique, "完全不同结构应通过!"
    print("    [PASS]")

    # ── 测试 3: 改名不换结构 → 语义标签应匹配 ──
    print("[3] 改名不换结构（ret→returns, amp→amplitude）:")
    is_unique2, fbk2, meta2 = check_code_uniqueness(C2, conn)
    for rid, scores in meta2["similarity_scores"].items():
        if rid == 1:
            print(f"    R{rid}: ast={scores['ast_sim']:.4f}, sem={scores['semantic_sim']:.4f}, "
                  f"combined={scores['combined']:.4f}")
            assert scores["semantic_sim"] > 0.8, f"语义相似度应高, 实际={scores['semantic_sim']:.4f}"
    print("    [PASS]")

    # ── 测试 4: 换 API 不换数学 → 语义标签应匹配 ──
    print("[4] 换 API 不换数学（groupby+pct_change → unstack+shift+stack）:")
    tags1 = _extract_semantic_tags(C1)
    tags3 = _extract_semantic_tags(C3)
    jac = _jaccard_similarity(tags1, tags3)
    print(f"    C1 tags: {tags1}")
    print(f"    C3 tags: {tags3}")
    print(f"    Jaccard: {jac:.4f}")
    assert jac > 0.2, f"语义标签应有重叠, 实际 Jaccard={jac:.4f}"
    print("    [PASS]")

    # ── 测试 5: 死锁检测 ──
    print("[5] 全部历史同模式死锁检测:")
    is_dl, rids, dl_fb = _detect_homogeneous_deadlock(conn, 3)
    print(f"    is_deadlock={is_dl} (预期 True), rounds={rids}")
    assert is_dl, "3 轮全是 C★ 应触发死锁!"
    print("    [PASS]")

    # ── 测试 6: 语义标签覆盖 ──
    print("\n[6] 语义标签覆盖:")
    print(f"    C1 tags: {_extract_semantic_tags(C1)}")
    print(f"    C3 tags: {_extract_semantic_tags(C3)}")
    print(f"    HURST tags: {_extract_semantic_tags(HURST)}")
    c1t = _extract_semantic_tags(C1)
    assert "multivariate_product" in c1t, "应检测到 multivariate_product"
    assert "cross_sectional_rank" in c1t, "应检测到 cross_sectional_rank"
    ht = _extract_semantic_tags(HURST)
    assert "multivariate_product" not in ht, "Hurst 不应有三元乘积"
    assert "rolling_any" in ht, "Hurst 应有 rolling"
    print("    [PASS]")

    conn.close()
    print(f"\n{'='*60}")
    print("全部 6 项测试通过.")

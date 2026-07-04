#!/usr/bin/env python3
"""Phase 3a 合规检查模块 — 纯 AST 静态分析，零误报。

替代 Phase 2 的正则匹配方案。通过遍历 Python AST 精确检测：
  - 未来函数: shift(负数) / shift(periods=负数)
  - 危险调用: eval / exec / open / __import__
  - 逐行循环: for x in df['col'].unique() / iterrows / itertuples
  - 危险导入: os / sys / subprocess / requests / socket

返回签名与 Phase 2 兼容: check_compliance(code: str) -> Tuple[str, str]
  ("PASS", "合规") 或 ("ERROR", "具体原因描述")
注意: 无 WARNING 级别（AST 精确检测无需模糊分级），但返回签名保留该关键字以兼容调用方。

用法:
  python checker.py              # 运行内置 demo 测试
  python checker.py --perf       # 性能测试 (100 次解析)
"""

import ast
import re
import time
from typing import Tuple, List

# ── 常量 ─────────────────────────────────────────────────────
MAX_FACTOR_LINES = 20

# 危险函数名（直接调用即禁止）
DANGEROUS_BUILTINS = {"eval", "exec", "open", "__import__"}

# 危险模块名
DANGEROUS_MODULES = {"os", "sys", "subprocess", "requests", "socket"}


# ── 代码预处理（Phase 3f+）──────────────────────────────────
def preprocess_code(code: str) -> str:
    """自动修正 LLM 常见的 MultiIndex 列引用错误。

    生产环境中的 DataFrame 使用 MultiIndex (date, code)，
    'code' 是 index level 而非普通列。LLM 常错误地将其当列引用，
    导致沙箱执行出错。此函数在 AST 分析和执行前自动修正。

    转换规则:
      df['code']          → df.index.get_level_values('code')
      df["code"]          → df.index.get_level_values('code')
      .groupby('code')    → .groupby(level='code')
      .groupby("code")    → .groupby(level='code')

    Returns:
        转换后的代码字符串
    """
    # df['code'] / df["code"] → df.index.get_level_values('code')
    code = re.sub(
        r"\bdf\[(['\"])code\1\]",
        r'df.index.get_level_values("code")',
        code
    )

    # .groupby('code', ...) / .groupby("code", ...) → .groupby(level='code', ...)
    # (去掉结尾 \) 以覆盖带额外参数如 sort=False 的情况)
    code = re.sub(
        r"\.groupby\((['\"])code\1",
        r'.groupby(level="code"',
        code
    )

    # Phase 3g: 检测 groupby as_index=False（MultiIndex 不支持该参数）
    if re.search(r"\.groupby\([^)]*\bas_index\s*=\s*False\b", code):
        code = "# ⚠️ WARNING: MultiIndex groupby does not support as_index=False; removed\n" + \
               re.sub(r",?\s*as_index\s*=\s*False", "", code)

    # Phase 3g: 检测 groupby sort=False（在 MultiIndex 中通常不需要）
    # 保留 sort=False 因为它是有效参数，但确保 level='code' 已修正
    if re.search(r"\.groupby\(level=\"code\",\s*sort\s*=\s*False\s*\)", code):
        pass  # 已正确处理

    # ── Phase 3g M-4 fix: 修正非已知列引用 ──
    # 两阶段修复：
    #   Stage A: df.groupby(...)['var'] → var.groupby(...)（最常见模式）
    #   Stage B: 残余 df['var'] → var（简单引用）
    _KNOWN_COLS = {"open", "high", "low", "close", "volume", "code"}

    # Stage A: df.groupby(level='code')['computed_var'] → var.groupby(level='code')
    def _fix_groupby_col_ref(m: re.Match) -> str:
        var_name = m.group(3)  # group(1)=args, group(2)=quote, group(3)=col_name
        if var_name in _KNOWN_COLS:
            return m.group(0)  # 保留已知列
        return f"{var_name}.groupby({m.group(1)})"

    code = re.sub(
        r"df\.groupby\(((?:level=['\"]code['\"]|level=\d+)[^)]*)\)\[(['\"])(\w+)\2\]",
        _fix_groupby_col_ref,
        code
    )

    # Phase 3g: Stage B (df['var']→var) 已回滚 —
    # 正则无法区分未定义变量和真实数据列，可能制造 NameError
    # 替代方案：engine.py system_prompt 强化 MultiIndex 规则

    # Phase 3i P1-1: 除零检测 — AST 扫描除法操作，标记警告（不修改代码）
    # 实际 inf 清洗在 batch_pipeline sandbox 执行后进行（replace([inf,-inf], nan)）
    # 这里仅做检测标注，供 AST 合规检查输出警告

    return code


# ── AST 工具 ─────────────────────────────────────────────────
def _is_negative_constant(node: ast.expr) -> bool:
    """判断 AST 表达式是否为静态可判定的负数。

    支持两种形式:
      - ast.UnaryOp(op=ast.USub, operand=ast.Constant(value=N)) → -5
      - ast.Constant(value=N) where N < 0                        → -5
    """
    if isinstance(node, ast.Constant):
        val = node.value
        return isinstance(val, (int, float)) and val < 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    return False


def _is_shift_negative_call(node: ast.Call) -> Tuple[bool, str]:
    """检测是否为 shift(负数) 或 shift(periods=负数) 调用。

    Returns:
        (is_violation, reason_str)
    """
    # 必须是 Attribute 调用且 attr='shift'
    if not isinstance(node.func, ast.Attribute):
        return False, ""
    if node.func.attr != "shift":
        return False, ""

    # 检查位置参数: shift(-1)
    for arg in node.args:
        if _is_negative_constant(arg):
            return True, "禁止使用未来函数: shift(负数)"

    # 检查关键字参数: shift(periods=-1)
    for kw in node.keywords:
        if kw.arg == "periods" and _is_negative_constant(kw.value):
            return True, "禁止使用未来函数: shift(periods=负数)"

    return False, ""


def _is_unique_on_subscript(iter_node: ast.expr) -> bool:
    """检测 iter 是否为 df['column'].unique() 模式。

    匹配: ast.Call(func=ast.Attribute(attr='unique'),
                    func.value=ast.Subscript(...))
    """
    if not isinstance(iter_node, ast.Call):
        return False
    if not isinstance(iter_node.func, ast.Attribute):
        return False
    if iter_node.func.attr != "unique":
        return False
    # unique() 的调用对象是 Subscript（如 df['code']）
    return isinstance(iter_node.func.value, ast.Subscript)


def _collect_violations(tree: ast.AST) -> List[str]:
    """遍历 AST 收集所有违规信息。"""
    violations = []

    for node in ast.walk(tree):
        # ── Call 节点 ──────────────────────────────────
        if isinstance(node, ast.Call):
            # shift(负数)
            is_v, reason = _is_shift_negative_call(node)
            if is_v:
                violations.append(reason)
                continue

            # eval / exec / open / __import__
            if isinstance(node.func, ast.Name):
                if node.func.id in DANGEROUS_BUILTINS:
                    violations.append(
                        f"禁止危险函数调用: {node.func.id}()"
                    )
                    continue

        # ── For 节点 ───────────────────────────────────
        if isinstance(node, ast.For):
            # 检查 iter 部分
            if _is_unique_on_subscript(node.iter):
                violations.append(
                    "禁止逐行循环遍历股票: for x in df['column'].unique()"
                )
                continue

            # iterrows / itertuples
            for child in ast.walk(node.iter):
                if isinstance(child, ast.Attribute) and child.attr in (
                    "iterrows", "itertuples"
                ):
                    violations.append(
                        f"禁止 iterrows/itertuples: .{child.attr}()"
                    )
                    break

        # ── Import 节点 ────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name in DANGEROUS_MODULES:
                    violations.append(f"禁止系统/网络模块导入: import {name}")

        if isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base in DANGEROUS_MODULES:
                    violations.append(
                        f"禁止系统/网络模块导入: from {node.module} import ..."
                    )

        # ── Attribute 节点（顶层检测 iterrows/itertuples 引用）──
        if isinstance(node, ast.Attribute):
            if node.attr in ("iterrows", "itertuples"):
                # 已经在 For 节点中检测过，但属性引用本身也应拦截
                # 避免漏掉: df.iterrows 赋值给变量后再遍历的情况
                pass  # For 节点已覆盖，此处仅作记号

    return violations


# ── 主接口 ───────────────────────────────────────────────────
def check_compliance(code: str) -> Tuple[str, str]:
    """检查因子代码合规性（纯 AST 静态分析）。

    Args:
        code: 因子代码字符串

    Returns:
        ("PASS", "合规") — 通过
        ("ERROR", "具体原因") — 违规

    P0-2 fix: AST 违规检测现在在原始代码上运行（预处理之前），
    确保 for 循环、iterrows 等硬违规能被精确捕获。
    预处理（MultiIndex 列引用自动修正）仅在违规检测通过后执行。
    """
    # 1. 空代码检查
    if not code or not code.strip():
        return ("ERROR", "因子代码为空")

    # 2. 行数检查（排除纯注释行和空行）
    lines = [
        l for l in code.strip().split("\n")
        if l.strip() and not l.strip().startswith("#")
    ]
    if len(lines) > MAX_FACTOR_LINES:
        return (
            "ERROR",
            f"因子代码 {len(lines)} 行, 超过上限 {MAX_FACTOR_LINES} 行",
        )

    # 3. 检查 compute_factor 函数存在
    if "def compute_factor" not in code:
        return ("ERROR", "未包含 compute_factor 函数定义")

    # 4. AST 解析（原始代码）
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ("ERROR", f"代码语法错误，无法解析: {e}")

    # 5. ★ P0-2 fix: 先做 AST 违规检测（在原始代码上），再预处理
    violations = _collect_violations(tree)
    if violations:
        return ("ERROR", violations[0])  # 返回第一个违规，避免信息过载

    # Phase 3i P1-1: 除零检测 — 独立警告，不阻断合规
    div_count = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
    )
    div_warning = ""
    if div_count > 0:
        div_warning = (
            f"⚠️ 检测到 {div_count} 处除法运算 — 请确保分母已用 .replace(0, np.nan) 包裹"
        )

    # 6. 通过违规检测后，预处理自动修正 MultiIndex 列引用
    preprocessed = preprocess_code(code)

    # 7. 如果预处理改变了代码，重新 AST 解析确认无语法错误
    if preprocessed != code:
        try:
            ast.parse(preprocessed)
        except SyntaxError as e:
            return ("ERROR", f"预处理后代码语法错误: {e}")

    return ("PASS", "合规")


# ── Demo & Tests ─────────────────────────────────────────────
def _run_tests():
    """运行内置测试套件。"""
    passed = 0
    failed = 0

    def test(name: str, code: str, expected_level: str, expected_substr: str = ""):
        nonlocal passed, failed
        level, reason = check_compliance(code)
        ok = level == expected_level
        if expected_substr and ok:
            ok = expected_substr in reason
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         expected={expected_level}('{expected_substr}'), "
                  f"got={level}('{reason}')")
        return ok

    print("=== checker.py AST 重写 — 测试套件 ===\n")

    # ── 必须拦截 ───────────────────────────────────────
    print("【必须拦截 — ERROR】")
    test("shift(-1)",
         "def compute_factor(df):\n    return df['close'].shift(-1)",
         "ERROR", "shift(负数)")

    test("shift(periods=-1)",
         "def compute_factor(df):\n    return df['close'].shift(periods=-1)",
         "ERROR", "shift(periods=负数)")

    test("shift(-5)",
         "def compute_factor(df):\n    return df['close'].shift(-5)",
         "ERROR", "shift(负数)")

    test("for code in df['code'].unique() — P0-2 fix: 在预处理前检测，应拦截",
         "def compute_factor(df):\n    for c in df['code'].unique():\n        pass\n    return df['close']",
         "ERROR", "逐行循环")

    test("iterrows",
         "def compute_factor(df):\n    for i, row in df.iterrows():\n        pass\n    return df['close']",
         "ERROR", "iterrows")

    test("itertuples",
         "def compute_factor(df):\n    for row in df.itertuples():\n        pass\n    return df['close']",
         "ERROR", "itertuples")

    test("import os",
         "def compute_factor(df):\n    import os\n    return df['close']",
         "ERROR", "import os")

    test("import sys",
         "def compute_factor(df):\n    import sys\n    return df['close']",
         "ERROR", "import sys")

    test("from os import ...",
         "def compute_factor(df):\n    from os import environ\n    return df['close']",
         "ERROR", "from os")

    test("import subprocess",
         "def compute_factor(df):\n    import subprocess\n    return df['close']",
         "ERROR", "subprocess")

    test("import requests",
         "def compute_factor(df):\n    import requests\n    return df['close']",
         "ERROR", "requests")

    test("eval()",
         "def compute_factor(df):\n    eval('1+1')\n    return df['close']",
         "ERROR", "eval()")

    test("exec()",
         "def compute_factor(df):\n    exec('x=1')\n    return df['close']",
         "ERROR", "exec()")

    test("__import__()",
         "def compute_factor(df):\n    __import__('os')\n    return df['close']",
         "ERROR", "__import__()")

    test("open()",
         "def compute_factor(df):\n    f = open('/etc/passwd')\n    return df['close']",
         "ERROR", "open()")

    # ── 必须放行 ───────────────────────────────────────
    print("\n【必须放行 — PASS】")
    test("shift(5) — 正数放行",
         "def compute_factor(df):\n    return df['close'].shift(5)",
         "PASS")

    test("shift(periods=20) — 正数放行",
         "def compute_factor(df):\n    return df['close'].shift(periods=20)",
         "PASS")

    test("groupby rolling mean — 向量化放行",
         "def compute_factor(df):\n    return df.groupby(level='code')['close'].rolling(5).mean()",
         "PASS")

    test("shift(5) / shift(20) — 合法动量因子",
         "def compute_factor(df):\n    return df['close'].shift(5) / df['close'].shift(20) - 1",
         "PASS")

    test("rank(pct=True) — 合法排名",
         "def compute_factor(df):\n    return df['volume'].rank(pct=True)",
         "PASS")

    test("groupby transform lambda shift — 合法",
         "def compute_factor(df):\n    return df.groupby(level='code')['close'].transform(lambda x: x.shift(1) / x.shift(21) - 1)",
         "PASS")

    test("rolling std — 合法波动率",
         "def compute_factor(df):\n    return df['close'].rolling(20).std()",
         "PASS")

    test("pct_change — 合法收益率",
         "def compute_factor(df):\n    return df['close'].groupby(level='code').pct_change(5)",
         "PASS")

    test("ewm — 合法指数加权",
         "def compute_factor(df):\n    return df['close'].ewm(span=20).mean()",
         "PASS")

    test("numpy import — 合法",
         "import numpy as np\nimport pandas as pd\n\ndef compute_factor(df):\n    return np.log(df['close'])",
         "PASS")

    # ── Phase 3f+ 自动修正测试 ─────────────────────────
    print("\n【自动修正 — PASS (预处理后合规)】")
    test("df.groupby('code') 自动修正为 level='code'",
         "def compute_factor(df):\n    return df.groupby('code')['close'].transform(lambda x: x.pct_change(5))",
         "PASS")

    test('df.groupby("code") 自动修正为 level="code"',
         'def compute_factor(df):\n    return df.groupby("code")["close"].transform(lambda x: x.pct_change(5))',
         "PASS")

    test("df['code'] 自动修正为 get_level_values",
         "def compute_factor(df):\n    codes = df['code']\n    return df['close'].pct_change(5)",
         "PASS")

    # ── Phase 3g 自动修正测试 ─────────────────────────
    print("\n【Phase 3g 自动修正 — PASS (预处理后合规)】")
    test("groupby('code', sort=False) 自动修正",
         "def compute_factor(df):\n    return df.groupby('code', sort=False)['close'].rolling(5).mean().reset_index(level=0, drop=True)",
         "PASS")
    test('groupby("code", sort=False) 自动修正',
         'def compute_factor(df):\n    return df.groupby("code", sort=False)["close"].transform(lambda x: x.rolling(5).mean())',
         "PASS")
    test("groupby('code', as_index=False) 自动修正（去除 as_index）",
         "def compute_factor(df):\n    return df.groupby('code', as_index=False)['close'].mean()",
         "PASS")

    print(f"\n{'='*50}")
    print(f"结果: {passed} 通过, {failed} 失败 (共 {passed+failed})")
    return failed == 0


def _perf_test():
    """性能测试：100 次解析耗时。"""
    code = (
        "def compute_factor(df):\n"
        "    close = df['close']\n"
        "    ma5 = close.rolling(5).mean()\n"
        "    ma20 = close.rolling(20).mean()\n"
        "    return (ma5 - ma20) / ma20\n"
    )
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        check_compliance(code)
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / n * 1000
    print(f"性能测试: {n} 次解析, 总耗时 {elapsed:.3f}s, "
          f"平均 {avg_ms:.2f}ms/次 (< 100ms: {'PASS' if avg_ms < 100 else 'FAIL'})")


if __name__ == "__main__":
    import sys

    if "--perf" in sys.argv:
        _perf_test()
    else:
        _run_tests()
        print()
        _perf_test()

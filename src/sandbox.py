#!/usr/bin/env python3
"""安全沙箱 — exec() + 受限命名空间，跨平台超时控制。

白名单: 仅 np/pd/df + 安全内置函数。
禁止: os, sys, subprocess, open, __import__, eval, exec, compile。
"""

import ast
import re
import sys
import threading
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from checker import preprocess_code  # P0-3: 消除重复代码，从 checker 导入

# ── 白名单（来自 PROBLEM v1.1 section 1.2）───────────────────
SAFE_BUILTINS: Dict = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip, "True": True, "False": False,
    "print": print, "None": None, "isinstance": isinstance, "type": type,
}

SANDBOX_TIMEOUT = 20  # Phase 3h: 15→20，反馈闭环比硬超时更重要


class SandboxTimeout(Exception):
    """沙箱执行超时异常。"""
    pass


def run_sandbox(code: str, df: pd.DataFrame, timeout: int = SANDBOX_TIMEOUT) -> pd.Series:
    """在受限命名空间中执行因子代码。

    Args:
        code: 因子代码字符串（必须包含 compute_factor 函数）
        df: 输入 DataFrame，MultiIndex (date, code)，columns=[open,high,low,close,volume]
        timeout: 超时秒数

    Returns:
        compute_factor 的返回值，pd.Series (MultiIndex date, code)

    Raises:
        SandboxTimeout: 执行超时
        RuntimeError: 沙箱违规或其他运行时错误
    """
    df_copy = df.copy()
    # P0-3: 使用 checker.preprocess_code 替代重复实现
    # P0-4: 仅当 DataFrame 有 MultiIndex 且包含 'code' level 时才预处理
    if isinstance(df_copy.index, pd.MultiIndex) and 'code' in df_copy.index.names:
        code = preprocess_code(code)
    sandbox_globals = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "pd": pd,
        "df": df_copy,
    }

    result_container: list = []
    error_container: list = []

    def _target():
        try:
            exec(code, sandbox_globals)
            if "compute_factor" not in sandbox_globals:
                error_container.append(RuntimeError("代码中未定义 compute_factor 函数"))
                return
            fn = sandbox_globals["compute_factor"]
            output = fn(df_copy)
            if not isinstance(output, pd.Series):
                error_container.append(TypeError(
                    f"compute_factor 必须返回 pd.Series, 实际返回 {type(output).__name__}"
                ))
                return
            result_container.append(output)
        except Exception as e:
            error_container.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise SandboxTimeout(f"因子代码执行超时（{timeout} 秒）")

    if error_container:
        raise RuntimeError(f"沙箱执行异常: {error_container[0]}")

    if not result_container:
        raise RuntimeError("沙箱未产生输出")

    return result_container[0]


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FactorLab sandbox.py 自检 ===\n")

    # 构造 mock 时序数据 (3 只股票 x 10 天)
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    stocks = ["600519", "000858", "002594"]
    data = []
    for s in stocks:
        price = 100 + np.cumsum(np.random.randn(10))
        for i, d in enumerate(dates):
            data.append({
                "date": d, "code": s,
                "open": price[i], "high": price[i] * 1.02, "low": price[i] * 0.98,
                "close": price[i], "volume": np.random.randint(1000000, 10000000),
            })
    df = pd.DataFrame(data).set_index(["date", "code"])

    # 1. 执行合法因子
    legal_code = """
def compute_factor(df):
    return df["close"].pct_change(5)
"""
    try:
        result = run_sandbox(legal_code, df)
        print(f"合法因子执行成功, 返回类型: {type(result).__name__}, 长度: {len(result)}")
        print(result)
    except Exception as e:
        print(f"合法因子执行失败: {e}")

    # 2. 测试 os 注入被拦截
    os_inject = """
def compute_factor(df):
    import os
    os.system("echo hacked")
    return df["close"]
"""
    try:
        result = run_sandbox(os_inject, df)
        print(f"os 注入未被拦截! 返回: {result}")
    except Exception as e:
        print(f"\nos 注入被拦截: {e}")

    # 3. 测试超时（无限循环）
    infinite_loop = """
def compute_factor(df):
    x = 0
    while True:
        x += 1
    return df["close"]
"""
    try:
        result = run_sandbox(infinite_loop, df, timeout=2)
        print(f"无限循环未被拦截! 返回: {result}")
    except SandboxTimeout:
        print(f"\n无限循环已超时拦截 [OK]")
    except Exception as e:
        print(f"\n无限循环被拦截: {type(e).__name__}: {e}")

    print("\n自检完成.")

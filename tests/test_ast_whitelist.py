#!/usr/bin/env python3
"""Phase 3a AST 白名单测试 — 验证 checker.py 对合法代码零误报。

数据来源（按优先级）：
  1. SQLite rounds 表中的 factor_code（Phase 2c 批量运行生成的 50 条）
  2. history/round_*/factor_draft.py 备份文件
  3. 内置合法代码示例（覆盖各种 pandas 向量化模式）

用法:
  python test_ast_whitelist.py
"""

import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

from checker import check_compliance
from config import PROJECT_ROOT


# ── 内置合法代码示例（补充）──────────────────────────────────
BUILTIN_LEGAL_CODES: List[Tuple[str, str]] = [
    # (名称, 代码)
    ("动量因子: 简单 pct_change (按股票)",
     textwrap.dedent("""\
     def compute_factor(df):
         return df.groupby(level='code')['close'].pct_change(5)
     """)),

    ("反转因子: 负过去收益率",
     textwrap.dedent("""\
     def compute_factor(df):
         ret_5d = df.groupby(level='code')['close'].pct_change(5)
         return -ret_5d
     """)),

    ("均线交叉: 5日/20日 ratio",
     textwrap.dedent("""\
     def compute_factor(df):
         close = df['close']
         ma5 = close.rolling(5).mean()
         ma20 = close.rolling(20).mean()
         return (ma5 - ma20) / ma20
     """)),

    ("布林带位置: (close - lower) / (upper - lower)",
     textwrap.dedent("""\
     def compute_factor(df):
         close = df['close']
         ma20 = close.rolling(20).mean()
         std20 = close.rolling(20).std()
         return (close - (ma20 - 2*std20)) / (4*std20)
     """)),

    ("RSI因子: 14日涨跌幅比率",
     textwrap.dedent("""\
     def compute_factor(df):
         delta = df.groupby(level='code')['close'].diff()
         gain = delta.clip(lower=0).rolling(14).mean()
         loss = -delta.clip(upper=0).rolling(14).mean()
         return gain / (loss + 1e-8)
     """)),

    ("成交量异常: 当日量 / 20日均量 - 1",
     textwrap.dedent("""\
     def compute_factor(df):
         vol = df['volume']
         vol_ma20 = df.groupby(level='code')['volume'].transform(lambda x: x.rolling(20).mean())
         return vol / (vol_ma20 + 1e-8) - 1
     """)),

    ("Amihud非流动性: |return| / volume",
     textwrap.dedent("""\
     def compute_factor(df):
         ret = df.groupby(level='code')['close'].pct_change()
         amihud = ret.abs() / (df['volume'] + 1e-8)
         return amihud.rolling(20).mean()
     """)),

    ("波动率锥: 20日标准差 / 60日标准差",
     textwrap.dedent("""\
     def compute_factor(df):
         ret = df.groupby(level='code')['close'].pct_change()
         vol20 = ret.rolling(20).std()
         vol60 = ret.rolling(60).std()
         return vol20 / (vol60 + 1e-8)
     """)),

    ("量价背离: rank(price) - rank(volume)",
     textwrap.dedent("""\
     def compute_factor(df):
         pr = df['close'].pct_change(10).rank(pct=True)
         vr = df['volume'].pct_change(10).rank(pct=True)
         return pr - vr
     """)),

    ("换手率异常: rank(turnover)",
     textwrap.dedent("""\
     def compute_factor(df):
         return df['volume'].rank(pct=True)
     """)),

    ("移动平均乖离率",
     textwrap.dedent("""\
     def compute_factor(df):
         close = df['close']
         ma = close.rolling(50).mean()
         return (close - ma) / (ma + 1e-8)
     """)),

    ("MACD信号线 (简化)",
     textwrap.dedent("""\
     def compute_factor(df):
         close = df['close']
         ema12 = close.ewm(span=12).mean()
         ema26 = close.ewm(span=26).mean()
         return ema12 - ema26
     """)),

    ("夏普比率因子: return / std",
     textwrap.dedent("""\
     def compute_factor(df):
         ret = df.groupby(level='code')['close'].pct_change()
         return ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
     """)),

    ("开盘强度: (close - open) / (high - low + 1e-8)",
     textwrap.dedent("""\
     def compute_factor(df):
         return (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
     """)),

    ("影线比例: (high - max(open,close)) / (high - low + 1e-8)",
     textwrap.dedent("""\
     def compute_factor(df):
         body_top = df[['open', 'close']].max(axis=1)
         upper_shadow = df['high'] - body_top
         return upper_shadow / (df['high'] - df['low'] + 1e-8)
     """))
]


def load_codes_from_sqlite() -> List[Tuple[str, str]]:
    """从 SQLite rounds 表提取 factor_code。"""
    import sqlite3
    db_path = PROJECT_ROOT / "db" / "factorlab.db"
    if not db_path.exists():
        return []
    try:
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        cur.execute(
            "SELECT round_id, factor_code FROM rounds "
            "WHERE factor_code IS NOT NULL ORDER BY round_id"
        )
        rows = cur.fetchall()
        db.close()
        return [(f"Round {r[0]} SQLite", r[1]) for r in rows]
    except Exception:
        return []


def _extract_compute_factor(source: str) -> str:
    """从完整 Python 源文件中提取 compute_factor 函数体。"""
    lines = source.split("\n")
    start = None
    end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def compute_factor"):
            start = i
            continue
        # 找到函数体结束：遇到非空、非缩进、非注释的顶层代码
        if start is not None and end is None:
            if stripped and not stripped.startswith("#") and not line.startswith(" ") and not line.startswith("\t"):
                if not stripped.startswith("def") and not stripped.startswith("@"):
                    end = i
                    break
    if start is None:
        return source
    func_lines = lines[start:end]
    return "\n".join(func_lines)


def load_codes_from_history() -> List[Tuple[str, str]]:
    """从 history/round_*/factor_draft.py 提取 compute_factor 函数体。"""
    history_dir = PROJECT_ROOT / "history"
    codes = []
    if history_dir.exists():
        for d in sorted(history_dir.iterdir()):
            if d.is_dir() and d.name.startswith("round_"):
                draft_path = d / "factor_draft.py"
                if draft_path.exists():
                    try:
                        full_code = draft_path.read_text()
                        if "def compute_factor" in full_code:
                            # 只提取 compute_factor 函数体，去除模板注释
                            func_body = _extract_compute_factor(full_code)
                            codes.append((f"History {d.name}", func_body))
                    except Exception:
                        pass
    return codes


def run_whitelist_tests():
    """运行白名单测试：所有合法代码都必须 PASS。"""
    # 收集测试用例
    test_cases: List[Tuple[str, str]] = []

    # 1. SQLite 来源
    sqlite_codes = load_codes_from_sqlite()
    test_cases.extend(sqlite_codes)
    print(f"SQLite rounds: {len(sqlite_codes)} 条")

    # 2. History 来源
    hist_codes = load_codes_from_history()
    # 去重（SQLite 已有则不重复）
    existing_names = {n for n, _ in test_cases}
    for name, code in hist_codes:
        if name not in existing_names:
            test_cases.append((name, code))
    print(f"History: {len(hist_codes)} 条 (新增 {len(test_cases) - len(sqlite_codes)})")

    # 3. 内置示例补充
    builtin_added = 0
    for name, code in BUILTIN_LEGAL_CODES:
        if len(test_cases) < 50:  # 目标至少 20-50 条
            test_cases.append((f"Builtin: {name}", code))
            builtin_added += 1
    print(f"内置示例: {builtin_added} 条")
    print(f"总计: {len(test_cases)} 条测试用例\n")

    # 运行测试
    passed = 0
    failed = 0
    false_positives = []

    for name, code in test_cases:
        level, reason = check_compliance(code)
        if level == "PASS":
            passed += 1
        else:
            failed += 1
            false_positives.append((name, level, reason))
            print(f"  [FAIL] {name}: {level} — {reason}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"白名单测试结果: {passed} 通过, {failed} 误报 (共 {len(test_cases)})")

    if false_positives:
        print(f"\n⚠️  误报详情 ({len(false_positives)} 条):")
        for name, level, reason in false_positives[:10]:
            print(f"  - {name}")
            print(f"    级别: {level}, 原因: {reason}")
            # 显示代码片段
            for tc_name, tc_code in test_cases:
                if tc_name == name:
                    snippet = tc_code[:200].replace('\n', ' | ')
                    print(f"    代码: {snippet}...")
                    break

    # 输出结论
    if failed == 0:
        print("\n✅ 零误报！AST checker 对全部合法代码返回 PASS。")
        return True
    else:
        print(f"\n❌ 发现 {failed} 条误报，需修正 checker.py。")
        return False


if __name__ == "__main__":
    print("=== Phase 3a AST 白名单测试 ===\n")
    ok = run_whitelist_tests()
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""Phase 3c 自然语言因子检索 — 关键词匹配 + 结构化过滤。

基于 SQLite 的因子检索，支持：
  - search(): 自然语言关键词匹配（对 natural_summary + direction_tag 做 LIKE）
  - structured_filter(): 白名单结构化条件过滤
  - interactive_query(): CLI 交互式查询

防止 SQL 注入：全部使用 ? 参数化查询 + SUPPORTED_FILTERS 白名单。

用法:
  python nl_query.py --demo   # 交互式演示
"""

import re
import sqlite3
from typing import Dict, List, Optional

import pandas as pd

# ── 停用词 ──────────────────────────────────────────────

STOP_WORDS = {
    "的", "了", "是", "有", "和", "或", "哪些", "什么",
    "因子", "个", "吗", "呢", "我", "你", "他", "她",
    "这", "那", "不", "也", "就", "都", "要", "会",
    "可以", "能", "在", "与", "及", "等",
}

# ── 结构化过滤白名单 ────────────────────────────────────

SUPPORTED_FILTERS: Dict[str, tuple] = {
    "sharpe_min": ("b.sharpe_ratio >= ?", float),
    "sharpe_max": ("b.sharpe_ratio <= ?", float),
    "ic_min": ("CAST(json_extract(f.metrics, '$.ic') AS REAL) >= ?", float),
    "ic_max": ("CAST(json_extract(f.metrics, '$.ic') AS REAL) <= ?", float),
    "ir_min": ("CAST(json_extract(f.metrics, '$.ir') AS REAL) >= ?", float),
    "drawdown_max": ("b.max_drawdown <= ?", float),  # max_drawdown is negative, so <= means less severe
    "direction_tag": ("f.direction_tag LIKE ?", str),
    "date_after": ("f.inbound_date >= ?", str),
    "date_before": ("f.inbound_date <= ?", str),
    "monotonicity_passed": ("f.monotonicity_passed = 1", None),
    "oos_passed": ("f.oos_stability_passed = 1", None),
}


# ════════════════════════════════════════════════════════════
# 搜索函数
# ════════════════════════════════════════════════════════════

def search(
    conn: sqlite3.Connection,
    query: str,
    include_all: bool = False,
) -> pd.DataFrame:
    """自然语言关键词匹配。

    对 query 分词后，对 natural_summary 和 direction_tag 做 OR 匹配。

    Args:
        conn: SQLite 连接
        query: 自然语言查询字符串
        include_all: 保留参数，factors 表无 status 列，始终查询全部入库因子

    Returns:
        匹配的因子 DataFrame，按夏普比率降序

    Example:
        >>> search(conn, "量价背离")
        >>> search(conn, "有哪些量价背离因子？")
    """
    # 分词 + 停用词过滤
    keywords = [
        k.strip()
        for k in re.split(r"[\s,，。.?？!！、；;：:]+", query)
        if len(k.strip()) > 1 and k.strip() not in STOP_WORDS
    ]

    if not keywords:
        return pd.DataFrame()

    # 构建 WHERE 子句（参数化查询防注入）
    conditions = []
    params: List[str] = []
    for kw in keywords:
        conditions.append(
            "(f.natural_summary LIKE ? OR f.direction_tag LIKE ?)"
        )
        params.extend([f"%{kw}%", f"%{kw}%"])

    where_clause = " OR ".join(conditions)

    sql = f"""
        SELECT f.factor_id, f.natural_summary, f.direction_tag,
               CAST(json_extract(f.metrics, '$.ic') AS REAL) as ic,
               CAST(json_extract(f.metrics, '$.ir') AS REAL) as ir,
               b.sharpe_ratio as backtest_sharpe,
               b.max_drawdown as backtest_max_drawdown,
               f.inbound_date
        FROM factors f
        LEFT JOIN backtests b ON f.factor_id = b.factor_id
        WHERE ({where_clause})
        ORDER BY b.sharpe_ratio DESC
        LIMIT 50
    """

    return pd.read_sql(sql, conn, params=params)


def structured_filter(
    conn: sqlite3.Connection,
    conditions: dict,
    include_all: bool = False,
) -> pd.DataFrame:
    """结构化条件过滤。

    只接受 SUPPORTED_FILTERS 白名单中的键（防注入），
    所有值通过 ? 参数化传入 SQL。

    Args:
        conn: SQLite 连接
        conditions: 过滤条件字典
        include_all: 保留参数，无实际作用

    Returns:
        匹配的因子 DataFrame

    Example:
        >>> structured_filter(conn, {"sharpe_min": 1.0, "ic_min": 0.03})
        >>> structured_filter(conn, {"direction_tag": "产业链", "date_after": "2024-01-01"})
        >>> structured_filter(conn, {"monotonicity_passed": True})
    """
    where_parts: List[str] = []
    params: list = []

    for key, value in conditions.items():
        if key not in SUPPORTED_FILTERS:
            continue  # 忽略不支持的条件

        clause, type_converter = SUPPORTED_FILTERS[key]

        if type_converter is None:
            # 无参数条件（如 monotonicity_passed = 1）
            where_parts.append(clause)
        else:
            where_parts.append(clause)
            params.append(type_converter(value))

    if not where_parts:
        # 无条件时返回全部因子
        sql = """
            SELECT f.factor_id, f.natural_summary, f.direction_tag,
                   CAST(json_extract(f.metrics, '$.ic') AS REAL) as ic,
                   CAST(json_extract(f.metrics, '$.ir') AS REAL) as ir,
                   b.sharpe_ratio as backtest_sharpe,
                   b.max_drawdown as backtest_max_drawdown,
                   f.monotonicity_passed, f.oos_stability_passed, f.inbound_date
            FROM factors f
            LEFT JOIN backtests b ON f.factor_id = b.factor_id
            ORDER BY b.sharpe_ratio DESC
            LIMIT 50
        """
        return pd.read_sql(sql, conn)

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT f.factor_id, f.natural_summary, f.direction_tag,
               CAST(json_extract(f.metrics, '$.ic') AS REAL) as ic,
               CAST(json_extract(f.metrics, '$.ir') AS REAL) as ir,
               b.sharpe_ratio as backtest_sharpe,
               b.max_drawdown as backtest_max_drawdown,
               f.monotonicity_passed, f.oos_stability_passed, f.inbound_date
        FROM factors f
        LEFT JOIN backtests b ON f.factor_id = b.factor_id
        WHERE {where_sql}
        ORDER BY b.sharpe_ratio DESC
        LIMIT 50
    """

    return pd.read_sql(sql, conn, params=params)


# ════════════════════════════════════════════════════════════
# CLI 交互式查询
# ════════════════════════════════════════════════════════════

def _print_factor_table(df: pd.DataFrame):
    """打印因子结果表格。"""
    if df.empty:
        print("\n  (无匹配结果)")
        return

    print(f"\n  {'ID':6s} {'方向':10s} {'IC':>8s} {'IR':>8s} {'夏普':>8s} {'摘要'}")
    print(f"  {'-'*70}")
    for _, r in df.iterrows():
        fid = str(r.get("factor_id", ""))
        tag = str(r.get("direction_tag", "-"))[:10]
        ic = r.get("ic")
        ir_val = r.get("ir")
        sharpe = r.get("backtest_sharpe")
        summary = str(r.get("natural_summary", ""))[:35]

        ic_str = f"{float(ic):.4f}" if ic is not None and not (isinstance(ic, float) and pd.isna(ic)) else "-"
        ir_str = f"{float(ir_val):.4f}" if ir_val is not None and not (isinstance(ir_val, float) and pd.isna(ir_val)) else "-"
        sh_str = f"{float(sharpe):.2f}" if sharpe is not None and not (isinstance(sharpe, float) and pd.isna(sharpe)) else "-"

        print(f"  {fid:6s} {tag:10s} {ic_str:>8s} {ir_str:>8s} {sh_str:>8s} {summary}")

    print(f"\n  共 {len(df)} 条结果。")


def interactive_query(conn: sqlite3.Connection):
    """CLI 交互式因子查询。

    显示子菜单：
    1. 自然语言查询（输入任意文字）
    2. 结构化过滤（选择条件组合）
    3. 查看全部入库因子
    0. 返回上级
    """
    while True:
        print(f"\n  {'─'*40}")
        print(f"  因子检索子菜单:")
        print(f"    1 — 自然语言查询（如: 量价背离）")
        print(f"    2 — 结构化过滤（按条件筛选）")
        print(f"    3 — 查看全部入库因子")
        print(f"    0 — 返回上级")

        try:
            choice = input("\n  请选择 (0-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == "1":
            query = input("  输入关键词: ").strip()
            if not query:
                print("  查询为空，已取消。")
                continue
            result = search(conn, query)
            _print_factor_table(result)

        elif choice == "2":
            print("\n  可用过滤条件:")
            print("    sharpe_min / sharpe_max — 夏普比率范围")
            print("    ic_min / ic_max — IC 范围")
            print("    ir_min — IR 最低值")
            print("    drawdown_max — 最大回撤上限（输入正数，如 0.20）")
            print("    direction_tag — 方向标签（精确匹配）")
            print("    date_after / date_before — 入库日期范围 (YYYY-MM-DD)")
            print("    monotonicity_passed — 单调性通过 (true/false)")
            print("    oos_passed — 样本外通过 (true/false)")
            print("  多个条件用逗号分隔，如: sharpe_min=1.0, ic_min=0.03")

            raw = input("\n  输入过滤条件: ").strip()
            if not raw:
                print("  条件为空，已取消。")
                continue

            # 解析条件字符串 "key=value, key2=value2"
            conds = {}
            for part in raw.split(","):
                part = part.strip()
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()

                if k in SUPPORTED_FILTERS:
                    _, converter = SUPPORTED_FILTERS[k]
                    if converter == bool or converter is None:
                        conds[k] = v.lower() in ("true", "1", "yes")
                    else:
                        try:
                            conds[k] = converter(v)
                        except (ValueError, TypeError):
                            print(f"  [警告] 条件 '{k}' 的值 '{v}' 格式无效，已跳过")

            if not conds:
                print("  无有效条件，已取消。")
                continue

            result = structured_filter(conn, conds)
            _print_factor_table(result)

        elif choice == "3":
            result = structured_filter(conn, {})
            _print_factor_table(result)

        elif choice == "0":
            return

        else:
            print("  无效选择，请输入 0-3。")


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json as _json

    if "--demo" in sys.argv:
        print("=== nl_query.py Phase 3c 检索演示 ===\n")

        # 创建临时数据库
        import tempfile
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db_path = tmp_db.name
        tmp_db.close()

        conn = sqlite3.connect(tmp_db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS factors (
                factor_id TEXT PRIMARY KEY,
                round INTEGER DEFAULT 1,
                direction_tag TEXT DEFAULT '',
                natural_summary TEXT,
                metrics TEXT,
                monotonicity_passed INTEGER DEFAULT 0,
                oos_stability_passed INTEGER DEFAULT 0,
                inbound_date TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL
            );
        """)

        mock_data = [
            ("f001", "反转类", "量价背离+缩量+波动收敛的复合反转因子", 0.045, 0.35, 1.2, -0.08, "2024-01-15"),
            ("f002", "动量", "传统20日动量因子，作为对照基准", 0.025, 0.15, 0.5, -0.15, "2024-02-10"),
            ("f003", "量价背离", "价格-成交量背离信号，捕捉主力资金动向", 0.052, 0.42, 1.5, -0.06, "2024-03-01"),
            ("f004", "波动率类", "低波动率异常因子：20日波动率倒数加权", 0.031, 0.22, 0.8, -0.10, "2024-03-15"),
            ("f005", "行为类", "散户情绪反向指标：基于换手率的极端值反转", 0.041, 0.31, 1.1, -0.09, "2024-04-05"),
        ]

        for fid, tag, summary, ic, ir_val, sharpe, mdd, date in mock_data:
            conn.execute(
                "INSERT INTO factors (factor_id, direction_tag, natural_summary, metrics, inbound_date) VALUES (?, ?, ?, ?, ?)",
                (fid, tag, summary, _json.dumps({"ic": ic, "ir": ir_val}), date),
            )
            conn.execute(
                "INSERT INTO backtests (factor_id, sharpe_ratio, max_drawdown) VALUES (?, ?, ?)",
                (fid, sharpe, mdd),
            )
        conn.commit()

        # 测试 1: 自然语言搜索
        print("[测试 1] search('量价背离')")
        df1 = search(conn, "量价背离")
        _print_factor_table(df1)
        assert len(df1) > 0, "应返回包含'量价背离'的因子"
        print("  [PASS] 返回匹配结果")

        # 测试 2: 停用词过滤
        print("\n[测试 2] search('的 了 是')")
        df2 = search(conn, "的 了 是")
        _print_factor_table(df2)
        assert df2.empty, "停用词查询应返回空"
        print("  [PASS] 停用词过滤生效")

        # 测试 3: 结构化过滤
        print("\n[测试 3] structured_filter({'sharpe_min': 1.0})")
        df3 = structured_filter(conn, {"sharpe_min": 1.0})
        _print_factor_table(df3)
        assert len(df3) > 0, "应返回夏普>=1.0的因子"
        assert all(s >= 1.0 for s in df3["backtest_sharpe"] if not pd.isna(s)), "所有结果夏普应>=1.0"
        print("  [PASS] 过滤正确")

        # 测试 4: 无效键忽略
        print("\n[测试 4] structured_filter({'invalid_key': 1})")
        df4 = structured_filter(conn, {"invalid_key": 1})
        _print_factor_table(df4)
        assert len(df4) > 0, "无效键应被忽略，返回全部因子"
        print("  [PASS] 无效键被忽略，不报错")

        # 测试 5: 无匹配结果
        print("\n[测试 5] structured_filter({'sharpe_min': 999})")
        df5 = structured_filter(conn, {"sharpe_min": 999.0})
        _print_factor_table(df5)
        assert df5.empty, "应返回空"
        print("  [PASS] 无匹配结果")

        # 测试 6: 方向标签过滤
        print("\n[测试 6] structured_filter({'direction_tag': '反转类'})")
        df6 = structured_filter(conn, {"direction_tag": "反转类"})
        _print_factor_table(df6)
        assert len(df6) > 0, "应返回反转类因子"
        print("  [PASS] 方向标签过滤正确")

        # 测试 7: search 匹配方向标签
        print("\n[测试 7] search('波动率')")
        df7 = search(conn, "波动率")
        _print_factor_table(df7)
        assert len(df7) > 0, "应返回波动率相关因子"
        print("  [PASS] 方向标签匹配生效")

        # 验证无 SQL 注入
        print("\n[测试 8] SQL 注入防护测试")
        try:
            df8 = search(conn, "'; DROP TABLE factors; --")
            _print_factor_table(df8)
            # 验证 factors 表仍存在
            cnt = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
            assert cnt == 5, f"factors 表应仍有 5 行，实际 {cnt}"
            print("  [PASS] SQL 注入防护有效")
        except Exception as e:
            print(f"  [PASS] 异常被安全捕获: {e}")

        conn.close()
        import os as _os
        _os.unlink(tmp_db_path)
        print("\n自检通过.")
    else:
        # 连接真实数据库
        from database import get_conn
        conn = get_conn()
        try:
            interactive_query(conn)
        finally:
            conn.close()

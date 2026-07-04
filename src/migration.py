#!/usr/bin/env python3
"""数据迁移工具 — Phase 1b JSON → Phase 2 SQLite。

用法:
  python migration.py            # 执行迁移
  python migration.py --report   # 仅查看迁移摘要
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from database import get_conn, init_db, insert_factor, insert_memory

from config import PROJECT_ROOT
FACTOR_POOL_PATH = PROJECT_ROOT / "factor_pool.json"
PROGRAM_PATH = PROJECT_ROOT / "program.md"


def migrate_factors(conn) -> int:
    """将 factor_pool.json 导入 factors 表。返回导入条数。"""
    if not FACTOR_POOL_PATH.exists():
        print("  factor_pool.json 不存在，跳过 factors 表")
        return 0

    with open(FACTOR_POOL_PATH, "r", encoding="utf-8") as f:
        pool = json.load(f)

    factors = pool.get("factors", [])
    count = 0
    for f in factors:
        insert_factor(conn, {
            "factor_id": f["factor_id"],
            "round": f.get("round", 0),
            "inbound_date": f.get("inbound_date", "2025-01-01"),
            "code_snapshot": f.get("code_snapshot", ""),
            "natural_summary": f.get("natural_summary", ""),
            "metrics": f.get("metrics", {}),
            "score_total": f.get("score_total", 0),
            "cost_adjusted_score": f.get("cost_adjusted_score", 0),
            "rank_snapshot": f.get("rank_snapshot", {}),
        })
        count += 1

    return count


def migrate_memory(conn) -> int:
    """将 program.md 第四章记忆导入 memory 表。返回导入条数。"""
    if not PROGRAM_PATH.exists():
        print("  program.md 不存在，跳过 memory 表")
        return 0

    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "<!-- AI_MEMORY_START -->"
    if marker not in content:
        print("  未找到 AI_MEMORY_START 标记")
        return 0

    memory_section = content.split(marker, 1)[1] if len(content.split(marker, 1)) > 1 else ""

    # 按 "## Round" 分割
    rounds = re.split(r"\n## Round (\d+)", memory_section)
    count = 0
    i = 1  # 跳过第一个空白段

    while i < len(rounds) - 1:
        try:
            round_num = int(rounds[i])
            block = rounds[i + 1]

            passed = "PASS" in block or "已入库" in block
            summary = ""
            fail_reasons = ""
            factor_type = ""

            # 提取摘要
            sm = re.search(r"###\s*因子摘要\s*\n(.+?)(?:\n###|\n---|\Z)", block, re.DOTALL)
            if sm:
                summary = sm.group(1).strip()[:500]

            # 提取失败原因
            fm = re.search(r"###\s*失败原因\s*\n(.+?)(?:\n###|\n---|\Z)", block, re.DOTALL)
            if fm:
                fail_reasons = fm.group(1).strip()[:500]

            # 提取因子类型
            tm = re.search(r"###\s*因子类型\s*\n(.+?)(?:\n|$)", block)
            if tm:
                factor_type = tm.group(1).strip()[:50]

            # 跳过系统初始化记忆
            if "系统初始化" in summary or "A 股量化环境背景" in block:
                i += 2
                continue

            insert_memory(conn, {
                "round_id": round_num,
                "timestamp": datetime.now().isoformat(),
                "factor_type": factor_type,
                "summary": summary,
                "passed": passed,
                "fail_reasons": fail_reasons,
                "suggestion": "",
            })
            count += 1
        except (ValueError, IndexError):
            pass
        i += 2

    return count


def run_migration(conn=None) -> dict:
    """执行完整迁移，返回报告。"""
    should_close = conn is None
    if conn is None:
        conn = get_conn()

    init_db(conn)

    print("=== 数据迁移 ===\n")

    print("[1] factors 表...")
    n_factors = migrate_factors(conn)
    print(f"    导入 {n_factors} 条")

    print("[2] memory 表...")
    n_memory = migrate_memory(conn)
    print(f"    导入 {n_memory} 条")

    # 验证
    f_count = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    m_count = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]

    report = {
        "factors_imported": n_factors,
        "memory_imported": n_memory,
        "factors_total": f_count,
        "memory_total": m_count,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n迁移完成: factors={f_count}, memory={m_count}")

    if should_close:
        conn.close()

    return report


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    conn = get_conn()
    init_db(conn)

    if "--report" in sys.argv:
        f_count = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
        m_count = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        print(f"factors: {f_count} 条")
        print(f"memory:  {m_count} 条")
    else:
        report = run_migration(conn)

    conn.close()

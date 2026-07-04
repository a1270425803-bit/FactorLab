#!/usr/bin/env python3
"""Phase 3c 集成测试 — combo_engine + html_reporter + nl_query + CLI 菜单。

用法:
  python test_phase3c_integration.py
"""

import json as _json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

passed = 0
failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")


# ════════════════════════════════════════════════════════════
print("=== Phase 3c 集成测试 ===\n")

# ── Setup：创建临时 SQLite 数据库 ──────────────────────
tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp_db_path = tmp_db.name
tmp_db.close()

conn = sqlite3.connect(tmp_db_path)
conn.execute("PRAGMA journal_mode=WAL")
conn.row_factory = sqlite3.Row

# 建表（使用与 database.py 相同的结构）
conn.executescript("""
    CREATE TABLE IF NOT EXISTS factors (
        factor_id TEXT PRIMARY KEY,
        round INTEGER NOT NULL DEFAULT 1,
        direction_tag TEXT DEFAULT '',
        inbound_date TEXT NOT NULL DEFAULT '2024-01-01',
        code_snapshot TEXT,
        natural_summary TEXT,
        metrics TEXT,
        score_total REAL DEFAULT 0,
        cost_adjusted_score REAL DEFAULT 0,
        rank_snapshot TEXT DEFAULT '{}',
        monotonicity REAL DEFAULT 0.0,
        monotonicity_passed INTEGER DEFAULT 0,
        oos_ic_train REAL DEFAULT 0.0,
        oos_ic_test REAL DEFAULT 0.0,
        oos_stability_passed INTEGER DEFAULT 0,
        ic_decay_ratio REAL DEFAULT 0.0,
        ic_decay_passed INTEGER DEFAULT 0,
        yearly_validation_passed INTEGER DEFAULT 0,
        yearly_observed TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS backtests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        factor_id TEXT NOT NULL,
        annual_return REAL DEFAULT 0,
        max_drawdown REAL DEFAULT 0,
        sharpe_ratio REAL DEFAULT 0,
        win_rate REAL DEFAULT 0,
        turnover_est REAL DEFAULT 0,
        avg_impact_cost_bps REAL DEFAULT 0.0,
        total_cost_annual REAL DEFAULT 0.0,
        layer_returns BLOB,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (factor_id) REFERENCES factors(factor_id)
    );
    CREATE TABLE IF NOT EXISTS rounds (
        round_id INTEGER NOT NULL,
        batch_run_id INTEGER,
        direction_tag TEXT DEFAULT '',
        status TEXT DEFAULT 'fail',
        started_at TEXT NOT NULL DEFAULT '2024-01-01',
        fail_reason TEXT DEFAULT '',
        api_cost REAL DEFAULT 0,
        factor_code TEXT DEFAULT '',
        summary TEXT DEFAULT '',
        steps TEXT DEFAULT '[]',
        PRIMARY KEY (round_id, batch_run_id)
    );
    CREATE TABLE IF NOT EXISTS batch_status (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_rounds INTEGER NOT NULL DEFAULT 50,
        completed_rounds INTEGER DEFAULT 0,
        cumulative_cost REAL DEFAULT 0,
        program_md5 TEXT DEFAULT '',
        status TEXT DEFAULT 'running',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id INTEGER NOT NULL,
        batch_run_id INTEGER DEFAULT 0,
        timestamp TEXT NOT NULL DEFAULT '2024-01-01',
        direction_tag TEXT DEFAULT '',
        factor_type TEXT DEFAULT '',
        summary TEXT DEFAULT '',
        passed INTEGER DEFAULT 0,
        fail_reasons TEXT DEFAULT '',
        suggestion TEXT DEFAULT ''
    );
""")

# Mock 因子数据
mock_factors = [
    ("f001", "反转类", 0.045, 0.35, 1.2, -0.08, 1, 1, 1, "[]", "2024-01-15"),
    ("f002", "动量", 0.025, 0.15, 0.5, -0.15, 1, 1, 1, "[]", "2024-02-10"),
    ("f003", "量价背离", 0.052, 0.42, 1.5, -0.06, 1, 1, 1, '["2023"]', "2024-03-01"),
]

for (fid, tag, ic_val, ir_val, sharpe, mdd, mono_p, oos_p, yv_p, yv_obs, date) in mock_factors:
    conn.execute(
        """INSERT INTO factors (factor_id, direction_tag, metrics, monotonicity_passed,
           oos_stability_passed, yearly_validation_passed, yearly_observed, inbound_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (fid, tag, _json.dumps({"ic": ic_val, "ir": ir_val}),
         mono_p, oos_p, yv_p, yv_obs, date),
    )
    conn.execute(
        "INSERT INTO backtests (factor_id, sharpe_ratio, max_drawdown, annual_return) VALUES (?, ?, ?, ?)",
        (fid, sharpe, mdd, ic_val * 0.3),
    )

# Mock rounds + batch_status
for i in range(1, 6):
    conn.execute(
        "INSERT INTO rounds (round_id, batch_run_id, status, api_cost) VALUES (?, 1, ?, ?)",
        (i, "inbound" if i % 2 == 1 else "fail", 0.03 + i * 0.002),
    )
conn.execute(
    "INSERT INTO batch_status (target_rounds, completed_rounds, cumulative_cost, status) VALUES (50, 5, 0.21, 'running')"
)
conn.commit()


# ════════════════════════════════════════════════════════════
# Test 1: combo_engine
# ════════════════════════════════════════════════════════════
print("1. combo_engine 测试")

try:
    from combo_engine import (
        ComboResult, _compute_weights, _compute_zscore,
        build_all_inbound, _compare_with_best_single,
    )

    # 1a: 权重计算
    icir_dict = {"f001": 0.5, "f002": -0.3, "f003": 1.2}
    weights = _compute_weights(icir_dict)
    check(abs(weights["f001"] - 0.5 / 1.7) < 0.01, f"正ICIR权重: f001={weights['f001']:.4f}")
    check(weights["f002"] == 0.0, f"负ICIR归零: f002={weights['f002']}")
    check(abs(weights["f003"] - 1.2 / 1.7) < 0.01, f"正ICIR权重: f003={weights['f003']:.4f}")
    check(abs(sum(weights.values()) - 1.0) < 0.01, f"权重和=1.0: {sum(weights.values()):.4f}")

    # 1b: 全部负ICIR
    all_neg = _compute_weights({"a": -0.5, "b": -0.3})
    check(all(s == 0.0 for s in all_neg.values()), "全部负ICIR→权重全0")

    # 1c: z-score
    np.random.seed(42)
    fv = pd.DataFrame(np.random.randn(10, 5), columns=[f"S{i}" for i in range(5)])
    z = _compute_zscore(fv)
    check(abs(z.mean(axis=1).mean()) < 0.01, f"z-score均值≈0: {z.mean(axis=1).mean():.6f}")
    check(abs(z.std(axis=1).mean() - 1.0) < 0.1, f"z-score标准差≈1: {z.std(axis=1).mean():.4f}")

    # 1d: 空因子库 → 优雅降级
    empty_conn = sqlite3.connect(":memory:")
    empty_conn.row_factory = sqlite3.Row
    empty_conn.execute("CREATE TABLE factors (factor_id TEXT, code_snapshot TEXT, metrics TEXT)")
    empty_result = build_all_inbound(empty_conn, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    check(empty_result.combo_factor is None, "空因子库→combo_factor=None")
    check(len(empty_result.weights) == 0, "空因子库→weights空")
    empty_conn.close()

    # 1e: vs_best_single
    from backtest import BacktestResult
    mock_bt = BacktestResult(sharpe_ratio=1.8)
    vs = _compare_with_best_single(conn, mock_bt)
    check(vs["combo_sharpe"] == 1.8, f"combo_sharpe={vs['combo_sharpe']}")
    check(vs["best_single_sharpe"] == 1.5, f"best_single_sharpe={vs['best_single_sharpe']} (最佳=f003)")
    check(vs["ratio"] > 0, f"ratio={vs['ratio']:.4f}")

except Exception as e:
    check(False, f"combo_engine 异常: {e}")


# ════════════════════════════════════════════════════════════
# Test 2: html_reporter 快速版
# ════════════════════════════════════════════════════════════
print("\n2. html_reporter 快速版测试")

try:
    from html_reporter import generate_quick_report, generate_full_report

    quick_path = generate_quick_report(conn, str(PROJECT_DIR / "reports" / "test_quick.html"))
    quick_size = os.path.getsize(quick_path)
    check(quick_size < 5_000_000, f"文件大小 < 5MB ({quick_size:,} bytes)")

    with open(quick_path, "r", encoding="utf-8") as f:
        content = f.read()
    check("因子库总览" in content, "含'因子库总览'")
    check("累计 API 成本" in content, "含'累计 API 成本'")
    check("统计摘要" in content, "含'统计摘要'")
    check("base64" in content, "base64 内嵌图表")
    check("<link" not in content and "http://" not in content, "无外部依赖")
except Exception as e:
    check(False, f"html_reporter 快速版异常: {e}")


# ════════════════════════════════════════════════════════════
# Test 3: html_reporter 完整版
# ════════════════════════════════════════════════════════════
print("\n3. html_reporter 完整版测试")

try:
    full_path = generate_full_report(conn, str(PROJECT_DIR / "reports" / "test_full.html"))
    full_size = os.path.getsize(full_path)
    check(full_size < 5_000_000, f"文件大小 < 5MB ({full_size:,} bytes)")

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    section_ids = ['id="a"', 'id="b"', 'id="c"', 'id="d"', 'id="e"', 'id="f"', 'id="g"']
    for sid in section_ids:
        check(sid in content, f"含 section {sid}")
    check("未计算" in content or "ICIR" in content, "f 部分含 ICIR 相关内容")
    check("<link" not in content, "CSS 内嵌（无外部 link）")
except Exception as e:
    check(False, f"html_reporter 完整版异常: {e}")


# ════════════════════════════════════════════════════════════
# Test 4: nl_query
# ════════════════════════════════════════════════════════════
print("\n4. nl_query 测试")

try:
    from nl_query import search, structured_filter

    # 4a: 关键词搜索
    df1 = search(conn, "量价")
    check(len(df1) >= 1, f"search('量价') 返回 {len(df1)} 条结果")

    # 4b: 停用词过滤
    df2 = search(conn, "的 了 是")
    check(df2.empty, "停用词查询返回空")

    # 4c: 结构化过滤（无匹配）
    df3 = structured_filter(conn, {"sharpe_min": 999})
    check(df3.empty, "sharpe_min=999返回空")

    # 4d: 无效键忽略
    df4 = structured_filter(conn, {"invalid_key": 1})
    check(len(df4) >= 1, f"无效键被忽略，返回全部 {len(df4)} 条")

    # 4e: 方向标签过滤
    df5 = structured_filter(conn, {"direction_tag": "反转类"})
    check(len(df5) == 1 and df5.iloc[0]["factor_id"] == "f001", "direction_tag过滤正确")

    # 4f: SQL 注入防护
    cnt_before = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    search(conn, "'; DROP TABLE factors; --")
    cnt_after = conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
    check(cnt_before == cnt_after, f"SQL注入防护: {cnt_before}→{cnt_after}")

    # 4g: 参数化查询验证（结构化过滤的值应通过 ? 传入）
    df6 = structured_filter(conn, {"ic_min": 0.04})
    check(len(df6) >= 1, f"ic_min=0.04 返回 {len(df6)} 条")
except Exception as e:
    check(False, f"nl_query 异常: {e}")


# ════════════════════════════════════════════════════════════
# Test 5: 冻结模块保护
# ════════════════════════════════════════════════════════════
print("\n5. 冻结模块保护测试")

frozen_modules = [
    "score.py", "sandbox.py", "data_fetcher.py",
    "checker.py", "backtest.py", "robustness_checker.py",
    "database.py", "logger.py", "data_fetcher_v2.py",
]
for mod in frozen_modules:
    mod_path = PROJECT_DIR / mod
    check(mod_path.exists(), f"{mod} 存在且未删除")

# Verify no unexpected imports in new modules
try:
    # combo_engine should not modify database.py
    import combo_engine
    check(True, "combo_engine 可导入")
except Exception as e:
    check(False, f"combo_engine 导入失败: {e}")


# ════════════════════════════════════════════════════════════
# Test 6: config.py REPORTS_DIR
# ════════════════════════════════════════════════════════════
print("\n6. config.py REPORTS_DIR 测试")

try:
    from config import REPORTS_DIR
    check(isinstance(REPORTS_DIR, str) and len(REPORTS_DIR) > 0, f"REPORTS_DIR='{REPORTS_DIR}'")
except Exception as e:
    check(False, f"config.REPORTS_DIR 异常: {e}")


# ════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════
conn.close()
os.unlink(tmp_db_path)

# 清理测试生成的报告文件
for test_file in ["test_quick.html", "test_full.html"]:
    fp = PROJECT_DIR / "reports" / test_file
    if fp.exists():
        fp.unlink()

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'='*60}")
print(f"  Phase 3c 集成测试: {passed}/{total} 通过"
      + (" ✅" if failed == 0 else f", {failed} 失败 ❌"))
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)

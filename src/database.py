#!/usr/bin/env python3
"""SQLite 数据库封装 — factors / rounds / memory / backtests / batch_status / pending_feedbacks 六表。

Phase 3e: 新增 pending_feedbacks 表（代码相似度门禁 feedback 传递通道）。

用法:
  python database.py          # 初始化数据库并打印表结构
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from config import PROJECT_ROOT
DB_PATH = PROJECT_ROOT / "db" / "factorlab.db"


def get_conn(path: Optional[Path] = None) -> sqlite3.Connection:
    """获取数据库连接（WAL 模式，支持并发读）。"""
    p = path or DB_PATH
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None, migrate: bool = False):
    """初始化五表结构（幂等）。

    Args:
        conn: 数据库连接（为 None 则自动创建并关闭）
        migrate: 是否执行 Phase 2c 迁移（DROP 旧表重建）。默认 False 仅创建新表。
    """
    c = conn or get_conn()
    close_after = (conn is None)

    # ── Phase 2c 迁移: 重建 rounds + memory + factors（加新字段）──
    if migrate:
        # m4 修复: 删除前检查行数并警告
        existing_data = False
        for tbl in ["factors", "rounds", "memory", "backtests", "batch_status", "pending_feedbacks"]:
            cnt = c.execute(
                f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{tbl}'"
            ).fetchone()[0]
            if cnt > 0:
                row_cnt = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                if row_cnt > 0:
                    existing_data = True
                    print(f"  [migrate] 警告: {tbl} 表有 {row_cnt} 行数据将被删除")
        if existing_data:
            print("  [migrate] 上述数据将在迁移中删除。如需保留请先备份 factorlab.db。")
        c.execute("PRAGMA foreign_keys=OFF")
        c.execute("DROP TABLE IF EXISTS backtests")
        c.execute("DROP TABLE IF EXISTS factors")
        c.execute("DROP TABLE IF EXISTS rounds")
        c.execute("DROP TABLE IF EXISTS memory")
        c.execute("DROP TABLE IF EXISTS batch_status")
        c.execute("PRAGMA foreign_keys=ON")

    c.executescript("""
        CREATE TABLE IF NOT EXISTS factors (
            factor_id    TEXT PRIMARY KEY,
            round        INTEGER NOT NULL,
            direction_tag TEXT DEFAULT '',   -- Phase 2c: 研究方向标签
            inbound_date TEXT NOT NULL,
            code_snapshot TEXT,
            natural_summary TEXT,
            metrics      TEXT,       -- JSON: {"ic": 0.045, ...}
            score_total  REAL,
            cost_adjusted_score REAL,
            rank_snapshot TEXT,      -- JSON: {"YYYY-MM-DD": {"code": rank}}
            -- Phase 3b: 稳健性字段
            monotonicity REAL DEFAULT 0.0,
            monotonicity_passed INTEGER DEFAULT 0,
            oos_ic_train REAL DEFAULT 0.0,
            oos_ic_test REAL DEFAULT 0.0,
            oos_stability_passed INTEGER DEFAULT 0,
            ic_decay_ratio REAL DEFAULT 0.0,
            ic_decay_passed INTEGER DEFAULT 0,
            yearly_validation_passed INTEGER DEFAULT 0,
            yearly_observed TEXT DEFAULT '',  -- JSON 列表字符串
            created_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rounds (
            round_id     INTEGER NOT NULL,
            batch_run_id INTEGER,              -- Phase 2c: 所属批量运行
            direction_tag TEXT DEFAULT '',      -- Phase 2c: 研究方向标签
            status       TEXT DEFAULT 'fail',   -- Phase 2c: inbound/fail/skip/error
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            fail_reason  TEXT,
            api_cost     REAL DEFAULT 0,
            factor_code  TEXT,
            summary      TEXT,
            steps        TEXT,       -- JSON: ["load_program", ...]
            PRIMARY KEY (round_id, batch_run_id)
        );

        CREATE TABLE IF NOT EXISTS memory (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id     INTEGER NOT NULL,
            batch_run_id INTEGER DEFAULT 0,    -- Phase 2c: 所属批量运行
            timestamp    TEXT NOT NULL,
            direction_tag TEXT DEFAULT '',      -- Phase 2c: 研究方向标签
            factor_type  TEXT,       -- 反转/动量/波动率/行为/量价背离/价格路径/其他
            summary      TEXT,
            passed       INTEGER,
            fail_reasons TEXT,
            suggestion   TEXT
        );

        CREATE TABLE IF NOT EXISTS backtests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_id    TEXT NOT NULL,
            annual_return REAL,
            max_drawdown  REAL,
            sharpe_ratio  REAL,
            win_rate      REAL,
            turnover_est  REAL,
            -- Phase 3b: 冲击成本字段
            avg_impact_cost_bps REAL DEFAULT 0.0,
            total_cost_annual REAL DEFAULT 0.0,
            layer_returns BLOB,        -- JSON 压缩存储
            created_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (factor_id) REFERENCES factors(factor_id)
        );

        CREATE TABLE IF NOT EXISTS batch_status (
            run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            target_rounds INTEGER NOT NULL,
            completed_rounds INTEGER DEFAULT 0,
            cumulative_cost REAL DEFAULT 0,
            program_md5  TEXT,          -- 启动时 program.md 前三章 MD5
            status       TEXT DEFAULT 'running',  -- running/paused/completed/fused
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS pending_feedbacks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id     INTEGER NOT NULL,
            feedback_text TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            consumed     INTEGER DEFAULT 0,
            FOREIGN KEY (round_id) REFERENCES rounds(round_id)
        );

        CREATE INDEX IF NOT EXISTS idx_rounds_status ON rounds(status);
        CREATE INDEX IF NOT EXISTS idx_rounds_batch ON rounds(batch_run_id);
        CREATE INDEX IF NOT EXISTS idx_memory_round ON memory(round_id);
        CREATE INDEX IF NOT EXISTS idx_memory_batch ON memory(batch_run_id);
        CREATE INDEX IF NOT EXISTS idx_backtests_factor ON backtests(factor_id);
    """)

    if close_after:
        c.close()


# ── Factors 表操作 ────────────────────────────────────────

# M3 修复: ORDER BY 白名单，防止 SQL 注入
_ALLOWED_SORT_COLUMNS = {
    "factor_id", "round", "score_total", "cost_adjusted_score",
    "inbound_date", "created_at", "direction_tag",
}


def insert_factor(conn: sqlite3.Connection, factor: dict):
    conn.execute(
        """INSERT OR REPLACE INTO factors
           (factor_id, round, direction_tag, inbound_date, code_snapshot, natural_summary,
            metrics, score_total, cost_adjusted_score, rank_snapshot,
            monotonicity, monotonicity_passed, oos_ic_train, oos_ic_test,
            oos_stability_passed, ic_decay_ratio, ic_decay_passed,
            yearly_validation_passed, yearly_observed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            factor["factor_id"],
            factor["round"],
            factor.get("direction_tag", ""),
            factor["inbound_date"],
            factor.get("code_snapshot", ""),
            factor.get("natural_summary", ""),
            json.dumps(factor.get("metrics", {}), ensure_ascii=False),
            factor.get("score_total", 0),
            factor.get("cost_adjusted_score", 0),
            json.dumps(factor.get("rank_snapshot", {}), ensure_ascii=False),
            factor.get("monotonicity", 0.0),
            int(factor.get("monotonicity_passed", False)),
            factor.get("oos_ic_train", 0.0),
            factor.get("oos_ic_test", 0.0),
            int(factor.get("oos_stability_passed", False)),
            factor.get("ic_decay_ratio", 0.0),
            int(factor.get("ic_decay_passed", False)),
            int(factor.get("yearly_validation_passed", True)),
            json.dumps(factor.get("yearly_observed", []), ensure_ascii=False),
        ),
    )
    conn.commit()


def query_factors(
    conn: sqlite3.Connection,
    min_ic: float = 0,
    sort_by: str = "score_total",
    limit: int = 50,
    direction_tag: str = "",
) -> list[dict]:
    """查询因子库，支持按方向筛选。"""
    where = ""
    params = []
    if min_ic > 0:
        where += " AND json_extract(metrics, '$.ic') >= ?"
        params.append(min_ic)
    if direction_tag:
        where += " AND direction_tag = ?"
        params.append(direction_tag)
    params.append(limit)
    # M3 修复: 白名单校验 sort_by，防止 SQL 注入
    if sort_by not in _ALLOWED_SORT_COLUMNS:
        sort_by = "score_total"
    rows = conn.execute(
        f"SELECT * FROM factors WHERE 1=1 {where} ORDER BY {sort_by} DESC LIMIT ?",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def count_factors(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]


# ── Rounds 表操作 ─────────────────────────────────────────

def insert_round(conn: sqlite3.Connection, round_data: dict) -> int:
    cur = conn.execute(
        """INSERT INTO rounds (round_id, batch_run_id, direction_tag, status,
           started_at, ended_at, fail_reason, api_cost, factor_code, summary, steps)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            round_data["round_id"],
            round_data.get("batch_run_id", 0),
            round_data.get("direction_tag", ""),
            round_data.get("status", "fail"),
            round_data["started_at"],
            round_data.get("ended_at", datetime.now().isoformat()),
            round_data.get("fail_reason", ""),
            round_data.get("api_cost", 0),
            round_data.get("factor_code", ""),
            round_data.get("summary", ""),
            json.dumps(round_data.get("steps", []), ensure_ascii=False),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_rounds_for_batch(conn: sqlite3.Connection, batch_run_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM rounds WHERE batch_run_id = ? ORDER BY round_id",
        (batch_run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_rounds_status(conn: sqlite3.Connection, n: int = 10) -> list[str]:
    """获取最近 n 轮的状态列表，用于中期熔断。"""
    rows = conn.execute(
        "SELECT status FROM rounds ORDER BY round_id DESC, batch_run_id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [r["status"] for r in rows]


# ── Memory 表操作 ─────────────────────────────────────────

def insert_memory(conn: sqlite3.Connection, mem: dict):
    conn.execute(
        """INSERT INTO memory (round_id, batch_run_id, timestamp, direction_tag,
           factor_type, summary, passed, fail_reasons, suggestion)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            mem["round_id"],
            mem.get("batch_run_id", 0),
            mem.get("timestamp", datetime.now().isoformat()),
            mem.get("direction_tag", ""),
            mem.get("factor_type", ""),
            mem.get("summary", ""),
            int(mem.get("passed", False)),
            mem.get("fail_reasons", ""),
            mem.get("suggestion", ""),
        ),
    )
    conn.commit()


def get_recent_memories(conn: sqlite3.Connection, n: int = 5) -> list[dict]:
    """获取最近 n 条记忆，用于 AI prompt 构建。"""
    rows = conn.execute(
        "SELECT * FROM memory ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]  # 时间正序


def get_recent_direction_tags(conn: sqlite3.Connection, n: int = 5) -> list[str]:
    """获取最近 n 轮的方向标签，用于里程碑事件 4。"""
    rows = conn.execute(
        "SELECT direction_tag FROM memory ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [r["direction_tag"] for r in reversed(rows)]


# ── Backtests 表操作 ──────────────────────────────────────

def insert_backtest(conn: sqlite3.Connection, bt: dict):
    # Phase 3b: 序列化 layer_returns 为 JSON（DataFrame → JSON string）
    layer_json = None
    lr = bt.get("layer_returns")
    if lr is not None:
        try:
            layer_json = lr.to_json(orient="index", date_format="iso")
        except Exception:
            layer_json = None

    conn.execute(
        """INSERT INTO backtests (factor_id, annual_return, max_drawdown, sharpe_ratio,
           win_rate, turnover_est, avg_impact_cost_bps, total_cost_annual, layer_returns)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bt["factor_id"],
            bt.get("annual_return", 0),
            bt.get("max_drawdown", 0),
            bt.get("sharpe_ratio", 0),
            bt.get("win_rate", 0),
            bt.get("turnover_est", 0),
            bt.get("avg_impact_cost_bps", 0.0),
            bt.get("total_cost_annual", 0.0),
            layer_json,
        ),
    )
    conn.commit()


def get_max_sharpe(conn: sqlite3.Connection) -> float:
    """获取当前最高回测夏普，用于里程碑事件 2。"""
    row = conn.execute("SELECT MAX(sharpe_ratio) FROM backtests").fetchone()
    return row[0] if row[0] else 0.0


# ── Batch Status 表操作（Phase 2c 新增）───────────────────

def create_batch(conn: sqlite3.Connection, target_rounds: int, program_md5: str) -> int:
    """创建新的批量运行记录，返回 run_id。"""
    cur = conn.execute(
        """INSERT INTO batch_status (target_rounds, program_md5)
           VALUES (?, ?)""",
        (target_rounds, program_md5),
    )
    conn.commit()
    return cur.lastrowid


def update_batch(conn: sqlite3.Connection, run_id: int, **kwargs):
    """更新批量运行状态。支持: completed_rounds, cumulative_cost, status。"""
    allowed = {"completed_rounds", "cumulative_cost", "status", "program_md5"}
    sets = []
    params = []
    for k, v in kwargs.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(run_id)
    conn.execute(
        f"UPDATE batch_status SET {', '.join(sets)} WHERE run_id = ?",
        params,
    )
    conn.commit()


def get_batch(conn: sqlite3.Connection, run_id: int) -> Optional[dict]:
    """获取单个批量运行记录。"""
    row = conn.execute(
        "SELECT * FROM batch_status WHERE run_id = ?", (run_id,)
    ).fetchone()
    return dict(row) if row else None


def get_latest_batch(conn: sqlite3.Connection) -> Optional[dict]:
    """获取最近一次批量运行（用于 --resume）。"""
    row = conn.execute(
        "SELECT * FROM batch_status ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_batch_round_summary(conn: sqlite3.Connection, batch_run_id: int) -> dict:
    """获取某次批量运行的轮次统计。"""
    total = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE batch_run_id = ?", (batch_run_id,)
    ).fetchone()[0]
    inbound = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE batch_run_id = ? AND status='inbound'",
        (batch_run_id,),
    ).fetchone()[0]
    fail = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE batch_run_id = ? AND status='fail'",
        (batch_run_id,),
    ).fetchone()[0]
    skip = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE batch_run_id = ? AND status='skip'",
        (batch_run_id,),
    ).fetchone()[0]
    error = conn.execute(
        "SELECT COUNT(*) FROM rounds WHERE batch_run_id = ? AND status='error'",
        (batch_run_id,),
    ).fetchone()[0]
    return {
        "total": total, "inbound": inbound, "fail": fail,
        "skip": skip, "error": error,
    }


# ── Phase 3e: Pending Feedbacks 表操作（代码相似度门禁）────────

def save_pending_similarity_feedback(
    conn: sqlite3.Connection, round_id: int, feedback_text: str
):
    """保存一条 similarity feedback 供下一轮消费。

    在代码相似度检测触发（is_unique=False）后调用，
    下一轮 generate_code() 前由 get_pending_similarity_feedback() 消费。
    """
    conn.execute(
        """INSERT INTO pending_feedbacks (round_id, feedback_text)
           VALUES (?, ?)""",
        (round_id, feedback_text),
    )
    conn.commit()


def get_pending_similarity_feedback(
    conn: sqlite3.Connection,
) -> Optional[str]:
    """获取最近一条未消费的 similarity feedback，并标记为已消费。

    Returns:
        feedback_text 或 None（无待消费 feedback）。
    """
    row = conn.execute(
        """SELECT id, feedback_text FROM pending_feedbacks
           WHERE consumed = 0
           ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return None
    # 标记已消费
    conn.execute(
        "UPDATE pending_feedbacks SET consumed = 1 WHERE id = ?",
        (row["id"],),
    )
    conn.commit()
    return row["feedback_text"]


# ── Phase 3b 迁移 ─────────────────────────────────────────

def migrate_v2_to_v3b(conn: sqlite3.Connection) -> bool:
    """从 Phase 2c (v2) 迁移到 Phase 3b (v3)。

    检测当前版本，按需执行 ALTER TABLE。
    返回：是否执行了迁移。
    """
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]
    if current_version >= 3:
        return False  # 已是最新

    # 检查表是否存在（防御性编程：应对早期版本无表的情况）
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )]

    # 新增 factors 表稳健性字段
    if "factors" in tables:
        new_cols = [
            ("monotonicity", "REAL DEFAULT 0.0"),
            ("monotonicity_passed", "INTEGER DEFAULT 0"),
            ("oos_ic_train", "REAL DEFAULT 0.0"),
            ("oos_ic_test", "REAL DEFAULT 0.0"),
            ("oos_stability_passed", "INTEGER DEFAULT 0"),
            ("ic_decay_ratio", "REAL DEFAULT 0.0"),
            ("ic_decay_passed", "INTEGER DEFAULT 0"),
            ("yearly_validation_passed", "INTEGER DEFAULT 0"),
            ("yearly_observed", "TEXT DEFAULT ''"),
        ]
        for col_name, col_type in new_cols:
            try:
                conn.execute(f"ALTER TABLE factors ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # 列已存在则跳过

    # 新增 backtests 表冲击成本字段
    if "backtests" in tables:
        new_cols = [
            ("avg_impact_cost_bps", "REAL DEFAULT 0.0"),
            ("total_cost_annual", "REAL DEFAULT 0.0"),
            ("layer_returns", "BLOB"),
        ]
        for col_name, col_type in new_cols:
            try:
                conn.execute(f"ALTER TABLE backtests ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

    # 更新版本号
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    return True


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== database.py 自检 (Phase 3b) ===\n")

    conn = get_conn()
    init_db(conn, migrate=True)

    # Phase 3b 迁移
    migrated = migrate_v2_to_v3b(conn)
    print(f"Phase 3b 迁移: {'已执行' if migrated else '已是最新 (跳过)'}")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    print(f"  当前 DB 版本: {version}")

    # 列出所有表
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("Tables:")
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        print(f"  {t[0]:20s} ({count} rows)")

    # 验证 Phase 3b 新增字段
    print("\n--- factors 表 Phase 3b 新字段 ---")
    factors_cols = conn.execute("PRAGMA table_info(factors)").fetchall()
    for c in factors_cols:
        if c["name"] in ("monotonicity", "monotonicity_passed", "oos_ic_train",
                          "oos_ic_test", "oos_stability_passed", "ic_decay_ratio",
                          "ic_decay_passed", "yearly_validation_passed", "yearly_observed"):
            print(f"  {c['name']:30s} {c['type']}")

    print("\n--- backtests 表 Phase 3b 新字段 ---")
    bt_cols = conn.execute("PRAGMA table_info(backtests)").fetchall()
    for c in bt_cols:
        if c["name"] in ("avg_impact_cost_bps", "total_cost_annual", "layer_returns"):
            print(f"  {c['name']:30s} {c['type']}")

    # 测试 Phase 3b 因子插入（含新字段）
    print("\n--- Phase 3b 因子插入测试 ---")
    insert_factor(conn, {
        "factor_id": "f_test_3b",
        "round": 1,
        "direction_tag": "反转类",
        "inbound_date": datetime.now().strftime("%Y-%m-%d"),
        "code_snapshot": "def compute_factor(df): ...",
        "natural_summary": "Phase 3b 测试因子",
        "metrics": {"ic": 0.04, "ir": 0.3},
        "score_total": 0.85,
        "cost_adjusted_score": 0.80,
        "rank_snapshot": {},
        "monotonicity": 0.45,
        "monotonicity_passed": True,
        "oos_ic_train": 0.04,
        "oos_ic_test": 0.03,
        "oos_stability_passed": True,
        "ic_decay_ratio": 0.65,
        "ic_decay_passed": True,
        "yearly_validation_passed": True,
        "yearly_observed": [],
    })
    print("  因子插入 [OK]")

    # 测试 Phase 3b backtest 插入（含新字段，无 pandas 依赖）
    try:
        import pandas as pd
        mock_layer = pd.DataFrame(
            {"L1": [0.01, 0.02], "L2": [0.02, 0.03], "L3": [0.03, 0.04],
             "L4": [0.04, 0.05], "L5": [0.05, 0.06]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
    except ImportError:
        mock_layer = None  # 测试无 layer_returns 场景

    insert_backtest(conn, {
        "factor_id": "f_test_3b",
        "annual_return": 0.15,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.2,
        "win_rate": 0.55,
        "turnover_est": 0.30,
        "avg_impact_cost_bps": 0.35,
        "total_cost_annual": 0.018,
        "layer_returns": mock_layer,
    })
    print("  回测插入 [OK]")

    # 验证读取
    row = conn.execute("SELECT * FROM factors WHERE factor_id='f_test_3b'").fetchone()
    print(f"  因子 monotonicity={row['monotonicity']}, monotonicity_passed={row['monotonicity_passed']} [OK]")

    bt_row = conn.execute("SELECT * FROM backtests WHERE factor_id='f_test_3b'").fetchone()
    print(f"  回测 avg_impact_cost_bps={bt_row['avg_impact_cost_bps']}, total_cost_annual={bt_row['total_cost_annual']} [OK]")

    conn.close()
    print("\n自检通过.")

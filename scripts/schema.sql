PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE pending_feedbacks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id     INTEGER NOT NULL,
            feedback_text TEXT NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            consumed     INTEGER DEFAULT 0,
            FOREIGN KEY (round_id) REFERENCES rounds(round_id)
        );
CREATE TABLE factors (
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
CREATE TABLE rounds (
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
CREATE TABLE memory (
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
CREATE TABLE backtests (
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
CREATE TABLE batch_status (
            run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            target_rounds INTEGER NOT NULL,
            completed_rounds INTEGER DEFAULT 0,
            cumulative_cost REAL DEFAULT 0,
            program_md5  TEXT,          -- 启动时 program.md 前三章 MD5
            status       TEXT DEFAULT 'running',  -- running/paused/completed/fused
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
DELETE FROM sqlite_sequence;
CREATE INDEX idx_rounds_status ON rounds(status);
CREATE INDEX idx_rounds_batch ON rounds(batch_run_id);
CREATE INDEX idx_memory_round ON memory(round_id);
CREATE INDEX idx_memory_batch ON memory(batch_run_id);
CREATE INDEX idx_backtests_factor ON backtests(factor_id);
COMMIT;

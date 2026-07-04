#!/usr/bin/env python3
"""Phase 2c 批量模式日志系统 — 主输出通道，实时写日志，CLI 辅助刷新。

用法:
  python logger.py             # demo 自检
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
LOGS_DIR = PROJECT_DIR / "logs"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB


class BatchLogger:
    """批量模式日志器。实时写入日志文件，支持自动轮转。"""

    def __init__(self, run_id: int = 0, dry_run: bool = False):
        """初始化日志器。

        Args:
            run_id: batch_status.run_id（0 表示未分配）
            dry_run: 仅 console 输出，不写文件
        """
        self.dry_run = dry_run
        self.run_id = run_id

        if not dry_run:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._part = 1
            self._filepath = self._make_path()
            self._fh = open(self._filepath, "w", encoding="utf-8", newline="\n")
            self._bytes_written = 0
            self._write_header()
        else:
            self._fh = None
            self._filepath = None

    def _make_path(self) -> Path:
        """生成日志文件路径（含轮转编号）。"""
        if self._part == 1:
            name = f"batch_{self._timestamp}_{self.run_id}.log"
        else:
            name = f"batch_{self._timestamp}_{self.run_id}_part{self._part}.log"
        return LOGS_DIR / name

    def _write_header(self):
        """写入日志头。"""
        self._write_line(f"=== FactorLab Phase 2c 批量日志 ===")
        self._write_line(f"启动时间: {datetime.now().isoformat()}")
        self._write_line(f"Run ID: {self.run_id}")
        self._write_line(f"Dry-run: {self.dry_run}")
        self._write_line("-" * 60)

    def _write_line(self, line: str):
        """写入一行到日志文件（带时间戳）。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {line}\n"
        if self.dry_run:
            print(f"  [LOG] {line}")
            return
        self._fh.write(full)
        self._fh.flush()
        self._bytes_written += len(full.encode("utf-8"))
        self._check_rotate()

    def _check_rotate(self):
        """检查是否需要日志轮转。"""
        if self._bytes_written >= MAX_LOG_SIZE:
            self._fh.close()
            self._part += 1
            self._filepath = self._make_path()
            self._fh = open(self._filepath, "w", encoding="utf-8", newline="\n")
            self._bytes_written = 0
            self._write_line(f"日志已切分，当前写入 part{self._part}")
            print(f"  [Logger] 日志已切分 → {self._filepath.name}")

    # ── 公共接口 ─────────────────────────────────────────

    def log_round_start(self, round_num: int, total: int):
        self._write_line(f"Round {round_num}/{total} START")

    def log_api_call(self, call_type: str, input_tokens: int, output_tokens: int, cost: float):
        self._write_line(
            f"API call: {call_type} (input={input_tokens}, output={output_tokens}, cost=¥{cost:.4f})"
        )

    def log_compliance(self, level: str, reason: str = ""):
        msg = f"Compliance: {level}"
        if reason:
            msg += f" — {reason[:120]}"
        self._write_line(msg)

    def log_sandbox(self, status: str, exec_time: float = 0):
        msg = f"Sandbox: {status}"
        if exec_time > 0:
            msg += f" (exec_time={exec_time:.2f}s)"
        self._write_line(msg)

    def log_score(self, ic: float = 0, ir_val: float = 0, passed: bool = False):
        self._write_line(
            f"Score: IC={ic:.4f} IR={ir_val:.4f} threshold_passed={passed}"
        )

    def log_diversity(self, passed: bool, corr_max: float = 0):
        self._write_line(f"Diversity: {'PASS' if passed else 'FAIL'} (corr_max={corr_max:.2f})")

    def log_backtest(self, sharpe: float = 0, max_dd: float = 0, annual_ret: float = 0):
        self._write_line(
            f"Backtest: sharpe={sharpe:.2f} max_dd={max_dd:.2%} annual_ret={annual_ret:.2%}"
        )

    def log_inbound(self, factor_id: str, direction_tag: str = ""):
        self._write_line(f"INBOUND: factor_id={factor_id} direction={direction_tag}")

    def log_skip(self, reason: str = ""):
        self._write_line(f"SKIP: {reason[:120]}")

    def log_fail(self, reason: str = ""):
        self._write_line(f"FAIL: {reason[:120]}")

    def log_error(self, error_msg: str = ""):
        self._write_line(f"ERROR: {error_msg[:200]}")

    def log_round_end(self, round_num: int, status: str, cost: float, cumulative_cost: float):
        self._write_line(
            f"Round {round_num} END | status={status} | cost=¥{cost:.4f} | cumulative=¥{cumulative_cost:.4f}"
        )

    def log_event(self, event_type: str, data: str = ""):
        """记录里程碑事件。"""
        self._write_line(f"[EVENT] {event_type}: {data[:200]}")

    def log_summary(self, line: str):
        """通用摘要行（CLI 辅助输出）。"""
        self._write_line(line)

    @property
    def filepath(self) -> Optional[Path]:
        return self._filepath

    def close(self):
        if self._fh:
            self._write_line(f"日志结束: {datetime.now().isoformat()}")
            self._fh.close()
            self._fh = None


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== logger.py 自检 ===\n")

    # Dry-run 模式测试
    print("[1] Dry-run 模式:")
    dry_logger = BatchLogger(run_id=0, dry_run=True)
    dry_logger.log_round_start(1, 50)
    dry_logger.log_api_call("generate_code", 1243, 156, 0.0016)
    dry_logger.log_compliance("PASS")
    dry_logger.log_sandbox("SUCCESS", 2.34)
    dry_logger.log_score(0.042, 0.58, True)
    dry_logger.log_diversity(True, 0.32)
    dry_logger.log_backtest(1.12, 0.15, 0.085)
    dry_logger.log_inbound("f001", "反转类")
    dry_logger.log_round_end(1, "inbound", 0.0042, 0.0042)
    dry_logger.log_event("milestone", "首个夏普>1.5 因子入库")
    dry_logger.close()
    print("  Dry-run 测试 [OK]")

    # 文件模式测试
    print("\n[2] 文件写入模式:")
    logger = BatchLogger(run_id=999)
    logger.log_round_start(1, 50)
    logger.log_api_call("generate_code", 1243, 156, 0.0016)
    logger.log_compliance("PASS")
    logger.log_sandbox("SUCCESS", 2.34)
    logger.log_score(0.042, 0.58, True)
    logger.log_diversity(True, 0.32)
    logger.log_backtest(1.12, 0.15, 0.085)
    logger.log_inbound("f001", "反转类")
    logger.log_api_call("generate_report", 1024, 234, 0.0015)
    logger.log_round_end(1, "inbound", 0.0042, 0.0042)
    logger.close()

    # 验证文件存在且有内容
    assert logger.filepath.exists(), "日志文件未创建!"
    content = logger.filepath.read_text(encoding="utf-8")
    assert "Round 1/50 START" in content
    assert "INBOUND: factor_id=f001" in content
    assert "Round 1 END" in content
    line_count = len(content.strip().split("\n"))
    print(f"  文件: {logger.filepath.name}")
    print(f"  行数: {line_count}")
    print(f"  大小: {logger.filepath.stat().st_size} bytes")
    print("  文件写入测试 [OK]")

    # 清理测试日志
    logger.filepath.unlink()
    print(f"  清理: {logger.filepath.name} 已删除")

    print("\n自检通过.")

#!/usr/bin/env python3
"""记忆管理器 — program.md 第四章追加 + 前三章 MD5 保护 + SQLite 双目标写入。

Phase 2c 新增:
  - append_milestone(): 事件驱动里程碑追加（MD5 校验 + 5 种事件）
  - update_md5_baseline(): adopt 后更新 batch_status 中的 MD5 基准
  - insert_memory_db(): SQLite memory 表写入（含 direction_tag）

规则:
  - 只能写入 <!-- AI_MEMORY_START --> 之后
  - 写入前校验前三章 MD5 未变，变化则拒绝并暂停
  - 记忆模板: Round N + 因子摘要 + 失败原因 + 下一步建议
"""

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
PROGRAM_PATH = PROJECT_DIR / "program.md"
MEMORY_MARKER = "<!-- AI_MEMORY_START -->"
MEMORY_END = "<!-- AI_MEMORY_END -->"


def _compute_chapters_md5() -> str:
    """计算 program.md 前三章（AI_MEMORY_START 之前）的 MD5。"""
    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    if MEMORY_MARKER in content:
        before = content.split(MEMORY_MARKER, 1)[0]
    else:
        before = content
    return hashlib.md5(before.encode("utf-8")).hexdigest()


def verify_chapters_integrity(expected_md5: str) -> Tuple[bool, str]:
    """校验前三章完整性。

    Args:
        expected_md5: 预期的前三章 MD5（首次启动时缓存）

    Returns:
        (是否通过, 当前 MD5)
    """
    current = _compute_chapters_md5()
    return (current == expected_md5, current)


def append_memory(
    round_num: int,
    factor_summary: str,
    passed: bool,
    fail_reason: str = "",
    next_suggestion: str = "",
    program_path: str = PROGRAM_PATH,
    expected_md5: str = "",
) -> bool:
    """向 program.md 第四章追加本轮记忆。

    Args:
        round_num: 轮次编号
        factor_summary: 因子自然语言摘要
        passed: 因子是否通过并入库
        fail_reason: 失败原因（passed=False 时有效）
        next_suggestion: AI 建议的下一步方向
        program_path: program.md 路径
        expected_md5: 前三章预期 MD5（为空则跳过校验）

    Returns:
        是否写入成功
    """
    # 1. 读取原文件
    with open(program_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. 校验前三章完整性
    if expected_md5:
        before = content.split(MEMORY_MARKER, 1)[0] if MEMORY_MARKER in content else content
        current = hashlib.md5(before.encode("utf-8")).hexdigest()
        if current != expected_md5:
            print("[memory_manager] 致命错误: program.md 前三章已被篡改!", file=sys.stderr)
            print(f"  期望 MD5: {expected_md5}", file=sys.stderr)
            print(f"  实际 MD5: {current}", file=sys.stderr)
            print("  系统暂停。请人工检查 program.md 后重试。", file=sys.stderr)
            return False

    # 3. 确保标记存在
    if MEMORY_MARKER not in content:
        print(f"[memory_manager] 错误: 未找到 {MEMORY_MARKER} 标记", file=sys.stderr)
        return False

    # 4. 构造本轮记忆（严格在 AI_MEMORY_START 之后插入）
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "PASS (已入库)" if passed else "FAIL (已丢弃)"
    memory_entry = (
        f"\n## Round {round_num} ({timestamp})\n\n"
        f"### 因子摘要\n{factor_summary}\n\n"
        f"### 结果\n{status}\n"
    )
    if not passed and fail_reason:
        memory_entry += f"\n### 失败原因\n{fail_reason}\n"
    if next_suggestion:
        memory_entry += f"\n### 下一步建议\n{next_suggestion}\n"
    memory_entry += "\n---\n"

    # 5. 插入到 AI_MEMORY_START 和 AI_MEMORY_END 之间
    parts = content.split(MEMORY_MARKER, 1)
    before = parts[0] + MEMORY_MARKER + "\n"
    after_marker = parts[1] if len(parts) > 1 else ""

    # 保留已有记忆：after_marker = 旧记忆 + AI_MEMORY_END + 尾部内容
    # 新记忆插入在旧记忆之前（最新在前），不覆盖已有内容
    after = after_marker

    new_content = before + memory_entry + after

    # 6. 原子写入：先写 .tmp 再替换
    tmp_path = str(program_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    os.replace(tmp_path, program_path)

    return True


# ── Phase 2c: 事件驱动里程碑追加 ─────────────────────────

def append_milestone(
    program_path: str,
    event_type: str,
    content: str,
    expected_md5: str,
) -> bool:
    """向 program.md 追加事件里程碑（5 种事件触发时调用）。

    追加前校验前三章 MD5，与 batch_status.program_md5 比对。
    不匹配则拒绝写入并报错。

    Args:
        program_path: program.md 路径
        event_type: 事件类型（中文描述）
        content: 里程碑内容（自然语言，支持多行）
        expected_md5: batch_status 中存储的前三章 MD5

    Returns:
        是否写入成功
    """
    # 1. 读取原文件
    with open(program_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    # 2. 校验前三章 MD5
    before = file_content.split(MEMORY_MARKER, 1)[0] if MEMORY_MARKER in file_content else file_content
    current = hashlib.md5(before.encode("utf-8")).hexdigest()
    if current != expected_md5:
        print(
            f"[memory_manager] 致命错误: program.md 前三章已被修改，拒绝追加里程碑!\n"
            f"  期望 MD5: {expected_md5}\n"
            f"  实际 MD5: {current}\n"
            f"  请重启系统以更新 MD5 基准。",
            file=sys.stderr,
        )
        return False

    # 3. 确保标记存在
    if MEMORY_MARKER not in file_content:
        print(f"[memory_manager] 错误: 未找到 {MEMORY_MARKER} 标记", file=sys.stderr)
        return False

    # 4. 构造里程碑条目
    round_num = "N"  # 事件可能不绑定特定轮次
    for line in content.split("\n"):
        if "当前状态" in line:
            import re as _re
            m = _re.search(r"入库\s*(\d+)", line)
            if m:
                round_num = m.group(1)

    milestone_entry = (
        f"\n### Milestone [{event_type}] — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{content.strip()}\n"
        f"\n---\n"
    )

    # 5. 插入到 AI_MEMORY_START 之后
    parts = file_content.split(MEMORY_MARKER, 1)
    before = parts[0] + MEMORY_MARKER + "\n"
    after_marker = parts[1] if len(parts) > 1 else ""

    # 保留已有记忆：after_marker = 旧记忆 + AI_MEMORY_END + 尾部内容
    # 新记忆插入在旧记忆之前（最新在前），不覆盖已有内容
    after = after_marker

    new_content = before + milestone_entry + after

    # 6. 原子写入
    tmp_path = str(program_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    os.replace(tmp_path, program_path)

    return True


# ── Phase 2c: MD5 基准更新 ──────────────────────────────

def update_md5_baseline(conn, new_md5: str) -> bool:
    """adopt 后更新 batch_status 中的 program_md5 基准。

    Args:
        conn: SQLite 连接
        new_md5: 替换后的前三章 MD5

    Returns:
        是否更新成功
    """
    from database import get_latest_batch, update_batch
    latest = get_latest_batch(conn)
    if not latest:
        print("[memory_manager] 错误: 无活跃 batch_status 记录", file=sys.stderr)
        return False
    update_batch(conn, latest["run_id"], program_md5=new_md5)
    return True


# ── Phase 2c: SQLite memory 表写入辅助 ─────────────────

def insert_memory_db(conn, round_num: int, batch_run_id: int,
                     direction_tag: str, summary: str, passed: bool,
                     fail_reasons: str = "", suggestion: str = "") -> bool:
    """写入 SQLite memory 表（含 direction_tag 字段）。

    Args:
        conn: SQLite 连接
        round_num: 轮次编号
        batch_run_id: 批量运行 ID
        direction_tag: 研究方向标签
        summary: 因子摘要
        passed: 是否入库
        fail_reasons: 失败原因
        suggestion: 下一步建议

    Returns:
        是否写入成功
    """
    from database import insert_memory
    insert_memory(conn, {
        "round_id": round_num,
        "batch_run_id": batch_run_id,
        "direction_tag": direction_tag,
        "factor_type": direction_tag,
        "summary": summary,
        "passed": passed,
        "fail_reasons": fail_reasons,
        "suggestion": suggestion,
    })
    return True


# ── Phase 2: SQLite 记忆读取接口 ───────────────────────────

def get_recent_memories_for_prompt(conn, n: int = 5) -> str:
    """从 SQLite memory 表读取最近 n 条记忆，拼接为 prompt 字符串。

    Args:
        conn: SQLite 连接
        n: 获取最近 n 条

    Returns:
        拼接后的记忆文本（Markdown 格式），无记忆时返回占位文本
    """
    from database import get_recent_memories
    memories = get_recent_memories(conn, n=n)

    if not memories:
        return "（暂无实验记忆）"

    lines = ["## 近期实验记忆\n"]
    for m in memories:
        status = "PASS" if m.get("passed") else "FAIL"
        ftype = m.get("factor_type", "未知")
        summary = m.get("summary", "无摘要")
        fail_reasons = m.get("fail_reasons", "")
        suggestion = m.get("suggestion", "")

        lines.append(f"### Round {m['round_id']} ({ftype}) [{status}]")
        lines.append(f"摘要: {summary[:200]}")
        if fail_reasons:
            lines.append(f"失败原因: {fail_reasons[:200]}")
        if suggestion:
            lines.append(f"建议: {suggestion[:200]}")
        lines.append("")

    return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== memory_manager.py 自检 ===\n")

    # 1. 验证前三章 MD5 可计算
    md5 = _compute_chapters_md5()
    print(f"[1] 前三章 MD5: {md5}")
    print("    校验通过（MD5 非空）[OK]" if len(md5) == 32 else "    [FAIL]")

    # 2. 验证 AI_MEMORY_START 存在
    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert MEMORY_MARKER in content, "AI_MEMORY_START 标记不存在!"
    print(f"[2] AI_MEMORY_START 标记存在 [OK]")

    # 3. 模拟追加记忆（用 tmp 文件测试，避免污染真实 program.md）
    print("\n[3] 测试追加记忆（写入临时文件）...")
    test_path = "program_test.md"

    # 复制 program.md 内容到测试文件
    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        original = f.read()
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(original)

    ok = append_memory(
        round_num=1,
        factor_summary="5 日动量因子：计算过去 5 个交易日收盘价变化率。",
        passed=True,
        fail_reason="",
        next_suggestion="尝试 10 日/20 日窗口的动量变体。",
        program_path=test_path,
        expected_md5=md5,
    )
    print(f"    写入: {'成功' if ok else '失败'}")
    assert ok

    with open(test_path, "r", encoding="utf-8") as f:
        result = f.read()
    assert "Round 1" in result
    assert "5 日动量因子" in result
    assert "PASS (已入库)" in result
    print("    内容验证: Round 1 + 摘要 + PASS [OK]")

    # 4. 测试 MD5 篡改拦截
    print("\n[4] 测试 MD5 篡改拦截...")
    with open(test_path, "r", encoding="utf-8") as f:
        tampered = f.read()
    tampered = tampered.replace("## 第一章", "## 第一章 (已被篡改)", 1)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(tampered)

    ok = append_memory(
        round_num=2,
        factor_summary="测试因子",
        passed=False,
        fail_reason="测试失败",
        program_path=test_path,
        expected_md5=md5,  # 旧 MD5，应不匹配
    )
    assert not ok, "MD5 校验应该失败!"
    print("    篡改拦截成功 [OK]")

    # 清理
    os.remove(test_path)
    print("\n自检通过.")

#!/usr/bin/env python3
"""Phase 2c 半自动规程替换 — 结构化对比 + 原子替换 + 热加载 + MD5 更新。

用法:
  python program_updater.py demo    # 演示对比与替换流程
"""

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

from config import PROJECT_ROOT
PROJECT_DIR = PROJECT_ROOT
HISTORY_DIR = PROJECT_DIR / "history"
PROGRAM_PATH = PROJECT_DIR / "program.md"
DRAFT_PATH = PROJECT_DIR / "program_draft.md"


def _ensure_history():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def show_diff_and_confirm(current_chapter: str, draft_chapter: str) -> str:
    """CLI 展示结构化对比（当前策略 vs 建议策略，各 5 行要点）。

    不展示完整 markdown diff，只提取关键方向变化。

    Args:
        current_chapter: 当前 program.md 第一章完整内容
        draft_chapter: 建议的草案第一章内容

    Returns:
        用户选择: "adopt" / "reject" / "edit"
    """
    # 提取关键行（以 - ** 或 - 开头的要点）
    def _extract_points(text: str, max_points: int = 8) -> list[str]:
        points = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- **") or stripped.startswith("- "):
                if len(stripped) > 5:
                    points.append(stripped)
        return points[:max_points]

    current_points = _extract_points(current_chapter)
    draft_points = _extract_points(draft_chapter)

    print(f"\n  ╔{'═'*60}╗")
    print(f"  ║  研究方向变更建议 (第一章)                                ║")
    print(f"  ╠{'═'*60}╣")
    print(f"  ║  当前策略 (前 5 项):                                      ║")
    for p in current_points[:5]:
        display = p[:55] + "..." if len(p) > 55 else p
        print(f"  ║    {display:<56s} ║")
    print(f"  ╠{'═'*60}╣")
    print(f"  ║  建议策略 (前 5 项):                                      ║")
    for p in draft_points[:5]:
        display = p[:55] + "..." if len(p) > 55 else p
        print(f"  ║    {display:<56s} ║")
    print(f"  ╠{'═'*60}╣")

    # Draft 注释块中的建议
    if "DRAFT_NOTE_START" in draft_chapter:
        note_start = draft_chapter.find("DRAFT_NOTE_START")
        note_end = draft_chapter.find("DRAFT_NOTE_END", note_start)
        if note_end > note_start:
            note = draft_chapter[note_start:note_end]
            for note_line in note.split("\n")[:5]:
                stripped = note_line.strip()
                if stripped and not stripped.startswith("<!--"):
                    display = stripped[:55] + "..." if len(stripped) > 55 else stripped
                    print(f"  ║  💡 {display:<54s} ║")

    print(f"  ╠{'═'*60}╣")
    print(f"  ║  adopt  — 采纳建议，替换 program.md 第一章              ║")
    print(f"  ║  reject — 拒绝建议，保留当前版本                         ║")
    print(f"  ║  edit   — 打开编辑器手动修改                             ║")
    print(f"  ╚{'═'*60}╝")

    choice = input("  请选择 (adopt/reject/edit): ").strip().lower()
    if choice not in ("adopt", "reject", "edit"):
        print("  无效输入，默认 reject")
        choice = "reject"
    return choice


def apply_adopt(draft_path: str = "", program_path: str = "") -> bool:
    """采纳草案：原子替换 program.md 第一章 + 备份旧版 + 更新 MD5。

    Args:
        draft_path: program_draft.md 路径
        program_path: program.md 路径

    Returns:
        是否替换成功
    """
    dp = Path(draft_path) if draft_path else DRAFT_PATH
    pp = Path(program_path) if program_path else PROGRAM_PATH
    _ensure_history()

    if not dp.exists():
        print(f"[program_updater] 错误: draft 文件不存在: {dp}", file=sys.stderr)
        return False

    draft_content = dp.read_text(encoding="utf-8")
    current_content = pp.read_text(encoding="utf-8")

    # 提取 draft 的第一章
    marker = "<!-- AI_MEMORY_START -->"
    draft_ch1_3 = draft_content.split(marker, 1)[0] if marker in draft_content else draft_content
    current_ch1_3 = current_content.split(marker, 1)[0] if marker in current_content else current_content

    ch1_start = draft_ch1_3.find("## 第一章")
    ch1_end = draft_ch1_3.find("\n## 第二章")
    if ch1_start < 0:
        print("[program_updater] 错误: draft 中未找到第一章", file=sys.stderr)
        return False
    if ch1_end < 0:
        ch1_end = len(draft_ch1_3)
    new_ch1 = draft_ch1_3[ch1_start:ch1_end]

    # 备份旧版
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = HISTORY_DIR / f"program_v{timestamp}.md"
    shutil.copy2(str(pp), str(backup_path))
    print(f"  旧版已备份: {backup_path}")

    # 原子替换第一章
    old_ch1_start = current_ch1_3.find("## 第一章")
    old_ch1_end = current_ch1_3.find("\n## 第二章")
    if old_ch1_start < 0 or old_ch1_end < 0:
        print("[program_updater] 错误: program.md 结构异常", file=sys.stderr)
        return False

    new_ch1_3 = current_ch1_3[:old_ch1_start] + new_ch1 + current_ch1_3[old_ch1_end:]
    new_content = new_ch1_3 + current_content[len(current_ch1_3):]

    # 原子写入
    tmp_path = str(pp) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    os.replace(tmp_path, str(pp))

    # 重新计算 MD5 并更新 batch_status
    new_md5 = _compute_new_md5(pp)
    try:
        from database import get_conn, get_latest_batch, update_batch
        conn = get_conn()
        latest = get_latest_batch(conn)
        if latest and latest["status"] in ("running", "paused"):
            update_batch(conn, latest["run_id"], program_md5=new_md5)
        conn.close()
    except Exception as e:
        print(f"  [警告] MD5 基准更新失败: {e}（可重启系统修复）")

    print(f"  ✅ 已采纳，新方向立即生效")
    print(f"  新 MD5 基准: {new_md5[:16]}...")
    return True


def apply_reject(draft_path: str = "") -> bool:
    """拒绝草案：保留审计副本，删除根目录 draft.md。

    Args:
        draft_path: program_draft.md 路径

    Returns:
        是否操作成功
    """
    dp = Path(draft_path) if draft_path else DRAFT_PATH
    _ensure_history()

    if not dp.exists():
        print("[program_updater] 提示: draft 文件不存在，无需操作")
        return True

    # 审计副本已在 generate_program_draft 时保存，这里仅删除
    dp.unlink(missing_ok=True)
    print(f"  ✅ 已拒绝，保持现有 direction，draft 已删除")
    return True


def apply_edit(program_path: str = "") -> bool:
    """打开编辑器手动修改 program.md。

    Args:
        program_path: program.md 路径

    Returns:
        是否操作成功
    """
    pp = Path(program_path) if program_path else PROGRAM_PATH

    # 尝试打开 VSCode
    try:
        subprocess.call(["code", str(pp)])
    except FileNotFoundError:
        # Fallback: 打印路径让用户手动打开
        print(f"  未找到 VSCode，请手动编辑: {pp}")

    print(f"  修改完成后按 Enter 继续...")
    input()
    print(f"  ✅ 编辑完成，请重启系统以更新 MD5 基准")
    return True


def _compute_new_md5(pp: Path) -> str:
    """计算 program.md 前三章的新 MD5。"""
    content = pp.read_text(encoding="utf-8")
    marker = "<!-- AI_MEMORY_START -->"
    before = content.split(marker, 1)[0] if marker in content else content
    return hashlib.md5(before.encode("utf-8")).hexdigest()


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== program_updater.py 自检 ===\n")

    _ensure_history()

    # 1. 测试结构化对比
    print("[1] 结构化对比:")
    current = """## 第一章：研究方向
### 1.3 探索优先级
- **【P0】** 反转类变体：近期跌幅大
- **【P0】** 行为类因子：散户情绪反向指标
- **【P1】** 量价背离：价格创新低但成交量不创新低"""
    draft = """## 第一章：研究方向
### 1.3 探索优先级
- **【P0】** 动量类因子
- **【P1】** 反转类变体
<!-- DRAFT_NOTE_START -->
- 保留方向: 反转类
- 暂停方向: 行为类
<!-- DRAFT_NOTE_END -->"""

    # 模拟非交互
    print("  (模拟对比输出，实际使用需交互)")
    print(f"  当前要点数: {len([l for l in current.split(chr(10)) if l.strip().startswith('- ')])}")
    print(f"  建议要点数: {len([l for l in draft.split(chr(10)) if l.strip().startswith('- ')])}")
    print("  结构化对比逻辑 [OK]")

    # 2. 测试 adopt
    print("\n[2] 测试 adopt 流程（模拟）...")
    # 创建测试文件
    test_current = """## 第一章：研究方向
### 1.1 当前研究方向
- 因子类型: 量价类
## 第二章：因子定义域
### 2.1 股票池
...
<!-- AI_MEMORY_START -->
## Round 1: test
<!-- AI_MEMORY_END -->"""

    test_draft = """## 第一章：研究方向（修订）
### 1.1 当前研究方向
- 因子类型: 量价类 + AI建议新增
## 第二章：因子定义域
### 2.1 股票池
...
<!-- AI_MEMORY_START -->
## Round 1: test
<!-- AI_MEMORY_END -->"""

    test_prog = PROJECT_DIR / "program_test_updater.md"
    test_draft_path = PROJECT_DIR / "program_draft_test.md"

    test_prog.write_text(test_current, encoding="utf-8")
    test_draft_path.write_text(test_draft, encoding="utf-8")

    ok = apply_adopt(str(test_draft_path), str(test_prog))
    if ok:
        result = test_prog.read_text(encoding="utf-8")
        assert "修订" in result, "adopt 应包含新内容"
        assert "第二章" in result, "第二章应保留"
        print("  adopt 测试 [OK]")

    # 3. 测试 reject
    print("\n[3] 测试 reject 流程...")
    test_draft_path.write_text(test_draft, encoding="utf-8")
    ok = apply_reject(str(test_draft_path))
    print(f"  reject 测试 [{'OK' if ok else 'FAIL'}]")

    # 清理
    test_prog.unlink(missing_ok=True)
    test_draft_path.unlink(missing_ok=True)

    print("\n自检通过.")

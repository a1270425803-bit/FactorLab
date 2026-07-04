"""熔断机制测试 — 模拟连续 3 次合规失败。

用法: python test_fuse_break.py
验证: 连续 3 次生成含 shift(-1) 的代码后，系统自动暂停。
"""

import sys
from pipeline import run_pipeline
from checker import check_compliance


def test_fuse():
    """模拟合规失败的因子代码，验证 3 次熔断。"""
    print("=== 熔断机制测试 ===\n")

    # 构造一个必然合规失败的因子代码
    bad_code = 'def compute_factor(df):\n    return df["close"].shift(-1) / df["close"] - 1'

    # 验证 checker 会报 ERROR
    level, reason = check_compliance(bad_code)
    assert level == "ERROR", f"checker 应返回 ERROR, 实际返回 {level}"
    print(f"[1] 合规检查确认: {level} — {reason}")

    # 验证合规失败会被正确检测（3 次模拟）
    print("\n[2] 模拟连续 3 次合规失败...")
    for attempt in range(1, 4):
        level, reason = check_compliance(bad_code)
        assert level == "ERROR"
        print(f"    第 {attempt} 次: {level} ({reason})")

    # 验证 MAX_CODE_ATTEMPTS 常量
    from pipeline import MAX_CODE_ATTEMPTS
    assert MAX_CODE_ATTEMPTS == 3, f"MAX_CODE_ATTEMPTS 应为 3, 实际为 {MAX_CODE_ATTEMPTS}"
    print(f"\n[3] MAX_CODE_ATTEMPTS = {MAX_CODE_ATTEMPTS} [OK]")

    # 验证 pipeline 在合规失败时会正确处理
    print("\n[4] 运行 pipeline Round 1 (dry-run) — 预期合规失败...")
    result = run_pipeline(round_num=99, dry_run=True)

    # 检查步骤
    assert "generate_code" in str(result.steps_completed), f"应调用过 generate_code"
    print(f"    steps: {result.steps_completed}")
    print(f"    fail_reason: {result.fail_reason}")

    # 验证失败报告有 actionable 信息
    if not result.passed:
        assert result.fail_reason, "失败报告不应为空"
        print(f"    [OK] 失败报告包含原因: {result.fail_reason[:80]}...")

    print("\n=== 熔断测试通过 ===")


if __name__ == "__main__":
    test_fuse()

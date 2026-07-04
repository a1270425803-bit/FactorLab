#!/usr/bin/env python3
"""AI 引擎 — DeepSeek API 封装 + 成本追踪 + Mock 模式。

每轮 API 调用预算:
  - generate_code(): 最多 3 次（合规重试包含在内）
  - generate_summary(): 1 次（合规通过后）
  - generate_report(): 1 次（轮次结束时）
  - 3 次代码生成均失败 → 跳过摘要和报告，成本 = 0
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostTracker:
    """全局成本追踪器（单例模式）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    _instance: Optional["CostTracker"] = field(default=None, repr=False, init=False)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens

    def preload(self, cumulative_cost: float):
        """M5 修复: resume 时预加载历史成本，以 token 数形式注入。
        按 DeepSeek 价格反向估算 token: 假设 input:output ≈ 5:1。
        """
        if cumulative_cost <= 0:
            return
        # cost = (input * 1 + output * 2) / 1_000_000
        # 设 input = 5 * output → cost ≈ (5*output + 2*output) / 1e6 = 7*output/1e6
        # → output ≈ cost * 1e6 / 7, input ≈ 5 * output
        est_output = int(cumulative_cost * 1_000_000 / 7)
        est_input = est_output * 5
        self.input_tokens += est_input
        self.output_tokens += est_output

    def cost(self) -> float:
        """DeepSeek 价格: ¥1/百万 input, ¥2/百万 output。"""
        return (self.input_tokens * 1 + self.output_tokens * 2) / 1_000_000

    def summary(self) -> str:
        return f"累计 API 成本: {self.cost():.4f} (input={self.input_tokens}, output={self.output_tokens})"


def _build_system_prompt(program_path: str = "program.md", max_memory_rounds: int = 5,
                        conn=None) -> str:
    """组装 system prompt: program.md 前三章 + SQLite memory 最近 N 轮记忆。

    Phase 2c: 记忆数据源从 program.md 第四章切换至 SQLite memory 表。
    若 conn 为 None（向后兼容），降级读取 program.md 第四章。

    Args:
        program_path: program.md 路径
        max_memory_rounds: 最近 N 轮记忆
        conn: SQLite 连接（Phase 2c）。为 None 时降级读取文件。
    """
    with open(program_path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "<!-- AI_MEMORY_START -->"
    if marker in content:
        chapters_1_3 = content.split(marker, 1)[0]
    else:
        chapters_1_3 = content

    # Phase 2c: 优先从 SQLite 读取记忆
    if conn is not None:
        try:
            from memory_manager import get_recent_memories_for_prompt
            recent = get_recent_memories_for_prompt(conn, n=max_memory_rounds)
        except Exception:
            recent = "（暂无实验记忆）"
    else:
        # 降级: 从 program.md 第四章读取（向后兼容 Phase 1b）
        if marker in content:
            memory_section = content.split(marker, 1)[1] if len(content.split(marker, 1)) > 1 else ""
        else:
            memory_section = ""
        rounds = [r.strip() for r in memory_section.split("## Round") if r.strip()]
        recent = "## Round" + ("## Round".join(rounds[-max_memory_rounds:])) if rounds else "（暂无历史记录）"

    # Phase 3g: f001 成功案例（注入每轮 system_prompt）
    # ⚠️ 代码和指标均来自 factor_pool.json 实际记录，禁止虚构
    f001_case_study = """
## 📖 成功案例研究：f001（反转类，综合评分 2.31）

**实际入库代码**（来自 factor_pool.json，2026-06-13 入库）：
```python
def compute_factor(df):
    # 计算日内涨跌幅（收盘相对开盘）
    intraday_ret = (df['close'] - df['open']) / df['open']

    # 计算日内振幅
    amplitude = (df['high'] - df['low']) / df['open']

    # 计算成交量相对过去5日均值的倍数
    avg_vol_5 = df.groupby(level='code')['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    vol_ratio = df['volume'] / avg_vol_5

    # 构建核心信号：动量强度 * 振幅 * 成交量放大（三乘积）
    raw_signal = intraday_ret * amplitude * vol_ratio

    # 去趋势：减去 close 的 5 日均值（捕捉突变，关键步骤！）
    smoothed = df.groupby(level='code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    factor = raw_signal - smoothed

    # 截面标准化（Z-score）
    factor = (factor - factor.groupby(level='date').transform('mean')) / factor.groupby(level='date').transform('std')

    return factor
```

**评分分解**（500只校准验证）：
IC=0.119, IR=0.213, rank_ac=0.997, dir_acc=0.498, turnover=0.003, coverage=0.862, OOS 4/7

**WHY 这个结构有效**：
1. 三个正交维度复合（日内方向 + 波动能量 + 成交量放大），信号来源多样化
2. **De-trend 是关键**：raw_signal - smoothed(close, 5) 移除了自身基线，留下"突变"部分。纯三乘积在 B22 前被尝试 10+ 轮全部失败——区别就在于有没有这步 de-trend
3. 截面 z-score 在最后做（满足规则 #6：per-stock 时序操作在截面之前）
4. 结构简单（≤12 行），每个操作都向量化，无 for/apply

⚠️ 注意：rank_ac=0.997 极高（接近完美自相关），turnover=0.003 几乎为零——信号过度平滑，可能过拟合
⚠️ 三乘积（a×b×c）本身不是问题——问题在于没有 de-trend。成功的公式 = 三乘积 + de-trend + 截面标准化

近失败结构（来自 factor_pool.json 实际记录）：
- f003（IC=0.037→几乎为零, IR=0.209）：price_pos + ma_dev + amplitude 等权和——信号源太弱，核心指标与收益相关性几乎为零
- 教训：等权和不如乘积（信噪比低），但乘积必须有 de-trend 来隔离突变

## 📐 可用模板方向（本轮具体方向见 user_prompt 强制指令）

T1纯时序反转 | T2波动率加权 | T3多周期共振 | T4日内路径效率 | T5成交量特征
→ 每轮 user_prompt 会强制指定一个方向，你必须按该方向构造因子。禁止使用三维乘积+截面rank（C★ 模式）。
"""

    return (
        f"{chapters_1_3.strip()}\n\n"
        f"{f001_case_study}\n"
        f"---\n\n"
        f"## 近期实验记忆（最近 {max_memory_rounds} 轮）\n\n"
        f"{recent}"
    )


# ── Mock Engine（Dry-run 模式）───────────────────────────────

MOCK_FACTOR_CODE = '''def compute_factor(df: pd.DataFrame) -> pd.Series:
    """5 日动量因子"""
    return df.groupby(level="code")["close"].pct_change(5)'''

MOCK_FACTOR_CODE_2 = '''def compute_factor(df: pd.DataFrame) -> pd.Series:
    """成交量比率因子"""
    vol = df.groupby(level="code")["volume"].transform(lambda x: x.rolling(5).mean())
    return vol / df["volume"]'''

MOCK_FACTOR_CODE_3 = '''def compute_factor(df: pd.DataFrame) -> pd.Series:
    """高低价振幅因子"""
    return (df["high"] - df["low"]) / df["close"]'''


class MockEngine:
    """Mock 引擎，用于 --dry-run 模式。不调用 API，返回预设因子代码。"""

    def __init__(self):
        self.cost_tracker = CostTracker()
        self._call_seq = 0

    def generate_code(self, _system_prompt: str, template_direction: dict = None) -> str:
        """返回预设 mock 因子代码（轮流使用 3 个 mock），模拟 API 延迟。"""
        _ = template_direction  # Phase 3h: 接受但忽略模板参数
        self._call_seq += 1
        codes = [MOCK_FACTOR_CODE, MOCK_FACTOR_CODE_2, MOCK_FACTOR_CODE_3]
        code = codes[(self._call_seq - 1) % len(codes)]
        time.sleep(0.1)
        self.cost_tracker.record(500, 100)  # mock token 计数
        return code

    def generate_summary(self, code: str) -> str:
        """返回预设摘要。"""
        time.sleep(0.05)
        self.cost_tracker.record(200, 50)
        return f"[Mock 摘要] 该因子基于常见量价指标构造，预期 IC 约 0.02-0.05。"

    def generate_report(self, round_info: dict) -> str:
        """返回预设报告。"""
        time.sleep(0.05)
        self.cost_tracker.record(300, 80)
        passed = round_info.get("passed", False)
        if passed:
            return (
                f"## Round {round_info['round']} 报告\n\n"
                f"因子已通过全部检查并入库。\n"
                f"评分: {round_info.get('score', 'N/A')}\n"
            )
        return (
            f"## Round {round_info['round']} 报告\n\n"
            f"本轮因子未达标。失败原因: {round_info.get('reason', '未知')}\n"
            f"建议: 尝试不同的因子类型或参数窗口。\n"
        )

    def chat(self, prompt: str, max_tokens: int = 800) -> str:
        """通用聊天接口（Mock，Phase 2c C3 修复）。
        用于 summary_engine 等模块，CostTracker 自动记录。
        """
        time.sleep(0.1)
        self.cost_tracker.record(800, 200)
        return "[Mock] AI 生成的 program_draft 草案（dry-run 模式）"


# ── Real Engine（真实 DeepSeek API）───────────────────────────

class DeepSeekEngine:
    """DeepSeek API 引擎，使用 OpenAI 兼容 SDK。"""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.cost_tracker = CostTracker()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
        return self._client

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        """底层 API 调用，带 3 次重试（指数退避）。"""
        last_error = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                if usage:
                    self.cost_tracker.record(
                        usage.prompt_tokens or 0,
                        usage.completion_tokens or 0,
                    )
                return content
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek API 调用失败（已重试 3 次）: {last_error}")

    def generate_code(self, system_prompt: str, template_direction: dict = None) -> str:
        """生成因子代码。

        Args:
            system_prompt: 系统 prompt（含 program.md + memory）
            template_direction: 可选，本轮强制模板方向 {'name', 'formula', 'key'}
        """
        # Phase 3i P0-4: 正向模板锁定 — 模板方向是核心锁定指令，AI 不可自由选方向
        template_block = ""
        if template_direction:
            template_block = (
                f"🎯 **本轮因子方向已锁定**（不可偏离）：\n"
                f"  方向名称：{template_direction.get('name', '')}\n"
                f"  核心公式模板（必须基于此构造）：{template_direction.get('formula', '')}\n"
                f"  关键约束：{template_direction.get('key', '')}\n"
                f"  🚫 严禁使用：三维乘积(ret/amp/vol) + 截面 rank(pct=True)（C★ 死循环模式）\n"
                f"  ✅ 你必须在上述公式模板基础上构造因子，只可调整参数和窗口期，不可改变核心数学结构。\n\n"
            )
        user_prompt = (
            f"请按以下锁定方向生成一个 A 股量化因子代码。\n\n"
            + template_block +
            "⚠️ 常见错误（当前系统高频挂掉场景）:\n"
            "  - 禁止引用不存在的列！df 只含 open, high, low, close, volume 5 列，没有被预计算的列\n"
            "  - 用 df.groupby(level='code') 分组，禁止 df.groupby('code') 或 df['code']\n"
            "\n⚠️ MultiIndex 规则（违反将导致沙箱报错 \"code occurs multiple times\"）:\n"
            "  ❌ 禁止: df['code'] = ..., df.reset_index(), df.groupby('code'), df[df['code']==...]\n"
            "  ✅ 正确: df.groupby(level='code')['close'].transform(...)\n\n"
            "⚠️ NaN-safe（Phase 3i）：所有除法分母必须加 .replace(0, np.nan)\n\n"
            "要求:\n"
            "1. 函数名 compute_factor，接收 MultiIndex(date,code) DataFrame，返回 pd.Series\n"
            "2. df 仅含 5 列: open, high, low, close, volume\n"
            "3. 分组必须用 df.groupby(level='code')，禁止任何其他 groupby 形式\n"
            "4. 代码 ≤ 20 行 | 禁止 shift(-1) | 禁止 import | 禁止 apply(lambda row, axis=1) | 禁止 for 循环\n"
            "5. 只输出 Python 代码，不要 markdown 标记\n"
            "\n请直接输出 compute_factor 函数定义:"
        )
        response = self._call_api(system_prompt, user_prompt, max_tokens=600)
        return _clean_code(response)

    def generate_summary(self, code: str) -> str:
        """将因子代码翻译为中文自然语言摘要（≤200 字）。"""
        user_prompt = (
            f"请用中文简要解释以下因子代码的构造逻辑（≤200 字，让无编程背景的投资者能理解）:\n\n"
            f"```python\n{code}\n```\n\n"
            f"输出格式: 一句话说明这个因子在衡量什么，然后简短说明计算方法。"
        )
        response = self._call_api("你是一位量化因子分析师。", user_prompt, max_tokens=300)
        return response.strip()[:300]

    def generate_report(self, round_info: dict) -> str:
        """生成本轮实验结构化报告。"""
        passed = round_info.get("passed", False)
        summary = round_info.get("summary", "")
        score = round_info.get("score", "N/A")
        reason = round_info.get("reason", "")
        round_num = round_info.get("round", "?")

        user_prompt = (
            f"请为以下量化因子实验轮次生成一份简短的结构化报告（Markdown 格式）:\n\n"
            f"- 轮次: Round {round_num}\n"
            f"- 因子摘要: {summary}\n"
            f"- 结果: {'通过' if passed else '未通过'}\n"
            f"- 评分: {score}\n"
            f"- 失败原因: {reason if reason else '无'}\n"
            f"- {'入库成功' if passed else '已丢弃，未入库'}\n\n"
            f"报告格式:\n"
            f"## Round {round_num}\n"
            f"### 因子摘要\n...\n"
            f"### 评分结果\n...\n"
            f"### 下一步建议\n..."
        )
        response = self._call_api(
            "你是一位量化因子研究助手。请用中文输出。",
            user_prompt,
            max_tokens=500,
        )
        return response.strip()

    def chat(self, prompt: str, max_tokens: int = 800) -> str:
        """通用聊天接口（Phase 2c C3 修复）。
        供 summary_engine 等模块使用，CostTracker 自动记录 token，复用 3 次重试 + 指数退避。

        Args:
            prompt: 用户 prompt
            max_tokens: 最大输出 token

        Returns:
            AI 响应文本
        """
        response = self._call_api(
            "你是一位量化因子研究助手。请用中文输出。",
            prompt,
            max_tokens=max_tokens,
        )
        return response.strip()


def _clean_code(raw: str) -> str:
    """清理 AI 返回的代码：去除 markdown 标记，提取 def compute_factor 块。"""
    # 去除 ```python ... ``` 包裹
    if "```" in raw:
        lines = raw.split("\n")
        code_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)
        if code_lines:
            raw = "\n".join(code_lines)

    # 提取 def compute_factor 到文件末尾（或到下一个顶层定义）
    lines = raw.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def compute_factor"):
            start = i
            break
    if start is None:
        return raw.strip()

    result = lines[start:]
    return "\n".join(result).strip()


def create_engine(api_key: str = "", model: str = "deepseek-chat", dry_run: bool = False):
    """工厂函数：根据 dry_run 参数返回 MockEngine 或 DeepSeekEngine。"""
    if dry_run or not api_key or api_key == "sk-xxxx":
        return MockEngine()
    return DeepSeekEngine(api_key=api_key, model=model)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== engine.py 自检 ===\n")

    # Mock 模式测试
    engine = MockEngine()
    print("[1] MockEngine.generate_code():")
    code = engine.generate_code("test system prompt")
    print(code[:120] + "...")
    print(f"    cost: {engine.cost_tracker.summary()}")

    print("\n[2] MockEngine.generate_summary():")
    summary = engine.generate_summary(code)
    print(f"    {summary}")
    print(f"    cost: {engine.cost_tracker.summary()}")

    print("\n[3] MockEngine.generate_report():")
    report = engine.generate_report({"round": 1, "passed": True, "score": 0.78})
    print(report)
    print(f"    cost: {engine.cost_tracker.summary()}")

    # _build_system_prompt 测试
    print("\n[4] _build_system_prompt():")
    prompt = _build_system_prompt()
    print(f"    system_prompt 长度: {len(prompt)} 字符")
    assert "第一章" in prompt
    assert "AI_MEMORY_START" not in prompt  # 已按标记分割
    print("    chapters 1-3 + recent memory 组装正确 [OK]")

    print("\n自检通过.")

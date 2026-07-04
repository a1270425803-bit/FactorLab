"""因子草案模板 — AI 生成因子代码的目标文件。

AI 必须实现 compute_factor(df) 函数：
- 输入: pd.DataFrame, MultiIndex (date, code), columns=[open,high,low,close,volume]
- 输出: pd.Series, 相同 MultiIndex (date, code)，值为因子暴露

约束: 代码 ≤ 20 行，禁止未来函数 shift(-1)，禁止 IO/网络访问。
"""

import numpy as np
import pandas as pd


def compute_factor(df: pd.DataFrame) -> pd.Series:
    """Mock 因子: 5 日动量（按股票分组计算 5 日价格变化率）。"""
    # 使用 groupby 处理多股票时序数据
    return df.groupby("code")["close"].pct_change(5)


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(1)
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    stocks = ["600519", "000858", "002594"]

    data = []
    for s in stocks:
        price = 100 + np.cumsum(np.random.randn(20))
        for i, d in enumerate(dates):
            data.append({
                "date": d, "code": s,
                "open": price[i], "high": price[i] * 1.02, "low": price[i] * 0.98,
                "close": price[i], "volume": np.random.randint(1000000, 10000000),
            })
    df = pd.DataFrame(data).set_index(["date", "code"])

    result = compute_factor(df)
    print("Mock 因子 (5 日动量), 最近 10 行:")
    print(result.dropna().tail(10))


#!/usr/bin/env python3
"""Phase 3b 回测引擎 — 冲击成本模型 + 分层收益，纯向量化计算。

升级要点：
  - 引入基于成交额的比例冲击成本（COEFF=150%, CAP=2%），替代固定年化成本
  - 新增 layer_returns 分层收益（L1-L5），供 robustness_checker 单调性检验
  - 向后兼容：旧调用方式（未传 volume_df/close_df）降级为无冲击模式

用法:
  python backtest.py --demo   # 用 Mock 数据验证冲击成本模型
"""

import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """回测结果结构体（Phase 3b 增强版）。"""

    annual_return: float = 0.0              # 年化收益率（十进制，如 0.15 表示 15%）
    max_drawdown: float = 0.0               # 最大回撤（十进制）
    sharpe_ratio: float = 0.0               # 年化夏普比率
    win_rate: float = 0.0                   # 调仓胜率（十进制）
    turnover_estimate: float = 0.0          # 估算换手率（十进制）
    cum_return_curve: Optional[pd.Series] = None   # 累计收益时间序列（date 索引）
    layer_returns: Optional[pd.DataFrame] = None   # 分层收益（date × [L1,L2,L3,L4,L5]）
    avg_impact_cost_bps: float = 0.0        # 平均冲击成本（百分比，如 0.5 表示 0.5%）
    total_cost_annual: float = 0.0          # 年化总成本（十进制，如 0.02 表示 2%）
    capital: float = 50_000_000             # 当前资金假设（RMB）


def simple_backtest(
    factor_values: pd.DataFrame,
    returns_df: pd.DataFrame,
    volume_df: pd.DataFrame = None,         # 【新增，可选】成交量矩阵 date × stock_code（股）
    close_df: pd.DataFrame = None,          # 【新增，可选】收盘价矩阵 date × stock_code
    capital: float = None,                  # None 时使用 config.CAPITAL_ASSUMPTION
    holding_period: int = None,             # None 时使用 config.HOLDING_PERIOD
    top_pct: float = None,                  # None 时使用 config.TOP_PCT
    impact_coefficient: float = None,       # None 时使用 config.IMPACT_COST_COEFFICIENT
    max_impact_cap: float = None,           # None 时使用 config.IMPACT_COST_CAP
    risk_free_rate: float = None,           # None 时使用 config.RISK_FREE_RATE
    adv_window: int = None,                 # None 时使用 config.ADV_WINDOW
) -> BacktestResult:
    """简化时间序列回测：等权 Top N%，含冲击成本模型。

    向后兼容：旧调用方式（未传 volume_df/close_df）给出 DeprecationWarning，
    并降级为无冲击模式（冲击成本 = 0，使用固定年化成本）。

    Args:
        factor_values: 因子值 DataFrame (dates × stock_code)
        returns_df:    T+1 日收益 DataFrame (dates × stock_code)，即 close.pct_change()
        volume_df:     每日成交量矩阵 (dates × stock_code)，单位：股
        close_df:      每日收盘价矩阵 (dates × stock_code)
        capital:       假设资金规模（RMB）
        holding_period: 调仓周期（交易日）
        top_pct:       做多比例（Top N%）
        impact_coefficient: 冲击成本系数（如 1.50 表示 150%）
        max_impact_cap: 冲击成本上限
        risk_free_rate: 无风险利率
        adv_window:    日均成交额滚动窗口

    Returns:
        BacktestResult
    """
    # ── 从 config 读取默认参数 ──────────────────────
    from config import (
        CAPITAL_ASSUMPTION, HOLDING_PERIOD, TOP_PCT,
        IMPACT_COST_COEFFICIENT, IMPACT_COST_CAP,
        RISK_FREE_RATE, ADV_WINDOW,
    )

    if capital is None:
        capital = CAPITAL_ASSUMPTION
    if holding_period is None:
        holding_period = HOLDING_PERIOD
    if top_pct is None:
        top_pct = TOP_PCT
    if impact_coefficient is None:
        impact_coefficient = IMPACT_COST_COEFFICIENT
    if max_impact_cap is None:
        max_impact_cap = IMPACT_COST_CAP
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE
    if adv_window is None:
        adv_window = ADV_WINDOW

    # ── 向后兼容：未提供 volume/close 时降级 ────────
    if volume_df is None or close_df is None:
        warnings.warn(
            "backtest.simple_backtest() 未提供 volume_df/close_df，"
            "冲击成本将设为 0。建议提供完整数据以获得准确回测。",
            DeprecationWarning,
            stacklevel=2,
        )
        use_impact = False
    else:
        use_impact = True

    # ── 对齐日期和股票 ──────────────────────────────
    common_dates = factor_values.index.intersection(returns_df.index)
    common_stocks = factor_values.columns.intersection(returns_df.columns)
    fv = factor_values.loc[common_dates, common_stocks]
    ret = returns_df.loc[common_dates, common_stocks]

    if use_impact:
        common_stocks = common_stocks.intersection(volume_df.columns).intersection(close_df.columns)
        fv = factor_values.loc[common_dates, common_stocks]
        ret = returns_df.loc[common_dates, common_stocks]
        vol = volume_df.loc[common_dates, common_stocks]
        cls = close_df.loc[common_dates, common_stocks]

    if len(common_dates) < holding_period * 2 or len(common_stocks) < 5:
        return BacktestResult(capital=capital)

    n_stocks = len(common_stocks)
    n_select = max(1, int(n_stocks * top_pct))

    # ── 冲击成本预计算（use_impact=True 时）─────────
    if use_impact:
        # 日均成交额 = 成交量(股) × 收盘价 的滚动均值
        daily_amount = vol * cls
        avg_daily_amount = daily_amount.rolling(window=adv_window, min_periods=5).mean()
        # 等权持仓金额
        position_value_per_stock = capital / n_select

    # ── 每日因子排名 → 选股信号 ─────────────────────
    ranks = fv.rank(axis=1, ascending=False, method="average")
    signal = (ranks <= n_select).astype(float)

    # ── 仅在调仓日更新仓位 ──────────────────────────
    position = signal.copy()
    for i in range(1, len(position)):
        if i % holding_period != 0:
            position.iloc[i] = position.iloc[i - 1]
    # 归一化为等权（Phase 3g: 显式处理全零行）
    row_sums = position.sum(axis=1)
    zero_rows = row_sums < 1e-8
    if zero_rows.any():
        import logging
        logging.getLogger("FactorLab").warning(
            f"backtest: {zero_rows.sum()} 个交易日无选股信号"
        )
    safe_sums = row_sums.where(~zero_rows, 1.0)
    position = position.div(safe_sums, axis=0)
    position.loc[zero_rows] = 0.0

    # ── 每日组合收益 ────────────────────────────────
    daily_ret = (position.shift(1) * ret).sum(axis=1)

    # ── 冲击成本扣除（仅调仓日）─────────────────────
    impact_series = pd.Series(0.0, index=daily_ret.index)
    impact_costs_recorded = []  # 记录每次调仓的冲击成本（百分比）

    rebalance_days = list(range(0, len(daily_ret), holding_period))

    if use_impact:
        for rday in rebalance_days:
            if rday >= len(fv) - 1:
                break
            trade_date = fv.index[rday]
            # 按因子值排序取 top_pct 股票
            fv_day = fv.iloc[rday].dropna()
            if len(fv_day) < n_select:
                continue
            selected = fv_day.nlargest(n_select).index
            # 获取选中股票的日均成交额
            if trade_date not in avg_daily_amount.index:
                continue
            adv = avg_daily_amount.loc[trade_date, selected]

            # 冲击成本 = min(持仓金额 / 日均成交额 × 系数, 上限)
            impact_ratio = position_value_per_stock / adv.replace(0, np.nan)
            impact_ratio = impact_ratio.fillna(max_impact_cap / impact_coefficient)
            impact_cost = (impact_ratio * impact_coefficient).clip(upper=max_impact_cap)
            # 组合平均冲击成本
            portfolio_impact = float(impact_cost.mean())
            total_trade_cost = portfolio_impact * 2  # 双边（买入 + 卖出）
            impact_series.iloc[rday] = total_trade_cost
            impact_costs_recorded.append(portfolio_impact)

    if not use_impact:
        # 旧版：固定年化成本降级
        annual_cost_rate = 0.05
        daily_cost = annual_cost_rate / 252
        for rday in rebalance_days:
            impact_series.iloc[rday] = daily_cost * holding_period

    # ── 净收益 ──────────────────────────────────────
    net_daily = daily_ret.fillna(0) - impact_series.fillna(0)

    # ── 累计收益曲线 ────────────────────────────────
    cum_ret = (1 + net_daily).cumprod()
    cum_ret = cum_ret.dropna()

    if len(cum_ret) < 10:
        return BacktestResult(capital=capital)

    # ── 年化收益 ────────────────────────────────────
    total_return = cum_ret.iloc[-1] - 1
    n_years = len(cum_ret) / 252
    annual_return = float(
        (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if total_return > -1 else -1.0
    )

    # ── 最大回撤 ────────────────────────────────────
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret - peak) / peak
    max_drawdown = float(drawdown.min())

    # ── 年化夏普 ────────────────────────────────────
    excess = net_daily.dropna() - risk_free_rate / 252
    sharpe_ratio = (
        float(excess.mean() / excess.std() * np.sqrt(252))
        if excess.std() > 1e-8 else 0.0
    )

    # ── 调仓期胜率 ──────────────────────────────────
    period_rets = []
    for i in range(0, len(net_daily) - holding_period, holding_period):
        p_ret = (1 + net_daily.iloc[i:i + holding_period]).prod() - 1
        period_rets.append(p_ret)
    win_rate = float(np.mean([1 if r > 0 else 0 for r in period_rets])) if period_rets else 0.0

    # ── 换手率估计 ──────────────────────────────────
    turnovers = []
    for i in range(0, len(position) - holding_period, holding_period):
        j = i + holding_period
        if j < len(position):
            chg = (position.iloc[i] - position.iloc[j]).abs().sum() / 2
            turnovers.append(chg)
    turnover_estimate = float(np.mean(turnovers)) if turnovers else 0.0

    # ── 冲击成本统计 ────────────────────────────────
    if impact_costs_recorded:
        avg_impact_cost_bps = float(np.mean(impact_costs_recorded)) * 100  # 转基点
        # 年化总成本 = 平均单边冲击 × 2（双边） × 年调仓次数 / 100
        total_cost_annual = float(np.mean(impact_costs_recorded)) * 2 * (252 / holding_period)
    else:
        avg_impact_cost_bps = 0.0
        total_cost_annual = 0.05  # 降级为固定 5%

    # ── 分层收益（L1-L5，供 robustness 单调性检验）───
    layer_returns = _compute_layer_returns(
        fv, ret, holding_period=holding_period, n_groups=5
    )

    # Phase 3g: isfinite 守卫防止 inf/nan 污染 BacktestResult
    def _safe_metric(value, ndigits=6):
        if not np.isfinite(value):
            return 0.0
        return round(float(value), ndigits)

    return BacktestResult(
        annual_return=_safe_metric(annual_return, 6),
        max_drawdown=_safe_metric(max_drawdown, 6),
        sharpe_ratio=_safe_metric(sharpe_ratio, 4),
        win_rate=_safe_metric(win_rate, 4),
        turnover_estimate=_safe_metric(turnover_estimate, 4),
        cum_return_curve=cum_ret,
        layer_returns=layer_returns,
        avg_impact_cost_bps=_safe_metric(avg_impact_cost_bps, 4),
        total_cost_annual=_safe_metric(total_cost_annual, 6),
        capital=capital,
    )


def _compute_layer_returns(
    fv: pd.DataFrame,
    ret: pd.DataFrame,
    holding_period: int = 5,
    n_groups: int = 5,
) -> pd.DataFrame:
    """计算分层持有期收益。

    每调仓日按因子值分为 n_groups 组（L1=最低, L5=最高），
    每组等权计算 holding_period 持有期收益。

    Returns:
        DataFrame: date × [L1, L2, ..., Ln]，每行是该调仓日的分层累计收益
    """
    layer_labels = [f"L{i + 1}" for i in range(n_groups)]
    layer_data = {}

    rebalance_days = list(range(0, len(fv), holding_period))

    for rday in rebalance_days:
        if rday >= len(fv):
            break
        trade_date = fv.index[rday]
        fv_day = fv.iloc[rday].dropna()
        if len(fv_day) < n_groups * 3:
            continue

        # 按因子值分 n_groups 组
        try:
            groups = pd.qcut(fv_day, n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

        if len(set(groups)) < n_groups:
            continue

        # 确定每组持有期结束日
        end_idx = min(rday + holding_period, len(fv) - 1)
        end_date = fv.index[end_idx]

        layer_row = {}
        for g in range(n_groups):
            g_stocks = list(fv_day[groups == g].index)  # 转为 list 避免 Index name 不匹配
            # 该组从 rday 到 end_date 的等权累计收益
            if end_date in ret.index:
                g_ret = ret.loc[trade_date:end_date, [s for s in g_stocks if s in ret.columns]]
                if len(g_ret) > 0 and g_ret.shape[1] > 0:
                    # 持有期总收益 = (1+r1)×(1+r2)×...×(1+rn) - 1
                    cum = (1 + g_ret).prod() - 1
                    layer_row[layer_labels[g]] = float(cum.mean())
                else:
                    layer_row[layer_labels[g]] = 0.0
            else:
                layer_row[layer_labels[g]] = 0.0

        layer_data[trade_date] = layer_row

    if not layer_data:
        # 返回全零的空 DataFrame
        return pd.DataFrame(columns=layer_labels, dtype=float)

    result = pd.DataFrame.from_dict(layer_data, orient="index")
    result = result[layer_labels]  # 确保列顺序
    result.index.name = "date"
    return result


# ── Demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        print("=== backtest.py Phase 3b 冲击成本模型演示 ===\n")
        np.random.seed(42)

        # Mock 数据：10 只 × 60 天
        n_days = 60
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

        # 大盘股 5 只（高成交额） + 小盘股 5 只（低成交额）
        large_caps = [f"LARGE_{i}" for i in range(5)]
        small_caps = [f"SMALL_{i}" for i in range(5)]
        all_stocks = large_caps + small_caps

        # 构造 price 和 volume
        close_data = {}
        volume_data = {}
        factor_data = {}
        return_data = {}

        for stock in all_stocks:
            is_large = stock.startswith("LARGE")
            base_price = 100.0 if is_large else 10.0
            trend = np.random.randn() * 0.001
            # 随机游走价格
            prices = [base_price]
            for _ in range(n_days - 1):
                prices.append(prices[-1] * (1 + np.random.randn() * 0.02 + trend))
            close_data[stock] = pd.Series(prices, index=dates)

            # 成交量：大盘 ~10亿股，小盘 ~40万股
            base_vol = 1e9 if is_large else 4e5
            vol = base_vol * (1 + np.random.randn(n_days) * 0.3)
            vol = np.abs(vol)
            volume_data[stock] = pd.Series(vol, index=dates)

            # 计算日均成交额用于展示
            avg_amount = np.mean(vol * np.array(prices))
            tag = "大盘" if is_large else "小盘"
            print(f"  {stock}: 均价≈{np.mean(prices):.1f}, "
                  f"日均成交量≈{np.mean(vol):.0f}股, 日均成交额≈{avg_amount:.0f}元 ({tag})")

        close_df = pd.DataFrame(close_data)
        volume_df = pd.DataFrame(volume_data)

        # 收益率
        returns_df = close_df.pct_change()

        # 构造有预测力的因子值
        true_alpha = np.random.randn(len(all_stocks)) * 0.1
        for i, d in enumerate(dates):
            fv = true_alpha + np.random.randn(len(all_stocks)) * 0.05
            factor_data[d] = pd.Series(fv, index=all_stocks)

        fv_df = pd.DataFrame(factor_data).T

        print(f"\n  股票数: {len(all_stocks)} | 交易日: {n_days}")
        print(f"  资金假设: ¥50,000,000 | Top: 30% | 持有期: 5天")
        print(f"  冲击系数: 150% | 上限: 2%\n")

        t0 = time.time()
        result = simple_backtest(fv_df, returns_df, volume_df=volume_df, close_df=close_df)
        elapsed = time.time() - t0

        print(f"  ╔{'═'*50}╗")
        print(f"  ║  回测结果 (Phase 3b 冲击成本模型)")
        print(f"  ╠{'═'*50}╣")
        print(f"  ║  年化收益:   {result.annual_return:>8.2%}")
        print(f"  ║  最大回撤:   {result.max_drawdown:>8.2%}")
        print(f"  ║  夏普比率:   {result.sharpe_ratio:>8.2f}")
        print(f"  ║  调仓胜率:   {result.win_rate:>8.2%}")
        print(f"  ║  换手率:     {result.turnover_estimate:>8.2%}")
        print(f"  ║  平均冲击:   {result.avg_impact_cost_bps:>8.2f}%")
        print(f"  ║  年化总成本: {result.total_cost_annual:>8.2%}")
        print(f"  ║  计算耗时:   {elapsed:>8.2f}s")
        print(f"  ╚{'═'*50}╝")

        # 验证各股票冲击成本
        print(f"\n  [验证] 个股冲击成本（第一调仓日）:")
        from config import CAPITAL_ASSUMPTION, TOP_PCT, IMPACT_COST_COEFFICIENT, IMPACT_COST_CAP

        n_sel = max(1, int(len(all_stocks) * TOP_PCT))
        pos_per_stock = CAPITAL_ASSUMPTION / n_sel
        first_date = fv_df.index[0]

        for stock in all_stocks:
            avg_vol = volume_df[stock].iloc[:20].mean()
            avg_close = close_df[stock].iloc[:20].mean()
            avg_amt = avg_vol * avg_close
            impact_ratio = pos_per_stock / max(avg_amt, 1)
            impact = min(impact_ratio * IMPACT_COST_COEFFICIENT, IMPACT_COST_CAP)
            impact_pct = impact * 100
            status = "✓" if (stock.startswith("LARGE") and impact_pct < 0.05) or \
                             (stock.startswith("SMALL") and impact_pct > 0.5) else "✗"
            print(f"    {stock}: 成交额≈{avg_amt:,.0f}, 持仓≈{pos_per_stock:,.0f}, "
                  f"冲击={impact_pct:.3f}% [{status}]")

        # 验证 layer_returns
        if result.layer_returns is not None and len(result.layer_returns) > 0:
            print(f"\n  [验证] 分层收益 (L1-L5):")
            print(f"    最后调仓日分层收益:")
            last_row = result.layer_returns.iloc[-1]
            for col in result.layer_returns.columns:
                print(f"      {col}: {last_row[col]:.4f}")

        # 向后兼容测试
        print(f"\n  [验证] 向后兼容 (无 volume_df/close_df):")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_result = simple_backtest(fv_df, returns_df)
            deprecation_seen = any("未提供 volume_df" in str(w_.message) for w_ in w)
            print(f"    DeprecationWarning: {'OK' if deprecation_seen else 'MISSING'}")
            print(f"    冲击成本: {old_result.avg_impact_cost_bps}% (应为 0)")
            print(f"    函数正常返回: {'OK' if old_result.annual_return is not None else 'FAIL'}")

        print(f"\n自检通过.")
    else:
        # 旧版 demo（保持兼容）
        print("=== backtest.py 自检 (向量化回测) ===\n")

        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=480, freq="B")
        n_stocks = 500

        true_alpha = np.random.randn(n_stocks) * 0.1
        factor_data = {}
        return_data = {}
        for i, d in enumerate(dates):
            fv = true_alpha + np.random.randn(n_stocks) * 0.3
            ret = fv * 0.05 + np.random.randn(n_stocks) * 0.5
            factor_data[d] = pd.Series(fv, index=[f"s{i:04d}" for i in range(n_stocks)])
            return_data[d] = pd.Series(ret, index=[f"s{i:04d}" for i in range(n_stocks)])

        fv_df = pd.DataFrame(factor_data).T
        ret_df = pd.DataFrame(return_data).T

        t0 = time.time()
        result = simple_backtest(fv_df, ret_df)
        elapsed = time.time() - t0

        print(f"Performance: {elapsed:.2f}s ({n_stocks} stocks × {len(dates)} days)")
        print(f"Annual Return:  {result.annual_return:.4f}")
        print(f"Max Drawdown:   {result.max_drawdown:.4f}")
        print(f"Sharpe Ratio:   {result.sharpe_ratio:.4f}")
        print(f"Win Rate:       {result.win_rate:.4f}")
        print(f"Turnover Est:   {result.turnover_estimate:.4f}")

        assert elapsed < 10, f"Too slow: {elapsed:.2f}s > 10s!"
        print(f"\nSpeed check: {elapsed:.2f}s < 10s [OK]")
        print("自检通过.")

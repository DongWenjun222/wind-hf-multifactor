from __future__ import annotations

"""单因子回测模块。

本文件负责评估每一个候选因子的独立交易效果：
- 把连续因子值转换成 -1/0/1 多空信号。
- 在训练集上判断因子正反方向，在测试集上评估样本外表现。
- 计算累计收益、年化收益、夏普、最大回撤、胜率、交易次数等指标。
- 生成 qcut 分组检验，观察因子值与未来收益是否具有单调关系。
- 调用 factor_library.py，把表现较好且低相关的因子沉淀到因子库。
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import BacktestConfig
from experiment_utils import (
    copy_existing_files,
    get_experiment_run_dir,
    snapshot_active_factor_library,
    write_factor_count_snapshot,
    write_run_config,
)
from runtime_utils import run_tracked
from factor_library import (
    build_factor_library,
    get_factor_library_dir,
    rank_single_factor_summary,
    save_factor_library,
)
from factors import (
    build_factors,
    build_single_factor_matrix,
    fetch_intraday_data,
    get_factor_id_map,
    get_factor_label_map,
    get_last_related_data_coverage,
    get_single_factor_new_factor_start_index,
    score_to_raw_signal,
    select_single_factor_columns,
    stop_wind,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def build_signal_from_score(
    score: pd.Series,
    threshold: float,
    score_name: str,
    direction: float = 1.0,
) -> pd.DataFrame:
    """把单个因子的连续分数转换成回测所需的信号表。

    composite_score：方向调整后的连续分数。
    raw_signal：根据阈值生成的 -1/0/1 目标仓位信号。
    position：实际持仓，使用 raw_signal.shift(1)，避免使用当前K线收盘后才知道的信号交易当前K线。
    """
    signal = pd.DataFrame(index=score.index)
    signal[score_name] = score
    directed_score = score * direction
    signal["composite_score"] = directed_score
    signal["raw_signal"] = score_to_raw_signal(
        directed_score,
        threshold,
    )
    signal["position"] = signal["raw_signal"].shift(1).fillna(0.0)
    return signal


def infer_annual_periods(index: pd.DatetimeIndex, annual_days: int) -> int:
    """根据真实数据频率推断年化周期数。

    高频数据每天K线数量可能受夜盘、节假日和缺失数据影响。
    这里用每日样本数中位数 * 年交易日数，得到更贴近实际的年化频率。
    """
    bars_per_day = (
        pd.Series(1, index=index)
        .groupby(index.normalize())
        .sum()
        .replace(0, np.nan)
        .dropna()
    )
    if bars_per_day.empty:
        return annual_days
    return int(max(annual_days, round(bars_per_day.median() * annual_days)))


def calculate_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    positions: pd.Series,
    annual_periods: int,
) -> Dict[str, float]:
    """计算回测绩效指标。

    指标包括累计收益、基准收益、年化收益、年化波动、夏普、最大回撤、
    胜率、交易次数、有效K线数和年化周期数。
    """
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
    positions = positions.reindex(strategy_returns.index).fillna(0.0)

    if strategy_returns.empty:
        raise ValueError("回测结果为空，无法计算绩效指标。")

    equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    total_return = equity.iloc[-1] - 1.0
    benchmark_total_return = benchmark_equity.iloc[-1] - 1.0
    annual_return = equity.iloc[-1] ** (annual_periods / len(strategy_returns)) - 1.0
    annual_vol = strategy_returns.std(ddof=0) * np.sqrt(annual_periods)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
    max_drawdown = drawdown.min()

    # 交易胜率按“连续持仓区间”粗略统计，而不是逐根K线统计。
    active_mask = positions != 0
    active_returns = strategy_returns[active_mask]
    previous_position = positions.shift(1).fillna(0.0)
    new_trade_flag = active_mask & (positions != previous_position)
    trade_id = new_trade_flag.cumsum().where(active_mask)
    trade_pnl = active_returns.groupby(trade_id[active_mask]).sum()
    win_rate = (trade_pnl > 0).mean() if not trade_pnl.empty else np.nan
    trade_count = int(len(trade_pnl))

    return {
        "累计收益": total_return,
        "基准累计收益": benchmark_total_return,
        "年化收益": annual_return,
        "年化波动": annual_vol,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "胜率": win_rate,
        "交易次数": float(trade_count),
        "样本K线数": float(len(strategy_returns)),
        "年化周期数": float(annual_periods),
    }


def empty_metrics() -> Dict[str, float]:
    """返回一组空绩效指标。

    当某个因子回测失败时使用，保证汇总表字段完整，不会因为单个因子失败中断全局流程。
    """
    return {
        "累计收益": np.nan,
        "基准累计收益": np.nan,
        "年化收益": np.nan,
        "年化波动": np.nan,
        "夏普比率": np.nan,
        "最大回撤": np.nan,
        "胜率": np.nan,
        "交易次数": 0.0,
        "样本K线数": 0.0,
        "年化周期数": np.nan,
    }


def safe_run_backtest(
    data: pd.DataFrame,
    signal: pd.DataFrame,
    config: Any,
) -> tuple[pd.DataFrame | None, Dict[str, float], str | None]:
    """安全包装 run_backtest。

    成功时返回回测明细、绩效指标和 None；
    失败时返回 None、空指标和错误信息，方便批量单因子测试继续往下跑。
    """
    try:
        backtest_df, metrics = run_backtest(data, signal, config)
        return backtest_df, metrics, None
    except Exception as exc:
        return None, empty_metrics(), str(exc)


def calculate_signal_coverage(backtest_df: pd.DataFrame | None) -> float:
    """计算原始信号非零的 K 线占比。"""
    if backtest_df is None or backtest_df.empty or "raw_signal" not in backtest_df.columns:
        return np.nan
    return float((backtest_df["raw_signal"].fillna(0.0) != 0).mean())


def calculate_position_coverage(backtest_df: pd.DataFrame | None) -> float:
    """计算实际持仓非零的 K 线占比。"""
    if backtest_df is None or backtest_df.empty or "position" not in backtest_df.columns:
        return np.nan
    return float((backtest_df["position"].fillna(0.0) != 0).mean())


def calculate_monthly_stability(backtest_df: pd.DataFrame | None) -> dict[str, float]:
    """计算单因子分月稳定性指标。

    月度收益用策略净收益按月复利聚合，收益集中度越高，说明越依赖少数月份贡献。
    """
    empty = {
        "月度样本数": 0.0,
        "盈利月份占比": np.nan,
        "月度平均收益": np.nan,
        "月度收益波动": np.nan,
        "最差月收益": np.nan,
        "最佳月收益": np.nan,
        "月度收益集中度": np.nan,
    }
    if (
        backtest_df is None
        or backtest_df.empty
        or "strategy_net_return" not in backtest_df.columns
        or not isinstance(backtest_df.index, pd.DatetimeIndex)
    ):
        return empty

    month_freq = get_supported_pandas_frequency(["ME", "M"])
    monthly_returns = (
        (1.0 + backtest_df["strategy_net_return"].fillna(0.0))
        .groupby(pd.Grouper(freq=month_freq))
        .prod()
        - 1.0
    ).dropna()
    monthly_returns = monthly_returns[monthly_returns.index.notna()]
    if monthly_returns.empty:
        return empty

    positive_returns = monthly_returns.clip(lower=0.0)
    total_positive_return = positive_returns.sum()
    concentration = (
        float(positive_returns.max() / total_positive_return)
        if total_positive_return > 0
        else np.nan
    )
    return {
        "月度样本数": float(len(monthly_returns)),
        "盈利月份占比": float((monthly_returns > 0).mean()),
        "月度平均收益": float(monthly_returns.mean()),
        "月度收益波动": float(monthly_returns.std(ddof=0)),
        "最差月收益": float(monthly_returns.min()),
        "最佳月收益": float(monthly_returns.max()),
        "月度收益集中度": concentration,
    }


def empty_predictive_metrics() -> dict[str, float]:
    """返回一组空预测能力指标。"""
    return {
        "IC": np.nan,
        "RankIC": np.nan,
        "ICIR": np.nan,
        "RankICIR": np.nan,
        "IC样本数": 0.0,
        "IC胜率": np.nan,
        "月度IC样本数": 0.0,
        "分组单调性": np.nan,
        "分组收益差": np.nan,
        "最高组平均收益": np.nan,
        "最低组平均收益": np.nan,
    }


def calculate_predictive_metrics(
    data: pd.DataFrame,
    score: pd.Series,
    period_index: pd.Index,
    qcut_summary: pd.DataFrame | None,
    direction: float = 1.0,
) -> dict[str, float]:
    """计算单因子的预测能力指标。

    这里衡量的是“当前 K 线结束后已知的因子值”对“下一根 K 线 open-to-close 收益”的预测关系。
    因此 future_return 使用 bar_return_oc.shift(-1)，只作为评估标签，不参与当期交易。
    """
    metrics = empty_predictive_metrics()
    if score.empty or len(period_index) == 0:
        return metrics

    directed_score = (score * direction).replace([np.inf, -np.inf], np.nan)
    future_return = calculate_bar_return_oc(data).shift(-1).replace([np.inf, -np.inf], np.nan)
    next_timestamp = pd.Series(data.index, index=data.index).shift(-1)
    frame = pd.DataFrame(
        {
            "score": directed_score,
            "future_return": future_return.reindex(directed_score.index),
            "next_timestamp": next_timestamp.reindex(directed_score.index),
        }
    ).reindex(period_index)
    frame = frame[frame["next_timestamp"].isin(period_index)].drop(columns=["next_timestamp"]).dropna()
    metrics["IC样本数"] = float(len(frame))
    if len(frame) >= 3 and frame["score"].nunique(dropna=True) > 1 and frame["future_return"].nunique(dropna=True) > 1:
        ic = frame["score"].corr(frame["future_return"], method="pearson")
        rank_ic = frame["score"].corr(frame["future_return"], method="spearman")
        metrics["IC"] = float(ic) if pd.notna(ic) else np.nan
        metrics["RankIC"] = float(rank_ic) if pd.notna(rank_ic) else np.nan

    if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
        month_freq = get_supported_pandas_frequency(["ME", "M"])
        monthly_ic_rows = []
        for _, month_frame in frame.groupby(pd.Grouper(freq=month_freq)):
            if (
                len(month_frame) >= 3
                and month_frame["score"].nunique(dropna=True) > 1
                and month_frame["future_return"].nunique(dropna=True) > 1
            ):
                monthly_ic_rows.append(
                    {
                        "ic": month_frame["score"].corr(month_frame["future_return"], method="pearson"),
                        "rank_ic": month_frame["score"].corr(month_frame["future_return"], method="spearman"),
                    }
                )
        monthly_ic = pd.DataFrame(monthly_ic_rows).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if not monthly_ic.empty:
            metrics["月度IC样本数"] = float(len(monthly_ic))
            if monthly_ic["ic"].notna().any():
                valid_ic = monthly_ic["ic"].dropna()
                ic_std = valid_ic.std(ddof=0)
                metrics["ICIR"] = float(valid_ic.mean() / ic_std) if ic_std > 0 else np.nan
                metrics["IC胜率"] = float((valid_ic > 0).mean())
            if monthly_ic["rank_ic"].notna().any():
                valid_rank_ic = monthly_ic["rank_ic"].dropna()
                rank_ic_std = valid_rank_ic.std(ddof=0)
                metrics["RankICIR"] = (
                    float(valid_rank_ic.mean() / rank_ic_std)
                    if rank_ic_std > 0
                    else np.nan
                )

    if qcut_summary is not None and not qcut_summary.empty and {"分组", "平均收益"}.issubset(qcut_summary.columns):
        group_returns = qcut_summary[["分组", "平均收益"]].copy()
        group_returns["group_id"] = (
            group_returns["分组"].astype(str).str.extract(r"Q(\d+)")[0].astype(float)
        )
        group_returns["平均收益"] = pd.to_numeric(group_returns["平均收益"], errors="coerce")
        group_returns = group_returns.dropna(subset=["group_id", "平均收益"]).sort_values("group_id")
        if len(group_returns) >= 3 and group_returns["平均收益"].nunique(dropna=True) > 1:
            monotonicity = group_returns["group_id"].corr(group_returns["平均收益"], method="spearman")
            metrics["分组单调性"] = float(monotonicity) if pd.notna(monotonicity) else np.nan
            metrics["最低组平均收益"] = float(group_returns.iloc[0]["平均收益"])
            metrics["最高组平均收益"] = float(group_returns.iloc[-1]["平均收益"])
            metrics["分组收益差"] = metrics["最高组平均收益"] - metrics["最低组平均收益"]

    return metrics


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """给一组指标统一加上样本前缀。"""
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def get_supported_pandas_frequency(candidates: list[str]) -> str:
    """返回当前 pandas 版本支持的第一个频率别名。"""
    for freq in candidates:
        try:
            pd.tseries.frequencies.to_offset(freq)
            return freq
        except ValueError:
            continue
    raise ValueError(f"当前 pandas 不支持这些频率别名: {candidates}")


def run_backtest(data: pd.DataFrame, signal: pd.DataFrame, config: Any):
    """执行基础向量化回测。

    核心假设：
    - position 已经向后移动一根K线，因此不会用未来信号交易当前K线。
    - 单根收益使用 open 到 close 的收益 bar_return_oc。
    - 成本按仓位变化 turnover * (commission + slippage) 计算。
    """
    df = data.join(signal, how="inner").copy()

    if "bar_return_oc" not in df.columns:
        if {"open", "close"}.issubset(df.columns):
            open_price = df["open"].replace(0, np.nan)
            df["bar_return_oc"] = df["close"] / open_price - 1.0
        else:
            raise KeyError("回测缺少 bar_return_oc，且原始数据中也没有 open/close 可用于重算。")

    df = df.dropna(subset=["position", "composite_score", "bar_return_oc"])

    turnover = df["position"].diff().abs().fillna(df["position"].abs())
    trading_cost = turnover * (config.commission_bps + config.slippage_bps) / 10000.0

    # 策略收益仍使用 open-to-close，因为信号在上一根K线结束后、下一根K线开盘执行。
    # 基准净值用于展示标的自身走势，使用 close-to-close 更直观，也避免长期排除隔夜跳空导致基准曲线失真。
    if "close" in df.columns:
        df["benchmark_return"] = df["close"].replace(0, np.nan).pct_change().fillna(0.0)
    else:
        df["benchmark_return"] = df["bar_return_oc"].fillna(0.0)
    df["strategy_gross_return"] = df["position"] * df["bar_return_oc"].fillna(0.0)
    df["strategy_net_return"] = df["strategy_gross_return"] - trading_cost
    df["turnover"] = turnover
    df["trading_cost"] = trading_cost
    df["nav"] = (1.0 + df["strategy_net_return"]).cumprod()
    df["benchmark_nav"] = (1.0 + df["benchmark_return"]).cumprod()
    df["drawdown"] = df["nav"] / df["nav"].cummax() - 1.0

    annual_periods = infer_annual_periods(df.index, config.annual_trading_days)
    metrics = calculate_metrics(
        strategy_returns=df["strategy_net_return"],
        benchmark_returns=df["benchmark_return"],
        positions=df["position"],
        annual_periods=annual_periods,
    )
    return df, metrics


def plot_backtest_result(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    title: str,
    score_label: str,
    file_name: str,
    qcut_df: pd.DataFrame | None = None,
    qcut_summary: pd.DataFrame | None = None,
) -> Path:
    """绘制单段回测图。

    图中包含策略净值、基准净值、回撤、因子分数/仓位，以及可选 qcut 分组净值。
    该函数也被综合因子回测复用。
    """
    has_qcut = qcut_df is not None and qcut_summary is not None
    nrows = 5 if has_qcut else 3
    height_ratios = [2.0, 1.0, 1.0, 1.6, 1.2] if has_qcut else [2.0, 1.0, 1.0]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(16, 18 if has_qcut else 12),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    axes[0].plot(backtest_df.index, backtest_df["nav"], label="策略净值", linewidth=1.6)
    axes[0].plot(
        backtest_df.index,
        backtest_df["benchmark_nav"],
        label="基准净值",
        linewidth=1.2,
        alpha=0.85,
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("净值")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(
        backtest_df.index,
        backtest_df["drawdown"],
        0,
        color="#d62728",
        alpha=0.35,
    )
    axes[1].set_ylabel("回撤")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        backtest_df.index,
        backtest_df["composite_score"],
        label=score_label,
        linewidth=1.0,
    )
    axes[2].step(
        backtest_df.index,
        backtest_df["position"],
        label="实际仓位",
        color="#ff7f0e",
        linewidth=1.0,
        where="mid",
    )
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("分数 / 仓位")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left")

    if has_qcut:
        group_nav = qcut_df["group_nav"].dropna(how="all")
        for group_name in group_nav.columns:
            axes[3].plot(group_nav.index, group_nav[group_name], label=group_name)

        axes[3].set_ylabel("分组净值")
        axes[3].grid(alpha=0.3)
        axes[3].legend(loc="upper left", ncol=3)

        bar_data = qcut_summary.set_index("分组")["平均收益"].astype(float)
        colors = np.where(bar_data >= 0, "#2ca02c", "#d62728")
        axes[4].bar(bar_data.index, bar_data.values, color=colors, alpha=0.85)
        axes[4].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        axes[4].set_ylabel("平均收益")
        axes[4].grid(axis="y", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plot_path = output_dir / file_name
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def rebuild_qcut_report_for_period(
    qcut_df: pd.DataFrame,
    period_index: pd.Index,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从全样本 qcut 结果中截取训练集或测试集分段报告。

    qcut 分组本身使用滚动历史分位，先在全样本时间线上计算，
    再按训练/测试索引切分，保证分组规则不偷看未来。
    """
    group_returns = qcut_df["group_return"].reindex(period_index).fillna(0.0)
    group_nav = (1.0 + group_returns).cumprod()
    effective_group = qcut_df["effective_group"].reindex(period_index)

    summary_rows = []
    for group_id in range(1, config.qcut_groups + 1):
        group_name = f"Q{group_id}"
        if group_name not in group_returns.columns:
            continue

        in_group = effective_group == float(group_id)
        active_returns = group_returns.loc[in_group, group_name]
        summary_rows.append(
            {
                "分组": group_name,
                "样本数": int(in_group.sum()),
                "平均收益": active_returns.mean()
                if not active_returns.empty
                else np.nan,
                "胜率": (active_returns > 0).mean()
                if not active_returns.empty
                else np.nan,
            }
        )

    period_qcut = pd.concat(
        {
            "group_return": group_returns,
            "group_nav": group_nav,
        },
        axis=1,
    )
    period_qcut["effective_group"] = effective_group
    return period_qcut, pd.DataFrame(summary_rows)


def plot_single_factor_train_test_result(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_qcut_df: pd.DataFrame,
    train_qcut_summary: pd.DataFrame,
    validation_qcut_df: pd.DataFrame,
    validation_qcut_summary: pd.DataFrame,
    test_qcut_df: pd.DataFrame,
    test_qcut_summary: pd.DataFrame,
    output_dir: Path,
    title: str,
    score_label: str,
    file_name: str,
) -> Path:
    """绘制单因子的训练集/验证集/最终测试集对比图。

    三列依次为训练集、验证集和最终测试集。
    每侧都包含净值、回撤、信号/仓位、qcut 分组净值和分组平均收益。
    """
    fig, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(30, 18),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.6, 1.2]},
    )
    fig.suptitle(title, fontsize=16)

    panels = [
        ("训练集", train_df, train_qcut_df, train_qcut_summary),
        ("验证集", validation_df, validation_qcut_df, validation_qcut_summary),
        ("最终测试集", test_df, test_qcut_df, test_qcut_summary),
    ]

    for col, (panel_name, backtest_df, qcut_df, qcut_summary) in enumerate(panels):
        axes[0, col].plot(
            backtest_df.index,
            backtest_df["nav"],
            label="策略净值",
            linewidth=1.6,
        )
        axes[0, col].plot(
            backtest_df.index,
            backtest_df["benchmark_nav"],
            label="基准净值",
            linewidth=1.2,
            alpha=0.85,
        )
        axes[0, col].set_title(panel_name)
        axes[0, col].set_ylabel("净值")
        axes[0, col].legend(loc="upper left")
        axes[0, col].grid(alpha=0.3)

        axes[1, col].fill_between(
            backtest_df.index,
            backtest_df["drawdown"],
            0,
            color="#d62728",
            alpha=0.35,
        )
        axes[1, col].set_ylabel("回撤")
        axes[1, col].grid(alpha=0.3)

        axes[2, col].plot(
            backtest_df.index,
            backtest_df["composite_score"],
            label=score_label,
            linewidth=1.0,
        )
        axes[2, col].step(
            backtest_df.index,
            backtest_df["position"],
            label="实际仓位",
            color="#ff7f0e",
            linewidth=1.0,
            where="mid",
        )
        axes[2, col].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        axes[2, col].set_ylabel("分数 / 仓位")
        axes[2, col].grid(alpha=0.3)
        axes[2, col].legend(loc="upper left")

        group_nav = qcut_df["group_nav"].dropna(how="all")
        for group_name in group_nav.columns:
            axes[3, col].plot(group_nav.index, group_nav[group_name], label=group_name)
        axes[3, col].set_ylabel("分组净值")
        axes[3, col].grid(alpha=0.3)
        axes[3, col].legend(loc="upper left", ncol=3)

        bar_data = qcut_summary.set_index("分组")["平均收益"].astype(float)
        colors = np.where(bar_data >= 0, "#2ca02c", "#d62728")
        axes[4, col].bar(bar_data.index, bar_data.values, color=colors, alpha=0.85)
        axes[4, col].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        axes[4, col].set_ylabel("平均收益")
        axes[4, col].grid(axis="y", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plot_path = output_dir / file_name
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def print_metrics(metrics: Dict[str, float]) -> None:
    """把绩效指标打印到控制台，主要用于调试和人工查看。"""
    for key, value in metrics.items():
        if "次数" in key or "K线数" in key or "周期数" in key:
            print(f"{key}: {value:.0f}")
        elif pd.isna(value):
            print(f"{key}: NaN")
        else:
            print(f"{key}: {value:.4f}")


def split_train_test_index(index: pd.Index, train_ratio: float) -> pd.Timestamp:
    """根据样本比例返回训练集/测试集切分时间点。"""
    if not 0 < train_ratio < 1:
        raise ValueError("auto_select_train_ratio 必须在 0 到 1 之间。")
    split_pos = int(len(index) * train_ratio)
    split_pos = min(max(split_pos, 1), len(index) - 1)
    return index[split_pos]


def split_train_validation_test_index(
    index: pd.Index,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """返回训练/验证/最终测试三段式切分时间点。"""
    if not 0 < train_ratio < 1:
        raise ValueError("auto_select_train_ratio 必须在 0 到 1 之间。")
    if not 0 <= validation_ratio < 1:
        raise ValueError("auto_select_validation_ratio 必须在 0 到 1 之间。")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("训练集比例 + 验证集比例必须小于 1，才能保留最终测试集。")

    train_pos = int(len(index) * train_ratio)
    validation_pos = int(len(index) * (train_ratio + validation_ratio))
    train_pos = min(max(train_pos, 1), len(index) - 2)
    validation_pos = min(max(validation_pos, train_pos + 1), len(index) - 1)
    return index[train_pos], index[validation_pos]


def choose_factor_direction(
    data: pd.DataFrame,
    score: pd.Series,
    factor_name: str,
    config: BacktestConfig,
    split_time: pd.Timestamp,
) -> tuple[float, dict[str, float]]:
    """在训练集上选择因子正向或反向。

    对同一个因子分别测试 +1 和 -1 两种方向，
    优先比较训练夏普，其次比较训练累计收益。
    选出的方向会固定用于测试集，避免在测试集上反复调方向。
    """
    if not config.auto_detect_factor_direction:
        return 1.0, {"训练正向夏普": np.nan, "训练反向夏普": np.nan}

    train_data = data.loc[data.index < split_time]
    direction_metrics = {}

    for direction, label in [(1.0, "正向"), (-1.0, "反向")]:
        signal = build_signal_from_score(
            score,
            config.signal_threshold,
            factor_name,
            direction=direction,
        )
        train_signal = signal.loc[signal.index < split_time]
        _, metrics, _ = safe_run_backtest(train_data, train_signal, config)
        direction_metrics[label] = metrics

    positive = direction_metrics["正向"]
    negative = direction_metrics["反向"]
    positive_score = (
        positive.get("夏普比率", np.nan),
        positive.get("累计收益", np.nan),
    )
    negative_score = (
        negative.get("夏普比率", np.nan),
        negative.get("累计收益", np.nan),
    )

    positive_score = tuple(-np.inf if pd.isna(value) else value for value in positive_score)
    negative_score = tuple(-np.inf if pd.isna(value) else value for value in negative_score)
    selected_direction = 1.0 if positive_score >= negative_score else -1.0

    return selected_direction, {
        "训练正向夏普": positive.get("夏普比率", np.nan),
        "训练反向夏普": negative.get("夏普比率", np.nan),
        "训练正向收益": positive.get("累计收益", np.nan),
        "训练反向收益": negative.get("累计收益", np.nan),
    }


def rolling_qcut_labels(
    score: pd.Series,
    groups: int,
    window: int,
    min_periods: int,
) -> pd.Series:
    """用滚动历史分位给因子值打 qcut 分组标签。

    传统全样本 qcut 会使用未来分布，容易产生信息泄露。
    这里每个时点只使用过去 window 根K线计算分位阈值。
    """
    clean_score = score.replace([np.inf, -np.inf], np.nan)
    effective_min_periods = min(min_periods, window)
    valid_history = (
        clean_score.rolling(window=window, min_periods=effective_min_periods).count()
        >= effective_min_periods
    )
    has_variation = (
        clean_score.rolling(window=window, min_periods=effective_min_periods).std() > 0
    )
    labels = pd.Series(1.0, index=score.index, dtype="float64")

    if groups > 1:
        quantile_thresholds = pd.DataFrame(index=score.index)
        for group_id in range(1, groups):
            quantile_thresholds[f"q{group_id}"] = clean_score.rolling(
                window=window,
                min_periods=effective_min_periods,
            ).quantile(group_id / groups)
            labels = labels.add(
                (clean_score > quantile_thresholds[f"q{group_id}"]).astype(float),
                fill_value=0.0,
            )
        missing_threshold = quantile_thresholds.isna().any(axis=1)
    else:
        missing_threshold = pd.Series(False, index=score.index)

    invalid_mask = (~valid_history) | (~has_variation) | clean_score.isna() | missing_threshold
    labels[invalid_mask] = np.nan
    return labels


def calculate_bar_return_oc(data: pd.DataFrame) -> pd.Series:
    """计算单根K线 open 到 close 的收益。"""
    open_price = data["open"].replace(0, np.nan)
    return data["close"] / open_price - 1.0


def build_qcut_group_nav(
    data: pd.DataFrame,
    score: pd.Series,
    config: BacktestConfig,
    direction: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建 qcut 分组净值和分组统计。

    每个分组表示“上一根K线产生的因子分组，在当前K线持有”的收益表现。
    使用 effective_group = qcut_label.shift(1) 来避免未来函数。
    """
    directed_score = score * direction
    qcut_label = rolling_qcut_labels(
        directed_score,
        groups=config.qcut_groups,
        window=config.qcut_window,
        min_periods=config.qcut_min_periods,
    )
    effective_group = qcut_label.shift(1)
    bar_return = calculate_bar_return_oc(data).reindex(score.index).fillna(0.0)

    group_returns = pd.DataFrame(index=score.index)
    group_nav = pd.DataFrame(index=score.index)
    summary_rows = []

    for group_id in range(1, config.qcut_groups + 1):
        group_name = f"Q{group_id}"
        in_group = effective_group == float(group_id)
        group_returns[group_name] = bar_return.where(in_group, 0.0)
        group_nav[group_name] = (1.0 + group_returns[group_name]).cumprod()

        active_returns = group_returns.loc[in_group, group_name]
        summary_rows.append(
            {
                "分组": group_name,
                "样本数": int(in_group.sum()),
                "平均收益": active_returns.mean()
                if not active_returns.empty
                else np.nan,
                "胜率": (active_returns > 0).mean()
                if not active_returns.empty
                else np.nan,
            }
        )

    result = pd.concat(
        {
            "group_return": group_returns,
            "group_nav": group_nav,
        },
        axis=1,
    )
    result["directed_score"] = directed_score
    result["qcut_label"] = qcut_label
    result["effective_group"] = effective_group
    return result, pd.DataFrame(summary_rows)


def get_single_factor_plot_top_n(config: BacktestConfig) -> int:
    """读取单因子 Top N 出图数量，并保证结果非负。"""
    return max(0, int(getattr(config, "single_factor_plot_top_n", 0) or 0))


def generate_top_single_factor_plots(
    active_library: pd.DataFrame,
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
    output_dir: Path,
    factor_label_map: dict[str, str],
    split_time: pd.Timestamp,
) -> dict[str, str]:
    """只给入库排名靠前的因子补充生成图表。

    当 single_factor_plot_all=False 时，主循环不再为每个因子出图。
    回测和入库筛选结束后，本函数只对 active_library 前 N 个因子重新生成可视化报告。
    """
    if active_library.empty or "因子" not in active_library.columns:
        return {}

    top_n = get_single_factor_plot_top_n(config)
    if top_n <= 0:
        return {}

    _, validation_end_time = split_train_validation_test_index(
        data.index,
        config.auto_select_train_ratio,
        getattr(config, "auto_select_validation_ratio", 0.0),
    )
    train_data = data.loc[data.index < split_time]
    validation_data = data.loc[(data.index >= split_time) & (data.index < validation_end_time)]
    backtest_data = data.loc[data.index >= validation_end_time]
    plot_paths: dict[str, str] = {}
    factor_names = active_library["因子"].dropna().head(top_n).tolist()

    for factor_name in tqdm(factor_names, desc="生成Top因子图表"):
        if factor_name not in factors.columns:
            continue

        try:
            factor_label = factor_label_map.get(factor_name, factor_name)
            direction, _ = choose_factor_direction(
                data,
                factors[factor_name],
                factor_name,
                config,
                split_time,
            )
            signal = build_signal_from_score(
                factors[factor_name],
                config.signal_threshold,
                factor_name,
                direction=direction,
            )
            train_df, _, train_error = safe_run_backtest(
                train_data,
                signal.loc[signal.index < split_time],
                config,
            )
            validation_df, _, validation_error = safe_run_backtest(
                validation_data,
                signal.loc[(signal.index >= split_time) & (signal.index < validation_end_time)],
                config,
            )
            backtest_df, _, test_error = safe_run_backtest(
                backtest_data,
                signal.loc[signal.index >= validation_end_time],
                config,
            )
            if train_df is None or validation_df is None or backtest_df is None:
                raise ValueError(
                    train_error
                    or validation_error
                    or test_error
                    or "训练集、验证集或测试集回测结果为空"
                )
            train_stability = calculate_monthly_stability(train_df)
            validation_stability = calculate_monthly_stability(validation_df)
            test_stability = calculate_monthly_stability(backtest_df)

            qcut_df, _ = build_qcut_group_nav(
                data,
                factors[factor_name],
                config,
                direction=direction,
            )
            train_qcut_df, train_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                train_data.index,
                config,
            )
            validation_qcut_df, validation_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                validation_data.index,
                config,
            )
            test_qcut_df, test_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                backtest_data.index,
                config,
            )
            plot_path = plot_single_factor_train_test_result(
                train_df,
                validation_df,
                backtest_df,
                train_qcut_df,
                train_qcut_summary,
                validation_qcut_df,
                validation_qcut_summary,
                test_qcut_df,
                test_qcut_summary,
                output_dir,
                f"{config.symbol} 单因子回测: {factor_label}",
                factor_label,
                f"{factor_label}_report.png",
            )
            plot_paths[factor_name] = str(plot_path)
        except Exception as exc:
            print(f"Top因子图表生成失败: {factor_name}, {exc}")

    return plot_paths


def run_single_factor_backtests(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """运行完整单因子批量回测流程。

    主要步骤：
    1. 根据配置选择本轮需要测试的因子。
    2. 对每个因子做方向选择、训练集回测、测试集回测和 qcut 检验。
    3. 汇总所有结果，按样本外夏普/累计收益排序。
    4. 调用因子库模块做收益门槛和相关性去重。
    5. 保存 active/all/rejected 因子库、单因子汇总、qcut 汇总和必要图表。
    """
    output_dir = Path(config.output_dir) / "single_factor"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = get_experiment_run_dir(config, "single")
    write_run_config(config, output_dir)
    if run_dir is not None:
        write_run_config(config, run_dir)
        write_factor_count_snapshot(factors, run_dir)
        coverage = get_last_related_data_coverage()
        if not coverage.empty:
            coverage.to_csv(
                run_dir / "related_data_coverage.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary_rows = []
    qcut_summary_rows = []
    factor_id_map = factors.attrs.get("factor_id_map") or get_factor_id_map(factors)
    factor_label_map = factors.attrs.get("factor_label_map") or get_factor_label_map(factors)
    if str(getattr(config, "single_factor_scope", "")).lower() == "new" and factors.attrs.get("factor_id_map"):
        factor_columns = list(factors.columns)
    else:
        factor_columns = select_single_factor_columns(factors, config)
    print(f"单因子回测范围: {config.single_factor_scope}, 因子数量: {len(factor_columns)}")
    if str(getattr(config, "single_factor_scope", "")).lower() == "new":
        start_index = get_single_factor_new_factor_start_index(config)
        print(f"当前品种新因子起始编号: {start_index}")
    if not factor_columns:
        raise ValueError("本轮没有可回测的单因子，请检查 single_factor_scope 和 single_factor_new_factor_start_index。")
    plot_all = bool(getattr(config, "single_factor_plot_all", False))
    plot_top_n = get_single_factor_plot_top_n(config)
    if plot_all:
        print("单因子图表生成: 全量生成")
    else:
        print(f"单因子图表生成: 仅对入库Top {plot_top_n} 因子生成")
    split_time, validation_end_time = split_train_validation_test_index(
        data.index,
        config.auto_select_train_ratio,
        getattr(config, "auto_select_validation_ratio", 0.0),
    )
    train_data = data.loc[data.index < split_time]
    validation_data = data.loc[(data.index >= split_time) & (data.index < validation_end_time)]
    backtest_data = data.loc[data.index >= validation_end_time]
    # 主循环只做必要计算；是否画图由配置控制，避免海量因子时生成过多 PNG。
    for factor_name in tqdm(factor_columns) :
        factor_label = factor_label_map[factor_name]
       # print(f"\n========== 单因子回测: {factor_name} ==========")
        summary_path = output_dir / "single_factor_summary.csv"
        try:
            direction, direction_metrics = choose_factor_direction(
                data,
                factors[factor_name],
                factor_name,
                config,
                split_time,
            )
            signal = build_signal_from_score(
                factors[factor_name],
                config.signal_threshold,
                factor_name,
                direction=direction,
            )
            train_signal = signal.loc[signal.index < split_time]
            validation_signal = signal.loc[
                (signal.index >= split_time) & (signal.index < validation_end_time)
            ]
            backtest_signal = signal.loc[signal.index >= validation_end_time]
            train_df, train_metrics, train_error = safe_run_backtest(
                train_data,
                train_signal,
                config,
            )
            validation_df, validation_metrics, validation_error = safe_run_backtest(
                validation_data,
                validation_signal,
                config,
            )
            backtest_df, metrics, test_error = safe_run_backtest(
                backtest_data,
                backtest_signal,
                config,
            )
            if train_df is None or validation_df is None or backtest_df is None:
                raise ValueError(
                    train_error
                    or validation_error
                    or test_error
                    or "训练集、验证集或测试集回测结果为空"
                )

            qcut_df, _ = build_qcut_group_nav(
                data,
                factors[factor_name],
                config,
                direction=direction,
            )
            train_qcut_df, train_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                train_data.index,
                config,
            )
            validation_qcut_df, validation_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                validation_data.index,
                config,
            )
            test_qcut_df, test_qcut_summary = rebuild_qcut_report_for_period(
                qcut_df,
                backtest_data.index,
                config,
            )
            train_predictive_metrics = calculate_predictive_metrics(
                data,
                factors[factor_name],
                train_data.index,
                train_qcut_summary,
                direction=direction,
            )
            validation_predictive_metrics = calculate_predictive_metrics(
                data,
                factors[factor_name],
                validation_data.index,
                validation_qcut_summary,
                direction=direction,
            )
            test_predictive_metrics = calculate_predictive_metrics(
                data,
                factors[factor_name],
                backtest_data.index,
                test_qcut_summary,
                direction=direction,
            )

            plot_path = ""
            if plot_all:
                plot_path = plot_single_factor_train_test_result(
                    train_df,
                    validation_df,
                    backtest_df,
                    train_qcut_df,
                    train_qcut_summary,
                    validation_qcut_df,
                    validation_qcut_summary,
                    test_qcut_df,
                    test_qcut_summary,
                    output_dir,
                    f"{config.symbol} 单因子回测: {factor_label}",
                    factor_label,
                    f"{factor_label}_report.png",
                )

            train_qcut_summary.insert(0, "因子标签", factor_label)
            train_qcut_summary.insert(0, "样本", "训练集")
            train_qcut_summary.insert(0, "因子", factor_name)
            train_qcut_summary.insert(0, "因子编号", factor_id_map[factor_name])
            validation_qcut_summary.insert(0, "因子标签", factor_label)
            validation_qcut_summary.insert(0, "样本", "验证集")
            validation_qcut_summary.insert(0, "因子", factor_name)
            validation_qcut_summary.insert(0, "因子编号", factor_id_map[factor_name])
            test_qcut_summary.insert(0, "因子标签", factor_label)
            test_qcut_summary.insert(0, "样本", "测试集")
            test_qcut_summary.insert(0, "因子", factor_name)
            test_qcut_summary.insert(0, "因子编号", factor_id_map[factor_name])
            qcut_summary_rows.extend(train_qcut_summary.to_dict("records"))
            qcut_summary_rows.extend(validation_qcut_summary.to_dict("records"))
            qcut_summary_rows.extend(test_qcut_summary.to_dict("records"))

            row = {
                "因子编号": factor_id_map[factor_name],
                "因子": factor_name,
                "因子标签": factor_label,
                "方向": "正向" if direction > 0 else "反向",
                "训练截止": str(split_time),
                "验证截止": str(validation_end_time),
                "训练累计收益": train_metrics["累计收益"],
                "训练年化收益": train_metrics["年化收益"],
                "训练年化波动": train_metrics["年化波动"],
                "训练夏普比率": train_metrics["夏普比率"],
                "训练最大回撤": train_metrics["最大回撤"],
                "训练胜率": train_metrics["胜率"],
                "训练交易次数": train_metrics["交易次数"],
                "训练样本K线数": train_metrics["样本K线数"],
                "训练信号覆盖率": calculate_signal_coverage(train_df),
                "训练持仓覆盖率": calculate_position_coverage(train_df),
                "验证累计收益": validation_metrics["累计收益"],
                "验证年化收益": validation_metrics["年化收益"],
                "验证年化波动": validation_metrics["年化波动"],
                "验证夏普比率": validation_metrics["夏普比率"],
                "验证最大回撤": validation_metrics["最大回撤"],
                "验证胜率": validation_metrics["胜率"],
                "验证交易次数": validation_metrics["交易次数"],
                "验证样本K线数": validation_metrics["样本K线数"],
                "验证信号覆盖率": calculate_signal_coverage(validation_df),
                "验证持仓覆盖率": calculate_position_coverage(validation_df),
                "测试累计收益": metrics["累计收益"],
                "测试年化收益": metrics["年化收益"],
                "测试年化波动": metrics["年化波动"],
                "测试夏普比率": metrics["夏普比率"],
                "测试最大回撤": metrics["最大回撤"],
                "测试胜率": metrics["胜率"],
                "测试交易次数": metrics["交易次数"],
                "测试样本K线数": metrics["样本K线数"],
                "测试信号覆盖率": calculate_signal_coverage(backtest_df),
                "测试持仓覆盖率": calculate_position_coverage(backtest_df),
                **prefix_metrics(train_stability, "训练"),
                **prefix_metrics(validation_stability, "验证"),
                **prefix_metrics(test_stability, "测试"),
                **prefix_metrics(train_predictive_metrics, "训练"),
                **prefix_metrics(validation_predictive_metrics, "验证"),
                **prefix_metrics(test_predictive_metrics, "测试"),
                **direction_metrics,
                "图片文件": str(plot_path) if plot_path else "",
                "错误": "",
            }
        except Exception as exc:
            row = {
                "因子编号": factor_id_map[factor_name],
                "因子": factor_name,
                "因子标签": factor_label,
                "方向": "",
                "训练截止": str(split_time),
                "验证截止": str(validation_end_time),
                "训练累计收益": np.nan,
                "训练年化收益": np.nan,
                "训练年化波动": np.nan,
                "训练夏普比率": np.nan,
                "训练最大回撤": np.nan,
                "训练胜率": np.nan,
                "训练交易次数": 0.0,
                "训练样本K线数": 0.0,
                "训练信号覆盖率": np.nan,
                "训练持仓覆盖率": np.nan,
                "验证累计收益": np.nan,
                "验证年化收益": np.nan,
                "验证年化波动": np.nan,
                "验证夏普比率": np.nan,
                "验证最大回撤": np.nan,
                "验证胜率": np.nan,
                "验证交易次数": 0.0,
                "验证样本K线数": 0.0,
                "验证信号覆盖率": np.nan,
                "验证持仓覆盖率": np.nan,
                "测试累计收益": np.nan,
                "测试年化收益": np.nan,
                "测试年化波动": np.nan,
                "测试夏普比率": np.nan,
                "测试最大回撤": np.nan,
                "测试胜率": np.nan,
                "测试交易次数": 0.0,
                "测试样本K线数": 0.0,
                "测试信号覆盖率": np.nan,
                "测试持仓覆盖率": np.nan,
                **prefix_metrics(calculate_monthly_stability(None), "训练"),
                **prefix_metrics(calculate_monthly_stability(None), "验证"),
                **prefix_metrics(calculate_monthly_stability(None), "测试"),
                **prefix_metrics(empty_predictive_metrics(), "训练"),
                **prefix_metrics(empty_predictive_metrics(), "验证"),
                **prefix_metrics(empty_predictive_metrics(), "测试"),
                "图片文件": "",
                "错误": str(exc),
            }
        summary_rows.append(row)
        #print_metrics(metrics)
        #print(f"图片: {plot_path}")

    summary = pd.DataFrame(summary_rows)
    active_summary, full_summary = rank_single_factor_summary(summary, config)
    active_library, library_all, rejected_library = build_factor_library(
        full_summary,
        factors,
        config,
    )
    if not plot_all:
        plot_paths = generate_top_single_factor_plots(
            active_library,
            data,
            factors,
            config,
            output_dir,
            factor_label_map,
            split_time,
        )
        if plot_paths:
            for frame in (active_library, library_all, full_summary):
                if not frame.empty and {"因子", "图片文件"}.issubset(frame.columns):
                    frame["图片文件"] = frame["因子"].map(plot_paths).fillna(frame["图片文件"])

    save_factor_library(active_library, library_all, rejected_library, config)
    full_summary_path = output_dir / "single_factor_all_summary.csv"
    active_library.to_csv(summary_path, index=False, encoding="utf-8-sig")
    full_summary.to_csv(full_summary_path, index=False, encoding="utf-8-sig")
    qcut_summary_path = output_dir / "qcut_group_summary.csv"
    pd.DataFrame(qcut_summary_rows).to_csv(
        qcut_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    library_dir = get_factor_library_dir(config)
    if run_dir is not None:
        snapshot_active_factor_library(config, run_dir)
        copy_existing_files(
            [
                summary_path,
                full_summary_path,
                qcut_summary_path,
                library_dir / "active_factors.csv",
                library_dir / "factor_library_all.csv",
                library_dir / "rejected_factors.csv",
            ],
            run_dir,
        )
    print(f"\n单因子入库汇总: {summary_path}")
    print(f"单因子全量汇总: {full_summary_path}")
    print(f"因子库active: {library_dir / 'active_factors.csv'}")
    print(f"因子库全量: {library_dir / 'factor_library_all.csv'}")
    print(f"因子库拒绝: {library_dir / 'rejected_factors.csv'}")
    print(f"qcut分组汇总: {qcut_summary_path}")
    return active_library


def main() -> None:
    """单因子回测脚本入口。"""
    config = BacktestConfig()

    def action() -> pd.DataFrame:
        try:
            print(f"读取 {config.symbol} 的 {config.bar_size} 分钟数据...")
            data = fetch_intraday_data(config)

            print("按需构建单因子矩阵...")
            factors = build_single_factor_matrix(data, config)

            return run_single_factor_backtests(data, factors, config)
        finally:
            stop_wind()

    run_tracked(config, "single", action)


if __name__ == "__main__":
    main()

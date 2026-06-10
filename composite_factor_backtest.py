from __future__ import annotations

"""综合因子 / XGBoost 滚动训练回测模块。

本文件把单因子信号合成为一个可交易的综合信号：
- 支持 selected / all / best 三种因子输入范围。
- 支持 signal / continuous / both 三种特征形式。
- 使用下一根K线涨跌方向作为分类目标，训练三分类 XGBoost。
- 采用滚动窗口训练和滚动预测，尽量模拟真实上线时只能使用历史数据的状态。
- 同时输出 XGBoost 策略和等权投票基准策略，方便判断模型是否真的优于简单合成。
"""

from pathlib import Path
from typing import Any
import warnings

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
from project_fingerprint import build_source_fingerprint_hash
from factor_library import get_factor_library_dir
from factors import (
    build_factors,
    fetch_intraday_data,
    get_factor_columns,
    get_factor_id_map,
    get_factor_label_map,
    get_last_related_data_coverage,
    score_to_raw_signal,
    stop_wind,
)
from single_factor_backtest import (
    build_signal_from_score,
    calculate_metrics,
    infer_annual_periods,
    plot_backtest_result,
    print_metrics,
    run_backtest,
    safe_run_backtest,
    choose_factor_direction,
    split_train_validation_test_index,
)


TARGET_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}
CLASS_TO_TARGET = {class_id: target for target, class_id in TARGET_TO_CLASS.items()}


def get_active_factor_library_path(config: BacktestConfig) -> Path:
    """返回综合模型使用的 active 因子库路径。"""
    if getattr(config, "use_frozen_active_library", False):
        frozen_path = getattr(config, "frozen_active_library_path", None)
        if not frozen_path:
            raise ValueError("use_frozen_active_library=True 时必须配置 frozen_active_library_path。")
        active_path = Path(frozen_path)
        if not active_path.is_absolute():
            active_path = Path(config.output_dir) / active_path
        return active_path
    return get_factor_library_dir(config) / "active_factors.csv"


def load_active_factor_names(config: BacktestConfig) -> list[str]:
    """先读取 active 因子名，用于综合回测按需构建因子矩阵。"""
    active_path = get_active_factor_library_path(config)
    if not active_path.exists():
        raise FileNotFoundError(
            f"没有找到 active 因子库: {active_path}。请先运行 single_factor_backtest.py 更新因子库。"
        )

    active_library = pd.read_csv(active_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {active_path}")
    return [str(factor_name) for factor_name in active_library["因子"].dropna().tolist()]


def build_factor_cache_meta(
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> dict[str, Any]:
    """生成 active 因子矩阵缓存的轻量校验信息。"""
    return {
        "symbol": config.symbol,
        "bar_size": int(config.bar_size),
        "start": str(data.index.min()) if len(data.index) else "",
        "end": str(data.index.max()) if len(data.index) else "",
        "rows": int(len(data.index)),
        "requested_factors": list(requested_factors),
        "zscore_window": int(config.zscore_window),
        "signal_threshold": float(config.signal_threshold),
        "enable_cross_asset_factors": bool(getattr(config, "enable_cross_asset_factors", False)),
        "enable_macro_state_factors": bool(getattr(config, "enable_macro_state_factors", False)),
        "related_symbols": list(getattr(config, "related_symbols", []) or []),
        "macro_state_symbols": list(getattr(config, "macro_state_symbols", []) or []),
        "factor_source_hash": build_source_fingerprint_hash(),
    }


def load_factor_matrix_cache(
    cache_path: Path,
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame | None:
    """读取并校验 active 因子矩阵缓存。"""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(cached, pd.DataFrame):
        return None
    expected_meta = build_factor_cache_meta(data, requested_factors, config)
    if cached.attrs.get("cache_meta") != expected_meta:
        return None
    if not cached.index.equals(data.index):
        return None
    missing_columns = sorted(set(requested_factors).difference(cached.columns))
    if missing_columns:
        return None
    return cached[list(requested_factors)].copy()


def save_factor_matrix_cache(
    factors: pd.DataFrame,
    cache_path: Path,
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> None:
    """保存 active 因子矩阵缓存，供下一次综合回测复用。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = factors.copy()
    cached.attrs["cache_meta"] = build_factor_cache_meta(data, requested_factors, config)
    cached.to_pickle(cache_path)


def load_active_factor_pool(
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> list[str]:
    """读取 active_factors.csv，并把它作为候选因子的硬边界。

    综合模型只考虑已经通过单因子入库流程的因子。
    xgboost_feature_scope 只在这个 active 池内部继续选择，而不是面对代码能生成的全部因子。
    """
    if getattr(config, "use_frozen_active_library", False):
        frozen_path = getattr(config, "frozen_active_library_path", None)
        if not frozen_path:
            raise ValueError("use_frozen_active_library=True 时必须配置 frozen_active_library_path。")
        active_path = Path(frozen_path)
        if not active_path.is_absolute():
            active_path = Path(config.output_dir) / active_path
    else:
        active_path = get_factor_library_dir(config) / "active_factors.csv"
    if not active_path.exists():
        raise FileNotFoundError(
            f"没有找到 active 因子库: {active_path}。请先运行 single_factor_backtest.py 更新因子库。"
        )

    active_library = pd.read_csv(active_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {active_path}")

    factor_columns = get_factor_columns(factors)
    factor_set = set(factor_columns)
    active_names = [
        str(factor_name)
        for factor_name in active_library["因子"].dropna().tolist()
        if str(factor_name) in factor_set
    ]
    if not active_names:
        raise ValueError(
            "active 因子库中没有任何因子能在当前 factors.py 生成。"
            "请检查 active 因子库、related_symbols 或重新运行单因子回测。"
        )

    missing_count = int(active_library["因子"].notna().sum()) - len(active_names)
    if missing_count > 0:
        print(f"active 因子库中有 {missing_count} 个因子当前不可生成，已自动跳过。")

    return active_names


def select_best_factors_on_training(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
    split_time: pd.Timestamp,
) -> tuple[list[str], pd.DataFrame]:
    """在静态训练集上选择表现最好的因子。

    这是非滚动 best 模式使用的旧路径：
    先对每个因子做训练集单因子回测，再按训练夏普和累计收益排序，
    最后取前 xgboost_best_top_n 个作为 XGBoost 候选特征。

    如果启用 walk-forward feature selection，则不会走这个函数，
    因为每个滚动窗口都会重新选因子。
    """
    train_data = data.loc[data.index < split_time]
    rows = []

    for factor_name in get_factor_columns(factors):
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
        _train_df, metrics, error = safe_run_backtest(train_data, train_signal, config)
        if error is not None:
            continue

        rows.append(
            {
                "因子": factor_name,
                "方向": direction,
                **metrics,
                **direction_metrics,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("训练段没有可用因子，无法选择 best 特征。")

    summary = summary.dropna(subset=["夏普比率", "累计收益"])
    summary = summary[
        (summary["夏普比率"] >= config.min_select_sharpe)
        & (summary["累计收益"] >= config.min_select_total_return)
    ]
    if summary.empty:
        raise ValueError("训练段没有因子满足 min_select_sharpe/min_select_total_return。")

    summary = summary.sort_values(["夏普比率", "累计收益"], ascending=False)
    selected = summary["因子"].head(config.xgboost_best_top_n).tolist()
    return selected, summary


def get_selected_factors(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
    split_time: pd.Timestamp,
) -> tuple[list[str], pd.DataFrame | None]:
    """根据 xgboost_feature_scope 决定 XGBoost 候选因子集合。

    active_factors.csv 是硬边界，所有模式都只能在 active 因子池内继续选择：
    - all：使用所有当前可生成的 active 因子。
    - selected：使用 config.selected_factors，但这些因子必须属于 active 因子池。
    - best：只在 active 因子池内做静态或滚动 best 选择。
    """
    active_factor_columns = load_active_factor_pool(factors, config)
    active_factors = factors[active_factor_columns]
    scope = config.xgboost_feature_scope.lower()

    if scope == "all":
        return active_factor_columns, None

    if scope == "best":
        if config.xgboost_walk_forward_feature_selection:
            return active_factor_columns, None
        return select_best_factors_on_training(data, active_factors, config, split_time)

    if scope != "selected":
        raise ValueError('xgboost_feature_scope 只能是 "selected", "all", 或 "best"。')

    if config.selected_factors:
        missing = sorted(set(config.selected_factors).difference(active_factor_columns))
        if missing:
            raise ValueError(
                "selected_factors 中存在不在 active 因子库内或当前不可生成的因子: "
                f"{missing}"
            )
        return list(config.selected_factors), None
    return active_factor_columns, None


def should_use_walk_forward_selection(config: BacktestConfig) -> bool:
    """判断是否启用滚动窗口内动态选因。"""
    return (
        config.xgboost_feature_scope.lower() == "best"
        and bool(config.xgboost_walk_forward_feature_selection)
    )


def build_factor_signal_features(
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """把多个因子的连续值转换成 -1/0/1 信号特征矩阵。

    这是当前框架的核心自变量之一：
    每个单因子的独立多空判断作为 XGBoost 输入，让模型学习不同因子信号组合下的下一根K线方向。
    """
    missing = sorted(set(selected_factors).difference(factors.columns))
    if missing:
        raise ValueError(f"因子数据缺少以下字段: {missing}")

    feature_data = {
        factor_name: score_to_raw_signal(
            factors[factor_name].replace([np.inf, -np.inf], np.nan),
            config.signal_threshold,
        )
        for factor_name in selected_factors
    }
    return pd.DataFrame(feature_data, index=factors.index)


def calculate_signal_streak(signal: pd.Series) -> pd.Series:
    """统计同方向非零信号连续出现的 K 线数量。"""
    clean_signal = signal.fillna(0.0)
    group_id = clean_signal.ne(clean_signal.shift(1)).cumsum()
    streak = clean_signal.groupby(group_id).cumcount() + 1
    return streak.where(clean_signal != 0.0, 0.0).astype("float64")


def build_factor_state_features(
    continuous_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    selected_factors: list[str],
) -> pd.DataFrame:
    """为 XGBoost 构造动态因子状态特征。

    这些特征帮助模型区分刚刚翻转的信号和已经持续一段时间的信号，
    也帮助模型识别因子是在增强还是在衰减。
    """
    state_parts = []
    for factor_name in selected_factors:
        value = continuous_features[factor_name]
        signal = signal_features[factor_name]
        state_parts.extend(
            [
                value.diff().rename(f"{factor_name}_value_diff_1"),
                value.shift(1).rename(f"{factor_name}_value_lag_1"),
                signal.shift(1).rename(f"{factor_name}_signal_lag_1"),
                signal.diff().rename(f"{factor_name}_signal_change_1"),
                calculate_signal_streak(signal).rename(f"{factor_name}_signal_streak"),
            ]
        )
    return pd.concat(state_parts, axis=1)


def build_features_for_factors(
    factors: pd.DataFrame,
    signal_features: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """根据配置构造 XGBoost 最终特征矩阵。

    signal：只使用 -1/0/1 信号。
    continuous：只使用标准化后的连续因子值。
    both：每个因子同时放入 value 和 signal 两个特征。
    """
    mode = config.xgboost_feature_mode.lower()
    if mode not in {"signal", "continuous", "both"}:
        raise ValueError('xgboost_feature_mode 只能是 "signal", "continuous", 或 "both"。')

    continuous_features = factors[selected_factors].replace([np.inf, -np.inf], np.nan)
    feature_parts = []

    if mode == "signal":
        feature_parts.append(signal_features[selected_factors])
    elif mode == "continuous":
        feature_parts.append(continuous_features)
    else:
        for factor_name in selected_factors:
            feature_parts.append(continuous_features[factor_name].rename(f"{factor_name}_value"))
            feature_parts.append(signal_features[factor_name].rename(f"{factor_name}_signal"))

    if getattr(config, "xgboost_include_factor_state_features", False):
        feature_parts.append(
            build_factor_state_features(
                continuous_features,
                signal_features,
                selected_factors,
            )
        )
    return pd.concat(feature_parts, axis=1)


def build_xgboost_features(
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """构造 XGBoost 特征矩阵的便捷函数。"""
    signal_features = build_factor_signal_features(factors, selected_factors, config)
    market_state = build_market_state_filter(data, config)
    return build_features_for_factors(factors, signal_features, selected_factors, config)


def get_feature_base_factor(feature_name: str, selected_factors: list[str]) -> str:
    """从模型特征名还原基础因子名。

    both 模式下会出现 xxx_value / xxx_signal，
    输出特征重要性时需要把它们映射回原始因子。
    """
    if feature_name in selected_factors:
        return feature_name
    for suffix in ("_value", "_signal"):
        if feature_name.endswith(suffix):
            base_name = feature_name[: -len(suffix)]
            if base_name in selected_factors:
                return base_name
    return feature_name


def get_xgboost_target_neutral_threshold(config: BacktestConfig) -> float:
    """返回 XGBoost 分类目标中的中性收益阈值。

    阈值单位最终转换为收益率。
    如果配置为 None，则用手续费 + 滑点作为中性区间，
    只有下一根K线收益超过交易成本附近的波动才标记为上涨或下跌。
    """
    neutral_bps = config.xgboost_target_neutral_bps
    if neutral_bps is None:
        neutral_bps = config.commission_bps + config.slippage_bps
    return max(0.0, float(neutral_bps)) / 10000.0


def select_factors_in_window(
    signal_features: pd.DataFrame,
    train_target: pd.Series,
    candidate_factors: list[str],
    config: BacktestConfig,
) -> tuple[list[str], pd.DataFrame]:
    """在单个滚动训练窗口内做特征选择。

    逻辑：
    1. 计算每个候选因子信号与训练目标的相关性。
    2. 先按绝对相关性筛出候选池。
    3. 再按因子之间的相关性做去重。
    4. 最多保留 xgboost_best_top_n 个因子。

    这样可以降低海量相似因子一起进入模型导致的过拟合风险。
    """
    frame = signal_features[candidate_factors].assign(__target__=train_target)
    frame = frame.dropna(subset=["__target__"])
    if frame.empty:
        return [], pd.DataFrame()

    target = frame["__target__"]
    factor_frame = frame[candidate_factors].replace([np.inf, -np.inf], np.nan)
    valid_factors = factor_frame.columns[factor_frame.std(ddof=0) > 0].tolist()
    if not valid_factors or target.std(ddof=0) == 0:
        return [], pd.DataFrame()

    factor_frame = factor_frame[valid_factors]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = factor_frame.corrwith(target).replace([np.inf, -np.inf], np.nan)
    summary = pd.DataFrame(
        {
            "因子": scores.index,
            "目标相关性": scores.values,
            "abs_score": scores.abs().values,
            "有效样本数": factor_frame.notna().sum().reindex(scores.index).values,
        }
    ).dropna(subset=["abs_score"])
    summary = summary[summary["有效样本数"] >= config.xgboost_min_train_samples]
    if summary.empty:
        return [], summary

    # 先放宽候选池，再做相关性去重；否则前几名高度相似时会挤掉其他信息源。
    candidate_limit = max(
        config.xgboost_best_top_n,
        config.xgboost_best_top_n * max(1, int(config.xgboost_candidate_multiplier)),
    )
    summary = summary.sort_values("abs_score", ascending=False).head(candidate_limit)

    selected = []
    selected_signals = factor_frame[summary["因子"].tolist()]
    max_corr = float(config.xgboost_max_feature_corr)
    for factor_name in summary["因子"]:
        if len(selected) >= config.xgboost_best_top_n:
            break
        if not selected:
            selected.append(factor_name)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pair_corr = selected_signals[selected].corrwith(selected_signals[factor_name]).abs()
        if pair_corr.dropna().lt(max_corr).all():
            selected.append(factor_name)

    summary["入选"] = summary["因子"].isin(selected)
    return selected, summary


def calculate_next_bar_direction(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """生成 XGBoost 的三分类预测目标。

    未来 horizon 根累计收益 > 中性阈值：1
    未来 horizon 根累计收益 < -中性阈值：-1
    落在中性区间：0
    """
    next_return = calculate_future_horizon_return(data, index, config)
    neutral_threshold = get_xgboost_target_neutral_threshold(config)
    direction = pd.Series(0.0, index=index, dtype="float64")
    direction[next_return > neutral_threshold] = 1.0
    direction[next_return < -neutral_threshold] = -1.0
    direction[next_return.isna()] = np.nan
    return direction


def get_xgboost_target_horizon(config: BacktestConfig) -> int:
    """返回配置中的预测目标跨度，并保证至少为一根 K 线。"""
    return max(1, int(getattr(config, "xgboost_target_horizon", 1) or 1))


def calculate_future_horizon_return(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """计算从下一根 K 线开盘到目标跨度收盘的未来累计收益。

    horizon=1 时等价于旧版下一根 K 线 open 到 close 目标。
    horizon=3 时使用 open[t+1] 到 close[t+3] 的收益，并对齐到时点 t。
    """
    horizon = get_xgboost_target_horizon(config)
    entry_open = data["open"].replace(0, np.nan).shift(-1)
    exit_close = data["close"].shift(-horizon)
    return (exit_close / entry_open - 1.0).reindex(index)


def calculate_next_bar_return(data: pd.DataFrame, index: pd.Index) -> pd.Series:
    """计算下一根K线 open 到 close 的收益，并对齐到当前时点。"""
    open_price = data["open"].replace(0, np.nan)
    return (data["close"] / open_price - 1.0).reindex(index).shift(-1)


def choose_signal_direction_on_training(
    raw_signal: pd.Series,
    next_return: pd.Series,
) -> float:
    """在训练窗口内判断一个合成信号是否需要反向。

    比较原始信号和反向信号在训练窗口的下一根K线收益贡献。
    该逻辑用于 XGBoost 概率信号和等权投票基准的方向校准。
    """
    aligned = pd.DataFrame(
        {
            "signal": raw_signal,
            "next_return": next_return.reindex(raw_signal.index),
        }
    ).dropna()
    if aligned.empty or aligned["signal"].abs().sum() == 0:
        return 1.0

    forward_return = (aligned["signal"] * aligned["next_return"]).sum()
    inverse_return = (-aligned["signal"] * aligned["next_return"]).sum()
    return -1.0 if inverse_return > forward_return else 1.0


def probabilities_to_trade_signal(
    probabilities: pd.DataFrame,
    config: BacktestConfig,
    min_edge: float | None = None,
    min_probability: float | None = None,
) -> pd.Series:
    """把 XGBoost 三分类概率转换成 -1/0/1 交易信号。

    使用 prob_up - prob_down 作为方向优势。
    只有优势超过 xgboost_trade_min_edge，且方向概率超过 xgboost_trade_min_probability 时才开仓。
    """
    edge = probabilities["prob_up"] - probabilities["prob_down"]
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    min_edge = max(
        0.0,
        float(config.xgboost_trade_min_edge if min_edge is None else min_edge),
    )
    min_probability = max(
        0.0,
        float(config.xgboost_trade_min_probability if min_probability is None else min_probability),
    )

    signal = pd.Series(0.0, index=probabilities.index, dtype="float64")
    long_mask = (edge >= min_edge) & (directional_probability >= min_probability)
    short_mask = (edge <= -min_edge) & (directional_probability >= min_probability)
    signal[long_mask] = 1.0
    signal[short_mask] = -1.0
    signal[probabilities[["prob_up", "prob_down"]].isna().any(axis=1)] = np.nan
    return signal


def build_market_state_filter(data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """仅使用当前和历史 K 线构造简单的交易质量过滤器。"""
    window = max(20, int(getattr(config, "xgboost_trade_filter_window", 120) or 120))
    min_periods = max(20, window // 3)
    use_filters = bool(getattr(config, "xgboost_trade_use_market_filters", False))

    abs_intrabar_return = (data["close"] / data["open"].replace(0, np.nan) - 1.0).abs()
    vol_rank = abs_intrabar_return.rolling(window=window, min_periods=min_periods).rank(pct=True)

    liquidity_series = None
    for column in ("amt", "amount", "volume"):
        if column in data.columns:
            liquidity_series = data[column].replace(0, np.nan).abs()
            if liquidity_series.notna().any():
                break
    if liquidity_series is None:
        liquidity_series = pd.Series(np.nan, index=data.index, dtype="float64")
    liquidity_rank = liquidity_series.rolling(window=window, min_periods=min_periods).rank(pct=True)

    trade_allowed = pd.Series(True, index=data.index, dtype="bool")
    if use_filters:
        min_vol_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_volatility_rank", 0.0) or 0.0)),
        )
        min_liq_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_liquidity_rank", 0.0) or 0.0)),
        )
        if min_vol_rank > 0:
            trade_allowed &= vol_rank.fillna(0.0) >= min_vol_rank
        if min_liq_rank > 0 and liquidity_rank.notna().any():
            trade_allowed &= liquidity_rank.fillna(0.0) >= min_liq_rank
    trade_allowed &= abs_intrabar_return.notna()

    return pd.DataFrame(
        {
            "volatility_rank": vol_rank.astype("float64"),
            "liquidity_rank": liquidity_rank.astype("float64"),
            "trade_allowed": trade_allowed.astype("float64"),
        },
        index=data.index,
    )


def build_confidence_position_size(
    probabilities: pd.DataFrame,
    config: BacktestConfig,
    min_edge: float | None = None,
    min_probability: float | None = None,
) -> pd.Series:
    """把模型置信度映射到 0 到 max_size 之间的目标仓位。"""
    edge = (probabilities["prob_up"] - probabilities["prob_down"]).abs()
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    min_edge = min(
        0.999,
        max(0.0, float(config.xgboost_trade_min_edge if min_edge is None else min_edge)),
    )
    min_probability = min(
        0.999,
        max(
            0.0,
            float(config.xgboost_trade_min_probability if min_probability is None else min_probability),
        ),
    )
    max_size = max(0.0, float(getattr(config, "xgboost_position_size_max", 1.0) or 1.0))

    edge_strength = ((edge - min_edge) / max(1e-9, 1.0 - min_edge)).clip(lower=0.0, upper=1.0)
    probability_strength = (
        (directional_probability - min_probability) / max(1e-9, 1.0 - min_probability)
    ).clip(lower=0.0, upper=1.0)
    confidence = edge_strength.combine(probability_strength, max).clip(lower=0.0, upper=1.0)

    if bool(getattr(config, "xgboost_use_dynamic_position_sizing", False)):
        power = max(0.25, float(getattr(config, "xgboost_position_size_power", 1.0) or 1.0))
        min_size = min(
            max_size,
            max(0.0, float(getattr(config, "xgboost_position_size_min", 0.0) or 0.0)),
        )
        size = min_size + (max_size - min_size) * confidence.pow(power)
        size = size.where(confidence > 0, 0.0)
    else:
        size = (confidence > 0).astype("float64") * max_size

    size[probabilities[["prob_up", "prob_down"]].isna().any(axis=1)] = np.nan
    return size.astype("float64")


def build_dynamic_confidence_position_size(
    probabilities: pd.DataFrame,
    min_edge: pd.Series,
    min_probability: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """使用逐行校准阈值，把置信度映射为仓位大小。"""
    edge = (probabilities["prob_up"] - probabilities["prob_down"]).abs()
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    edge_threshold = min_edge.reindex(probabilities.index).fillna(float(config.xgboost_trade_min_edge))
    probability_threshold = min_probability.reindex(probabilities.index).fillna(
        float(config.xgboost_trade_min_probability)
    )
    max_size = max(0.0, float(getattr(config, "xgboost_position_size_max", 1.0) or 1.0))

    edge_strength = (
        (edge - edge_threshold) / (1.0 - edge_threshold).clip(lower=1e-9)
    ).clip(lower=0.0, upper=1.0)
    probability_strength = (
        (directional_probability - probability_threshold)
        / (1.0 - probability_threshold).clip(lower=1e-9)
    ).clip(lower=0.0, upper=1.0)
    confidence = edge_strength.combine(probability_strength, max).clip(lower=0.0, upper=1.0)

    if bool(getattr(config, "xgboost_use_dynamic_position_sizing", False)):
        power = max(0.25, float(getattr(config, "xgboost_position_size_power", 1.0) or 1.0))
        min_size = min(
            max_size,
            max(0.0, float(getattr(config, "xgboost_position_size_min", 0.0) or 0.0)),
        )
        size = min_size + (max_size - min_size) * confidence.pow(power)
        size = size.where(confidence > 0, 0.0)
    else:
        size = (confidence > 0).astype("float64") * max_size

    size[probabilities[["prob_up", "prob_down"]].isna().any(axis=1)] = np.nan
    return size.astype("float64")


def apply_position_rules(
    target_position: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """对模型目标仓位应用交易执行层规则。

    规则只使用当前和历史目标仓位，不读取未来收益：
    - 最小持仓期：降低刚开仓后立刻反向或清仓。
    - 反转冷却：刚退出后等待若干根 K 线再重新开仓。
    - 小变化忽略：同方向仓位微调不频繁交易。
    - 可选仓位平滑：让目标仓位逐步靠近模型输出。
    """
    clean_target = target_position.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    if not bool(getattr(config, "xgboost_use_position_rules", False)):
        return clean_target

    min_holding_bars = max(0, int(getattr(config, "xgboost_min_holding_bars", 0) or 0))
    cooldown_bars = max(0, int(getattr(config, "xgboost_reentry_cooldown_bars", 0) or 0))
    min_change = max(0.0, float(getattr(config, "xgboost_min_position_change", 0.0) or 0.0))
    alpha = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_position_smoothing_alpha", 1.0) or 1.0)),
    )

    filtered_values: list[float] = []
    current_position = 0.0
    holding_bars = 0
    cooldown_remaining = 0

    for desired_position in clean_target:
        desired_position = float(desired_position)
        current_sign = np.sign(current_position)
        desired_sign = np.sign(desired_position)
        next_position = desired_position

        if cooldown_remaining > 0 and current_sign == 0 and desired_sign != 0:
            next_position = 0.0
            desired_sign = 0.0
            cooldown_remaining -= 1
        elif cooldown_remaining > 0 and current_sign == 0:
            cooldown_remaining -= 1

        if current_sign != 0:
            is_exit = desired_sign == 0
            is_reverse = desired_sign != 0 and desired_sign != current_sign
            is_same_direction = desired_sign == current_sign

            if holding_bars < min_holding_bars and (is_exit or is_reverse):
                next_position = current_position
            elif is_reverse:
                next_position = 0.0 if cooldown_bars > 0 else desired_position
                cooldown_remaining = cooldown_bars
            elif is_exit:
                next_position = 0.0
                cooldown_remaining = cooldown_bars
            elif is_same_direction and abs(desired_position - current_position) < min_change:
                next_position = current_position

        if alpha < 1.0:
            next_position = alpha * next_position + (1.0 - alpha) * current_position

        if abs(next_position) < 1e-12:
            next_position = 0.0

        next_sign = np.sign(next_position)
        if next_sign == 0:
            holding_bars = 0
        elif next_sign == current_sign:
            holding_bars += 1
        else:
            holding_bars = 1

        current_position = float(next_position)
        filtered_values.append(current_position)

    return pd.Series(filtered_values, index=target_position.index, dtype="float64")


def build_confidence_rank_filter(
    confidence_score: pd.Series,
    config: BacktestConfig,
) -> pd.DataFrame:
    """基于近期置信度分位构造自适应过滤器。"""
    window = max(20, int(getattr(config, "xgboost_trade_confidence_rank_window", 240) or 240))
    min_periods = max(20, window // 3)
    rank = confidence_score.abs().rolling(window=window, min_periods=min_periods).rank(pct=True)

    allowed = pd.Series(True, index=confidence_score.index, dtype="bool")
    if bool(getattr(config, "xgboost_trade_use_confidence_rank_filter", False)):
        min_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_confidence_rank", 0.0) or 0.0)),
        )
        if min_rank > 0:
            allowed &= rank.fillna(0.0) >= min_rank
    allowed &= confidence_score.notna()

    return pd.DataFrame(
        {
            "confidence_rank": rank.astype("float64"),
            "confidence_trade_allowed": allowed.astype("float64"),
        },
        index=confidence_score.index,
    )


def calibrate_trade_thresholds_on_training(
    probabilities: pd.DataFrame,
    forward_return: pd.Series,
    trade_allowed: pd.Series,
    config: BacktestConfig,
) -> tuple[float, float]:
    """根据近期训练窗口选择交易阈值。"""
    base_edge = max(0.0, float(config.xgboost_trade_min_edge))
    base_probability = max(0.0, float(config.xgboost_trade_min_probability))
    if not bool(getattr(config, "xgboost_auto_calibrate_trade_thresholds", False)):
        return base_edge, base_probability

    edge_grid = sorted(
        {
            base_edge,
            *[
                max(base_edge, float(value))
                for value in (getattr(config, "xgboost_trade_edge_grid", []) or [])
            ],
        }
    )
    probability_grid = sorted(
        {
            base_probability,
            *[
                max(base_probability, float(value))
                for value in (getattr(config, "xgboost_trade_probability_grid", []) or [])
            ],
        }
    )
    min_trades = max(1, int(getattr(config, "xgboost_threshold_min_trades", 20) or 20))

    aligned_return = forward_return.reindex(probabilities.index)
    allowed = trade_allowed.reindex(probabilities.index).fillna(False).astype(bool)
    best_score = -np.inf
    best_pair = (base_edge, base_probability)

    for edge_threshold in edge_grid:
        for probability_threshold in probability_grid:
            raw_signal = probabilities_to_trade_signal(
                probabilities,
                config,
                min_edge=edge_threshold,
                min_probability=probability_threshold,
            ).where(allowed, 0.0)
            trade_mask = raw_signal.fillna(0.0) != 0
            if int(trade_mask.sum()) < min_trades:
                continue
            strategy_return = (raw_signal * aligned_return).replace([np.inf, -np.inf], np.nan).dropna()
            if strategy_return.empty:
                continue
            volatility = float(strategy_return.std(ddof=0))
            score = float(strategy_return.mean() / volatility) if volatility > 0 else float(strategy_return.mean())
            if score > best_score:
                best_score = score
                best_pair = (float(edge_threshold), float(probability_threshold))

    return best_pair


def vote_score_to_trade_signal(vote_score: pd.Series, config: BacktestConfig) -> pd.Series:
    """把等权投票分数转换成 -1/0/1 交易信号。"""
    min_abs_score = max(0.0, float(config.benchmark_vote_min_abs_score))
    signal = pd.Series(0.0, index=vote_score.index, dtype="float64")
    signal[vote_score > min_abs_score] = 1.0
    signal[vote_score < -min_abs_score] = -1.0
    signal[vote_score.isna()] = np.nan
    return signal


def build_backtest_signal_from_columns(
    signal: pd.DataFrame,
    score_column: str,
    raw_signal_column: str,
    position_column: str,
) -> pd.DataFrame:
    """把综合信号表中的指定列整理成 run_backtest 需要的标准格式。"""
    return pd.DataFrame(
        {
            "composite_score": signal[score_column],
            "raw_signal": signal[raw_signal_column],
            "position": signal[position_column],
        },
        index=signal.index,
    )


def train_xgboost_classifier(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
) -> Any:
    """训练单个 XGBoost 三分类模型。

    输入标签为 -1/0/1，内部映射成 XGBoost 需要的 0/1/2 类别。
    模型参数全部来自 BacktestConfig，便于统一调参和复现实验。
    """
    import xgboost as xgb

    validation_matrix = None
    validation_size = 0
    if bool(getattr(config, "xgboost_use_validation_early_stopping", False)):
        validation_ratio = min(
            0.40,
            max(0.05, float(getattr(config, "xgboost_validation_ratio", 0.20) or 0.20)),
        )
        validation_size = int(len(train_features) * validation_ratio)
        validation_size = min(validation_size, max(0, len(train_features) - 50))

    if validation_size > 0:
        fit_features = train_features.iloc[:-validation_size]
        fit_target = train_target.reindex(fit_features.index)
        fit_weight = sample_weight.reindex(fit_features.index).fillna(1.0) if sample_weight is not None else None
        validation_features = train_features.iloc[-validation_size:]
        validation_target = train_target.reindex(validation_features.index)
        validation_weight = (
            sample_weight.reindex(validation_features.index).fillna(1.0)
            if sample_weight is not None
            else None
        )
        if fit_target.nunique() < 2 or validation_target.dropna().empty:
            fit_features = train_features
            fit_target = train_target
            fit_weight = sample_weight
            validation_features = None
            validation_target = None
            validation_weight = None
    else:
        fit_features = train_features
        fit_target = train_target
        fit_weight = sample_weight
        validation_features = None
        validation_target = None
        validation_weight = None

    train_label = fit_target.map(TARGET_TO_CLASS)
    train_matrix = xgb.DMatrix(
        fit_features[feature_columns],
        label=train_label,
        weight=None if fit_weight is None else fit_weight.reindex(fit_features.index).fillna(1.0),
        feature_names=feature_columns,
    )
    if validation_features is not None and validation_target is not None:
        validation_matrix = xgb.DMatrix(
            validation_features[feature_columns],
            label=validation_target.map(TARGET_TO_CLASS),
            weight=(
                None
                if validation_weight is None
                else validation_weight.reindex(validation_features.index).fillna(1.0)
            ),
            feature_names=feature_columns,
        )

    evals = [(train_matrix, "train")]
    train_kwargs: dict[str, Any] = {"verbose_eval": False}
    early_stopping_rounds = int(getattr(config, "xgboost_early_stopping_rounds", 0) or 0)
    if validation_matrix is not None and early_stopping_rounds > 0:
        evals.append((validation_matrix, "validation"))
        train_kwargs["early_stopping_rounds"] = early_stopping_rounds

    return xgb.train(
        params={
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": str(getattr(config, "xgboost_tree_method", "hist") or "hist"),
            "max_depth": int(config.xgboost_max_depth),
            "eta": float(config.xgboost_learning_rate),
            "subsample": float(config.xgboost_subsample),
            "colsample_bytree": float(config.xgboost_colsample_bytree),
            "min_child_weight": float(config.xgboost_min_child_weight),
            "gamma": float(config.xgboost_gamma),
            "lambda": float(config.xgboost_reg_lambda),
            "alpha": float(config.xgboost_reg_alpha),
            "seed": int(config.xgboost_random_state),
            "nthread": int(getattr(config, "xgboost_nthread", -1) or -1),
            "verbosity": 0,
        },
        dtrain=train_matrix,
        evals=evals,
        num_boost_round=int(config.xgboost_n_estimators),
        **train_kwargs,
    )


def predict_xgboost_probability(model: Any, matrix: Any) -> np.ndarray:
    """在可用时使用早停得到的最佳轮次预测概率。"""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        return model.predict(matrix, iteration_range=(0, int(best_iteration) + 1))
    return model.predict(matrix)


def get_enabled_composite_models(config: BacktestConfig) -> list[str]:
    """读取需要滚动对比的综合模型列表，并做轻量规范化。"""
    raw_models = getattr(config, "composite_model_names", ["xgboost"]) or ["xgboost"]
    alias = {
        "logistic": "logistic_regression",
        "lr": "logistic_regression",
        "rf": "random_forest",
        "et": "extra_trees",
    }
    models: list[str] = []
    for model_name in raw_models:
        normalized = alias.get(str(model_name).strip().lower(), str(model_name).strip().lower())
        if normalized and normalized not in models:
            models.append(normalized)
    return models or ["xgboost"]


def train_sklearn_classifier(
    model_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
) -> Any:
    """训练 sklearn 三分类模型，用作 XGBoost 之外的模型对照。"""
    train_label = train_target.map(TARGET_TO_CLASS).astype(int)
    weight = None if sample_weight is None else sample_weight.reindex(train_features.index).fillna(1.0)

    if model_name == "logistic_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(
                C=float(getattr(config, "composite_logistic_c", 1.0) or 1.0),
                max_iter=int(getattr(config, "composite_logistic_max_iter", 1000) or 1000),
                multi_class="auto",
                random_state=int(config.xgboost_random_state),
            ),
        )
        fit_kwargs = {"logisticregression__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            RandomForestClassifier(
                n_estimators=int(getattr(config, "composite_sklearn_n_estimators", 120) or 120),
                max_depth=int(config.xgboost_max_depth) if int(config.xgboost_max_depth) > 0 else None,
                min_samples_leaf=max(1, int(getattr(config, "xgboost_min_child_weight", 1) or 1)),
                random_state=int(config.xgboost_random_state),
                n_jobs=int(getattr(config, "composite_sklearn_n_jobs", -1) or -1),
                class_weight=None,
            ),
        )
        fit_kwargs = {"randomforestclassifier__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            ExtraTreesClassifier(
                n_estimators=int(getattr(config, "composite_sklearn_n_estimators", 120) or 120),
                max_depth=int(config.xgboost_max_depth) if int(config.xgboost_max_depth) > 0 else None,
                min_samples_leaf=max(1, int(getattr(config, "xgboost_min_child_weight", 1) or 1)),
                random_state=int(config.xgboost_random_state),
                n_jobs=int(getattr(config, "composite_sklearn_n_jobs", -1) or -1),
                class_weight=None,
            ),
        )
        fit_kwargs = {"extratreesclassifier__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    raise ValueError(
        "未知综合模型: "
        f"{model_name}。可选: xgboost, logistic_regression, random_forest, extra_trees。"
    )


def train_composite_classifier(
    model_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
) -> Any:
    """按模型名称训练综合三分类模型。"""
    if model_name == "xgboost":
        return train_xgboost_classifier(
            train_features,
            train_target,
            feature_columns,
            config,
            sample_weight=sample_weight,
        )
    return train_sklearn_classifier(
        model_name,
        train_features,
        train_target,
        feature_columns,
        config,
        sample_weight=sample_weight,
    )


def predict_composite_probability(
    model_name: str,
    model: Any,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """统一输出三分类概率，列顺序固定为 down/flat/up。"""
    if model_name == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(features[feature_columns], feature_names=feature_columns)
        return predict_xgboost_probability(model, matrix)

    raw_probability = model.predict_proba(features[feature_columns])
    probability = np.zeros((len(features), 3), dtype="float64")
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)
    if classes is None:
        raise ValueError(f"{model_name} 没有 classes_，无法对齐三分类概率。")
    for source_column, class_id in enumerate(classes):
        if 0 <= int(class_id) <= 2:
            probability[:, int(class_id)] = raw_probability[:, source_column]
    row_sum = probability.sum(axis=1)
    empty_rows = row_sum <= 0
    if empty_rows.any():
        probability[empty_rows, 1] = 1.0
    return probability


def get_model_feature_importance(
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> pd.Series:
    """提取不同模型的特征重要性，无法提取时使用等权兜底。"""
    if model_name == "xgboost":
        return pd.Series(model.get_score(importance_type="gain"), dtype="float64")

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_columns, dtype="float64")
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype="float64")
        importance = np.abs(coef).mean(axis=0)
        return pd.Series(importance, index=feature_columns, dtype="float64")
    return pd.Series(1.0, index=feature_columns, dtype="float64")


def build_training_sample_weights(
    train_target: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """构造逐样本权重，让 XGBoost 更关注方向性标签。"""
    neutral_weight = max(
        0.0,
        float(getattr(config, "xgboost_train_neutral_class_weight", 1.0) or 1.0),
    )
    nonzero_weight = max(
        0.0,
        float(getattr(config, "xgboost_train_nonzero_class_weight", 1.0) or 1.0),
    )
    weights = pd.Series(nonzero_weight, index=train_target.index, dtype="float64")
    weights[train_target == 0] = neutral_weight
    weights[train_target.isna()] = 0.0
    return weights


def build_xgboost_rolling_signal(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
    predict_index: pd.Index,
    model_name: str = "xgboost",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame | None]:
    """执行指定模型的滚动训练、滚动预测并生成交易信号。

    每个预测时点只使用它之前的 train_window 根K线训练模型。
    每隔 xgboost_retrain_every 根K线重新训练一次，中间复用上一次模型。
    如果启用 walk-forward feature selection，则每次重新训练前都会在当前训练窗口内重新筛因子。

    返回：
    - signal：包含预测方向、概率、交易信号、持仓和基准投票信号。
    - feature_importance：滚动模型累计特征重要性。
    - features：最终构造的特征矩阵。
    - selection_summary：滚动选因明细，未启用时可能为 None。
    """
    model_name = str(model_name).strip().lower()
    if model_name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError("未安装 xgboost，请先安装: python -m pip install xgboost") from exc

    signal_features = build_factor_signal_features(factors, selected_factors, config)
    market_state = build_market_state_filter(data, config)
    use_walk_forward_selection = should_use_walk_forward_selection(config)
    if use_walk_forward_selection:
        features = pd.DataFrame(index=factors.index)
        feature_columns: list[str] = []
    else:
        features = build_features_for_factors(factors, signal_features, selected_factors, config)
        feature_columns = features.columns.tolist()
    target = calculate_next_bar_direction(data, factors.index, config)
    next_bar_return = calculate_next_bar_return(data, factors.index)
    predict_index = pd.Index(predict_index).intersection(factors.index)
    predict_positions = factors.index.get_indexer(predict_index)
    predict_positions = predict_positions[predict_positions >= 0]

    predicted_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    vote_score = pd.Series(np.nan, index=factors.index, dtype="float64")
    xgboost_signal_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    vote_signal_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    calibrated_min_edge = pd.Series(np.nan, index=factors.index, dtype="float64")
    calibrated_min_probability = pd.Series(np.nan, index=factors.index, dtype="float64")
    probabilities = pd.DataFrame(
        np.nan,
        index=factors.index,
        columns=["prob_down", "prob_flat", "prob_up"],
        dtype="float64",
    )
    importance_sum = pd.Series(dtype="float64")
    model_count = 0
    model = None
    active_feature_columns: list[str] = []
    active_selected_factors = list(selected_factors)
    active_xgboost_signal_direction = 1.0
    active_vote_signal_direction = 1.0
    active_min_edge = float(config.xgboost_trade_min_edge)
    active_min_probability = float(config.xgboost_trade_min_probability)
    selection_rows = []

    train_window = int(config.xgboost_train_window)
    min_train_samples = int(config.xgboost_min_train_samples)
    retrain_every = max(1, int(config.xgboost_retrain_every))
    target_horizon = get_xgboost_target_horizon(config)

    # 对每个样本外时点滚动预测；position 永远是当前预测点。
    # 多周期目标需要额外剔除训练窗口尾部尚未完全落地的标签，避免未来函数。
    for step, position in enumerate(predict_positions):
        print(
            f"{model_name}滚动预测进度: "
            f"{step + 1}/{len(predict_positions)} 个K线时点；"
            f"active候选因子={len(selected_factors)}；"
            f"本轮最多选因={config.xgboost_best_top_n}",
            end="\r",
        )
        train_start = max(0, position - train_window)
        train_end = position - target_horizon + 1
        if train_end <= train_start:
            continue
        train_target = target.iloc[train_start:train_end]

        # 首次预测或到达重训间隔时，使用当前历史窗口重新训练模型。
        if model is None or step % retrain_every == 0:
            if use_walk_forward_selection:
                active_selected_factors, window_selection = select_factors_in_window(
                    signal_features.iloc[train_start:train_end],
                    train_target,
                    selected_factors,
                    config,
                )
                if len(active_selected_factors) == 0:
                    continue
                features = build_features_for_factors(
                    factors,
                    signal_features,
                    active_selected_factors,
                    config,
                )
                feature_columns = features.columns.tolist()
                active_feature_columns = feature_columns

                if not window_selection.empty:
                    selected_labels = ",".join(active_selected_factors)
                    window_selection = window_selection.copy()
                    window_selection.insert(0, "预测时间", factors.index[position])
                    window_selection.insert(1, "训练开始", factors.index[train_start])
                    window_selection.insert(2, "训练结束", factors.index[train_end - 1])
                    window_selection["本轮入选因子"] = selected_labels
                    selection_rows.extend(window_selection.to_dict("records"))
            else:
                active_feature_columns = feature_columns

            train_features = features.iloc[train_start:train_end]
            train_frame = train_features.assign(__target__=train_target).dropna(subset=["__target__"])
            if bool(getattr(config, "xgboost_train_use_market_filters", False)):
                train_allowed_mask = (
                    market_state.loc[train_frame.index, "trade_allowed"].fillna(0.0) > 0
                )
                train_frame = train_frame.loc[train_allowed_mask]
            min_directional_samples = max(
                0,
                int(getattr(config, "xgboost_train_min_directional_samples", 0) or 0),
            )
            directional_sample_count = int((train_frame["__target__"] != 0).sum())
            if directional_sample_count < min_directional_samples:
                continue
            if len(train_frame) < min_train_samples or train_frame["__target__"].nunique() < 2:
                continue
            train_trade_allowed = market_state.loc[train_frame.index, "trade_allowed"].fillna(0.0) > 0
            train_sample_weight = build_training_sample_weights(train_frame["__target__"], config)

            model = train_composite_classifier(
                model_name,
                train_frame[active_feature_columns],
                train_frame["__target__"],
                active_feature_columns,
                config,
                sample_weight=train_sample_weight,
            )
            # 用训练窗口内的预测表现判断概率信号是否需要整体反向。
            if (
                config.xgboost_auto_calibrate_signal_direction
                or config.xgboost_auto_calibrate_trade_thresholds
            ):
                train_probability = predict_composite_probability(
                    model_name,
                    model,
                    train_frame[active_feature_columns],
                    active_feature_columns,
                )
                train_probability = pd.DataFrame(
                    train_probability,
                    index=train_frame.index,
                    columns=["prob_down", "prob_flat", "prob_up"],
                )
                active_min_edge, active_min_probability = calibrate_trade_thresholds_on_training(
                    train_probability,
                    next_bar_return,
                    train_trade_allowed,
                    config,
                )
                train_raw_signal = probabilities_to_trade_signal(
                    train_probability,
                    config,
                    min_edge=active_min_edge,
                    min_probability=active_min_probability,
                )
                train_raw_signal = train_raw_signal.where(train_trade_allowed, 0.0)
                if config.xgboost_auto_calibrate_signal_direction:
                    active_xgboost_signal_direction = choose_signal_direction_on_training(
                        train_raw_signal,
                        next_bar_return,
                    )
                else:
                    active_xgboost_signal_direction = 1.0
            else:
                active_min_edge = float(config.xgboost_trade_min_edge)
                active_min_probability = float(config.xgboost_trade_min_probability)
                active_xgboost_signal_direction = 1.0

            # 等权投票基准也可单独做方向校准，便于与 XGBoost 策略公平对照。
            if config.benchmark_vote_auto_calibrate_direction and active_selected_factors:
                train_vote_score = signal_features.loc[train_frame.index, active_selected_factors].mean(axis=1)
                train_vote_signal = vote_score_to_trade_signal(train_vote_score, config)
                train_vote_signal = train_vote_signal.where(train_trade_allowed, 0.0)
                active_vote_signal_direction = choose_signal_direction_on_training(
                    train_vote_signal,
                    next_bar_return,
                )
            else:
                active_vote_signal_direction = 1.0

            model_importance = get_model_feature_importance(model_name, model, active_feature_columns)
            importance_sum = importance_sum.add(pd.Series(model_importance), fill_value=0.0)
            model_count += 1
        elif use_walk_forward_selection and not active_feature_columns:
            continue

        # 当前时点只做一次预测，预测结果会在回测里 shift 成下一根K线实际持仓。
        current_features = features.iloc[[position]]
        class_probability = predict_composite_probability(
            model_name,
            model,
            current_features[active_feature_columns],
            active_feature_columns,
        )[0]
        probabilities.iloc[position] = class_probability
        predicted_class = int(np.argmax(class_probability))
        predicted_direction.iloc[position] = CLASS_TO_TARGET[predicted_class]
        xgboost_signal_direction.iloc[position] = active_xgboost_signal_direction
        calibrated_min_edge.iloc[position] = active_min_edge
        calibrated_min_probability.iloc[position] = active_min_probability

        if active_selected_factors:
            vote_score.iloc[position] = (
                active_vote_signal_direction
                * signal_features.iloc[position][active_selected_factors].mean()
            )
            vote_signal_direction.iloc[position] = active_vote_signal_direction

    if predicted_direction.reindex(predict_index).dropna().empty:
        raise ValueError(
            f"{model_name} 滚动窗口没有生成有效预测，请调小 xgboost_min_train_samples "
            "或 xgboost_train_window。"
        )

    if model_count > 0 and importance_sum.sum() > 0:
        feature_importance = importance_sum / importance_sum.sum()
    else:
        fallback_columns = active_feature_columns or feature_columns
        feature_importance = pd.Series(1.0 / len(fallback_columns), index=fallback_columns)

    signal = pd.DataFrame(index=factors.index)
    signal["xgboost_predicted_direction"] = predicted_direction
    signal["target_direction"] = target
    signal["future_horizon_return"] = calculate_future_horizon_return(data, factors.index, config)
    signal = signal.join(probabilities)
    signal["prob_edge"] = signal["prob_up"] - signal["prob_down"]
    signal["directional_probability"] = signal[["prob_up", "prob_down"]].max(axis=1)
    signal["calibrated_min_edge"] = calibrated_min_edge
    signal["calibrated_min_probability"] = calibrated_min_probability
    signal["xgboost_signal_direction"] = xgboost_signal_direction
    signal["calibrated_prob_edge"] = signal["prob_edge"] * signal["xgboost_signal_direction"]
    signal["composite_score"] = signal["calibrated_prob_edge"]
    signal = signal.join(market_state)
    confidence_filter = build_confidence_rank_filter(signal["calibrated_prob_edge"], config)
    signal = signal.join(confidence_filter)
    signal["trade_allowed"] = (
        signal["trade_allowed"].fillna(0.0) > 0
    ) & (signal["confidence_trade_allowed"].fillna(0.0) > 0)
    signal["trade_allowed"] = signal["trade_allowed"].astype("float64")
    signal["position_size"] = build_dynamic_confidence_position_size(
        probabilities,
        signal["calibrated_min_edge"],
        signal["calibrated_min_probability"],
        config,
    ).fillna(0.0)
    edge_threshold = signal["calibrated_min_edge"].fillna(float(config.xgboost_trade_min_edge))
    probability_threshold = signal["calibrated_min_probability"].fillna(
        float(config.xgboost_trade_min_probability)
    )
    raw_signal = pd.Series(0.0, index=signal.index, dtype="float64")
    raw_signal[
        (signal["prob_edge"] >= edge_threshold)
        & (signal["directional_probability"] >= probability_threshold)
    ] = 1.0
    raw_signal[
        (signal["prob_edge"] <= -edge_threshold)
        & (signal["directional_probability"] >= probability_threshold)
    ] = -1.0
    signal["raw_signal"] = (raw_signal * signal["xgboost_signal_direction"]).fillna(0.0)
    signal["raw_signal"] = signal["raw_signal"].where(signal["trade_allowed"].fillna(0.0) > 0, 0.0)
    signal["calibrated_predicted_direction"] = (
        signal["xgboost_predicted_direction"] * signal["xgboost_signal_direction"]
    )
    signal["raw_signal_before_position_rules"] = signal["raw_signal"]
    signal["target_position_before_rules"] = signal["raw_signal"] * signal["position_size"]
    signal["target_position"] = apply_position_rules(
        signal["target_position_before_rules"],
        config,
    )
    signal["raw_signal"] = np.sign(signal["target_position"]).astype("float64")
    signal["position"] = signal["target_position"].shift(1).fillna(0.0)
    signal["benchmark_vote_score"] = vote_score
    signal["benchmark_vote_signal_direction"] = vote_signal_direction
    signal["benchmark_vote_raw_signal"] = vote_score_to_trade_signal(vote_score, config).fillna(0.0)
    signal["benchmark_vote_raw_signal"] = signal["benchmark_vote_raw_signal"].where(
        signal["trade_allowed"].fillna(0.0) > 0,
        0.0,
    )
    signal["benchmark_vote_raw_signal_before_position_rules"] = signal["benchmark_vote_raw_signal"]
    signal["benchmark_vote_target_position_before_rules"] = (
        signal["benchmark_vote_raw_signal"] * signal["position_size"]
    )
    signal["benchmark_vote_target_position"] = apply_position_rules(
        signal["benchmark_vote_target_position_before_rules"],
        config,
    )
    signal["benchmark_vote_raw_signal"] = np.sign(signal["benchmark_vote_target_position"]).astype("float64")
    signal["benchmark_vote_position"] = signal["benchmark_vote_target_position"].shift(1).fillna(0.0)

    extra_columns = []
    for factor_name in selected_factors:
        if factor_name in active_selected_factors:
            extra_columns.append(factors[[factor_name]])
            extra_columns.append(signal_features[[factor_name]].rename(columns={factor_name: f"{factor_name}_signal"}))
    if not use_walk_forward_selection:
        feature_extra_columns = []
        for feature_name in feature_columns:
            if feature_name not in signal.columns:
                feature_extra_columns.append(feature_name)
        if feature_extra_columns:
            extra_columns.append(
                features[feature_extra_columns].rename(
                    columns={feature_name: f"feature_{feature_name}" for feature_name in feature_extra_columns}
                )
            )
    if extra_columns:
        signal = pd.concat([signal, *extra_columns], axis=1).copy()

    selection_summary = pd.DataFrame(selection_rows) if selection_rows else None
    return signal, feature_importance, features, selection_summary


def save_prediction_diagnostics(backtest_df: pd.DataFrame, output_dir: Path) -> None:
    """保存独立于策略盈亏的模型预测质量诊断。"""
    required = {
        "target_direction",
        "future_horizon_return",
        "xgboost_predicted_direction",
        "calibrated_predicted_direction",
        "raw_signal",
        "calibrated_prob_edge",
    }
    if not required.issubset(backtest_df.columns):
        return

    optional_columns = [
        "trade_allowed",
        "confidence_trade_allowed",
        "confidence_rank",
        "calibrated_min_edge",
        "calibrated_min_probability",
    ]
    diagnostic_columns = list(required) + [col for col in optional_columns if col in backtest_df.columns]
    valid = backtest_df[diagnostic_columns].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_direction", "future_horizon_return", "xgboost_predicted_direction"]
    )
    if valid.empty:
        return

    raw_pred = valid["xgboost_predicted_direction"]
    calibrated_pred = valid["calibrated_predicted_direction"]
    target = valid["target_direction"]
    traded = valid["raw_signal"] != 0
    directional_target = target != 0

    diagnostics = pd.Series(
        {
            "样本数": int(len(valid)),
            "目标上涨占比": float((target > 0).mean()),
            "目标下跌占比": float((target < 0).mean()),
            "目标中性占比": float((target == 0).mean()),
            "原始三分类准确率": float((raw_pred == target).mean()),
            "校准后三分类准确率": float((calibrated_pred == target).mean()),
            "非中性目标方向准确率": float(
                (np.sign(calibrated_pred[directional_target]) == np.sign(target[directional_target])).mean()
            )
            if directional_target.any()
            else np.nan,
            "交易信号样本数": int(traded.sum()),
            "交易信号方向准确率": float(
                (np.sign(valid.loc[traded, "raw_signal"]) == np.sign(target[traded])).mean()
            )
            if traded.any()
            else np.nan,
            "可交易样本数": int((valid["trade_allowed"].fillna(0.0) > 0).sum())
            if "trade_allowed" in valid.columns
            else np.nan,
            "可交易样本方向准确率": float(
                (
                    np.sign(
                        valid.loc[valid["trade_allowed"].fillna(0.0) > 0, "calibrated_predicted_direction"]
                    )
                    == np.sign(target[valid["trade_allowed"].fillna(0.0) > 0])
                ).mean()
            )
            if "trade_allowed" in valid.columns and (valid["trade_allowed"].fillna(0.0) > 0).any()
            else np.nan,
            "高置信度样本数": int((valid["confidence_trade_allowed"].fillna(0.0) > 0).sum())
            if "confidence_trade_allowed" in valid.columns
            else np.nan,
            "概率差与未来收益相关性": float(
                valid["calibrated_prob_edge"].corr(valid["future_horizon_return"])
            ),
            "平均未来Horizon收益": float(valid["future_horizon_return"].mean()),
            "交易样本平均未来Horizon收益": float(valid.loc[traded, "future_horizon_return"].mean())
            if traded.any()
            else np.nan,
        },
        name="value",
    )
    diagnostics.to_csv(output_dir / "composite_prediction_diagnostics.csv", encoding="utf-8-sig")

    confusion = pd.crosstab(
        target,
        calibrated_pred,
        rownames=["真实方向"],
        colnames=["校准预测方向"],
        dropna=False,
    )
    confusion.to_csv(output_dir / "composite_prediction_confusion_matrix.csv", encoding="utf-8-sig")

    bins = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
    labels = [
        "<=-0.20",
        "-0.20~-0.10",
        "-0.10~-0.05",
        "-0.05~0",
        "0~0.05",
        "0.05~0.10",
        "0.10~0.20",
        ">0.20",
    ]
    valid = valid.copy()
    valid["概率差分箱"] = pd.cut(valid["calibrated_prob_edge"], bins=bins, labels=labels)
    edge_rows = []
    for bucket, bucket_df in valid.groupby("概率差分箱", observed=False):
        if bucket_df.empty:
            continue
        edge_rows.append(
            {
                "概率差分箱": bucket,
                "样本数": int(len(bucket_df)),
                "平均概率差": bucket_df["calibrated_prob_edge"].mean(),
                "平均未来Horizon收益": bucket_df["future_horizon_return"].mean(),
                "未来上涨占比": (bucket_df["future_horizon_return"] > 0).mean(),
                "未来下跌占比": (bucket_df["future_horizon_return"] < 0).mean(),
                "平均真实方向": bucket_df["target_direction"].mean(),
                "交易信号占比": (bucket_df["raw_signal"] != 0).mean(),
            }
        )
    pd.DataFrame(edge_rows).to_csv(
        output_dir / "composite_xgboost_edge_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_train_test_backtest_result(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    title: str,
    score_label: str,
    file_name: str,
) -> Path:
    """并排绘制训练集、验证集和最终测试集回测面板。"""
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(30, 16),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.0]},
    )
    fig.suptitle(title, fontsize=16)

    panels = [
        ("训练集滚动预测", train_df),
        ("验证集滚动预测", validation_df),
        ("最终测试集滚动预测", test_df),
    ]
    for col, (panel_name, backtest_df) in enumerate(panels):
        axes[0, col].plot(backtest_df.index, backtest_df["nav"], label="策略净值", linewidth=1.6)
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
        axes[2, col].legend(loc="upper left")
        axes[2, col].grid(alpha=0.3)

        if "future_horizon_return" in backtest_df.columns:
            axes[3, col].scatter(
                backtest_df["calibrated_prob_edge"],
                backtest_df["future_horizon_return"],
                s=8,
                alpha=0.35,
            )
            axes[3, col].axhline(0, color="black", linewidth=0.8, alpha=0.7)
            axes[3, col].axvline(0, color="black", linewidth=0.8, alpha=0.7)
            axes[3, col].set_xlabel("校准概率差")
            axes[3, col].set_ylabel("未来Horizon收益")
            axes[3, col].grid(alpha=0.3)
        else:
            axes[3, col].axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plot_path = output_dir / file_name
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def calculate_backtest_segment_metrics(
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float]:
    """基于已有回测明细计算某个分段的绩效指标。"""
    annual_periods = infer_annual_periods(
        pd.DatetimeIndex(segment_df.index),
        config.annual_trading_days,
    )
    return calculate_metrics(
        segment_df["strategy_net_return"],
        segment_df["benchmark_return"],
        segment_df["position"],
        annual_periods,
    )


def append_segment_metric_row(
    rows: list[dict[str, Any]],
    segment_type: str,
    segment_name: str,
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """把一个分段的绩效指标追加到稳健性报告行列表。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if segment_df.empty or not required_columns.issubset(segment_df.columns):
        return
    try:
        metrics = calculate_backtest_segment_metrics(segment_df, config)
    except Exception:
        return
    rows.append(
        {
            "分段类型": segment_type,
            "分段": segment_name,
            "开始时间": segment_df.index.min(),
            "结束时间": segment_df.index.max(),
            **metrics,
        }
    )


def get_supported_pandas_frequency(candidates: list[str]) -> str:
    """在不同 pandas 版本之间选择可用的频率别名。"""
    for freq in candidates:
        try:
            pd.tseries.frequencies.to_offset(freq)
            return freq
        except ValueError:
            continue
    raise ValueError(f"当前 pandas 不支持这些频率别名: {candidates}")


def save_composite_robustness_report(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """保存综合策略分段稳健性报告，帮助识别收益是否过度集中。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if backtest_df.empty or not required_columns.issubset(backtest_df.columns):
        return

    rows: list[dict[str, Any]] = []
    cleaned = backtest_df.replace([np.inf, -np.inf], np.nan).copy()
    append_segment_metric_row(rows, "整体", "最终测试集", cleaned.dropna(subset=["strategy_net_return"]), config)

    month_freq = get_supported_pandas_frequency(["ME", "M"])
    quarter_freq = get_supported_pandas_frequency(["QE", "Q"])
    for period_name, freq in [("月度", month_freq), ("季度", quarter_freq)]:
        for period, segment_df in cleaned.groupby(pd.Grouper(freq=freq)):
            append_segment_metric_row(
                rows,
                period_name,
                str(period.date()) if pd.notna(period) else "未知",
                segment_df.dropna(subset=["strategy_net_return"]),
                config,
            )

    benchmark_abs_return = cleaned["benchmark_return"].abs().replace([np.inf, -np.inf], np.nan)
    if benchmark_abs_return.notna().sum() >= 20:
        vol_rank = benchmark_abs_return.rolling(120, min_periods=20).rank(pct=True)
        regime_map = {
            "低波动": vol_rank <= 0.33,
            "中波动": (vol_rank > 0.33) & (vol_rank <= 0.66),
            "高波动": vol_rank > 0.66,
        }
        for regime_name, mask in regime_map.items():
            append_segment_metric_row(
                rows,
                "波动状态",
                regime_name,
                cleaned.loc[mask.fillna(False)].dropna(subset=["strategy_net_return"]),
                config,
            )

    if "calibrated_prob_edge" in cleaned.columns:
        confidence = cleaned["calibrated_prob_edge"].abs().replace([np.inf, -np.inf], np.nan)
        if confidence.notna().sum() >= 20:
            rank = confidence.rolling(120, min_periods=20).rank(pct=True)
            regime_map = {
                "低置信度": rank <= 0.33,
                "中置信度": (rank > 0.33) & (rank <= 0.66),
                "高置信度": rank > 0.66,
            }
            for regime_name, mask in regime_map.items():
                append_segment_metric_row(
                    rows,
                    "模型置信度",
                    regime_name,
                    cleaned.loc[mask.fillna(False)].dropna(subset=["strategy_net_return"]),
                    config,
                )

    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "composite_robustness_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_cost_stress_report(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """基于同一组持仓重算不同交易成本下的策略表现。"""
    required_columns = {"strategy_gross_return", "turnover", "benchmark_return", "position"}
    if backtest_df.empty or not required_columns.issubset(backtest_df.columns):
        return

    annual_periods = infer_annual_periods(
        pd.DatetimeIndex(backtest_df.index),
        config.annual_trading_days,
    )
    rows = []
    for cost_bps in getattr(config, "cost_stress_bps_list", []) or []:
        cost_bps = float(cost_bps)
        net_return = backtest_df["strategy_gross_return"].fillna(0.0) - (
            backtest_df["turnover"].fillna(0.0) * cost_bps / 10000.0
        )
        metrics = calculate_metrics(
            net_return,
            backtest_df["benchmark_return"],
            backtest_df["position"],
            annual_periods,
        )
        rows.append(
            {
                "单边总成本bps": cost_bps,
                "平均单根成本": float((backtest_df["turnover"].fillna(0.0) * cost_bps / 10000.0).mean()),
                **metrics,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "composite_cost_stress_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_composite_outputs(
    backtest_df: pd.DataFrame,
    metrics: dict[str, float],
    selected_factors: list[str],
    feature_importance: pd.Series,
    output_dir: Path,
    factor_id_map: dict[str, int],
    factor_label_map: dict[str, str],
    split_time: pd.Timestamp,
    validation_end_time: pd.Timestamp,
    config: BacktestConfig,
    selection_summary: pd.DataFrame | None = None,
) -> None:
    """保存综合因子回测的全部输出文件。

    输出包括：
    - composite_detail.csv：逐K线信号、持仓、收益、净值明细。
    - composite_summary.csv：模型配置和绩效摘要。
    - composite_xgboost_feature_importance.csv：特征重要性。
    - composite_xgboost_feature_selection.csv：滚动选因明细。
    - composite_xgboost_edge_diagnostics.csv：概率优势分桶诊断。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output_dir / "composite_detail.csv", encoding="utf-8-sig")

    summary = pd.Series(metrics, name="value")
    summary.loc["模型"] = "xgboost_rolling_multiclass"
    summary.loc["训练验证切分时间"] = str(split_time)
    summary.loc["验证测试切分时间"] = str(validation_end_time)
    summary.loc["预测目标"] = "future_horizon_return_direction_with_cost_neutral(-1,0,1)"
    summary.loc["预测目标跨度K线数"] = get_xgboost_target_horizon(config)
    summary.loc["训练标签隔离K线数"] = max(0, get_xgboost_target_horizon(config) - 1)
    summary.loc["目标中性区间bps"] = (
        config.xgboost_target_neutral_bps
        if config.xgboost_target_neutral_bps is not None
        else config.commission_bps + config.slippage_bps
    )
    summary.loc["手续费bps"] = config.commission_bps
    summary.loc["滑点bps"] = config.slippage_bps
    summary.loc["候选池来源"] = (
        getattr(config, "frozen_active_library_path", None)
        if getattr(config, "use_frozen_active_library", False)
        else "active_factors.csv"
    )
    summary.loc["使用冻结因子库"] = bool(getattr(config, "use_frozen_active_library", False))
    summary.loc["特征范围"] = config.xgboost_feature_scope
    summary.loc["特征模式"] = config.xgboost_feature_mode
    summary.loc["使用因子状态特征"] = config.xgboost_include_factor_state_features
    summary.loc["滚动选因"] = config.xgboost_walk_forward_feature_selection
    summary.loc["候选因子数量"] = len(selected_factors)
    summary.loc["树最小子节点权重"] = config.xgboost_min_child_weight
    summary.loc["分裂最小损失下降"] = config.xgboost_gamma
    summary.loc["L2正则"] = config.xgboost_reg_lambda
    summary.loc["L1正则"] = config.xgboost_reg_alpha
    summary.loc["启用验证集EarlyStopping"] = config.xgboost_use_validation_early_stopping
    summary.loc["验证集比例"] = config.xgboost_validation_ratio
    summary.loc["EarlyStopping轮数"] = config.xgboost_early_stopping_rounds
    summary.loc["交易最小概率差"] = config.xgboost_trade_min_edge
    summary.loc["交易最小方向概率"] = config.xgboost_trade_min_probability
    summary.loc["自动校准交易阈值"] = config.xgboost_auto_calibrate_trade_thresholds
    summary.loc["阈值校准最少交易数"] = config.xgboost_threshold_min_trades
    summary.loc["启用市场状态过滤"] = config.xgboost_trade_use_market_filters
    summary.loc["过滤窗口"] = config.xgboost_trade_filter_window
    summary.loc["最小波动率分位"] = config.xgboost_trade_min_volatility_rank
    summary.loc["最小流动性分位"] = config.xgboost_trade_min_liquidity_rank
    summary.loc["动态仓位"] = config.xgboost_use_dynamic_position_sizing
    summary.loc["最小动态仓位"] = config.xgboost_position_size_min
    summary.loc["仓位幂次"] = config.xgboost_position_size_power
    summary.loc["最大仓位"] = config.xgboost_position_size_max
    summary.loc["启用持仓规则优化"] = bool(getattr(config, "xgboost_use_position_rules", False))
    summary.loc["最小持仓K线数"] = int(getattr(config, "xgboost_min_holding_bars", 0) or 0)
    summary.loc["反转冷却K线数"] = int(getattr(config, "xgboost_reentry_cooldown_bars", 0) or 0)
    summary.loc["最小仓位变化阈值"] = float(getattr(config, "xgboost_min_position_change", 0.0) or 0.0)
    summary.loc["仓位平滑系数"] = float(getattr(config, "xgboost_position_smoothing_alpha", 1.0) or 1.0)
    summary.loc["启用置信度分位过滤"] = config.xgboost_trade_use_confidence_rank_filter
    summary.loc["置信度分位窗口"] = config.xgboost_trade_confidence_rank_window
    summary.loc["最小置信度分位"] = config.xgboost_trade_min_confidence_rank
    summary.loc["训练集启用市场过滤"] = config.xgboost_train_use_market_filters
    summary.loc["训练最少方向样本数"] = config.xgboost_train_min_directional_samples
    summary.loc["训练非中性类别权重"] = config.xgboost_train_nonzero_class_weight
    summary.loc["训练中性类别权重"] = config.xgboost_train_neutral_class_weight
    summary.loc["XGBoost自动校准方向"] = config.xgboost_auto_calibrate_signal_direction
    summary.loc["等权投票自动校准方向"] = config.benchmark_vote_auto_calibrate_direction
    if "raw_signal" in backtest_df.columns:
        raw_signal = backtest_df["raw_signal"].fillna(0.0)
        summary.loc["信号覆盖率"] = float((raw_signal != 0).mean())
        summary.loc["做多信号数"] = int((raw_signal > 0).sum())
        summary.loc["做空信号数"] = int((raw_signal < 0).sum())
        summary.loc["空仓信号数"] = int((raw_signal == 0).sum())
    if "raw_signal_before_position_rules" in backtest_df.columns:
        raw_signal_before_rules = backtest_df["raw_signal_before_position_rules"].fillna(0.0)
        summary.loc["规则前信号覆盖率"] = float((raw_signal_before_rules != 0).mean())
    if "calibrated_prob_edge" in backtest_df.columns:
        summary.loc["平均校准概率差"] = float(backtest_df["calibrated_prob_edge"].mean())
        summary.loc["平均绝对校准概率差"] = float(backtest_df["calibrated_prob_edge"].abs().mean())
    if "trade_allowed" in backtest_df.columns:
        summary.loc["过滤后可交易覆盖率"] = float(backtest_df["trade_allowed"].fillna(0.0).mean())
    if "position_size" in backtest_df.columns:
        summary.loc["平均目标仓位大小"] = float(backtest_df["position_size"].fillna(0.0).mean())
    if {"target_position", "target_position_before_rules"}.issubset(backtest_df.columns):
        target_position = backtest_df["target_position"].fillna(0.0)
        target_before_rules = backtest_df["target_position_before_rules"].fillna(0.0)
        summary.loc["规则后目标仓位覆盖率"] = float((target_position != 0).mean())
        summary.loc["规则前目标仓位覆盖率"] = float((target_before_rules != 0).mean())
        summary.loc["规则后目标仓位平均绝对值"] = float(target_position.abs().mean())
        summary.loc["规则前目标仓位平均绝对值"] = float(target_before_rules.abs().mean())
        summary.loc["规则后目标仓位平均变化"] = float(target_position.diff().abs().fillna(target_position.abs()).mean())
        summary.loc["规则前目标仓位平均变化"] = float(
            target_before_rules.diff().abs().fillna(target_before_rules.abs()).mean()
        )
    if "position" in backtest_df.columns:
        position = backtest_df["position"].fillna(0.0)
        summary.loc["实际持仓覆盖率"] = float((position != 0).mean())
        summary.loc["平均实际仓位绝对值"] = float(position.abs().mean())
        summary.loc["非零持仓平均绝对仓位"] = (
            float(position[position != 0].abs().mean()) if (position != 0).any() else 0.0
        )
    if "confidence_rank" in backtest_df.columns:
        summary.loc["平均置信度分位"] = float(backtest_df["confidence_rank"].fillna(0.0).mean())
    if "calibrated_min_edge" in backtest_df.columns:
        summary.loc["平均校准概率差阈值"] = float(backtest_df["calibrated_min_edge"].dropna().mean())
    if "calibrated_min_probability" in backtest_df.columns:
        summary.loc["平均校准方向概率阈值"] = float(
            backtest_df["calibrated_min_probability"].dropna().mean()
        )
    if not should_use_walk_forward_selection(config):
        summary.loc["选中因子编号"] = ",".join(str(factor_id_map[factor]) for factor in selected_factors)
        summary.loc["选中因子标签"] = ",".join(factor_label_map[factor] for factor in selected_factors)
        summary.loc["选中因子"] = ",".join(selected_factors)
    summary.to_csv(output_dir / "composite_summary.csv", encoding="utf-8-sig")

    importance_df = feature_importance.rename("importance").reset_index()
    importance_df.columns = ["特征", "importance"]
    importance_df["因子"] = importance_df["特征"].map(
        lambda feature_name: get_feature_base_factor(feature_name, selected_factors)
    )
    importance_df.insert(0, "因子编号", importance_df["因子"].map(factor_id_map))
    importance_df.insert(1, "因子标签", importance_df["因子"].map(factor_label_map))
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(
        output_dir / "composite_xgboost_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    factor_contribution = (
        importance_df.groupby(["因子编号", "因子标签", "因子"], dropna=False)["importance"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "重要性合计",
                "mean": "平均特征重要性",
                "count": "特征数量",
            }
        )
        .sort_values("重要性合计", ascending=False)
    )
    total_importance = factor_contribution["重要性合计"].sum()
    factor_contribution["重要性占比"] = (
        factor_contribution["重要性合计"] / total_importance
        if total_importance > 0
        else np.nan
    )
    factor_contribution.to_csv(
        output_dir / "composite_factor_contribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if selection_summary is not None:
        selection_summary.to_csv(
            output_dir / "composite_xgboost_feature_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_prediction_diagnostics(backtest_df, output_dir)
    save_composite_robustness_report(backtest_df, output_dir, config)
    save_cost_stress_report(backtest_df, output_dir, config)


def run_composite_backtest(config: BacktestConfig) -> dict[str, float]:
    """运行一次指定配置下的综合因子回测，并返回测试集绩效指标。"""
    output_dir = Path(config.output_dir) / "composite_factor"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = get_experiment_run_dir(config, "composite")
    write_run_config(config, output_dir)
    if run_dir is not None:
        write_run_config(config, run_dir)

    try:
        print(f"读取 {config.symbol} 的 {config.bar_size} 分钟数据...")
        data = fetch_intraday_data(config)

        print("构建因子...")
        requested_factors = None
        if bool(getattr(config, "composite_build_active_only", True)):
            requested_factors = load_active_factor_names(config)
            print(f"综合回测按 active 因子按需构建: {len(requested_factors)} 个候选因子")
        cache_path = output_dir / "active_factor_matrix_cache.pkl"
        factors = None
        if requested_factors is not None and bool(getattr(config, "composite_use_factor_cache", True)):
            factors = load_factor_matrix_cache(cache_path, data, requested_factors, config)
            if factors is not None:
                print(f"已读取 active 因子矩阵缓存: {cache_path}")
        if factors is None:
            factors = build_factors(data, config, requested_factors=requested_factors)
            if requested_factors is not None and bool(getattr(config, "composite_use_factor_cache", True)):
                save_factor_matrix_cache(factors, cache_path, data, requested_factors, config)
                print(f"已保存 active 因子矩阵缓存: {cache_path}")
        if run_dir is not None:
            write_factor_count_snapshot(factors, run_dir)
            snapshot_active_factor_library(config, run_dir)
            coverage = get_last_related_data_coverage()
            if not coverage.empty:
                coverage.to_csv(
                    output_dir / "related_data_coverage.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                coverage.to_csv(
                    run_dir / "related_data_coverage.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        split_time, validation_end_time = split_train_validation_test_index(
            data.index,
            config.auto_select_train_ratio,
            getattr(config, "auto_select_validation_ratio", 0.0),
        )
        selected_factors, selection_summary = get_selected_factors(
            data,
            factors,
            config,
            split_time,
        )
        factor_id_map = get_factor_id_map(factors)
        factor_label_map = get_factor_label_map(factors)
        train_data = data.loc[data.index < split_time]
        train_factors = factors.loc[factors.index < split_time]
        validation_data = data.loc[(data.index >= split_time) & (data.index < validation_end_time)]
        validation_factors = factors.loc[
            (factors.index >= split_time) & (factors.index < validation_end_time)
        ]
        backtest_data = data.loc[data.index >= validation_end_time]
        backtest_factors = factors.loc[factors.index >= validation_end_time]

        if should_use_walk_forward_selection(config):
            print(
                "XGBoost候选因子池: active_factors.csv；可用active因子数量: "
                f"{len(selected_factors)}，每次重训滚动选取 Top {config.xgboost_best_top_n}"
            )
        else:
            print(
                "XGBoost特征因子(active池内): "
                + ", ".join(factor_label_map[factor] for factor in selected_factors)
            )
        print(
            "滚动训练参数: "
            f"window={config.xgboost_train_window}, "
            f"min_samples={config.xgboost_min_train_samples}, "
            f"retrain_every={config.xgboost_retrain_every}"
        )

        predict_start = min(
            max(1, int(config.xgboost_min_train_samples)),
            max(0, len(factors.index) - 1),
        )
        predict_index = factors.index[predict_start:]
        model_names = get_enabled_composite_models(config)
        print("综合模型对比: " + ", ".join(model_names))
        comparison_rows: list[dict[str, Any]] = []
        generated_paths: list[Path] = []
        primary_metrics: dict[str, float] | None = None
        benchmark_vote_metrics: dict[str, float] | None = None

        for model_name in model_names:
            try:
                signal, feature_importance, _, rolling_selection_summary = build_xgboost_rolling_signal(
                    data,
                    factors,
                    selected_factors,
                    config,
                    predict_index,
                    model_name=model_name,
                )
            except ImportError as exc:
                print(f"\n跳过模型 {model_name}: {exc}")
                comparison_rows.append({"模型": model_name, "错误": str(exc)})
                continue
            train_signal = signal.loc[train_factors.index]
            validation_signal = signal.loc[validation_factors.index]
            backtest_signal = signal.loc[backtest_factors.index]
            train_df, train_metrics = run_backtest(train_data, train_signal, config)
            validation_df, validation_metrics = run_backtest(
                validation_data,
                validation_signal,
                config,
            )
            backtest_df, metrics = run_backtest(backtest_data, backtest_signal, config)
            comparison_rows.extend(
                [
                    {"模型": f"{model_name}_train", **train_metrics},
                    {"模型": f"{model_name}_validation", **validation_metrics},
                    {"模型": model_name, **metrics},
                ]
            )

            report_filename = (
                "composite_report.png"
                if model_name == "xgboost"
                else f"composite_report_{model_name}.png"
            )
            plot_path = plot_train_test_backtest_result(
                train_df,
                validation_df,
                backtest_df,
                output_dir,
                f"{config.symbol} {model_name}多因子滚动窗口回测",
                "预测方向(-1/0/1)",
                report_filename,
            )
            generated_paths.append(plot_path)

            should_save_primary_outputs = model_name == "xgboost" or primary_metrics is None
            if should_save_primary_outputs:
                save_composite_outputs(
                    backtest_df,
                    metrics,
                    selected_factors,
                    feature_importance,
                    output_dir,
                    factor_id_map,
                    factor_label_map,
                    split_time,
                    validation_end_time,
                    config,
                    selection_summary=(
                        rolling_selection_summary
                        if rolling_selection_summary is not None
                        else selection_summary
                    ),
                )
                primary_metrics = metrics

            train_benchmark_vote_signal = build_backtest_signal_from_columns(
                train_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            validation_benchmark_vote_signal = build_backtest_signal_from_columns(
                validation_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            benchmark_vote_signal = build_backtest_signal_from_columns(
                backtest_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            train_benchmark_vote_df, train_benchmark_vote_metrics = run_backtest(
                train_data,
                train_benchmark_vote_signal,
                config,
            )
            validation_benchmark_vote_df, validation_benchmark_vote_metrics = run_backtest(
                validation_data,
                validation_benchmark_vote_signal,
                config,
            )
            benchmark_vote_df, current_benchmark_vote_metrics = run_backtest(
                backtest_data,
                benchmark_vote_signal,
                config,
            )
            if benchmark_vote_metrics is None:
                benchmark_vote_metrics = current_benchmark_vote_metrics
                benchmark_plot_path = plot_train_test_backtest_result(
                    train_benchmark_vote_df,
                    validation_benchmark_vote_df,
                    benchmark_vote_df,
                    output_dir,
                    f"{config.symbol} 等权投票基准滚动窗口回测",
                    "等权投票分数",
                    "benchmark_vote_report.png",
                )
                generated_paths.append(benchmark_plot_path)
                comparison_rows.extend(
                    [
                        {"模型": "equal_vote_benchmark_train", **train_benchmark_vote_metrics},
                        {
                            "模型": "equal_vote_benchmark_validation",
                            **validation_benchmark_vote_metrics,
                        },
                        {"模型": "equal_vote_benchmark", **current_benchmark_vote_metrics},
                    ]
                )

        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "composite_model_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if run_dir is not None:
            copy_existing_files(
                [
                    output_dir / "composite_detail.csv",
                    output_dir / "composite_summary.csv",
                    output_dir / "composite_model_comparison.csv",
                    output_dir / "composite_xgboost_feature_importance.csv",
                    output_dir / "composite_factor_contribution.csv",
                    output_dir / "composite_xgboost_feature_selection.csv",
                    output_dir / "composite_xgboost_edge_diagnostics.csv",
                    output_dir / "composite_robustness_report.csv",
                    output_dir / "composite_cost_stress_report.csv",
                    output_dir / "composite_prediction_diagnostics.csv",
                    output_dir / "composite_prediction_confusion_matrix.csv",
                    output_dir / "related_data_coverage.csv",
                    *generated_paths,
                ],
                run_dir,
            )

        metrics = primary_metrics or {}
        print("\n========== 多模型综合因子滚动窗口回测 ==========")
        print_metrics(metrics)
        print("\n========== 等权投票基准回测 ==========")
        print_metrics(benchmark_vote_metrics or {})
        print("\n输出文件:")
        print(output_dir / "composite_detail.csv")
        print(output_dir / "composite_summary.csv")
        print(output_dir / "composite_model_comparison.csv")
        print(output_dir / "composite_xgboost_feature_importance.csv")
        print(output_dir / "composite_factor_contribution.csv")
        print(output_dir / "composite_robustness_report.csv")
        print(output_dir / "composite_cost_stress_report.csv")
        for path in generated_paths:
            print(path)
        return metrics
    finally:
        # 资源释放由脚本入口或批量入口统一负责，这里只保持异常传播和语法结构完整。
        pass


def main() -> None:
    """综合因子回测脚本入口。"""
    config = BacktestConfig()

    def action() -> dict[str, float]:
        try:
            return run_composite_backtest(config)
        finally:
            stop_wind()

    run_tracked(config, "composite", action)


if __name__ == "__main__":
    main()

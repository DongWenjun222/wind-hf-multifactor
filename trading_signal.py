from __future__ import annotations

"""
最新交易信号导出工具。

默认 compute 模式复用综合回测的滚动模型、特征、校准和交易规则现场生成最新信号；
vote 模式才会根据 active 因子库表现做加权投票；
detail 模式才会读取已经生成的 composite_detail.csv 最后一行做快速对照。
输出既保留模型原始目标，也给出最终执行建议、调仓动作、是否建议交易和原因。
"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from composite_factor_backtest import (
    build_xgboost_rolling_signal,
    get_enabled_composite_models,
    get_selected_factors,
)
from config import BacktestConfig, resolve_symbol_universe
from framework.factor_library import conservative_pair, get_factor_library_dir
from framework.factors import build_factors, fetch_intraday_data, safe_symbol_name, score_to_raw_signal, stop_wind
from framework.runtime_utils import configure_warning_output
from single_factor_backtest import split_train_validation_test_index


SIGNAL_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amt",
    "amount",
    "prob_down",
    "prob_flat",
    "prob_up",
    "prob_edge",
    "directional_probability",
    "calibrated_prob_edge",
    "calibrated_min_edge",
    "calibrated_min_probability",
    "trade_allowed",
    "confidence_rank",
    "confidence_trade_allowed",
    "position_size",
    "raw_signal",
    "raw_signal_before_position_rules",
    "target_position_before_rules",
    "target_position",
    "position",
    "benchmark_vote_score",
    "benchmark_vote_raw_signal",
    "benchmark_vote_target_position",
    "benchmark_vote_position",
    "volatility_rank",
    "liquidity_rank",
    "trend_strength",
    "rolling_regime_return",
    "volatility_regime",
    "liquidity_regime",
    "trend_regime",
    "session_regime",
]


def normalize_signal_symbols(symbols: str | list[str] | None, config: BacktestConfig) -> list[str]:
    """解析交易信号需要导出的品种列表。"""
    if symbols:
        return resolve_symbol_universe(symbols)
    configured_symbols = getattr(config, "symbols", None)
    if configured_symbols:
        return resolve_symbol_universe(configured_symbols)
    return [config.symbol]


def get_symbol_composite_detail_path(config: BacktestConfig, symbol: str, source: str) -> Path:
    """根据来源模式返回某个品种的综合回测明细路径。"""
    source = str(source).lower()
    single_path = Path(config.output_dir) / "composite_factor" / "composite_detail.csv"
    if source == "single":
        return single_path

    symbol_dir = safe_symbol_name(symbol).upper()
    multi_path = (
        Path(config.output_dir)
        / getattr(config, "multi_symbol_output_subdir", "by_symbol")
        / symbol_dir
        / "composite_factor"
        / "composite_detail.csv"
    )
    if source == "multi":
        return multi_path

    config_symbol = str(getattr(config, "symbol", "")).upper()
    current_symbol = str(symbol).upper()
    if multi_path.exists():
        return multi_path
    if current_symbol == config_symbol:
        return single_path
    return multi_path


def get_symbol_output_dir(config: BacktestConfig, symbol: str) -> Path:
    """返回多品种模式下某个品种自己的输出目录。"""
    return (
        Path(config.output_dir)
        / getattr(config, "multi_symbol_output_subdir", "by_symbol")
        / safe_symbol_name(symbol).upper()
    )


def build_symbol_compute_config(config: BacktestConfig, symbol: str, source: str) -> BacktestConfig:
    """为现场计算交易信号创建单品种配置。"""
    source = str(source).lower()
    symbol_config = replace(config, symbol=str(symbol).upper())
    multi_output_dir = get_symbol_output_dir(config, symbol)

    if source == "multi":
        symbol_config.output_dir = str(multi_output_dir)
        return symbol_config

    if source == "single":
        return symbol_config

    if multi_output_dir.exists():
        symbol_config.output_dir = str(multi_output_dir)
        return symbol_config

    if str(symbol).upper() == str(getattr(config, "symbol", "")).upper():
        return symbol_config

    symbol_config.output_dir = str(multi_output_dir)
    return symbol_config


def read_latest_signal_row(detail_path: Path) -> pd.Series:
    """读取 composite_detail.csv 中最新一根有有效信号的记录。"""
    if not detail_path.exists():
        raise FileNotFoundError(f"没有找到综合回测明细: {detail_path}")
    detail = pd.read_csv(detail_path, index_col=0, parse_dates=True)
    if detail.empty:
        raise ValueError(f"综合回测明细为空: {detail_path}")

    signal_like_columns = [
        column
        for column in ["target_position", "raw_signal", "calibrated_prob_edge", "position"]
        if column in detail.columns
    ]
    if signal_like_columns:
        valid = detail.dropna(subset=signal_like_columns, how="all")
        if not valid.empty:
            detail = valid
    return detail.iloc[-1]


def get_active_library_path(config: BacktestConfig) -> Path:
    """返回交易信号使用的 active 因子库路径。"""
    if bool(getattr(config, "use_frozen_active_library", False)):
        frozen_path = getattr(config, "frozen_active_library_path", None)
        if not frozen_path:
            raise ValueError("use_frozen_active_library=True 时必须配置 frozen_active_library_path。")
        active_path = Path(frozen_path)
        if not active_path.is_absolute():
            active_path = Path(config.output_dir) / active_path
        return active_path
    return get_factor_library_dir(config) / "active_factors.csv"


def load_active_library_for_signal(config: BacktestConfig) -> pd.DataFrame:
    """读取 active 因子库，供交易信号现场合成使用。"""
    active_path = get_active_library_path(config)
    if not active_path.exists():
        raise FileNotFoundError(f"没有找到 active 因子库: {active_path}")
    active_library = pd.read_csv(active_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {active_path}")
    active_library = active_library.dropna(subset=["因子"]).copy()
    active_library["因子"] = active_library["因子"].astype(str)
    if active_library.empty:
        raise ValueError(f"active 因子库为空: {active_path}")
    return active_library


def get_factor_weight_series(active_library: pd.DataFrame, available_factors: list[str]) -> pd.Series:
    """根据可追溯的训练/验证表现生成合成权重，最终测试集不参与。"""
    library = active_library.set_index("因子").reindex(available_factors)
    raw_weight = pd.Series(np.nan, index=available_factors, dtype="float64")

    traceable = library.get(
        "入库数据可追溯",
        pd.Series(False, index=available_factors),
    )
    traceable = traceable.map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    for column in ("初筛科研综合评分", "初筛预测能力评分"):
        if column in library.columns:
            score = pd.to_numeric(library[column], errors="coerce").where(traceable)
            raw_weight = raw_weight.fillna(score)

    raw_weight = raw_weight.fillna(
        conservative_pair(library, "训练夏普比率", "验证夏普比率")
    )
    raw_weight = raw_weight.fillna(
        conservative_pair(library, "训练累计收益", "验证累计收益")
    )
    raw_weight = raw_weight.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if raw_weight.sum() <= 0:
        raw_weight = pd.Series(1.0, index=available_factors, dtype="float64")
    return raw_weight / raw_weight.sum()


def get_factor_direction_series(active_library: pd.DataFrame, available_factors: list[str]) -> pd.Series:
    """读取单因子方向，正向为 1，反向为 -1。"""
    library = active_library.set_index("因子").reindex(available_factors)
    direction_text = library.get("方向", pd.Series("", index=available_factors)).fillna("")
    direction = pd.Series(1.0, index=available_factors, dtype="float64")
    direction[direction_text.astype(str).str.contains("反")] = -1.0
    return direction


def build_weighted_factor_signal_frame(
    factors: pd.DataFrame,
    active_library: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """用 active 因子的最新多空信号加权合成交易信号。"""
    available_factors = [factor for factor in active_library["因子"].tolist() if factor in factors.columns]
    if not available_factors:
        raise ValueError("active 因子库中的因子在当前因子矩阵中均不可用。")

    if bool(factors.attrs.get("precomputed_factor_signals", False)):
        factor_signals = factors[available_factors].replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    else:
        factor_signals = pd.DataFrame(
            {
                factor: score_to_raw_signal(
                    factors[factor].replace([np.inf, -np.inf], np.nan),
                    config.signal_threshold,
                )
                for factor in available_factors
            },
            index=factors.index,
        )
    direction = get_factor_direction_series(active_library, available_factors)
    weights = get_factor_weight_series(active_library, available_factors)
    directed_signals = factor_signals.mul(direction, axis=1)
    vote_score = directed_signals.mul(weights, axis=1).sum(axis=1, min_count=1)
    min_abs_score = max(0.0, float(getattr(config, "benchmark_vote_min_abs_score", 0.0) or 0.0))
    raw_signal = pd.Series(0.0, index=factors.index, dtype="float64")
    raw_signal[vote_score > min_abs_score] = 1.0
    raw_signal[vote_score < -min_abs_score] = -1.0
    position_size = pd.Series(1.0, index=factors.index, dtype="float64")
    target_position = raw_signal * position_size
    position = target_position.shift(1).fillna(0.0)
    confidence_rank = vote_score.abs().rolling(
        int(getattr(config, "xgboost_trade_confidence_rank_window", 240) or 240),
        min_periods=20,
    ).rank(pct=True)
    confidence_rank = confidence_rank.fillna(vote_score.abs().rank(pct=True))

    signal = pd.DataFrame(
        {
            "xgboost_predicted_direction": raw_signal,
            "target_direction": np.nan,
            "future_horizon_return": np.nan,
            "prob_down": np.nan,
            "prob_flat": np.nan,
            "prob_up": np.nan,
            "prob_edge": vote_score,
            "directional_probability": vote_score.abs(),
            "calibrated_min_edge": min_abs_score,
            "calibrated_min_probability": 0.0,
            "xgboost_signal_direction": 1.0,
            "calibrated_prob_edge": vote_score,
            "composite_score": vote_score,
            "trade_allowed": 1.0,
            "confidence_rank": confidence_rank,
            "confidence_trade_allowed": 1.0,
            "position_size": position_size,
            "raw_signal": raw_signal,
            "raw_signal_before_position_rules": raw_signal,
            "target_position_before_rules": target_position,
            "target_position": target_position,
            "position": position,
            "benchmark_vote_score": vote_score,
            "benchmark_vote_signal_direction": 1.0,
            "benchmark_vote_raw_signal": raw_signal,
            "benchmark_vote_target_position": target_position,
            "benchmark_vote_position": position,
            "active_factor_count": len(available_factors),
            "active_factor_signal_coverage": factor_signals.ne(0).mean(axis=1),
        },
        index=factors.index,
    )
    return signal


def load_cached_factor_inputs(
    symbol_config: BacktestConfig,
    active_factors: list[str],
    price_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str] | None:
    """优先读取已缓存的 active 因子矩阵和行情字段，用于快速现场预测。"""
    if not bool(getattr(symbol_config, "trading_signal_use_factor_cache", True)):
        return None

    output_dir = Path(symbol_config.output_dir) / "composite_factor"
    cache_path = output_dir / "active_factor_matrix_cache.pkl"
    detail_path = output_dir / "composite_detail.csv"

    cached_factors: pd.DataFrame | None = None
    source = ""
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
        except Exception:
            cached = None
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            available_factors = [factor for factor in active_factors if factor in cached.columns]
            if available_factors:
                cached_factors = cached[available_factors].copy()
                source = f"现场计算:因子矩阵缓存:{cache_path}"

    if cached_factors is not None and price_data is not None:
        price_columns = [
            column
            for column in ["open", "high", "low", "close", "volume", "amt", "amount"]
            if column in price_data.columns
        ]
        if {"open", "close"}.issubset(price_columns):
            common_index = cached_factors.index.intersection(price_data.index)
            if not common_index.empty:
                data = price_data.loc[common_index, price_columns].copy()
                factors = cached_factors.loc[common_index].copy()
                factors.attrs.update(cached_factors.attrs)
                data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "close"])
                factors = factors.loc[data.index]
                if not data.empty and not factors.empty:
                    return data, factors, source

    if not detail_path.exists():
        return None
    try:
        detail = pd.read_csv(detail_path, index_col=0, parse_dates=True)
    except Exception:
        return None
    if detail.empty:
        return None
    price_columns = [
        column
        for column in ["open", "high", "low", "close", "volume", "amt", "amount"]
        if column in detail.columns
    ]
    if "close" not in price_columns or "open" not in price_columns:
        return None

    if cached_factors is None:
        value_columns = {
            factor: f"feature_{factor}_value"
            for factor in active_factors
            if f"feature_{factor}_value" in detail.columns
        }
        direct_columns = {factor: factor for factor in active_factors if factor in detail.columns}
        selected_value_columns = {**direct_columns, **value_columns}
        if selected_value_columns:
            cached_factors = detail[list(selected_value_columns.values())].rename(
                columns={source_column: factor for factor, source_column in selected_value_columns.items()}
            )
            source = f"现场计算:明细因子值缓存:{detail_path}"

    if cached_factors is None:
        signal_columns = {}
        for factor in active_factors:
            feature_signal = f"feature_{factor}_signal"
            raw_signal = f"{factor}_signal"
            if feature_signal in detail.columns:
                signal_columns[factor] = feature_signal
            elif raw_signal in detail.columns:
                signal_columns[factor] = raw_signal
        if signal_columns:
            cached_factors = detail[list(signal_columns.values())].rename(
                columns={source_column: factor for factor, source_column in signal_columns.items()}
            )
            cached_factors.attrs["precomputed_factor_signals"] = True
            source = f"现场计算:明细因子信号缓存:{detail_path}"

    if cached_factors is None or cached_factors.empty:
        return None

    common_index = cached_factors.index.intersection(detail.index)
    if common_index.empty:
        return None

    data = detail.loc[common_index, price_columns].copy()
    factors = cached_factors.loc[common_index].copy()
    factors.attrs.update(cached_factors.attrs)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "close"])
    factors = factors.loc[data.index]
    if data.empty or factors.empty:
        return None

    return data, factors, source


def load_live_factor_inputs(
    config: BacktestConfig,
    symbol: str,
    source: str,
) -> tuple[BacktestConfig, pd.DataFrame, pd.DataFrame, list[str], str]:
    """读取模型现场预测需要的配置、行情、因子和 active 候选池。"""
    symbol_config = build_symbol_compute_config(config, symbol, source)
    active_library = load_active_library_for_signal(symbol_config)
    active_factors = active_library["因子"].tolist()
    if not active_factors:
        raise ValueError(f"{symbol} 没有可用 active 因子。")

    print(f"现场读取 {symbol} 模型输入: active因子={len(active_factors)}")
    latest_data = fetch_intraday_data(symbol_config)
    if latest_data.empty:
        raise ValueError(f"{symbol} 行情数据为空。")

    cached_inputs = load_cached_factor_inputs(
        symbol_config,
        active_factors,
        price_data=latest_data,
    )
    if cached_inputs is not None:
        data, factors, input_source = cached_inputs
        cache_is_stale = pd.Timestamp(data.index.max()) < pd.Timestamp(latest_data.index.max())
        precomputed_signals = bool(factors.attrs.get("precomputed_factor_signals", False))
        feature_mode = str(symbol_config.xgboost_feature_mode).lower()
        signal_cache_incompatible = precomputed_signals and (
            feature_mode != "signal"
            or bool(getattr(symbol_config, "xgboost_include_factor_state_features", False))
        )
        if cache_is_stale or signal_cache_incompatible:
            reason = "因子缓存早于最新行情" if cache_is_stale else "信号缓存无法还原当前连续/状态特征"
            if not bool(getattr(symbol_config, "trading_signal_rebuild_missing_factors", False)):
                raise ValueError(
                    f"{symbol} {reason}；请先重新运行 composite/multi，"
                    "或设置 trading_signal_rebuild_missing_factors=True 现场重建。"
                )
            cached_inputs = None

    if cached_inputs is None:
        if not bool(getattr(symbol_config, "trading_signal_rebuild_missing_factors", False)):
            raise FileNotFoundError(
                f"{symbol} 没有可复用的因子缓存或明细因子列；"
                "请先运行该品种 composite/multi 流程，或设置 trading_signal_rebuild_missing_factors=True。"
            )
        data = latest_data
        factors = build_factors(data, symbol_config, requested_factors=active_factors)
        input_source = f"现场计算:实时构建:{Path(symbol_config.output_dir)}"
    if factors.empty:
        raise ValueError(f"{symbol} active 因子矩阵为空。")
    return symbol_config, data, factors, active_factors, input_source


def get_live_model_predict_index(
    factor_index: pd.Index,
    config: BacktestConfig,
) -> pd.Index:
    """生成与历史重训节奏对齐、且覆盖交易规则历史的实时预测区间。"""
    if len(factor_index) < 2:
        raise ValueError("模型现场预测至少需要两根 K 线。")
    predict_start = min(
        max(1, int(config.xgboost_min_train_samples)),
        max(0, len(factor_index) - 1),
    )
    available_steps = len(factor_index) - predict_start
    if available_steps <= 0:
        raise ValueError("可用样本不足，无法生成模型现场预测区间。")

    confidence_window = max(
        20,
        int(getattr(config, "xgboost_trade_confidence_rank_window", 240) or 240),
    )
    configured_history = max(
        1,
        int(getattr(config, "trading_signal_model_history_bars", confidence_window) or confidence_window),
    )
    position_history = (
        int(getattr(config, "xgboost_min_holding_bars", 0) or 0)
        + int(getattr(config, "xgboost_reentry_cooldown_bars", 0) or 0)
        + 2
    )
    required_history = max(confidence_window, configured_history, position_history)
    retrain_every = max(1, int(config.xgboost_retrain_every))
    latest_step = available_steps - 1
    desired_start_step = max(0, latest_step - required_history + 1)
    # 从重训边界开始，保证本段内的模型更新时间与完整历史回测一致。
    aligned_start_step = (desired_start_step // retrain_every) * retrain_every
    start_position = predict_start + aligned_start_step
    return factor_index[start_position:]


def build_latest_computed_signal_row(
    config: BacktestConfig,
    symbol: str,
    source: str,
) -> tuple[pd.Series, str]:
    """复用综合回测滚动模型，现场生成最新模型信号。"""
    symbol_config, data, factors, _active_factors, input_source = load_live_factor_inputs(
        config,
        symbol,
        source,
    )
    split_time, _ = split_train_validation_test_index(
        data.index,
        symbol_config.auto_select_train_ratio,
        symbol_config.auto_select_validation_ratio,
    )
    selected_factors, _ = get_selected_factors(data, factors, symbol_config, split_time)
    model_names = get_enabled_composite_models(symbol_config)
    model_name = "xgboost" if "xgboost" in model_names else model_names[0]
    predict_index = get_live_model_predict_index(factors.index, symbol_config)
    print(
        f"现场模型预测 {symbol}: model={model_name}, "
        f"候选因子={len(selected_factors)}, 预测历史={len(predict_index)}"
    )
    signal, _importance, _features, _selection = build_xgboost_rolling_signal(
        data,
        factors,
        selected_factors,
        symbol_config,
        predict_index,
        model_name=model_name,
    )
    valid = signal.dropna(subset=["target_position", "raw_signal"], how="all")
    if valid.empty:
        raise ValueError(f"{symbol} 模型现场计算没有生成有效交易信号。")

    latest_time = valid.index[-1]
    row = signal.loc[latest_time].copy()
    for column in ["open", "high", "low", "close", "volume", "amt", "amount"]:
        if column in data.columns:
            row[column] = data.loc[latest_time, column]
    row.name = latest_time
    return row, f"现场模型:{model_name}:{input_source}"


def build_latest_vote_signal_row(
    config: BacktestConfig,
    symbol: str,
    source: str,
) -> tuple[pd.Series, str]:
    """使用 active 因子历史表现加权投票，保留为显式基准模式。"""
    symbol_config, data, factors, _active_factors, input_source = load_live_factor_inputs(
        config,
        symbol,
        source,
    )
    active_library = load_active_library_for_signal(symbol_config)

    signal = build_weighted_factor_signal_frame(factors, active_library, symbol_config)
    valid = signal.dropna(subset=["target_position", "raw_signal"], how="all")
    if valid.empty:
        raise ValueError(f"{symbol} 现场计算没有生成有效交易信号。")

    latest_time = valid.index[-1]
    row = signal.loc[latest_time].copy()
    for column in ["open", "high", "low", "close", "volume", "amt", "amount"]:
        if column in data.columns:
            row[column] = data.loc[latest_time, column]
    row.name = latest_time
    return row, f"现场投票:{input_source}"


def safe_float(value: Any, default: float = np.nan) -> float:
    """安全转换为 float。"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool = True) -> bool:
    """把 1/0、True/False、字符串等安全转换为 bool。"""
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "否", "不允许"}
    return bool(value)


def get_optional_value(row: pd.Series, column: str, default: Any = np.nan) -> Any:
    """读取可选列。"""
    return row[column] if column in row.index else default


def infer_next_target_position(row: pd.Series) -> float:
    """推导模型给出的下一根 K 线目标仓位。"""
    if "target_position" in row.index and pd.notna(row["target_position"]):
        return safe_float(row["target_position"], 0.0)
    raw_signal = safe_float(get_optional_value(row, "raw_signal", 0.0), 0.0)
    position_size = safe_float(get_optional_value(row, "position_size", 1.0), 1.0)
    return raw_signal * position_size


def describe_direction(value: float, eps: float = 1e-9) -> str:
    """把仓位或信号数值转成中文方向。"""
    if value > eps:
        return "做多"
    if value < -eps:
        return "做空"
    return "空仓"


def describe_adjustment(current_position: float, target_position: float, eps: float = 1e-9) -> str:
    """把当前仓位和目标仓位转换成更贴近执行的动作。"""
    if abs(target_position - current_position) <= eps:
        return "不交易"
    if abs(current_position) <= eps and target_position > eps:
        return "开多"
    if abs(current_position) <= eps and target_position < -eps:
        return "开空"
    if current_position > eps and abs(target_position) <= eps:
        return "平多"
    if current_position < -eps and abs(target_position) <= eps:
        return "平空"
    if current_position > eps and target_position < -eps:
        return "平多并反手做空"
    if current_position < -eps and target_position > eps:
        return "平空并反手做多"
    if target_position > current_position:
        return "加多" if target_position > eps else "减空"
    return "减多" if target_position > eps else "加空"


def get_confidence_level(confidence_rank: float, min_rank: float) -> str:
    """根据置信度分位给出可读等级。"""
    if pd.isna(confidence_rank):
        return "未知"
    if confidence_rank >= 0.8:
        return "强"
    if confidence_rank >= max(0.5, min_rank):
        return "中"
    if confidence_rank >= min_rank:
        return "弱"
    return "过低"


def is_exposure_reducing(current_position: float, target_position: float, eps: float = 1e-9) -> bool:
    """判断从当前仓位走向目标仓位是否属于降低风险暴露。"""
    if abs(current_position) <= eps:
        return abs(target_position) <= eps
    if np.sign(current_position) != np.sign(target_position) and abs(target_position) > eps:
        return False
    return abs(target_position) <= abs(current_position) + eps


def clamp_to_reduce_only(current_position: float, target_position: float, eps: float = 1e-9) -> float:
    """当过滤器禁止加风险时，只允许减仓、平仓，不允许开仓或反手。"""
    if is_exposure_reducing(current_position, target_position, eps=eps):
        return target_position
    if abs(current_position) > eps and np.sign(current_position) != np.sign(target_position):
        return 0.0
    return current_position


def is_signal_expired(row_time: Any, config: BacktestConfig) -> tuple[bool, float]:
    """根据配置判断信号是否过期，默认不启用过期检查。"""
    max_age_minutes = int(getattr(config, "trading_signal_max_age_minutes", 0) or 0)
    if max_age_minutes <= 0:
        return False, np.nan
    try:
        signal_time = pd.Timestamp(row_time)
        reference_time = pd.Timestamp(getattr(config, "end_time", pd.Timestamp.now()))
        age_minutes = (reference_time - signal_time).total_seconds() / 60
        return age_minutes > max_age_minutes, age_minutes
    except Exception:
        return False, np.nan


def apply_execution_filters(
    row: pd.Series,
    config: BacktestConfig,
    current_position: float,
    model_target_position: float,
) -> dict[str, Any]:
    """把模型目标仓位转换成最终执行建议。"""
    min_adjustment = float(getattr(config, "trading_signal_min_adjustment", 0.10))
    min_confidence_rank = float(getattr(config, "trading_signal_min_confidence_rank", 0.30))
    use_quality_gate = bool(getattr(config, "trading_signal_use_quality_gate", True))
    block_low_liquidity = bool(getattr(config, "trading_signal_block_low_liquidity_open", True))
    block_when_model_denies = bool(getattr(config, "trading_signal_block_when_model_denies_trade", True))

    reasons: list[str] = []
    warnings: list[str] = []
    executable_target = model_target_position

    trade_allowed = safe_bool(get_optional_value(row, "trade_allowed", True), True)
    confidence_allowed = safe_bool(get_optional_value(row, "confidence_trade_allowed", True), True)
    confidence_rank = safe_float(get_optional_value(row, "confidence_rank", np.nan), np.nan)
    liquidity_regime = str(get_optional_value(row, "liquidity_regime", "") or "")

    expired, age_minutes = is_signal_expired(row.name, config)
    if expired:
        reasons.append("信号已过期")
    if block_when_model_denies and not trade_allowed:
        reasons.append("模型交易过滤器不允许交易")
    if block_when_model_denies and not confidence_allowed:
        reasons.append("模型置信度过滤器不允许交易")
    if use_quality_gate and pd.notna(confidence_rank) and confidence_rank < min_confidence_rank:
        reasons.append(f"置信度分位低于阈值({confidence_rank:.3f} < {min_confidence_rank:.3f})")
    if block_low_liquidity and "低" in liquidity_regime:
        reasons.append("当前处于低流动性状态")

    if reasons:
        executable_target = clamp_to_reduce_only(current_position, model_target_position)
        if abs(executable_target - model_target_position) > 1e-9:
            warnings.append("过滤器触发：只允许减仓/平仓，不允许新增风险")

    adjustment = executable_target - current_position
    if abs(adjustment) < min_adjustment:
        if abs(adjustment) > 1e-9:
            reasons.append(f"调仓幅度小于最小交易阈值({abs(adjustment):.3f} < {min_adjustment:.3f})")
        executable_target = current_position
        adjustment = 0.0

    should_trade = abs(adjustment) > 1e-9
    if should_trade and not reasons:
        decision = "建议交易"
    elif should_trade:
        decision = "仅风险收缩交易"
    else:
        decision = "不建议交易"

    return {
        "执行建议目标仓位": executable_target,
        "执行建议调仓量": adjustment,
        "是否建议交易": "是" if should_trade else "否",
        "交易决策": decision,
        "执行动作": describe_adjustment(current_position, executable_target),
        "执行方向": describe_direction(executable_target),
        "信号置信度等级": get_confidence_level(confidence_rank, min_confidence_rank),
        "信号年龄分钟": age_minutes,
        "交易过滤原因": "；".join(reasons) if reasons else "通过",
        "交易提醒": "；".join(warnings),
    }


def build_symbol_signal(
    config: BacktestConfig,
    symbol: str,
    source: str = "auto",
    mode: str = "compute",
) -> dict[str, Any]:
    """生成单个品种的最新交易信号行。"""
    mode = str(mode or getattr(config, "trading_signal_mode", "compute")).lower()
    if mode == "detail":
        detail_path = get_symbol_composite_detail_path(config, symbol, source)
        row = read_latest_signal_row(detail_path)
        signal_source = str(detail_path)
    elif mode in {"compute", "model"}:
        row, signal_source = build_latest_computed_signal_row(config, symbol, source)
    elif mode == "vote":
        row, signal_source = build_latest_vote_signal_row(config, symbol, source)
    else:
        raise ValueError("trading_signal mode 只能是 compute/model、vote 或 detail。")

    signal_time = row.name
    current_position = safe_float(get_optional_value(row, "position", 0.0), 0.0)
    model_target_position = infer_next_target_position(row)
    model_adjustment = model_target_position - current_position
    execution = apply_execution_filters(row, config, current_position, model_target_position)

    result: dict[str, Any] = {
        "品种": symbol,
        "信号时间": signal_time,
        "信号生成模式": mode,
        "信号来源": signal_source,
        "最新收盘价": safe_float(get_optional_value(row, "close", np.nan)),
        "当前实际仓位": current_position,
        "模型下一根目标仓位": model_target_position,
        "模型调仓量": model_adjustment,
        "模型交易方向": describe_direction(model_target_position),
        "模型调仓动作": describe_adjustment(current_position, model_target_position),
        **execution,
    }
    for column in SIGNAL_COLUMNS:
        if column in row.index and column not in result:
            result[column] = row[column]
    return result


def build_error_signal_row(symbol: str, error: Exception) -> dict[str, Any]:
    """把单品种读取失败也转成一行，便于多品种批量导出时定位问题。"""
    return {
        "品种": symbol,
        "信号时间": "",
        "信号生成模式": "",
        "信号来源": "",
        "最新收盘价": np.nan,
        "当前实际仓位": np.nan,
        "模型下一根目标仓位": np.nan,
        "模型调仓量": np.nan,
        "模型交易方向": "",
        "模型调仓动作": "",
        "执行建议目标仓位": np.nan,
        "执行建议调仓量": np.nan,
        "是否建议交易": "否",
        "交易决策": "读取失败",
        "执行动作": "",
        "执行方向": "",
        "信号置信度等级": "",
        "信号年龄分钟": np.nan,
        "交易过滤原因": "",
        "交易提醒": "",
        "错误": str(error),
    }


def build_trading_signals(
    config: BacktestConfig,
    symbols: list[str],
    source: str = "auto",
    mode: str = "compute",
) -> pd.DataFrame:
    """批量生成最新交易信号表。"""
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            rows.append(build_symbol_signal(config, symbol, source=source, mode=mode))
        except Exception as exc:
            rows.append(build_error_signal_row(symbol, exc))
    return pd.DataFrame(rows)


def save_trading_signals(
    signals: pd.DataFrame,
    config: BacktestConfig,
    output_path: str | None = None,
) -> Path:
    """保存最新交易信号表。"""
    if output_path:
        path = Path(output_path)
    else:
        path = Path(config.output_dir) / "trading_signals" / "trading_signals_latest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_trading_signal_export(
    config: BacktestConfig,
    symbols: str | list[str] | None = None,
    source: str = "auto",
    mode: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """生成并保存最新交易信号。"""
    selected_symbols = normalize_signal_symbols(symbols, config)
    signal_mode = str(mode or getattr(config, "trading_signal_mode", "compute") or "compute").lower()
    try:
        signals = build_trading_signals(config, selected_symbols, source=source, mode=signal_mode)
    finally:
        if signal_mode in {"compute", "model", "vote"}:
            stop_wind()
    path = save_trading_signals(signals, config, output_path)
    print(f"最新交易信号已保存: {path}")
    if not signals.empty:
        display_columns = [
            column
            for column in [
                "品种",
                "信号时间",
                "信号生成模式",
                "交易决策",
                "是否建议交易",
                "执行动作",
                "当前实际仓位",
                "执行建议目标仓位",
                "执行建议调仓量",
                "信号置信度等级",
                "交易过滤原因",
                "错误",
            ]
            if column in signals.columns
        ]
        print(signals[display_columns].to_string(index=False))
    return signals


def build_parser() -> argparse.ArgumentParser:
    """创建独立脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="导出最新单品种或多品种交易信号")
    parser.add_argument("--symbols", help="逗号分隔品种列表；不填则使用 config.symbols 全品种池。")
    parser.add_argument(
        "--mode",
        choices=["compute", "model", "vote", "detail"],
        default=None,
        help=(
            "信号生成模式：compute/model 复用综合回测滚动模型；"
            "vote 使用 active 因子加权投票；detail 读取已有 composite_detail.csv。"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "single", "multi"],
        default="auto",
        help="信号来源：single 读根目录 composite_factor；multi 读 by_symbol；auto 优先 by_symbol。",
    )
    parser.add_argument("--output", help="输出 CSV 路径；不填则写入 output_dir/trading_signals/trading_signals_latest.csv。")
    parser.add_argument("--output-dir", help="覆盖 config.output_dir。")
    return parser


def main() -> None:
    """独立脚本入口。"""
    args = build_parser().parse_args()
    config = BacktestConfig()
    configure_warning_output(config)
    if args.output_dir:
        config.output_dir = args.output_dir
    run_trading_signal_export(
        config,
        symbols=args.symbols,
        source=args.source,
        mode=args.mode,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

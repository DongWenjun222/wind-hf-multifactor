from __future__ import annotations

"""滚动模型校准工具。

校准器只使用当前预测时点之前已经成熟的标签，既支持普通时间序列索引，
也支持 pooled long-format 中同一时间包含多个品种的样本。
"""

from typing import Any, Iterable

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = ["prob_down", "prob_flat", "prob_up"]
TARGET_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}


def normalize_class_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    """清洗并归一化三分类概率，同时保留原索引。"""
    frame = probabilities.reindex(columns=PROBABILITY_COLUMNS).astype("float64")
    frame = frame.replace([np.inf, -np.inf], np.nan).clip(1e-12, 1.0)
    row_sum = frame.sum(axis=1, min_count=len(PROBABILITY_COLUMNS)).replace(0.0, np.nan)
    return frame.div(row_sum, axis=0)


def temperature_scale_multiclass(
    probabilities: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    """使用单参数温度缩放校准三分类概率。"""
    normalized = normalize_class_probabilities(probabilities)
    temperature = max(1e-6, float(temperature))
    values = normalized.to_numpy(dtype="float64")
    valid = np.isfinite(values).all(axis=1)
    output = np.full_like(values, np.nan, dtype="float64")
    if valid.any():
        logits = np.log(np.clip(values[valid], 1e-12, 1.0)) / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        scaled = np.exp(logits)
        output[valid] = scaled / scaled.sum(axis=1, keepdims=True)
    return pd.DataFrame(output, index=normalized.index, columns=PROBABILITY_COLUMNS)


def temperature_scale_binary(probability: pd.Series, temperature: float) -> pd.Series:
    """对二分类概率的 logit 使用温度缩放。"""
    clean = pd.to_numeric(probability, errors="coerce").astype("float64")
    temperature = max(1e-6, float(temperature))
    clipped = clean.clip(1e-8, 1.0 - 1e-8)
    logit = np.log(clipped / (1.0 - clipped)) / temperature
    scaled = 1.0 / (1.0 + np.exp(-logit.clip(-40.0, 40.0)))
    return scaled.where(clean.notna())


def _temperature_grid(config: Any) -> list[float]:
    raw = getattr(config, "composite_probability_temperature_grid", None) or [
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
    ]
    values = sorted({max(1e-6, float(value)) for value in raw})
    if 1.0 not in values:
        values.append(1.0)
        values.sort()
    return values


def _best_multiclass_temperature(
    probabilities: pd.DataFrame,
    target: pd.Series,
    grid: Iterable[float],
) -> float:
    labels = pd.to_numeric(target, errors="coerce").map(TARGET_TO_CLASS)
    normalized = normalize_class_probabilities(probabilities)
    valid = labels.notna() & normalized.notna().all(axis=1)
    if not valid.any():
        return 1.0
    label_values = labels.loc[valid].astype(int).to_numpy()
    best_temperature = 1.0
    best_key = (float("inf"), float("inf"))
    for temperature in grid:
        calibrated = temperature_scale_multiclass(
            normalized.loc[valid],
            temperature,
        ).to_numpy()
        loss = float(
            -np.log(
                np.clip(
                    calibrated[np.arange(len(label_values)), label_values],
                    1e-12,
                    1.0,
                )
            ).mean()
        )
        key = (loss, abs(np.log(float(temperature))))
        if key < best_key:
            best_key = key
            best_temperature = float(temperature)
    return best_temperature


def _best_binary_temperature(
    probability: pd.Series,
    target: pd.Series,
    grid: Iterable[float],
) -> float:
    clean_probability = pd.to_numeric(probability, errors="coerce")
    clean_target = pd.to_numeric(target, errors="coerce")
    valid = clean_probability.notna() & clean_target.notna()
    if not valid.any():
        return 1.0
    target_values = clean_target.loc[valid].clip(0.0, 1.0)
    best_temperature = 1.0
    best_key = (float("inf"), float("inf"))
    for temperature in grid:
        calibrated = temperature_scale_binary(
            clean_probability.loc[valid],
            temperature,
        ).clip(1e-12, 1.0 - 1e-12)
        loss = float(
            -(
                target_values * np.log(calibrated)
                + (1.0 - target_values) * np.log(1.0 - calibrated)
            ).mean()
        )
        key = (loss, abs(np.log(float(temperature))))
        if key < best_key:
            best_key = key
            best_temperature = float(temperature)
    return best_temperature


def _resolve_timing(
    index: pd.Index,
    horizon: int,
    prediction_times: pd.Series | None,
    label_available_times: pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.Index]:
    """构造预测时间和标签成熟时间；默认使用行位置表达时间。"""
    if prediction_times is None:
        times = pd.Series(np.arange(len(index), dtype="int64"), index=index)
    else:
        times = pd.Series(prediction_times, index=index)
    if label_available_times is None:
        available = pd.Series(
            np.arange(len(index), dtype="int64") + max(1, int(horizon)),
            index=index,
        )
    else:
        available = pd.Series(label_available_times, index=index)
    unique_times = pd.Index(times.dropna().unique()).sort_values()
    return times, available, unique_times


def _history_mask(
    times: pd.Series,
    available: pd.Series,
    unique_times: pd.Index,
    step: int,
    current_time: Any,
    window: int,
) -> pd.Series:
    mask = times.lt(current_time) & available.le(current_time)
    if window > 0 and step > window:
        mask &= times.ge(unique_times[step - window])
    return mask.fillna(False)


def rolling_temperature_calibrate_multiclass(
    probabilities: pd.DataFrame,
    target: pd.Series,
    config: Any,
    *,
    horizon: int = 1,
    prediction_times: pd.Series | None = None,
    label_available_times: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """用严格滞后的历史 OOS 预测滚动校准三分类概率。"""
    raw = normalize_class_probabilities(probabilities)
    calibrated = raw.copy()
    temperatures = pd.Series(1.0, index=raw.index, dtype="float64")
    history_counts = pd.Series(0.0, index=raw.index, dtype="float64")
    if not bool(getattr(config, "composite_probability_calibration_enabled", True)):
        return calibrated, temperatures, history_counts

    times, available, unique_times = _resolve_timing(
        raw.index,
        horizon,
        prediction_times,
        label_available_times,
    )
    window = max(1, int(getattr(config, "composite_probability_calibration_window", 480) or 480))
    min_history = max(
        5,
        int(getattr(config, "composite_probability_calibration_min_history", 120) or 120),
    )
    retrain_every = max(
        1,
        int(getattr(config, "composite_probability_calibration_retrain_every", 25) or 25),
    )
    grid = _temperature_grid(config)
    current_temperature = 1.0
    current_history_count = 0
    last_fit_step = -10**9

    for step, current_time in enumerate(unique_times):
        current_mask = times.eq(current_time)
        if step - last_fit_step >= retrain_every:
            history = _history_mask(times, available, unique_times, step, current_time, window)
            valid_history = history & target.notna() & raw.notna().all(axis=1)
            current_history_count = int(valid_history.sum())
            if current_history_count >= min_history:
                current_temperature = _best_multiclass_temperature(
                    raw.loc[valid_history],
                    target.loc[valid_history],
                    grid,
                )
            last_fit_step = step
        calibrated.loc[current_mask] = temperature_scale_multiclass(
            raw.loc[current_mask],
            current_temperature,
        )
        temperatures.loc[current_mask] = current_temperature
        history_counts.loc[current_mask] = current_history_count
    return calibrated, temperatures, history_counts


def rolling_temperature_calibrate_binary(
    probability: pd.Series,
    target: pd.Series,
    config: Any,
    *,
    horizon: int = 1,
    prediction_times: pd.Series | None = None,
    label_available_times: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """用严格滞后的历史 OOS 标签滚动校准二分类概率。"""
    raw = pd.to_numeric(probability, errors="coerce").astype("float64")
    calibrated = raw.copy()
    temperatures = pd.Series(1.0, index=raw.index, dtype="float64")
    history_counts = pd.Series(0.0, index=raw.index, dtype="float64")
    if not bool(getattr(config, "composite_probability_calibration_enabled", True)):
        return calibrated, temperatures, history_counts

    times, available, unique_times = _resolve_timing(
        raw.index,
        horizon,
        prediction_times,
        label_available_times,
    )
    window = max(1, int(getattr(config, "composite_probability_calibration_window", 480) or 480))
    min_history = max(
        5,
        int(getattr(config, "composite_probability_calibration_min_history", 120) or 120),
    )
    retrain_every = max(
        1,
        int(getattr(config, "composite_probability_calibration_retrain_every", 25) or 25),
    )
    grid = _temperature_grid(config)
    current_temperature = 1.0
    current_history_count = 0
    last_fit_step = -10**9

    for step, current_time in enumerate(unique_times):
        current_mask = times.eq(current_time)
        if step - last_fit_step >= retrain_every:
            history = _history_mask(times, available, unique_times, step, current_time, window)
            valid_history = history & target.notna() & raw.notna()
            current_history_count = int(valid_history.sum())
            if current_history_count >= min_history:
                current_temperature = _best_binary_temperature(
                    raw.loc[valid_history],
                    target.loc[valid_history],
                    grid,
                )
            last_fit_step = step
        calibrated.loc[current_mask] = temperature_scale_binary(
            raw.loc[current_mask],
            current_temperature,
        )
        temperatures.loc[current_mask] = current_temperature
        history_counts.loc[current_mask] = current_history_count
    return calibrated, temperatures, history_counts


def rolling_calibrate_edge_to_return(
    raw_edge: pd.Series,
    realized_standardized_return: pd.Series,
    config: Any,
    *,
    horizon: int = 1,
    prediction_times: pd.Series | None = None,
    label_available_times: pd.Series | None = None,
    fallback_to_raw: bool = False,
) -> pd.DataFrame:
    """把任意模型边际滚动映射到预期标准化净收益。"""
    raw = pd.to_numeric(raw_edge, errors="coerce").astype("float64")
    realized = pd.to_numeric(realized_standardized_return, errors="coerce").astype("float64")
    calibrated = raw.copy() if fallback_to_raw else pd.Series(np.nan, index=raw.index, dtype="float64")
    slope_series = pd.Series(np.nan, index=raw.index, dtype="float64")
    intercept_series = pd.Series(np.nan, index=raw.index, dtype="float64")
    history_counts = pd.Series(0.0, index=raw.index, dtype="float64")
    if not bool(getattr(config, "composite_edge_calibration_enabled", True)):
        return pd.DataFrame(
            {
                "model_edge_score": raw,
                "edge_calibration_slope": 1.0,
                "edge_calibration_intercept": 0.0,
                "edge_calibration_history_count": 0.0,
            },
            index=raw.index,
        )

    times, available, unique_times = _resolve_timing(
        raw.index,
        horizon,
        prediction_times,
        label_available_times,
    )
    window = max(1, int(getattr(config, "composite_edge_calibration_window", 480) or 480))
    min_history = max(5, int(getattr(config, "composite_edge_calibration_min_history", 120) or 120))
    retrain_every = max(
        1,
        int(getattr(config, "composite_edge_calibration_retrain_every", 25) or 25),
    )
    prior_count = max(0.0, float(getattr(config, "composite_edge_calibration_prior_count", 40.0) or 0.0))
    slope_limit = max(0.1, float(getattr(config, "composite_edge_calibration_slope_limit", 3.0) or 3.0))
    score_clip = max(0.1, float(getattr(config, "composite_edge_score_clip", 3.0) or 3.0))
    current_slope = np.nan
    current_intercept = np.nan
    current_history_count = 0
    last_fit_step = -10**9

    for step, current_time in enumerate(unique_times):
        current_mask = times.eq(current_time)
        if step - last_fit_step >= retrain_every:
            history = _history_mask(times, available, unique_times, step, current_time, window)
            valid_history = history & raw.notna() & realized.notna()
            current_history_count = int(valid_history.sum())
            if current_history_count >= min_history:
                x = raw.loc[valid_history].to_numpy(dtype="float64")
                y = realized.loc[valid_history].to_numpy(dtype="float64")
                x_mean = float(np.mean(x))
                y_mean = float(np.mean(y))
                variance = float(np.mean((x - x_mean) ** 2))
                covariance = float(np.mean((x - x_mean) * (y - y_mean)))
                raw_slope = covariance / variance if variance > 1e-12 else 0.0
                reliability = (
                    current_history_count / (current_history_count + prior_count)
                    if prior_count > 0
                    else 1.0
                )
                current_slope = float(np.clip(raw_slope * reliability, -slope_limit, slope_limit))
                current_intercept = float((y_mean - raw_slope * x_mean) * reliability)
            last_fit_step = step
        if np.isfinite(current_slope) and np.isfinite(current_intercept):
            calibrated.loc[current_mask] = (
                current_intercept + current_slope * raw.loc[current_mask]
            ).clip(-score_clip, score_clip)
        slope_series.loc[current_mask] = current_slope
        intercept_series.loc[current_mask] = current_intercept
        history_counts.loc[current_mask] = current_history_count

    return pd.DataFrame(
        {
            "model_edge_score": calibrated,
            "edge_calibration_slope": slope_series,
            "edge_calibration_intercept": intercept_series,
            "edge_calibration_history_count": history_counts,
        },
        index=raw.index,
    )


def calculate_binary_probability_metrics(
    probability: pd.Series,
    target: pd.Series,
    bins: int = 10,
) -> dict[str, float]:
    """计算两阶段第一阶段所需的二分类概率评价。"""
    p = pd.to_numeric(probability, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    valid = p.notna() & y.notna()
    if not valid.any():
        return {
            "可交易概率Brier": np.nan,
            "可交易概率LogLoss": np.nan,
            "可交易概率ECE": np.nan,
        }
    p = p.loc[valid].clip(1e-12, 1.0 - 1e-12)
    y = y.loc[valid].clip(0.0, 1.0)
    brier = float(((p - y) ** 2).mean())
    logloss = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())
    bucket = pd.cut(p, bins=np.linspace(0.0, 1.0, max(2, int(bins)) + 1), include_lowest=True)
    calibration = pd.DataFrame({"p": p, "y": y, "bucket": bucket}).groupby(
        "bucket",
        observed=True,
    ).agg(probability=("p", "mean"), realized=("y", "mean"), count=("y", "size"))
    ece = float(
        (
            (calibration["probability"] - calibration["realized"]).abs()
            * calibration["count"]
        ).sum()
        / calibration["count"].sum()
    )
    return {
        "可交易概率Brier": brier,
        "可交易概率LogLoss": logloss,
        "可交易概率ECE": ece,
    }

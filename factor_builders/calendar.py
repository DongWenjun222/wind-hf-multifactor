from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    align_macro_daily_to_main,
    align_related_data_to_main,
    calculate_related_data_coverage,
    rolling_zscore,
    safe_symbol_name,
)

def add_calendar_seasonality_factors(
    df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """生成交易日历和季节性因子。

    这些因子只使用 K 线时间戳，以及当前和历史已经可见的量价状态；
    适合捕捉日内时段效应、周内/月内/年内季节性和季节性与量价状态的交互。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df.index)

    index = pd.DatetimeIndex(df.index)
    factor_specs: list[tuple[str, pd.Series]] = []
    minute_of_day = pd.Series(index.hour * 60 + index.minute, index=index, dtype="float64")
    day_angle = 2.0 * np.pi * minute_of_day / 1440.0

    calendar_date = pd.Series(index.normalize(), index=index)
    trading_date = calendar_date.where(index.hour < 21, calendar_date + pd.Timedelta(days=1))
    trading_datetime = pd.DatetimeIndex(trading_date)
    day_of_week = pd.Series(trading_datetime.dayofweek, index=index, dtype="float64")
    month = pd.Series(trading_datetime.month, index=index, dtype="float64")
    day_of_month = pd.Series(trading_datetime.day, index=index, dtype="float64")
    day_of_year = pd.Series(trading_datetime.dayofyear, index=index, dtype="float64")
    quarter = pd.Series(trading_datetime.quarter, index=index, dtype="float64")
    days_in_month = pd.Series(trading_datetime.days_in_month, index=index, dtype="float64")
    days_to_month_end = days_in_month - day_of_month
    day_of_month_centered = (day_of_month - 1.0) / (days_in_month - 1.0).replace(0, np.nan) - 0.5
    week_angle = 2.0 * np.pi * day_of_week / 5.0
    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0
    year_angle = 2.0 * np.pi * day_of_year / 366.0
    quarter_angle = 2.0 * np.pi * (quarter - 1.0) / 4.0

    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    volume = df.get("volume", pd.Series(np.nan, index=index)).replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    bar_return = df.get("bar_return_cc", close.pct_change())
    intrabar_return = df.get("bar_return_oc", close / open_price - 1.0)
    range_pct = (high - low) / open_price
    body_pct = (close - open_price) / open_price
    close_location = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    gap_return = open_price / close.shift(1).replace(0, np.nan) - 1.0
    volume_change = volume.pct_change()
    amount_change = amount.pct_change()
    return_rolling_mean = bar_return.rolling(20, min_periods=5).mean()
    return_rolling_vol = bar_return.rolling(20, min_periods=5).std()
    intrabar_rolling_mean = intrabar_return.rolling(20, min_periods=5).mean()
    range_rolling_mean = range_pct.rolling(20, min_periods=5).mean()
    volume_rolling_mean = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    volume_ratio = volume / volume_rolling_mean - 1.0
    amount_ratio = amount / amount.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1.0

    raw_calendar_inputs = {
        "time_sin": pd.Series(np.sin(day_angle), index=index),
        "time_cos": pd.Series(np.cos(day_angle), index=index),
        "week_sin": pd.Series(np.sin(week_angle), index=index),
        "week_cos": pd.Series(np.cos(week_angle), index=index),
        "month_sin": pd.Series(np.sin(month_angle), index=index),
        "month_cos": pd.Series(np.cos(month_angle), index=index),
        "year_sin": pd.Series(np.sin(year_angle), index=index),
        "year_cos": pd.Series(np.cos(year_angle), index=index),
        "quarter_sin": pd.Series(np.sin(quarter_angle), index=index),
        "quarter_cos": pd.Series(np.cos(quarter_angle), index=index),
        "day_of_month_centered": day_of_month_centered,
        "days_to_month_end_norm": days_to_month_end / days_in_month.replace(0, np.nan) - 0.5,
        "night_session": pd.Series(((index.hour >= 21) | (index.hour < 9)).astype(float), index=index),
        "morning_session": pd.Series(((index.hour >= 9) & (index.hour < 11)).astype(float), index=index),
        "midday_session": pd.Series(((index.hour >= 11) & (index.hour < 14)).astype(float), index=index),
        "afternoon_session": pd.Series(((index.hour >= 14) & (index.hour < 16)).astype(float), index=index),
        "night_open": pd.Series(((index.hour == 21) & (index.minute <= 30)).astype(float), index=index),
        "day_open": pd.Series(((index.hour == 9) & (index.minute <= 30)).astype(float), index=index),
        "day_close": pd.Series(((index.hour == 14) & (index.minute >= 30)).astype(float), index=index),
        "monday_or_postweekend": (day_of_week <= 0).astype(float),
        "friday_or_preweekend": (day_of_week >= 4).astype(float),
        "month_start": (day_of_month <= 3).astype(float),
        "month_end": (days_to_month_end <= 3).astype(float),
        "quarter_start": ((month % 3 == 1) & (day_of_month <= 5)).astype(float),
        "quarter_end": ((month % 3 == 0) & (days_to_month_end <= 5)).astype(float),
        "year_start": ((month == 1) & (day_of_month <= 10)).astype(float),
        "year_end": ((month == 12) & (days_to_month_end <= 10)).astype(float),
    }

    for name, value in raw_calendar_inputs.items():
        factor_specs.append((f"calendar_{name}", value))

    seasonal_inputs = {
        name: raw_calendar_inputs[name]
        for name in [
            "time_sin",
            "time_cos",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "year_sin",
            "year_cos",
            "night_session",
            "month_end",
            "quarter_end",
        ]
    }
    state_inputs = {
        "ret_mean20": return_rolling_mean,
        "ret_vol20": return_rolling_vol,
        "intrabar_mean20": intrabar_rolling_mean,
        "range_mean20": range_rolling_mean,
        "volume_ratio20": volume_ratio,
        "abs_ret": bar_return.abs(),
    }
    for seasonal_name, seasonal_value in seasonal_inputs.items():
        for state_name, state_value in state_inputs.items():
            factor_specs.append(
                (
                    f"calendar_x_{seasonal_name}_{state_name}",
                    seasonal_value * state_value,
                )
            )

    extension_specs: list[tuple[str, pd.Series]] = []
    harmonic_inputs = {
        "time": minute_of_day / 1440.0,
        "week": day_of_week / 5.0,
        "month": (day_of_month - 1.0) / days_in_month.replace(0, np.nan),
        "year": day_of_year / 366.0,
        "quarter": (quarter - 1.0) / 4.0,
    }
    calendar_multipliers: dict[str, pd.Series] = {
        "night": raw_calendar_inputs["night_session"],
        "morning": raw_calendar_inputs["morning_session"],
        "midday": raw_calendar_inputs["midday_session"],
        "afternoon": raw_calendar_inputs["afternoon_session"],
        "day_open": raw_calendar_inputs["day_open"],
        "day_close": raw_calendar_inputs["day_close"],
        "month_start": raw_calendar_inputs["month_start"],
        "month_end": raw_calendar_inputs["month_end"],
        "quarter_start": raw_calendar_inputs["quarter_start"],
        "quarter_end": raw_calendar_inputs["quarter_end"],
        "year_start": raw_calendar_inputs["year_start"],
        "year_end": raw_calendar_inputs["year_end"],
        "preweekend": raw_calendar_inputs["friday_or_preweekend"],
        "postweekend": raw_calendar_inputs["monday_or_postweekend"],
        "month_progress": (day_of_month - 1.0) / days_in_month.replace(0, np.nan) - 0.5,
        "month_remaining": days_to_month_end / days_in_month.replace(0, np.nan) - 0.5,
        "year_progress": day_of_year / 366.0 - 0.5,
        "week_progress": day_of_week / 5.0 - 0.5,
        "day_progress": minute_of_day / 1440.0 - 0.5,
    }
    for cycle_name, cycle_value in harmonic_inputs.items():
        for harmonic in range(1, 21):
            angle = 2.0 * np.pi * harmonic * cycle_value
            calendar_multipliers[f"harm_sin_{cycle_name}_{harmonic}"] = pd.Series(np.sin(angle), index=index)
            calendar_multipliers[f"harm_cos_{cycle_name}_{harmonic}"] = pd.Series(np.cos(angle), index=index)

    market_state_inputs = {
        "ret": bar_return,
        "intrabar": intrabar_return,
        "abs_ret": bar_return.abs(),
        "range": range_pct,
        "body": body_pct,
        "gap": gap_return,
        "volume_chg": volume_change,
        "amount_chg": amount_change,
        "volume_ratio": volume_ratio,
        "amount_ratio": amount_ratio,
        "location": close_location,
        "ret_mean20": return_rolling_mean,
        "ret_vol20": return_rolling_vol,
        "intrabar_mean20": intrabar_rolling_mean,
        "range_mean20": range_rolling_mean,
    }
    windows = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233]
    state_features: list[tuple[str, pd.Series]] = []
    for state_name, state_value in market_state_inputs.items():
        state_features.append((f"level_{state_name}", state_value))
        for window in windows:
            min_periods = max(2, window // 3)
            state_mean = state_value.rolling(window, min_periods=min_periods).mean()
            state_std = state_value.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            state_abs_mean = state_value.abs().rolling(window, min_periods=min_periods).mean().replace(0, np.nan)
            state_features.extend(
                [
                    (f"mean_{state_name}_{window}", state_mean),
                    (f"std_{state_name}_{window}", state_std),
                    (f"zdist_{state_name}_{window}", (state_value - state_mean) / state_std),
                    (f"diff_mean_{state_name}_{window}", state_value.diff().rolling(window, min_periods=min_periods).mean()),
                    (f"last_mean_gap_{state_name}_{window}", state_value / state_mean.replace(0, np.nan) - 1.0),
                    (f"mean_abs_ratio_{state_name}_{window}", state_mean / state_abs_mean),
                ]
            )

    for multiplier_name, multiplier_value in calendar_multipliers.items():
        for state_name, state_value in state_features:
            extension_specs.append(
                (
                    f"calendar_ext_{multiplier_name}_{state_name}",
                    multiplier_value * state_value,
                )
            )
            if len(extension_specs) >= 20000:
                break
        if len(extension_specs) >= 20000:
            break

    factor_specs.extend(extension_specs)

    calendar_raw = pd.DataFrame(
        {factor_name: raw_factor for factor_name, raw_factor in factor_specs},
        index=df.index,
    )
    min_periods = max(20, int(config.zscore_window) // 3)
    rolling_mean = calendar_raw.rolling(config.zscore_window, min_periods=min_periods).mean()
    rolling_std = calendar_raw.rolling(config.zscore_window, min_periods=min_periods).std()
    return ((calendar_raw - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-3, 3)

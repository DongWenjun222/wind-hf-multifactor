from __future__ import annotations

"""四类各三万个按需扩展因子。"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import align_related_data_to_main, rolling_zscore


FAMILY_EXPANSION_COUNT = 30_000
FAMILY_EXPANSION_TOTAL_COUNT = 120_000
FAMILY_PREFIXES = ("paramx", "calendarx", "noncrossx", "crossx")
FAMILY_WINDOWS = (2, 3, 4, 5, 8, 10, 13, 21, 34, 55, 89, 110, 144, 178, 233)
FAMILY_LAGS = (0, 1, 2, 5)
FAMILY_SCALES = (1, 2, 3, 4, 6)
FAMILY_TRANSFORMS = (
    "meangap", "stdratio", "zdist", "rank", "momentum",
    "slope", "skewgap", "kurtgap", "autocorr", "efficiency",
)
FAMILY_SOURCES = {
    "paramx": (
        "ret", "gap", "range", "body", "location",
        "volchg", "amtchg", "pressure", "illiq", "vwapgap",
    ),
    "calendarx": (
        "dowsin", "dowcos", "monthsin", "monthcos", "doysin",
        "doycos", "monthend", "quarterend", "weekphase", "yearphase",
    ),
    "noncrossx": (
        "absret", "signedrange", "bodyrange", "wickbalance", "retvol",
        "retamt", "pressurechg", "locationmom", "illiqsign", "trangeaccel",
    ),
    "crossx": (
        "relret", "relabsret", "relvol", "relrange", "returnspread",
        "rangespread", "volspread", "basketmom", "dispersion", "leadgap",
    ),
}


@lru_cache(maxsize=1)
def get_family_expansion_names() -> tuple[str, ...]:
    """返回四个连续名称块，每块严格三万个。"""
    blocks: list[tuple[str, ...]] = []
    for family in FAMILY_PREFIXES:
        block = tuple(
            f"{family}_{source}_{transform}_w{window}_l{lag}_s{scale}"
            for source in FAMILY_SOURCES[family]
            for transform in FAMILY_TRANSFORMS
            for window in FAMILY_WINDOWS
            for lag in FAMILY_LAGS
            for scale in FAMILY_SCALES
        )
        if len(block) != FAMILY_EXPANSION_COUNT:
            raise RuntimeError(f"{family} 因子名称数量异常: {len(block)}")
        blocks.append(block)
    names = tuple(name for block in blocks for name in block)
    if len(names) != FAMILY_EXPANSION_TOTAL_COUNT or len(set(names)) != len(names):
        raise RuntimeError("四类扩展因子名称数量或唯一性异常")
    return names


def _parse_name(name: str) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in FAMILY_PREFIXES:
        return None
    family, source, transform = parts[:3]
    if source not in FAMILY_SOURCES[family] or transform not in FAMILY_TRANSFORMS:
        return None
    try:
        window = int(parts[3][1:])
        lag = int(parts[4][1:])
        scale = int(parts[5][1:])
    except (ValueError, IndexError):
        return None
    if window not in FAMILY_WINDOWS or lag not in FAMILY_LAGS or scale not in FAMILY_SCALES:
        return None
    return family, source, transform, window, lag, scale


def _market_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    volume = df.get("volume", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    ret = close.pct_change()
    spread = (high - low).replace(0, np.nan)
    range_pct = spread / open_price
    body = (close - open_price) / open_price
    location = ((close - low) - (high - close)) / spread
    volchg = volume.pct_change()
    amtchg = amount.pct_change()
    vwap = amount / volume
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1) / open_price
    return {
        "ret": ret,
        "gap": open_price / close.shift() - 1.0,
        "range": range_pct,
        "body": body,
        "location": location,
        "volchg": volchg,
        "amtchg": amtchg,
        "pressure": location * range_pct,
        "illiq": ret.abs() / amount,
        "vwapgap": close / vwap.replace(0, np.nan) - 1.0,
        "absret": ret.abs(),
        "signedrange": np.sign(ret) * range_pct,
        "bodyrange": body / range_pct.replace(0, np.nan),
        "wickbalance": location * body.abs(),
        "retvol": ret * volchg,
        "retamt": ret * amtchg,
        "pressurechg": (location * range_pct).diff(),
        "locationmom": location * ret,
        "illiqsign": np.sign(ret) * ret.abs() / amount,
        "trangeaccel": true_range.diff(),
    }


def _calendar_components(df: pd.DataFrame) -> dict[str, pd.Series]:
    index = pd.DatetimeIndex(df.index)
    ret = df["close"].replace(0, np.nan).pct_change().fillna(0.0)
    dow = index.dayofweek.to_numpy(dtype="float64")
    month = index.month.to_numpy(dtype="float64")
    doy = index.dayofyear.to_numpy(dtype="float64")
    week = index.isocalendar().week.to_numpy(dtype="float64")
    days_in_month = index.days_in_month.to_numpy(dtype="float64")
    day = index.day.to_numpy(dtype="float64")

    def series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=df.index, dtype="float64") * (1.0 + ret)

    return {
        "dowsin": series(np.sin(2 * np.pi * dow / 5.0)),
        "dowcos": series(np.cos(2 * np.pi * dow / 5.0)),
        "monthsin": series(np.sin(2 * np.pi * month / 12.0)),
        "monthcos": series(np.cos(2 * np.pi * month / 12.0)),
        "doysin": series(np.sin(2 * np.pi * doy / 365.25)),
        "doycos": series(np.cos(2 * np.pi * doy / 365.25)),
        "monthend": series((days_in_month - day) / np.maximum(days_in_month, 1.0)),
        "quarterend": series(((month - 1.0) % 3.0 + day / np.maximum(days_in_month, 1.0)) / 3.0),
        "weekphase": series(np.sin(2 * np.pi * week / 52.0)),
        "yearphase": series(doy / 365.25 - 0.5),
    }


def _cross_components(
    df: pd.DataFrame,
    related_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> dict[str, pd.Series]:
    main_close = df["close"].replace(0, np.nan)
    main_ret = main_close.pct_change()
    main_range = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    main_volume = df.get("volume", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    related_returns: list[pd.Series] = []
    related_ranges: list[pd.Series] = []
    related_volumes: list[pd.Series] = []
    max_ffill = max(0, int(getattr(config, "cross_asset_max_ffill_bars", 2) or 0))
    for related in related_data_map.values():
        if related.empty or "close" not in related.columns:
            continue
        aligned = align_related_data_to_main(related, df.index, max_ffill)
        close = aligned["close"].replace(0, np.nan)
        related_returns.append(close.pct_change())
        if {"high", "low", "open"}.issubset(aligned.columns):
            related_ranges.append(
                (aligned["high"] - aligned["low"]) / aligned["open"].replace(0, np.nan)
            )
        if "volume" in aligned.columns:
            related_volumes.append(aligned["volume"].replace(0, np.nan).pct_change())
    if not related_returns:
        return {}
    return_frame = pd.concat(related_returns, axis=1)
    basket_ret = return_frame.mean(axis=1)
    basket_range = (
        pd.concat(related_ranges, axis=1).mean(axis=1)
        if related_ranges
        else basket_ret.abs()
    )
    basket_volume = (
        pd.concat(related_volumes, axis=1).mean(axis=1)
        if related_volumes
        else pd.Series(np.nan, index=df.index)
    )
    main_volchg = main_volume.pct_change()
    return {
        "relret": basket_ret,
        "relabsret": basket_ret.abs(),
        "relvol": basket_volume,
        "relrange": basket_range,
        "returnspread": main_ret - basket_ret,
        "rangespread": main_range - basket_range,
        "volspread": main_volchg - basket_volume,
        "basketmom": basket_ret.rolling(5).sum(),
        "dispersion": return_frame.std(axis=1),
        "leadgap": basket_ret.shift(1) - main_ret,
    }


def _transform(value: pd.Series, transform: str, window: int, scale: int) -> pd.Series:
    slow = max(window + 1, window * scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow).std().replace(0, np.nan)
    if transform == "meangap":
        return fast_mean - slow_mean
    if transform == "stdratio":
        return fast_std / slow_std - 1.0
    if transform == "zdist":
        return (value - slow_mean) / slow_std
    if transform == "rank":
        return value.rolling(slow).rank(pct=True) - 0.5
    if transform == "momentum":
        return value.rolling(window).sum() - value.rolling(slow).sum() * window / slow
    if transform == "slope":
        return fast_mean.diff(max(1, scale))
    if transform == "skewgap":
        return value.rolling(window).skew() - value.rolling(slow).skew()
    if transform == "kurtgap":
        return value.rolling(window).kurt() - value.rolling(slow).kurt()
    if transform == "autocorr":
        return value.rolling(slow).corr(value.shift(max(1, scale)))
    return value.rolling(window).sum() / value.abs().rolling(slow).sum().replace(0, np.nan)


def add_family_expansion_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | list[str] | None,
    related_data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """仅计算明确请求的四类扩展因子。"""
    parsed = [(name, _parse_name(name)) for name in dict.fromkeys(requested_factors or [])]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)
    requested_families = {spec[0] for _, spec in parsed}
    source_maps: dict[str, dict[str, pd.Series]] = {}
    market = _market_components(df)
    if "paramx" in requested_families:
        source_maps["paramx"] = {name: market[name] for name in FAMILY_SOURCES["paramx"]}
    if "noncrossx" in requested_families:
        source_maps["noncrossx"] = {name: market[name] for name in FAMILY_SOURCES["noncrossx"]}
    if "calendarx" in requested_families:
        source_maps["calendarx"] = _calendar_components(df)
    if "crossx" in requested_families:
        source_maps["crossx"] = _cross_components(df, related_data_map or {}, config)

    cache: dict[tuple[str, str, str, int, int, int], pd.Series] = {}
    columns: dict[str, pd.Series] = {}
    for name, spec in parsed:
        family, source, transform, window, lag, scale = spec
        if source not in source_maps.get(family, {}):
            continue
        if spec not in cache:
            raw = _transform(source_maps[family][source].shift(lag), transform, window, scale)
            cache[spec] = rolling_zscore(raw, config.zscore_window)
        columns[name] = cache[spec]
    return pd.DataFrame(columns, index=df.index)

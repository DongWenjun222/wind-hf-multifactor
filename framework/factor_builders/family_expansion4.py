from __future__ import annotations

"""五类各两万个第四批按需扩展因子。"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore
from .family_expansion import _calendar_components, _cross_components, _market_components


FOURTH_FAMILY_COUNT = 20_000
FOURTH_FAMILY_TOTAL_COUNT = 100_000
FOURTH_FAMILY_PREFIXES = ("crossw", "noncrossw", "expanded5", "paramw", "calendarw")
FOURTH_FAMILY_WINDOWS = (2, 4, 7, 11, 18, 29, 47, 76, 123, 199)
FOURTH_FAMILY_LAGS = (0, 1, 4, 8)
FOURTH_FAMILY_SCALES = (1, 2, 3, 5, 7)
FOURTH_FAMILY_TRANSFORMS = (
    "dualmomentum",
    "downsidez",
    "upsidez",
    "semivolratio",
    "changepersist",
    "shockdecay",
    "curvature",
    "efficiencygap",
    "quantileskew",
    "autocorrgap",
)
FOURTH_FAMILY_SOURCES = {
    "crossw": (
        "relret", "relabsret", "relvol", "relrange", "returnspread",
        "rangespread", "volspread", "basketmom", "dispersion", "leadgap",
    ),
    "noncrossw": (
        "returnliquidity", "rangepressure", "bodyvolume", "locationvolume",
        "gapvolume", "wickpressure", "returnamount", "rangeilliquidity",
        "bodyaccel", "pressureaccel",
    ),
    "expanded5": (
        "absret", "signedrange", "bodyrange", "wickbalance", "retvol",
        "retamt", "pressurechg", "locationmom", "illiqsign", "trangeaccel",
    ),
    "paramw": (
        "ret", "gap", "range", "body", "location",
        "volchg", "amtchg", "pressure", "illiq", "vwapgap",
    ),
    "calendarw": (
        "dowsin", "dowcos", "monthsin", "monthcos", "doysin",
        "doycos", "monthend", "quarterend", "weekphase", "yearphase",
    ),
}


@lru_cache(maxsize=1)
def get_fourth_family_expansion_names() -> tuple[str, ...]:
    """返回五个连续名称块，每块严格两万个。"""
    blocks: list[tuple[str, ...]] = []
    for family in FOURTH_FAMILY_PREFIXES:
        block = tuple(
            f"{family}_{source}_{transform}_w{window}_l{lag}_s{scale}"
            for source in FOURTH_FAMILY_SOURCES[family]
            for transform in FOURTH_FAMILY_TRANSFORMS
            for window in FOURTH_FAMILY_WINDOWS
            for lag in FOURTH_FAMILY_LAGS
            for scale in FOURTH_FAMILY_SCALES
        )
        if len(block) != FOURTH_FAMILY_COUNT:
            raise RuntimeError(f"{family} 第四批因子名称数量异常: {len(block)}")
        blocks.append(block)
    names = tuple(name for block in blocks for name in block)
    if len(names) != FOURTH_FAMILY_TOTAL_COUNT or len(set(names)) != len(names):
        raise RuntimeError("五类第四批扩展因子名称数量或唯一性异常")
    return names


def _parse_name(name: str) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in FOURTH_FAMILY_PREFIXES:
        return None
    family, source, transform = parts[:3]
    if source not in FOURTH_FAMILY_SOURCES[family] or transform not in FOURTH_FAMILY_TRANSFORMS:
        return None
    try:
        window = int(parts[3][1:])
        lag = int(parts[4][1:])
        scale = int(parts[5][1:])
    except (ValueError, IndexError):
        return None
    if (
        window not in FOURTH_FAMILY_WINDOWS
        or lag not in FOURTH_FAMILY_LAGS
        or scale not in FOURTH_FAMILY_SCALES
    ):
        return None
    return family, source, transform, window, lag, scale


def _non_cross_components(market: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """构造强调流动性、成交确认和局部加速度的单品种中间量。"""
    return {
        "returnliquidity": market["ret"] / (1.0 + market["illiq"].abs()),
        "rangepressure": market["range"] * market["pressure"],
        "bodyvolume": market["body"] * market["volchg"],
        "locationvolume": market["location"] * market["volchg"],
        "gapvolume": market["gap"] * market["volchg"],
        "wickpressure": market["wickbalance"] * market["pressure"],
        "returnamount": market["ret"] * market["amtchg"],
        "rangeilliquidity": market["range"] * market["illiq"],
        "bodyaccel": market["body"].diff() * np.sign(market["ret"]),
        "pressureaccel": market["pressure"].diff(),
    }


def _transform(value: pd.Series, transform: str, window: int, scale: int) -> pd.Series:
    slow = max(window + 1, window * scale)
    step = max(1, scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow).std().replace(0, np.nan)
    if transform == "dualmomentum":
        return (fast_mean - slow_mean) / slow_std
    if transform == "downsidez":
        downside_std = value.where(value < 0, 0.0).rolling(slow).std().replace(0, np.nan)
        return (value - fast_mean) / downside_std
    if transform == "upsidez":
        upside_std = value.where(value > 0, 0.0).rolling(slow).std().replace(0, np.nan)
        return (value - fast_mean) / upside_std
    if transform == "semivolratio":
        upside_std = value.where(value > 0, 0.0).rolling(slow).std()
        downside_std = value.where(value < 0, 0.0).rolling(slow).std().replace(0, np.nan)
        return upside_std / downside_std - 1.0
    if transform == "changepersist":
        change = value.diff(step)
        return np.sign(change).rolling(slow).mean() * change.abs().rolling(window).mean()
    if transform == "shockdecay":
        center = value.ewm(span=slow, adjust=False, min_periods=window).mean()
        deviation = (value - center).abs().ewm(
            span=slow,
            adjust=False,
            min_periods=window,
        ).mean().replace(0, np.nan)
        return (value - center) / deviation
    if transform == "curvature":
        slope = fast_mean.diff(step)
        return (slope - slope.shift(step)) / slow_std
    if transform == "efficiencygap":
        fast_path = value.diff().abs().rolling(window).sum().replace(0, np.nan)
        slow_path = value.diff().abs().rolling(slow).sum().replace(0, np.nan)
        fast_efficiency = value.diff(window).abs() / fast_path
        slow_efficiency = value.diff(slow).abs() / slow_path
        return fast_efficiency - slow_efficiency
    if transform == "quantileskew":
        lower = value.rolling(slow).quantile(0.1)
        median = value.rolling(slow).median()
        upper = value.rolling(slow).quantile(0.9)
        return (upper + lower - 2.0 * median) / (upper - lower).replace(0, np.nan)
    fast_corr = value.rolling(window).corr(value.shift(step))
    slow_corr = value.rolling(slow).corr(value.shift(step))
    return fast_corr - slow_corr


def add_fourth_family_expansion_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | list[str] | None,
    related_data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """仅计算明确请求的五类第四批扩展因子。"""
    parsed = [(name, _parse_name(name)) for name in dict.fromkeys(requested_factors or [])]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)

    requested_families = {spec[0] for _, spec in parsed}
    market = _market_components(df)
    source_maps: dict[str, dict[str, pd.Series]] = {}
    if "paramw" in requested_families:
        source_maps["paramw"] = {name: market[name] for name in FOURTH_FAMILY_SOURCES["paramw"]}
    if "expanded5" in requested_families:
        source_maps["expanded5"] = {
            name: market[name] for name in FOURTH_FAMILY_SOURCES["expanded5"]
        }
    if "noncrossw" in requested_families:
        source_maps["noncrossw"] = _non_cross_components(market)
    if "calendarw" in requested_families:
        source_maps["calendarw"] = _calendar_components(df)
    if "crossw" in requested_families:
        source_maps["crossw"] = _cross_components(df, related_data_map or {}, config)

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

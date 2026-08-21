from __future__ import annotations

"""五类各两万个第二批按需扩展因子。"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore
from .family_expansion import _calendar_components, _cross_components, _market_components


SECOND_FAMILY_COUNT = 20_000
SECOND_FAMILY_TOTAL_COUNT = 100_000
SECOND_FAMILY_PREFIXES = ("crossy", "noncrossy", "expanded3", "paramy", "calendary")
SECOND_FAMILY_WINDOWS = (3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
SECOND_FAMILY_LAGS = (0, 1, 2, 5)
SECOND_FAMILY_SCALES = (1, 2, 3, 4, 6)
SECOND_FAMILY_TRANSFORMS = (
    "mediangap",
    "madz",
    "ewmshock",
    "qposition",
    "volmean",
    "signpersist",
    "downshare",
    "changez",
    "trendquality",
    "revstretch",
)
SECOND_FAMILY_SOURCES = {
    "crossy": (
        "relret", "relabsret", "relvol", "relrange", "returnspread",
        "rangespread", "volspread", "basketmom", "dispersion", "leadgap",
    ),
    "noncrossy": (
        "retrange", "bodypressure", "wickbody", "volpressure", "amtpressure",
        "illiqmomentum", "locationaccel", "rangeaccel", "volumerange", "gappressure",
    ),
    "expanded3": (
        "absret", "signedrange", "bodyrange", "wickbalance", "retvol",
        "retamt", "pressurechg", "locationmom", "illiqsign", "trangeaccel",
    ),
    "paramy": (
        "ret", "gap", "range", "body", "location",
        "volchg", "amtchg", "pressure", "illiq", "vwapgap",
    ),
    "calendary": (
        "dowsin", "dowcos", "monthsin", "monthcos", "doysin",
        "doycos", "monthend", "quarterend", "weekphase", "yearphase",
    ),
}


@lru_cache(maxsize=1)
def get_second_family_expansion_names() -> tuple[str, ...]:
    """返回五个连续名称块，每块严格两万个。"""
    blocks: list[tuple[str, ...]] = []
    for family in SECOND_FAMILY_PREFIXES:
        block = tuple(
            f"{family}_{source}_{transform}_w{window}_l{lag}_s{scale}"
            for source in SECOND_FAMILY_SOURCES[family]
            for transform in SECOND_FAMILY_TRANSFORMS
            for window in SECOND_FAMILY_WINDOWS
            for lag in SECOND_FAMILY_LAGS
            for scale in SECOND_FAMILY_SCALES
        )
        if len(block) != SECOND_FAMILY_COUNT:
            raise RuntimeError(f"{family} 第二批因子名称数量异常: {len(block)}")
        blocks.append(block)
    names = tuple(name for block in blocks for name in block)
    if len(names) != SECOND_FAMILY_TOTAL_COUNT or len(set(names)) != len(names):
        raise RuntimeError("五类第二批扩展因子名称数量或唯一性异常")
    return names


def _parse_name(name: str) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in SECOND_FAMILY_PREFIXES:
        return None
    family, source, transform = parts[:3]
    if source not in SECOND_FAMILY_SOURCES[family] or transform not in SECOND_FAMILY_TRANSFORMS:
        return None
    try:
        window = int(parts[3][1:])
        lag = int(parts[4][1:])
        scale = int(parts[5][1:])
    except (ValueError, IndexError):
        return None
    if (
        window not in SECOND_FAMILY_WINDOWS
        or lag not in SECOND_FAMILY_LAGS
        or scale not in SECOND_FAMILY_SCALES
    ):
        return None
    return family, source, transform, window, lag, scale


def _non_cross_components(market: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {
        "retrange": market["ret"] * market["range"],
        "bodypressure": market["body"] * market["pressure"],
        "wickbody": market["wickbalance"] * market["body"],
        "volpressure": market["volchg"] * market["pressure"],
        "amtpressure": market["amtchg"] * market["pressure"],
        "illiqmomentum": market["illiq"] * market["ret"],
        "locationaccel": market["location"].diff() * market["ret"],
        "rangeaccel": market["range"].diff(),
        "volumerange": market["volchg"] * market["range"],
        "gappressure": market["gap"] * market["pressure"],
    }


def _transform(value: pd.Series, transform: str, window: int, scale: int) -> pd.Series:
    slow = max(window + 1, window * scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow).std().replace(0, np.nan)
    slow_median = value.rolling(slow).median()
    if transform == "mediangap":
        return value.rolling(window).median() - slow_median
    if transform == "madz":
        mad = (value - slow_median).abs().rolling(slow).median().replace(0, np.nan)
        return (value - slow_median) / mad
    if transform == "ewmshock":
        ewm = value.ewm(span=slow, adjust=False, min_periods=max(2, window)).mean()
        return (value - ewm) / slow_std
    if transform == "qposition":
        lower = value.rolling(slow).quantile(0.25)
        upper = value.rolling(slow).quantile(0.75)
        return (value - lower) / (upper - lower).replace(0, np.nan) - 0.5
    if transform == "volmean":
        return fast_mean / slow_std
    if transform == "signpersist":
        return np.sign(value).rolling(slow).mean()
    if transform == "downshare":
        absolute_sum = value.abs().rolling(slow).sum().replace(0, np.nan)
        return value.where(value < 0, 0.0).abs().rolling(slow).sum() / absolute_sum
    if transform == "changez":
        change = value.diff(max(1, scale))
        return change / change.rolling(slow).std().replace(0, np.nan)
    if transform == "trendquality":
        absolute_sum = value.abs().rolling(slow).sum().replace(0, np.nan)
        return value.rolling(slow).sum() / absolute_sum
    return -(value - fast_mean) / fast_std * np.sign(fast_mean - slow_mean)


def add_second_family_expansion_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | list[str] | None,
    related_data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """仅计算明确请求的五类第二批扩展因子。"""
    parsed = [(name, _parse_name(name)) for name in dict.fromkeys(requested_factors or [])]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)

    requested_families = {spec[0] for _, spec in parsed}
    market = _market_components(df)
    source_maps: dict[str, dict[str, pd.Series]] = {}
    if "paramy" in requested_families:
        source_maps["paramy"] = {name: market[name] for name in SECOND_FAMILY_SOURCES["paramy"]}
    if "expanded3" in requested_families:
        source_maps["expanded3"] = {
            name: market[name] for name in SECOND_FAMILY_SOURCES["expanded3"]
        }
    if "noncrossy" in requested_families:
        source_maps["noncrossy"] = _non_cross_components(market)
    if "calendary" in requested_families:
        source_maps["calendary"] = _calendar_components(df)
    if "crossy" in requested_families:
        source_maps["crossy"] = _cross_components(df, related_data_map or {}, config)

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

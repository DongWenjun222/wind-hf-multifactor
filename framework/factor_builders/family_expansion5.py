from __future__ import annotations

"""Fifth on-demand expansion block with 20,000 factors per family."""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore
from .family_expansion import _calendar_components, _cross_components, _market_components
from .family_expansion4 import _non_cross_components


FIFTH_FAMILY_COUNT = 20_000
FIFTH_FAMILY_TOTAL_COUNT = 100_000
FIFTH_FAMILY_PREFIXES = ("crossv", "noncrossv", "expanded6", "paramv", "calendarv")
FIFTH_FAMILY_WINDOWS = (3, 6, 9, 14, 22, 35, 57, 92, 149, 241)
FIFTH_FAMILY_LAGS = (0, 2, 6, 12)
FIFTH_FAMILY_SCALES = (1, 2, 4, 6, 9)
FIFTH_FAMILY_TRANSFORMS = (
    "medianrevert",
    "robustmomentum",
    "voladjustedchange",
    "tailbalance",
    "signentropy",
    "trendstrength",
    "drawdownpressure",
    "breakoutdistance",
    "ewmgap",
    "serialshock",
)
FIFTH_FAMILY_SOURCES = {
    "crossv": (
        "relret", "relabsret", "relvol", "relrange", "returnspread",
        "rangespread", "volspread", "basketmom", "dispersion", "leadgap",
    ),
    "noncrossv": (
        "returnliquidity", "rangepressure", "bodyvolume", "locationvolume",
        "gapvolume", "wickpressure", "returnamount", "rangeilliquidity",
        "bodyaccel", "pressureaccel",
    ),
    "expanded6": (
        "absret", "signedrange", "bodyrange", "wickbalance", "retvol",
        "retamt", "pressurechg", "locationmom", "illiqsign", "trangeaccel",
    ),
    "paramv": (
        "ret", "gap", "range", "body", "location",
        "volchg", "amtchg", "pressure", "illiq", "vwapgap",
    ),
    "calendarv": (
        "dowsin", "dowcos", "monthsin", "monthcos", "doysin",
        "doycos", "monthend", "quarterend", "weekphase", "yearphase",
    ),
}


@lru_cache(maxsize=1)
def get_fifth_family_expansion_names() -> tuple[str, ...]:
    """Return five stable contiguous blocks containing 20,000 names each."""
    blocks: list[tuple[str, ...]] = []
    for family in FIFTH_FAMILY_PREFIXES:
        block = tuple(
            f"{family}_{source}_{transform}_w{window}_l{lag}_s{scale}"
            for source in FIFTH_FAMILY_SOURCES[family]
            for transform in FIFTH_FAMILY_TRANSFORMS
            for window in FIFTH_FAMILY_WINDOWS
            for lag in FIFTH_FAMILY_LAGS
            for scale in FIFTH_FAMILY_SCALES
        )
        if len(block) != FIFTH_FAMILY_COUNT:
            raise RuntimeError(f"Unexpected fifth-block count for {family}: {len(block)}")
        blocks.append(block)
    names = tuple(name for block in blocks for name in block)
    if len(names) != FIFTH_FAMILY_TOTAL_COUNT or len(set(names)) != len(names):
        raise RuntimeError("Fifth expansion names are not complete and unique")
    return names


def _parse_name(name: str) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in FIFTH_FAMILY_PREFIXES:
        return None
    family, source, transform = parts[:3]
    if source not in FIFTH_FAMILY_SOURCES[family] or transform not in FIFTH_FAMILY_TRANSFORMS:
        return None
    try:
        window = int(parts[3][1:])
        lag = int(parts[4][1:])
        scale = int(parts[5][1:])
    except (ValueError, IndexError):
        return None
    if (
        window not in FIFTH_FAMILY_WINDOWS
        or lag not in FIFTH_FAMILY_LAGS
        or scale not in FIFTH_FAMILY_SCALES
    ):
        return None
    return family, source, transform, window, lag, scale


def _transform(value: pd.Series, transform: str, window: int, scale: int) -> pd.Series:
    slow = max(window + 1, window * scale)
    step = max(1, scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow).mean()
    slow_std = value.rolling(slow).std().replace(0, np.nan)
    median = value.rolling(slow).median()

    if transform == "medianrevert":
        mad = (value - median).abs().rolling(slow).median().replace(0, np.nan)
        return (median - value) / mad
    if transform == "robustmomentum":
        lower = value.rolling(slow).quantile(0.25)
        upper = value.rolling(slow).quantile(0.75)
        return (fast_mean - slow_mean) / (upper - lower).replace(0, np.nan)
    if transform == "voladjustedchange":
        return value.diff(step) / slow_std
    if transform == "tailbalance":
        lower = value.rolling(slow).quantile(0.1)
        upper = value.rolling(slow).quantile(0.9)
        return ((value > upper).astype(float) - (value < lower).astype(float)).rolling(window).mean()
    if transform == "signentropy":
        positive_rate = value.diff().gt(0).rolling(slow).mean().clip(1e-8, 1 - 1e-8)
        entropy = -positive_rate * np.log(positive_rate) - (1 - positive_rate) * np.log(1 - positive_rate)
        return np.sign(fast_mean - slow_mean) * (np.log(2.0) - entropy)
    if transform == "trendstrength":
        path = value.diff().abs().rolling(slow).sum().replace(0, np.nan)
        return value.diff(slow) / path
    if transform == "drawdownpressure":
        rolling_max = value.rolling(slow).max()
        drawdown = value - rolling_max
        return drawdown / slow_std
    if transform == "breakoutdistance":
        lower = value.rolling(slow).min()
        upper = value.rolling(slow).max()
        return (value - lower) / (upper - lower).replace(0, np.nan) - 0.5
    if transform == "ewmgap":
        fast_ewm = value.ewm(span=window, adjust=False, min_periods=window).mean()
        slow_ewm = value.ewm(span=slow, adjust=False, min_periods=window).mean()
        return (fast_ewm - slow_ewm) / slow_std
    shock = value.diff(step) / slow_std
    return shock * np.sign(shock.shift(step))


def add_fifth_family_expansion_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | list[str] | None,
    related_data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build only explicitly requested fifth-block factors."""
    parsed = [(name, _parse_name(name)) for name in dict.fromkeys(requested_factors or [])]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)

    requested_families = {spec[0] for _, spec in parsed}
    market = _market_components(df)
    source_maps: dict[str, dict[str, pd.Series]] = {}
    if "paramv" in requested_families:
        source_maps["paramv"] = {name: market[name] for name in FIFTH_FAMILY_SOURCES["paramv"]}
    if "expanded6" in requested_families:
        source_maps["expanded6"] = {
            name: market[name] for name in FIFTH_FAMILY_SOURCES["expanded6"]
        }
    if "noncrossv" in requested_families:
        source_maps["noncrossv"] = _non_cross_components(market)
    if "calendarv" in requested_families:
        source_maps["calendarv"] = _calendar_components(df)
    if "crossv" in requested_families:
        source_maps["crossv"] = _cross_components(df, related_data_map or {}, config)

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

from __future__ import annotations

"""五类各一万个第三批按需扩展因子。"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore
from .family_expansion import _calendar_components, _cross_components, _market_components


THIRD_FAMILY_COUNT = 10_000
THIRD_FAMILY_TOTAL_COUNT = 50_000
THIRD_FAMILY_PREFIXES = ("crossz", "noncrossz", "expanded4", "paramz", "calendarz")
THIRD_FAMILY_WINDOWS = (4, 6, 9, 14, 22, 35, 56, 90, 145, 234)
THIRD_FAMILY_LAGS = (0, 3)
THIRD_FAMILY_SCALES = (1, 2, 3, 5, 8)
THIRD_FAMILY_TRANSFORMS = (
    "ewmspread",
    "robustslope",
    "voladjusted",
    "tailbalance",
    "meanrevert",
    "breakoutdist",
    "signentropy",
    "acceleration",
    "corrdecay",
    "drawbalance",
)
THIRD_FAMILY_SOURCES = {
    "crossz": (
        "relret", "relabsret", "relvol", "relrange", "returnspread",
        "rangespread", "volspread", "basketmom", "dispersion", "leadgap",
    ),
    "noncrossz": (
        "returnpressure", "rangeimbalance", "liquidityshock", "vwapmomentum",
        "volumeaccel", "amountaccel", "bodylocation", "gapreversal",
        "pressurevol", "trueaccel",
    ),
    "expanded4": (
        "absret", "signedrange", "bodyrange", "wickbalance", "retvol",
        "retamt", "pressurechg", "locationmom", "illiqsign", "trangeaccel",
    ),
    "paramz": (
        "ret", "gap", "range", "body", "location",
        "volchg", "amtchg", "pressure", "illiq", "vwapgap",
    ),
    "calendarz": (
        "dowsin", "dowcos", "monthsin", "monthcos", "doysin",
        "doycos", "monthend", "quarterend", "weekphase", "yearphase",
    ),
}


@lru_cache(maxsize=1)
def get_third_family_expansion_names() -> tuple[str, ...]:
    """返回五个连续名称块，每块严格一万个。"""
    blocks: list[tuple[str, ...]] = []
    for family in THIRD_FAMILY_PREFIXES:
        block = tuple(
            f"{family}_{source}_{transform}_w{window}_l{lag}_s{scale}"
            for source in THIRD_FAMILY_SOURCES[family]
            for transform in THIRD_FAMILY_TRANSFORMS
            for window in THIRD_FAMILY_WINDOWS
            for lag in THIRD_FAMILY_LAGS
            for scale in THIRD_FAMILY_SCALES
        )
        if len(block) != THIRD_FAMILY_COUNT:
            raise RuntimeError(f"{family} 第三批因子名称数量异常: {len(block)}")
        blocks.append(block)
    names = tuple(name for block in blocks for name in block)
    if len(names) != THIRD_FAMILY_TOTAL_COUNT or len(set(names)) != len(names):
        raise RuntimeError("五类第三批扩展因子名称数量或唯一性异常")
    return names


def _parse_name(name: str) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in THIRD_FAMILY_PREFIXES:
        return None
    family, source, transform = parts[:3]
    if source not in THIRD_FAMILY_SOURCES[family] or transform not in THIRD_FAMILY_TRANSFORMS:
        return None
    try:
        window = int(parts[3][1:])
        lag = int(parts[4][1:])
        scale = int(parts[5][1:])
    except (ValueError, IndexError):
        return None
    if (
        window not in THIRD_FAMILY_WINDOWS
        or lag not in THIRD_FAMILY_LAGS
        or scale not in THIRD_FAMILY_SCALES
    ):
        return None
    return family, source, transform, window, lag, scale


def _non_cross_components(market: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """构造比第二批更强调交互和加速度的单品种中间量。"""
    return {
        "returnpressure": market["ret"] * market["pressure"],
        "rangeimbalance": market["signedrange"] * market["wickbalance"],
        "liquidityshock": market["illiq"].diff(),
        "vwapmomentum": market["vwapgap"] * market["ret"],
        "volumeaccel": market["volchg"].diff(),
        "amountaccel": market["amtchg"].diff(),
        "bodylocation": market["body"] * market["location"],
        "gapreversal": -market["gap"] * market["ret"],
        "pressurevol": market["pressure"] * market["absret"],
        "trueaccel": market["trangeaccel"] * np.sign(market["ret"]),
    }


def _transform(value: pd.Series, transform: str, window: int, scale: int) -> pd.Series:
    slow = max(window + 1, window * scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow).std().replace(0, np.nan)
    if transform == "ewmspread":
        fast_ewm = value.ewm(span=window, adjust=False, min_periods=window).mean()
        slow_ewm = value.ewm(span=slow, adjust=False, min_periods=window).mean()
        return (fast_ewm - slow_ewm) / slow_std
    if transform == "robustslope":
        median = value.rolling(slow).median()
        mad = (value - median).abs().rolling(slow).median().replace(0, np.nan)
        return median.diff(max(1, scale)) / mad
    if transform == "voladjusted":
        return fast_mean / fast_std - slow_mean / slow_std
    if transform == "tailbalance":
        upper = value.rolling(slow).quantile(0.8)
        lower = value.rolling(slow).quantile(0.2)
        return (value.gt(upper).astype(float) - value.lt(lower).astype(float)).rolling(window).mean()
    if transform == "meanrevert":
        trend_strength = (fast_mean - slow_mean).abs() / slow_std
        return -(value - fast_mean) / fast_std * trend_strength
    if transform == "breakoutdist":
        lower = value.rolling(slow).min()
        upper = value.rolling(slow).max()
        return (value - lower) / (upper - lower).replace(0, np.nan) - 0.5
    if transform == "signentropy":
        positive_share = value.gt(0).astype(float).rolling(slow).mean().clip(1e-6, 1 - 1e-6)
        entropy = -(
            positive_share * np.log(positive_share)
            + (1 - positive_share) * np.log(1 - positive_share)
        )
        return np.sign(fast_mean) * (np.log(2.0) - entropy)
    if transform == "acceleration":
        step = max(1, scale)
        return fast_mean.diff(step) - slow_mean.diff(step)
    if transform == "corrdecay":
        step = max(1, scale)
        fast_corr = value.rolling(window).corr(value.shift(step))
        slow_corr = value.rolling(slow).corr(value.shift(step))
        return fast_corr - slow_corr
    positive = value.clip(lower=0).rolling(slow).sum()
    negative = -value.clip(upper=0).rolling(slow).sum()
    return (positive - negative) / (positive + negative).replace(0, np.nan)


def add_third_family_expansion_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | list[str] | None,
    related_data_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """仅计算明确请求的五类第三批扩展因子。"""
    parsed = [(name, _parse_name(name)) for name in dict.fromkeys(requested_factors or [])]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)

    requested_families = {spec[0] for _, spec in parsed}
    market = _market_components(df)
    source_maps: dict[str, dict[str, pd.Series]] = {}
    if "paramz" in requested_families:
        source_maps["paramz"] = {name: market[name] for name in THIRD_FAMILY_SOURCES["paramz"]}
    if "expanded4" in requested_families:
        source_maps["expanded4"] = {
            name: market[name] for name in THIRD_FAMILY_SOURCES["expanded4"]
        }
    if "noncrossz" in requested_families:
        source_maps["noncrossz"] = _non_cross_components(market)
    if "calendarz" in requested_families:
        source_maps["calendarz"] = _calendar_components(df)
    if "crossz" in requested_families:
        source_maps["crossz"] = _cross_components(df, related_data_map or {}, config)

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

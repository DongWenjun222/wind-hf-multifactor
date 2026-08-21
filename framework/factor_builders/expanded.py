from __future__ import annotations

"""二十万级扩展量价因子名称空间。

该家族只支持按需构建。名称目录可以廉价生成稳定名称，但实际计算仅
处理 ``requested_factors`` 中出现的列，避免在长行情上创建不可承受的宽矩阵。
"""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore


LEGACY_EXPANDED_FACTOR_COUNT = 100_000
SECOND_EXPANDED_FACTOR_COUNT = 100_000
EXPANDED_FACTOR_COUNT = LEGACY_EXPANDED_FACTOR_COUNT + SECOND_EXPANDED_FACTOR_COUNT
EXPANDED_WINDOWS = (2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233)
EXPANDED_LAGS = (0, 1, 2, 3, 5)
EXPANDED_SCALES = (1, 2, 3, 4, 6)
EXPANDED_SOURCES = (
    "retcc", "retoc", "gap", "range", "body", "upwick", "lowwick", "location",
    "volchg", "amtchg", "signedvol", "signedamt", "vwapgap", "pressure", "illiq",
    "trange", "typret", "hlgap", "retvol", "rangevol",
)
EXPANDED_TRANSFORMS = (
    "meangap", "stdratio", "zdist", "rank", "ewmgap",
    "momgap", "slope", "skewgap", "autocorr", "efficiency",
)
EXPANDED2_TRANSFORMS = (
    "absmean", "volsign", "zchange", "rankchange", "ewmaccel",
    "momratio", "sloperatio", "kurtgap", "partialauto", "trendquality",
)


@lru_cache(maxsize=1)
def get_expanded_factor_names() -> tuple[str, ...]:
    """返回稳定有序的二十万个扩展因子名称，旧十万名称顺序保持不变。"""
    legacy_names = tuple(
        f"expanded_{source}_{transform}_w{window}_l{lag}_s{scale}"
        for source in EXPANDED_SOURCES
        for transform in EXPANDED_TRANSFORMS
        for window in EXPANDED_WINDOWS
        for lag in EXPANDED_LAGS
        for scale in EXPANDED_SCALES
    )
    second_names = tuple(
        f"expanded2_{source}_{transform}_w{window}_l{lag}_s{scale}"
        for source in EXPANDED_SOURCES
        for transform in EXPANDED2_TRANSFORMS
        for window in EXPANDED_WINDOWS
        for lag in EXPANDED_LAGS
        for scale in EXPANDED_SCALES
    )
    if len(legacy_names) != LEGACY_EXPANDED_FACTOR_COUNT:
        raise RuntimeError(f"旧扩展因子名称数量异常: {len(legacy_names)}")
    if len(second_names) != SECOND_EXPANDED_FACTOR_COUNT:
        raise RuntimeError(f"第二代扩展因子名称数量异常: {len(second_names)}")
    names = legacy_names + second_names
    if len(names) != EXPANDED_FACTOR_COUNT:
        raise RuntimeError(f"扩展因子名称数量异常: {len(names)}")
    return names


def _parse_expanded_factor_name(
    name: str,
) -> tuple[str, str, str, int, int, int] | None:
    parts = str(name).split("_")
    if len(parts) != 6 or parts[0] not in {"expanded", "expanded2"}:
        return None
    family, source, transform = parts[0], parts[1], parts[2]
    valid_transforms = EXPANDED_TRANSFORMS if family == "expanded" else EXPANDED2_TRANSFORMS
    if source not in EXPANDED_SOURCES or transform not in valid_transforms:
        return None
    try:
        window = int(parts[3][1:]) if parts[3].startswith("w") else -1
        lag = int(parts[4][1:]) if parts[4].startswith("l") else -1
        scale = int(parts[5][1:]) if parts[5].startswith("s") else -1
    except ValueError:
        return None
    if window not in EXPANDED_WINDOWS or lag not in EXPANDED_LAGS or scale not in EXPANDED_SCALES:
        return None
    return family, source, transform, window, lag, scale


def _build_source_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    volume = df.get("volume", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    retcc = df.get("bar_return_cc", close.pct_change())
    retoc = df.get("bar_return_oc", close / open_price - 1.0)
    spread = (high - low).replace(0, np.nan)
    range_pct = spread / open_price
    body = (close - open_price) / open_price
    upwick = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    lowwick = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    location = ((close - low) - (high - close)) / spread
    volchg = volume.pct_change()
    amtchg = amount.pct_change()
    typical = (high + low + close) / 3.0
    vwap = amount / volume
    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1) / open_price
    return {
        "retcc": retcc,
        "retoc": retoc,
        "gap": open_price / close.shift(1) - 1.0,
        "range": range_pct,
        "body": body,
        "upwick": upwick,
        "lowwick": lowwick,
        "location": location,
        "volchg": volchg,
        "amtchg": amtchg,
        "signedvol": np.sign(retcc).fillna(0.0) * volchg,
        "signedamt": np.sign(retcc).fillna(0.0) * amtchg,
        "vwapgap": close / vwap.replace(0, np.nan) - 1.0,
        "pressure": location * range_pct,
        "illiq": retcc.abs() / amount,
        "trange": true_range,
        "typret": typical.pct_change(),
        "hlgap": high.pct_change() - low.pct_change(),
        "retvol": retcc * volchg,
        "rangevol": range_pct * volchg,
    }


def _transform_source(
    value: pd.Series,
    transform: str,
    window: int,
    scale: int,
) -> pd.Series:
    slow_window = max(window + 1, window * scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow_window).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow_window).std().replace(0, np.nan)
    if transform == "meangap":
        return fast_mean - slow_mean
    if transform == "stdratio":
        return fast_std / slow_std - 1.0
    if transform == "zdist":
        return (value - slow_mean) / slow_std
    if transform == "rank":
        return value.rolling(slow_window).rank(pct=True) - 0.5
    if transform == "ewmgap":
        fast_ewm = value.ewm(span=window, adjust=False, min_periods=window).mean()
        slow_ewm = value.ewm(span=slow_window, adjust=False, min_periods=window).mean()
        return fast_ewm - slow_ewm
    if transform == "momgap":
        return value.rolling(window).sum() - value.rolling(slow_window).sum() * window / slow_window
    if transform == "slope":
        return fast_mean.diff(max(1, scale))
    if transform == "skewgap":
        return value.rolling(window).skew() - value.rolling(slow_window).skew()
    if transform == "autocorr":
        return value.rolling(slow_window).corr(value.shift(scale))
    fast_sum = value.rolling(window).sum()
    slow_abs_sum = value.abs().rolling(slow_window).sum().replace(0, np.nan)
    return fast_sum / slow_abs_sum


def _transform_source_v2(
    value: pd.Series,
    transform: str,
    window: int,
    scale: int,
) -> pd.Series:
    """第二代非线性与稳定性变换，和旧扩展家族保持计算差异。"""
    slow_window = max(window + 1, window * scale)
    fast_mean = value.rolling(window).mean()
    slow_mean = value.rolling(slow_window).mean()
    fast_std = value.rolling(window).std().replace(0, np.nan)
    slow_std = value.rolling(slow_window).std().replace(0, np.nan)
    if transform == "absmean":
        return value.abs().rolling(window).mean() - value.abs().rolling(slow_window).mean()
    if transform == "volsign":
        return np.sign(fast_mean) * (fast_std / slow_std - 1.0)
    if transform == "zchange":
        zscore = (value - slow_mean) / slow_std
        return zscore.diff(max(1, scale))
    if transform == "rankchange":
        rank = value.rolling(slow_window).rank(pct=True)
        return rank - rank.shift(max(1, scale))
    if transform == "ewmaccel":
        fast_ewm = value.ewm(span=window, adjust=False, min_periods=window).mean()
        slow_ewm = value.ewm(span=slow_window, adjust=False, min_periods=window).mean()
        return fast_ewm.diff(max(1, scale)) - slow_ewm.diff(max(1, scale))
    if transform == "momratio":
        fast_sum = value.rolling(window).sum()
        slow_abs_sum = value.abs().rolling(slow_window).sum().replace(0, np.nan)
        return fast_sum / slow_abs_sum
    if transform == "sloperatio":
        return fast_mean.diff(max(1, scale)) / slow_std
    if transform == "kurtgap":
        return value.rolling(window).kurt() - value.rolling(slow_window).kurt()
    if transform == "partialauto":
        first_corr = value.rolling(slow_window).corr(value.shift(max(1, scale)))
        second_corr = value.rolling(slow_window).corr(value.shift(max(2, scale + 1)))
        return first_corr - second_corr
    direction = value.rolling(window).sum().abs()
    path = value.abs().rolling(slow_window).sum().replace(0, np.nan)
    return np.sign(fast_mean) * direction / path


def add_expanded_factors(
    df: pd.DataFrame,
    config: Any,
    requested_factors: set[str] | None = None,
) -> pd.DataFrame:
    """仅计算明确请求的扩展因子。"""
    requested = list(dict.fromkeys(requested_factors or []))
    if not requested:
        return pd.DataFrame(index=df.index)
    parsed = [(name, _parse_expanded_factor_name(name)) for name in requested]
    parsed = [(name, spec) for name, spec in parsed if spec is not None]
    if not parsed:
        return pd.DataFrame(index=df.index)

    sources = _build_source_series(df)
    transformed_cache: dict[tuple[str, str, str, int, int, int], pd.Series] = {}
    columns: dict[str, pd.Series] = {}
    for name, spec in parsed:
        family, source, transform, window, lag, scale = spec
        cache_key = (family, source, transform, window, lag, scale)
        if cache_key not in transformed_cache:
            shifted_source = sources[source].shift(lag)
            raw = (
                _transform_source(shifted_source, transform, window, scale)
                if family == "expanded"
                else _transform_source_v2(shifted_source, transform, window, scale)
            )
            transformed_cache[cache_key] = rolling_zscore(raw, config.zscore_window)
        columns[name] = transformed_cache[cache_key]
    return pd.DataFrame(columns, index=df.index)

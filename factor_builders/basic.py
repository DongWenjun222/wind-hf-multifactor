from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import rolling_zscore


def add_basic_factors(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    """生成最基础的单品种量价因子。"""
    momentum_raw = df["close"].pct_change(12)
    reversal_raw = -(df["close"] / df["close"].rolling(6).mean() - 1.0)

    rolling_high = df["high"].rolling(20).max()
    rolling_low = df["low"].rolling(20).min()
    price_range = (rolling_high - rolling_low).replace(0, np.nan)
    breakout_raw = ((df["close"] - rolling_low) / price_range - 0.5) * 2.0

    volume_ratio = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    volume_confirm_raw = np.sign(df["bar_return_cc"]) * (volume_ratio - 1.0)

    return pd.DataFrame(
        {
            "momentum": rolling_zscore(momentum_raw, config.zscore_window),
            "reversal": rolling_zscore(reversal_raw, config.zscore_window),
            "breakout": rolling_zscore(breakout_raw, config.zscore_window),
            "volume_confirm": rolling_zscore(volume_confirm_raw, config.zscore_window),
        },
        index=df.index,
    )

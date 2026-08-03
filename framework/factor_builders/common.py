from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_symbol_name(symbol: str) -> str:
    """把 Wind 标的代码转换成适合用于因子名的安全片段。"""
    return symbol.replace(".", "_").replace("/", "_").replace("-", "_").lower()

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 zscore 标准化，并把极端值截断到 [-3, 3]。

    高频因子中异常值比较常见，截断可以降低极端点对模型和单因子信号的影响。
    """
    min_periods = max(20, window // 3)
    mean_ = series.rolling(window=window, min_periods=min_periods).mean()
    std_ = series.rolling(window=window, min_periods=min_periods).std()
    zscore = (series - mean_) / std_.replace(0, np.nan)
    return zscore.clip(-3, 3)

def align_related_data_to_main(
    related_data: pd.DataFrame,
    main_index: pd.Index,
    max_ffill_bars: int,
) -> pd.DataFrame:
    """把相关品种行情对齐到主标的时间轴，且不使用向后填充。"""
    aligned = related_data.reindex(main_index)
    if max_ffill_bars > 0:
        aligned = aligned.ffill(limit=max_ffill_bars)
    return aligned

def calculate_related_data_coverage(
    symbol: str,
    related_data: pd.DataFrame,
    main_index: pd.Index,
    max_ffill_bars: int,
) -> dict[str, Any]:
    """计算单个相关品种的数据覆盖率诊断。"""
    before_ffill = related_data.reindex(main_index)
    after_ffill = align_related_data_to_main(related_data, main_index, max_ffill_bars)
    before_valid = before_ffill["close"].notna() if "close" in before_ffill.columns else pd.Series(False, index=main_index)
    after_valid = after_ffill["close"].notna() if "close" in after_ffill.columns else pd.Series(False, index=main_index)
    ffill_added = after_valid & ~before_valid
    missing_after_ffill = ~after_valid

    if len(missing_after_ffill):
        missing_groups = missing_after_ffill.ne(missing_after_ffill.shift(fill_value=False)).cumsum()
        max_consecutive_missing = int(
            missing_after_ffill.groupby(missing_groups).sum().max()
        )
    else:
        max_consecutive_missing = 0

    clean_related_index = pd.DatetimeIndex(related_data.index).sort_values()
    main_datetime_index = pd.DatetimeIndex(main_index)
    if len(clean_related_index) and len(main_datetime_index):
        matched_positions = clean_related_index.searchsorted(main_datetime_index, side="right") - 1
        valid_match = matched_positions >= 0
        lag_minutes = pd.Series(np.nan, index=main_index, dtype="float64")
        if valid_match.any():
            matched_times = clean_related_index[matched_positions[valid_match]]
            lag_minutes.loc[valid_match] = (
                main_datetime_index[valid_match] - matched_times
            ).total_seconds() / 60.0
        usable_lag = lag_minutes.loc[after_valid]
    else:
        usable_lag = pd.Series(dtype="float64")

    return {
        "symbol": symbol,
        "raw_rows": int(len(related_data)),
        "raw_start": str(related_data.index.min()) if len(related_data) else "",
        "raw_end": str(related_data.index.max()) if len(related_data) else "",
        "raw_duplicate_timestamps": int(pd.Index(related_data.index).duplicated().sum()),
        "main_rows": int(len(main_index)),
        "direct_aligned_rows": int(before_valid.sum()),
        "ffill_added_rows": int(ffill_added.sum()),
        "usable_rows": int(after_valid.sum()),
        "missing_rows_after_ffill": int(missing_after_ffill.sum()),
        "max_consecutive_missing_after_ffill": max_consecutive_missing,
        "direct_coverage_rate": float(before_valid.mean()) if len(before_valid) else np.nan,
        "usable_coverage_rate": float(after_valid.mean()) if len(after_valid) else np.nan,
        "missing_rate_after_ffill": float(missing_after_ffill.mean()) if len(missing_after_ffill) else np.nan,
        "ffill_share_in_usable": float(ffill_added.sum() / after_valid.sum()) if after_valid.sum() else np.nan,
        "max_alignment_lag_minutes": float(usable_lag.max()) if not usable_lag.empty else np.nan,
        "avg_alignment_lag_minutes": float(usable_lag.mean()) if not usable_lag.empty else np.nan,
    }

def align_macro_daily_to_main(
    macro_data: pd.DataFrame,
    main_index: pd.Index,
    lag_daily_bars: int,
) -> pd.Series:
    """把日频宏观数据对齐到分钟线，并按配置滞后，避免使用当天未知数据。"""
    macro_close = macro_data["close"].sort_index().copy()
    lag_daily_bars = max(0, int(lag_daily_bars or 0))
    if lag_daily_bars:
        macro_close = macro_close.shift(lag_daily_bars)
    main_datetime_index = pd.DatetimeIndex(main_index)
    normalized_main_dates = main_datetime_index.normalize()
    aligned = macro_close.reindex(normalized_main_dates, method="ffill")
    return pd.Series(aligned.to_numpy(dtype="float64"), index=main_index)

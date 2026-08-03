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

def add_macro_state_factors(
    main_data: pd.DataFrame,
    macro_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """基于 Wind 日频宏观/指数/利率代理数据生成宏观状态因子。"""
    if not macro_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_return = main_close.pct_change()
    main_range = (main_data["high"] - main_data["low"]) / main_data["open"].replace(0, np.nan)
    windows = list(getattr(config, "macro_state_windows", []) or [])
    lag_daily_bars = max(0, int(getattr(config, "macro_state_lag_daily_bars", 1) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []

    for symbol, macro_data in macro_data_map.items():
        if macro_data.empty or "close" not in macro_data.columns:
            continue
        symbol_key = safe_symbol_name(symbol)
        macro_close = align_macro_daily_to_main(macro_data, main_data.index, lag_daily_bars)
        macro_return = macro_close.pct_change()
        macro_log_level = np.log(macro_close.replace(0, np.nan))
        macro_change = macro_close.diff()

        factor_specs.extend(
            [
                (f"macro_ret_1_{symbol_key}", macro_return),
                (f"macro_level_change_{symbol_key}", macro_change),
                (f"macro_log_level_{symbol_key}", macro_log_level),
                (f"macro_main_ret_spread_1_{symbol_key}", main_return - macro_return),
                (f"macro_main_ret_product_1_{symbol_key}", main_return * macro_return),
                (f"macro_range_product_1_{symbol_key}", main_range * macro_return.abs()),
            ]
        )

        for window in windows:
            min_periods = max(2, window // 3)
            macro_mean = macro_return.rolling(window, min_periods=min_periods).mean()
            macro_std = macro_return.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            macro_level_mean = macro_log_level.rolling(window, min_periods=min_periods).mean()
            macro_level_std = macro_log_level.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            main_vol = main_return.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            macro_vol = macro_return.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            macro_corr = main_return.rolling(window, min_periods=min_periods).corr(macro_return)
            macro_beta = main_return.rolling(window, min_periods=min_periods).cov(macro_return) / macro_return.rolling(window, min_periods=min_periods).var().replace(0, np.nan)
            factor_specs.extend(
                [
                    (f"macro_ret_mean_{symbol_key}_{window}", macro_mean),
                    (f"macro_ret_vol_{symbol_key}_{window}", macro_std),
                    (f"macro_ret_zdist_{symbol_key}_{window}", (macro_return - macro_mean) / macro_std),
                    (f"macro_level_zdist_{symbol_key}_{window}", (macro_log_level - macro_level_mean) / macro_level_std),
                    (f"macro_momentum_{symbol_key}_{window}", macro_close.pct_change(window)),
                    (f"macro_vol_spread_{symbol_key}_{window}", main_vol / macro_vol - 1.0),
                    (f"macro_corr_main_{symbol_key}_{window}", macro_corr),
                    (f"macro_beta_main_{symbol_key}_{window}", macro_beta),
                    (f"macro_beta_resid_{symbol_key}_{window}", (main_return - macro_beta * macro_return).rolling(window, min_periods=min_periods).mean()),
                    (f"macro_main_ret_spread_{symbol_key}_{window}", main_return.rolling(window, min_periods=min_periods).mean() - macro_mean),
                    (f"macro_risk_on_product_{symbol_key}_{window}", macro_return.rolling(window, min_periods=min_periods).mean() * main_range.rolling(window, min_periods=min_periods).mean()),
                    (f"macro_abs_shock_{symbol_key}_{window}", macro_return.abs() / macro_return.abs().rolling(window, min_periods=min_periods).mean().replace(0, np.nan) - 1.0),
                ]
            )

    if not factor_specs:
        return pd.DataFrame(index=main_data.index)
    macro_factors = {
        factor_name: rolling_zscore(raw_factor, config.zscore_window)
        for factor_name, raw_factor in factor_specs
    }
    return pd.DataFrame(macro_factors, index=main_data.index)

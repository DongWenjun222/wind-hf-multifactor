from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import align_macro_daily_to_main, rolling_zscore, safe_symbol_name


def add_external_daily_factors(
    main_data: pd.DataFrame,
    external_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """基于 Wind 外部日频数据生成商品基本面/期限结构类因子。

    外部数据会先按配置滞后，再对齐到分钟线，避免使用当天尚未公开的低频数据。
    """
    if not external_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_return = main_close.pct_change()
    main_range = (main_data["high"] - main_data["low"]) / main_data["open"].replace(0, np.nan)
    windows = list(getattr(config, "external_daily_windows", []) or [])
    default_lag = max(0, int(getattr(config, "external_daily_lag_daily_bars", 1) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []

    for source_name, external_data in external_data_map.items():
        if external_data.empty or "value" not in external_data.columns:
            continue
        source_key = safe_symbol_name(str(source_name))
        lag_daily_bars = int(external_data.attrs.get("lag_daily_bars", default_lag) or 0)
        daily_value = external_data.rename(columns={"value": "close"})
        aligned_value = align_macro_daily_to_main(daily_value, main_data.index, lag_daily_bars)
        value_change = aligned_value.diff()
        value_pct_change = aligned_value.pct_change()
        value_log = np.log(aligned_value.replace(0, np.nan))

        factor_specs.extend(
            [
                (f"external_level_{source_key}", aligned_value),
                (f"external_change_1_{source_key}", value_change),
                (f"external_pct_change_1_{source_key}", value_pct_change),
                (f"external_log_level_{source_key}", value_log),
                (f"external_main_ret_spread_1_{source_key}", main_return - value_pct_change),
                (f"external_main_ret_product_1_{source_key}", main_return * value_pct_change),
                (f"external_range_product_1_{source_key}", main_range * value_pct_change.abs()),
            ]
        )

        for window in windows:
            min_periods = max(2, window // 3)
            value_mean = aligned_value.rolling(window, min_periods=min_periods).mean()
            value_std = aligned_value.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            change_mean = value_change.rolling(window, min_periods=min_periods).mean()
            change_std = value_change.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            pct_mean = value_pct_change.rolling(window, min_periods=min_periods).mean()
            pct_std = value_pct_change.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            main_vol = main_return.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            external_vol = value_pct_change.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
            corr_main = main_return.rolling(window, min_periods=min_periods).corr(value_pct_change)
            beta_main = (
                main_return.rolling(window, min_periods=min_periods).cov(value_pct_change)
                / value_pct_change.rolling(window, min_periods=min_periods).var().replace(0, np.nan)
            )

            factor_specs.extend(
                [
                    (f"external_zdist_{source_key}_{window}", (aligned_value - value_mean) / value_std),
                    (f"external_change_mean_{source_key}_{window}", change_mean),
                    (f"external_change_zdist_{source_key}_{window}", (value_change - change_mean) / change_std),
                    (f"external_pct_mean_{source_key}_{window}", pct_mean),
                    (f"external_pct_vol_{source_key}_{window}", pct_std),
                    (f"external_pct_zdist_{source_key}_{window}", (value_pct_change - pct_mean) / pct_std),
                    (f"external_momentum_{source_key}_{window}", aligned_value.pct_change(window)),
                    (f"external_shock_abs_{source_key}_{window}", value_change.abs() / value_change.abs().rolling(window, min_periods=min_periods).mean().replace(0, np.nan) - 1.0),
                    (f"external_vol_spread_{source_key}_{window}", main_vol / external_vol - 1.0),
                    (f"external_corr_main_{source_key}_{window}", corr_main),
                    (f"external_beta_main_{source_key}_{window}", beta_main),
                    (f"external_beta_resid_{source_key}_{window}", (main_return - beta_main * value_pct_change).rolling(window, min_periods=min_periods).mean()),
                    (f"external_main_ret_spread_{source_key}_{window}", main_return.rolling(window, min_periods=min_periods).mean() - pct_mean),
                    (f"external_range_interaction_{source_key}_{window}", main_range.rolling(window, min_periods=min_periods).mean() * value_pct_change.abs().rolling(window, min_periods=min_periods).mean()),
                ]
            )

    if not factor_specs:
        return pd.DataFrame(index=main_data.index)
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=main_data.index,
    )

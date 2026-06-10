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

LAST_RELATED_DATA_COVERAGE = pd.DataFrame()

def get_last_related_data_coverage() -> pd.DataFrame:
    """返回最近一次跨品种因子构建产生的数据覆盖率诊断。"""
    return LAST_RELATED_DATA_COVERAGE.copy()

def add_cross_asset_factors(
    main_data: pd.DataFrame,
    related_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """基于相关期货 OHLCV 行情生成跨品种因子。

    这些因子只使用当前和历史相关品种 K 线。相关品种数据会先对齐到主标的时间轴，
    再按照严格的 K 线数量限制做向前填充。这里不会使用向后填充，
    因此未来的相关品种数值不会被移动到过去。
    """
    global LAST_RELATED_DATA_COVERAGE
    LAST_RELATED_DATA_COVERAGE = pd.DataFrame()

    if not related_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_volume = main_data.get("volume", pd.Series(np.nan, index=main_data.index)).replace(0, np.nan)
    main_return = main_close.pct_change()
    main_range = (main_data["high"] - main_data["low"]) / main_data["open"].replace(0, np.nan)
    windows = list(getattr(config, "cross_asset_factor_windows", []) or [])
    max_ffill_bars = max(0, int(getattr(config, "cross_asset_max_ffill_bars", 0) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []
    cross_expansion_specs: list[tuple[str, pd.Series]] = []
    coverage_rows = []

    for symbol, related_data in related_data_map.items():
        coverage_rows.append(
            calculate_related_data_coverage(
                symbol,
                related_data,
                main_data.index,
                max_ffill_bars,
            )
        )
        related = align_related_data_to_main(related_data, main_data.index, max_ffill_bars)
        if related.empty or "close" not in related.columns:
            continue

        symbol_key = safe_symbol_name(symbol)
        related_close = related["close"].replace(0, np.nan)
        related_open = related["open"].replace(0, np.nan)
        related_high = related["high"]
        related_low = related["low"]
        related_volume = related.get("volume", pd.Series(np.nan, index=related.index)).replace(0, np.nan)
        related_return = related_close.pct_change()
        related_intrabar_return = related_close / related_open - 1.0
        related_range = (related_high - related_low) / related_open
        price_ratio = main_close / related_close
        relative_return = main_return - related_return
        volume_ratio = main_volume / related_volume
        signed_related_volume = np.sign(related_return).fillna(0.0) * related_volume.fillna(0.0)

        for window in windows:
            half_window = max(2, window // 2)
            related_vol = related_return.rolling(window).std().replace(0, np.nan)
            main_vol = main_return.rolling(window).std().replace(0, np.nan)
            related_return_var = related_return.rolling(window).var().replace(0, np.nan)
            rolling_corr = main_return.rolling(window).corr(related_return)
            rolling_beta = main_return.rolling(window).cov(related_return) / related_return_var
            beta_residual = main_return - rolling_beta * related_return
            relative_strength = main_close.pct_change(window) - related_close.pct_change(window)
            price_ratio_ma = price_ratio.rolling(window).mean().replace(0, np.nan)
            price_ratio_std = price_ratio.rolling(window).std().replace(0, np.nan)
            volume_ratio_ma = volume_ratio.rolling(window).mean().replace(0, np.nan)
            related_volume_ma = related_volume.rolling(window).mean().replace(0, np.nan)

            factor_specs.extend(
                [
                    (f"cross_relret_{symbol_key}_{window}", relative_return.rolling(window).mean()),
                    (f"cross_relstrength_{symbol_key}_{window}", relative_strength),
                    (f"cross_ratio_mom_{symbol_key}_{window}", price_ratio.pct_change(window)),
                    (f"cross_ratio_zdist_{symbol_key}_{window}", (price_ratio - price_ratio_ma) / price_ratio_std),
                    (f"cross_corr_{symbol_key}_{window}", rolling_corr),
                    (f"cross_beta_{symbol_key}_{window}", rolling_beta),
                    (f"cross_beta_resid_{symbol_key}_{window}", beta_residual.rolling(window).mean()),
                    (f"cross_lead_return_{symbol_key}_{window}", related_return.shift(1).rolling(window).mean()),
                    (f"cross_lead_intrabar_{symbol_key}_{window}", related_intrabar_return.shift(1).rolling(window).mean()),
                    (f"cross_volume_ratio_{symbol_key}_{window}", volume_ratio / volume_ratio_ma - 1.0),
                    (f"cross_volume_pressure_{symbol_key}_{window}", signed_related_volume.rolling(window).sum() / related_volume.rolling(window).sum().replace(0, np.nan)),
                    (f"cross_vol_spread_{symbol_key}_{window}", main_vol / related_vol - 1.0),
                    (f"cross_range_spread_{symbol_key}_{window}", main_range.rolling(window).mean() - related_range.rolling(window).mean()),
                    (f"cross_related_volume_shock_{symbol_key}_{window}", related_volume / related_volume_ma - 1.0),
                    (f"cross_fast_slow_relret_{symbol_key}_{window}", relative_return.rolling(half_window).mean() - relative_return.rolling(window).mean()),
                ]
            )

            cross_expansion_inputs = {
                "relative_return": relative_return,
                "related_return": related_return,
                "related_intrabar": related_intrabar_return,
                "related_return_lag1": related_return.shift(1),
                "related_intrabar_lag1": related_intrabar_return.shift(1),
                "price_ratio_return": price_ratio.pct_change(),
                "price_ratio_level": price_ratio / price_ratio_ma - 1.0,
                "volume_ratio_level": volume_ratio / volume_ratio_ma - 1.0,
                "volume_ratio_change": volume_ratio.pct_change(),
                "related_range": related_range,
                "range_spread": main_range - related_range,
                "vol_spread": main_vol / related_vol - 1.0,
                "rolling_corr": rolling_corr,
                "rolling_beta": rolling_beta,
                "beta_residual": beta_residual,
                "relative_strength": relative_strength,
                "signed_related_volume_ratio": signed_related_volume / related_volume.replace(0, np.nan),
                "related_volume_change": related_volume.pct_change(),
                "related_volume_shock": related_volume / related_volume_ma - 1.0,
                "main_minus_related_abs_return": main_return.abs() - related_return.abs(),
                "main_related_return_product": main_return * related_return,
                "main_related_range_product": main_range * related_range,
                "related_return_rank": related_return.rolling(window).rank(pct=True) - 0.5,
                "relative_return_rank": relative_return.rolling(window).rank(pct=True) - 0.5,
                "lead_lag_return_gap": related_return.shift(1) - main_return,
            }
            for input_name, input_value in cross_expansion_inputs.items():
                input_ma = input_value.rolling(window).mean()
                input_std = input_value.rolling(window).std().replace(0, np.nan)
                input_abs_sum = input_value.abs().rolling(window).sum().replace(0, np.nan)
                input_half_ma = input_value.rolling(half_window).mean()
                input_half_std = input_value.rolling(half_window).std()
                input_ewm = input_value.ewm(span=window, adjust=False, min_periods=half_window).mean()
                input_positive_sum = input_value.where(input_value > 0, 0.0).rolling(window).sum()
                cross_expansion_specs.extend(
                    [
                        (f"crossmega_mean_{symbol_key}_{input_name}_{window}", input_ma),
                        (f"crossmega_std_{symbol_key}_{input_name}_{window}", input_std),
                        (f"crossmega_median_{symbol_key}_{input_name}_{window}", input_value.rolling(window).median()),
                        (f"crossmega_rank_{symbol_key}_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                        (f"crossmega_zdist_{symbol_key}_{input_name}_{window}", (input_value - input_ma) / input_std),
                        (f"crossmega_half_full_gap_{symbol_key}_{input_name}_{window}", input_half_ma / input_ma.replace(0, np.nan) - 1.0),
                        (f"crossmega_half_std_ratio_{symbol_key}_{input_name}_{window}", input_half_std / input_std - 1.0),
                        (f"crossmega_ewm_gap_{symbol_key}_{input_name}_{window}", input_value - input_ewm),
                        (f"crossmega_diff_mean_{symbol_key}_{input_name}_{window}", input_value.diff().rolling(window).mean()),
                        (f"crossmega_sum_abs_share_{symbol_key}_{input_name}_{window}", input_value.rolling(window).sum() / input_abs_sum),
                        (f"crossmega_positive_share_{symbol_key}_{input_name}_{window}", input_positive_sum / input_abs_sum),
                        (f"crossmega_quantile_spread_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                        (f"crossmega_main_return_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_return)),
                    ]
                )

    factor_specs = factor_specs + cross_expansion_specs[:10000]

    max_factors = getattr(config, "cross_asset_max_factors", None)
    if max_factors is not None:
        factor_specs = factor_specs[: max(0, int(max_factors))]

    cross_factors = {
        factor_name: rolling_zscore(raw_factor, config.zscore_window)
        for factor_name, raw_factor in factor_specs
    }
    LAST_RELATED_DATA_COVERAGE = pd.DataFrame(coverage_rows)
    return pd.DataFrame(cross_factors, index=main_data.index)

def add_complex_cross_asset_factors(
    main_data: pd.DataFrame,
    related_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """追加生成更复杂的跨品种结构因子。"""
    if not related_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_open = main_data["open"].replace(0, np.nan)
    main_high = main_data["high"].replace(0, np.nan)
    main_low = main_data["low"].replace(0, np.nan)
    main_volume = main_data.get("volume", pd.Series(np.nan, index=main_data.index)).replace(0, np.nan)
    main_amount = main_data.get("amt", main_data.get("amount", main_close * main_volume)).replace(0, np.nan)
    main_return = main_close.pct_change()
    main_intrabar = main_close / main_open - 1.0
    main_range = (main_high - main_low) / main_open
    main_volume_change = main_volume.pct_change()
    main_amount_change = main_amount.pct_change()
    main_close_location = ((main_close - main_low) - (main_high - main_close)) / (main_high - main_low).replace(0, np.nan)
    windows = list(getattr(config, "cross_asset_factor_windows", []) or [])
    max_ffill_bars = max(0, int(getattr(config, "cross_asset_max_ffill_bars", 0) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []

    for symbol, related_data in related_data_map.items():
        related = align_related_data_to_main(related_data, main_data.index, max_ffill_bars)
        if related.empty or "close" not in related.columns:
            continue

        symbol_key = safe_symbol_name(symbol)
        related_close = related["close"].replace(0, np.nan)
        related_open = related["open"].replace(0, np.nan)
        related_high = related["high"].replace(0, np.nan)
        related_low = related["low"].replace(0, np.nan)
        related_volume = related.get("volume", pd.Series(np.nan, index=related.index)).replace(0, np.nan)
        related_amount = related.get("amt", related.get("amount", related_close * related_volume)).replace(0, np.nan)
        related_return = related_close.pct_change()
        related_intrabar = related_close / related_open - 1.0
        related_range = (related_high - related_low) / related_open
        related_volume_change = related_volume.pct_change()
        related_amount_change = related_amount.pct_change()
        related_close_location = (
            ((related_close - related_low) - (related_high - related_close))
            / (related_high - related_low).replace(0, np.nan)
        )
        price_ratio = main_close / related_close
        volume_ratio = main_volume / related_volume
        amount_ratio = main_amount / related_amount

        for window in windows:
            half_window = max(2, window // 2)
            rolling_beta = (
                main_return.rolling(window).cov(related_return)
                / related_return.rolling(window).var().replace(0, np.nan)
            )
            rolling_corr = main_return.rolling(window).corr(related_return)
            cross_inputs = {
                "relative_return": main_return - related_return,
                "relative_intrabar": main_intrabar - related_intrabar,
                "lead_related_return": related_return.shift(1),
                "lead_related_intrabar": related_intrabar.shift(1),
                "price_ratio_return": price_ratio.pct_change(),
                "price_ratio_gap": price_ratio / price_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "volume_ratio_gap": volume_ratio / volume_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "amount_ratio_gap": amount_ratio / amount_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "range_spread": main_range - related_range,
                "volume_change_spread": main_volume_change - related_volume_change,
                "amount_change_spread": main_amount_change - related_amount_change,
                "location_spread": main_close_location - related_close_location,
                "beta_residual": main_return - rolling_beta * related_return,
                "corr_level": rolling_corr,
                "beta_level": rolling_beta,
                "return_product": main_return * related_return,
                "range_product": main_range * related_range,
                "main_abs_minus_related_abs": main_return.abs() - related_return.abs(),
                "lead_lag_gap": related_return.shift(1) - main_return,
                "relative_strength": main_close.pct_change(window) - related_close.pct_change(window),
                "related_volume_shock": related_volume / related_volume.rolling(window).mean().replace(0, np.nan) - 1.0,
                "related_amount_shock": related_amount / related_amount.rolling(window).mean().replace(0, np.nan) - 1.0,
                "related_range_rank": related_range.rolling(window).rank(pct=True) - 0.5,
                "relative_return_rank": (main_return - related_return).rolling(window).rank(pct=True) - 0.5,
                "related_pressure": related_close_location * related_range,
            }
            for input_name, input_value in cross_inputs.items():
                rolling_mean = input_value.rolling(window).mean()
                rolling_std = input_value.rolling(window).std().replace(0, np.nan)
                rolling_abs_sum = input_value.abs().rolling(window).sum().replace(0, np.nan)
                half_mean = input_value.rolling(half_window).mean()
                half_std = input_value.rolling(half_window).std()
                ewm_mean = input_value.ewm(span=window, adjust=False, min_periods=half_window).mean()
                factor_specs.extend(
                    [
                        (f"crossultra_mean_{symbol_key}_{input_name}_{window}", rolling_mean),
                        (f"crossultra_std_{symbol_key}_{input_name}_{window}", rolling_std),
                        (f"crossultra_rank_{symbol_key}_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                        (f"crossultra_zdist_{symbol_key}_{input_name}_{window}", (input_value - rolling_mean) / rolling_std),
                        (f"crossultra_half_mean_gap_{symbol_key}_{input_name}_{window}", half_mean / rolling_mean.replace(0, np.nan) - 1.0),
                        (f"crossultra_half_std_ratio_{symbol_key}_{input_name}_{window}", half_std / rolling_std - 1.0),
                        (f"crossultra_ewm_gap_{symbol_key}_{input_name}_{window}", input_value - ewm_mean),
                        (f"crossultra_diff_mean_{symbol_key}_{input_name}_{window}", input_value.diff().rolling(window).mean()),
                        (f"crossultra_sum_abs_share_{symbol_key}_{input_name}_{window}", input_value.rolling(window).sum() / rolling_abs_sum),
                        (f"crossultra_quantile_spread_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                        (f"crossultra_main_return_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_return)),
                        (f"crossultra_related_return_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(related_return)),
                        (f"crossultra_volume_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_volume_change)),
                    ]
                )

    factor_specs = factor_specs[:10000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=main_data.index,
    )

def add_hyper_cross_asset_factors(
    main_data: pd.DataFrame,
    related_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """追加生成更高阶的跨品种交互因子。"""
    if not related_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_open = main_data["open"].replace(0, np.nan)
    main_high = main_data["high"].replace(0, np.nan)
    main_low = main_data["low"].replace(0, np.nan)
    main_volume = main_data.get("volume", pd.Series(np.nan, index=main_data.index)).replace(0, np.nan)
    main_amount = main_data.get("amt", main_data.get("amount", main_close * main_volume)).replace(0, np.nan)
    main_return = main_close.pct_change()
    main_intrabar = main_close / main_open - 1.0
    main_range = (main_high - main_low) / main_open
    main_volume_change = main_volume.pct_change()
    main_amount_change = main_amount.pct_change()
    main_location = ((main_close - main_low) - (main_high - main_close)) / (main_high - main_low).replace(0, np.nan)
    windows = list(getattr(config, "cross_asset_factor_windows", []) or [])
    max_ffill_bars = max(0, int(getattr(config, "cross_asset_max_ffill_bars", 0) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []

    for symbol, related_data in related_data_map.items():
        related = align_related_data_to_main(related_data, main_data.index, max_ffill_bars)
        if related.empty or "close" not in related.columns:
            continue

        symbol_key = safe_symbol_name(symbol)
        related_close = related["close"].replace(0, np.nan)
        related_open = related["open"].replace(0, np.nan)
        related_high = related["high"].replace(0, np.nan)
        related_low = related["low"].replace(0, np.nan)
        related_volume = related.get("volume", pd.Series(np.nan, index=related.index)).replace(0, np.nan)
        related_amount = related.get("amt", related.get("amount", related_close * related_volume)).replace(0, np.nan)
        related_return = related_close.pct_change()
        related_intrabar = related_close / related_open - 1.0
        related_range = (related_high - related_low) / related_open
        related_volume_change = related_volume.pct_change()
        related_amount_change = related_amount.pct_change()
        related_location = (
            ((related_close - related_low) - (related_high - related_close))
            / (related_high - related_low).replace(0, np.nan)
        )
        price_ratio = main_close / related_close
        volume_ratio = main_volume / related_volume
        amount_ratio = main_amount / related_amount

        for window in windows:
            half_window = max(2, window // 2)
            double_window = min(240, max(window + 1, window * 2))
            beta = main_return.rolling(window).cov(related_return) / related_return.rolling(window).var().replace(0, np.nan)
            corr = main_return.rolling(window).corr(related_return)
            cross_inputs = {
                "rel_ret_x_corr": (main_return - related_return) * corr,
                "rel_ret_x_beta": (main_return - related_return) * beta,
                "lead_ret_x_main": related_return.shift(1) * main_return,
                "lead_intrabar_x_main": related_intrabar.shift(1) * main_intrabar,
                "price_ratio_mom": price_ratio.pct_change(),
                "price_ratio_reversal": price_ratio / price_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "volume_ratio_mom": volume_ratio.pct_change(),
                "amount_ratio_mom": amount_ratio.pct_change(),
                "range_spread_x_corr": (main_range - related_range) * corr,
                "location_spread_x_range": (main_location - related_location) * (main_range - related_range),
                "volume_spread_x_ret": (main_volume_change - related_volume_change) * (main_return - related_return),
                "amount_spread_x_ret": (main_amount_change - related_amount_change) * (main_return - related_return),
                "beta_residual": main_return - beta * related_return,
                "corr_change": corr.diff(),
                "beta_change": beta.diff(),
                "relative_abs_gap": main_return.abs() - related_return.abs(),
                "relative_range_gap": main_range - related_range,
                "related_pressure": related_location * related_range,
                "main_related_pressure_gap": main_location * main_range - related_location * related_range,
                "lead_lag_pressure": related_return.shift(1) - main_return,
                "price_volume_ratio_mix": price_ratio.pct_change() * volume_ratio.pct_change(),
                "price_amount_ratio_mix": price_ratio.pct_change() * amount_ratio.pct_change(),
                "related_volume_shock": related_volume / related_volume.rolling(window).mean().replace(0, np.nan) - 1.0,
                "related_amount_shock": related_amount / related_amount.rolling(window).mean().replace(0, np.nan) - 1.0,
                "relative_strength": main_close.pct_change(window) - related_close.pct_change(window),
            }
            for input_name, input_value in cross_inputs.items():
                rolling_mean = input_value.rolling(window).mean()
                rolling_std = input_value.rolling(window).std().replace(0, np.nan)
                slow_mean = input_value.rolling(double_window).mean()
                slow_std = input_value.rolling(double_window).std().replace(0, np.nan)
                half_mean = input_value.rolling(half_window).mean()
                ewm_fast = input_value.ewm(span=half_window, adjust=False, min_periods=half_window).mean()
                ewm_slow = input_value.ewm(span=window, adjust=False, min_periods=half_window).mean()
                abs_sum = input_value.abs().rolling(window).sum().replace(0, np.nan)
                factor_specs.extend(
                    [
                        (f"crosshyper_mean_{symbol_key}_{input_name}_{window}", rolling_mean),
                        (f"crosshyper_std_{symbol_key}_{input_name}_{window}", rolling_std),
                        (f"crosshyper_zdist_{symbol_key}_{input_name}_{window}", (input_value - rolling_mean) / rolling_std),
                        (f"crosshyper_rank_{symbol_key}_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                        (f"crosshyper_slow_zdist_{symbol_key}_{input_name}_{window}", (rolling_mean - slow_mean) / slow_std),
                        (f"crosshyper_fast_slow_gap_{symbol_key}_{input_name}_{window}", half_mean / slow_mean.replace(0, np.nan) - 1.0),
                        (f"crosshyper_ewm_cross_{symbol_key}_{input_name}_{window}", ewm_fast - ewm_slow),
                        (f"crosshyper_abs_share_{symbol_key}_{input_name}_{window}", input_value.rolling(window).sum() / abs_sum),
                        (f"crosshyper_tail_spread_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.9) - input_value.rolling(window).quantile(0.1)),
                        (f"crosshyper_iqr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                        (f"crosshyper_main_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_return)),
                        (f"crosshyper_related_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(related_return)),
                        (f"crosshyper_volume_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_volume_change)),
                    ]
                )

    factor_specs = factor_specs[:10000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=main_data.index,
    )

def add_omega_cross_asset_factors(
    main_data: pd.DataFrame,
    related_data_map: dict[str, pd.DataFrame],
    config: Any,
) -> pd.DataFrame:
    """追加生成第四批跨品种复杂交互因子。"""
    if not related_data_map:
        return pd.DataFrame(index=main_data.index)

    main_close = main_data["close"].replace(0, np.nan)
    main_open = main_data["open"].replace(0, np.nan)
    main_high = main_data["high"].replace(0, np.nan)
    main_low = main_data["low"].replace(0, np.nan)
    main_volume = main_data.get("volume", pd.Series(np.nan, index=main_data.index)).replace(0, np.nan)
    main_amount = main_data.get("amt", main_data.get("amount", main_close * main_volume)).replace(0, np.nan)
    main_return = main_close.pct_change()
    main_intrabar = main_close / main_open - 1.0
    main_range = (main_high - main_low) / main_open
    main_volume_change = main_volume.pct_change()
    main_amount_change = main_amount.pct_change()
    main_location = ((main_close - main_low) - (main_high - main_close)) / (main_high - main_low).replace(0, np.nan)
    windows = list(getattr(config, "cross_asset_factor_windows", []) or [])
    max_ffill_bars = max(0, int(getattr(config, "cross_asset_max_ffill_bars", 0) or 0))
    factor_specs: list[tuple[str, pd.Series]] = []

    for symbol, related_data in related_data_map.items():
        related = align_related_data_to_main(related_data, main_data.index, max_ffill_bars)
        if related.empty or "close" not in related.columns:
            continue

        symbol_key = safe_symbol_name(symbol)
        related_close = related["close"].replace(0, np.nan)
        related_open = related["open"].replace(0, np.nan)
        related_high = related["high"].replace(0, np.nan)
        related_low = related["low"].replace(0, np.nan)
        related_volume = related.get("volume", pd.Series(np.nan, index=related.index)).replace(0, np.nan)
        related_amount = related.get("amt", related.get("amount", related_close * related_volume)).replace(0, np.nan)
        related_return = related_close.pct_change()
        related_intrabar = related_close / related_open - 1.0
        related_range = (related_high - related_low) / related_open
        related_volume_change = related_volume.pct_change()
        related_amount_change = related_amount.pct_change()
        related_location = (
            ((related_close - related_low) - (related_high - related_close))
            / (related_high - related_low).replace(0, np.nan)
        )
        price_ratio = main_close / related_close
        volume_ratio = main_volume / related_volume
        amount_ratio = main_amount / related_amount

        for window in windows:
            half_window = max(2, window // 2)
            beta = main_return.rolling(window).cov(related_return) / related_return.rolling(window).var().replace(0, np.nan)
            corr = main_return.rolling(window).corr(related_return)
            cross_inputs = {
                "rel_ret": main_return - related_return,
                "rel_intrabar": main_intrabar - related_intrabar,
                "lead_ret": related_return.shift(1),
                "lead_intrabar": related_intrabar.shift(1),
                "price_ratio_ret": price_ratio.pct_change(),
                "price_ratio_gap": price_ratio / price_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "volume_ratio_ret": volume_ratio.pct_change(),
                "volume_ratio_gap": volume_ratio / volume_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "amount_ratio_ret": amount_ratio.pct_change(),
                "amount_ratio_gap": amount_ratio / amount_ratio.rolling(window).mean().replace(0, np.nan) - 1.0,
                "range_spread": main_range - related_range,
                "location_spread": main_location - related_location,
                "volume_change_spread": main_volume_change - related_volume_change,
                "amount_change_spread": main_amount_change - related_amount_change,
                "beta_residual": main_return - beta * related_return,
                "corr_level": corr,
                "beta_level": beta,
                "corr_change": corr.diff(),
                "beta_change": beta.diff(),
                "ret_product": main_return * related_return,
                "range_product": main_range * related_range,
                "pressure_gap": main_location * main_range - related_location * related_range,
                "lead_lag_gap": related_return.shift(1) - main_return,
                "relative_strength": main_close.pct_change(window) - related_close.pct_change(window),
                "related_pressure": related_location * related_range,
            }
            for input_name, input_value in cross_inputs.items():
                rolling_mean = input_value.rolling(window).mean()
                rolling_std = input_value.rolling(window).std().replace(0, np.nan)
                slow_mean = input_value.rolling(min(240, max(window + 1, window * 2))).mean()
                slow_std = input_value.rolling(min(240, max(window + 1, window * 2))).std().replace(0, np.nan)
                abs_sum = input_value.abs().rolling(window).sum().replace(0, np.nan)
                factor_specs.extend(
                    [
                        (f"crossomega_mean_{symbol_key}_{input_name}_{window}", rolling_mean),
                        (f"crossomega_std_{symbol_key}_{input_name}_{window}", rolling_std),
                        (f"crossomega_zdist_{symbol_key}_{input_name}_{window}", (input_value - rolling_mean) / rolling_std),
                        (f"crossomega_rank_{symbol_key}_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                        (f"crossomega_slow_zdist_{symbol_key}_{input_name}_{window}", (rolling_mean - slow_mean) / slow_std),
                        (f"crossomega_fast_slow_gap_{symbol_key}_{input_name}_{window}", input_value.rolling(half_window).mean() / slow_mean.replace(0, np.nan) - 1.0),
                        (f"crossomega_ewm_cross_{symbol_key}_{input_name}_{window}", input_value.ewm(span=half_window, adjust=False, min_periods=half_window).mean() - input_value.ewm(span=window, adjust=False, min_periods=half_window).mean()),
                        (f"crossomega_abs_share_{symbol_key}_{input_name}_{window}", input_value.rolling(window).sum() / abs_sum),
                        (f"crossomega_pos_rate_{symbol_key}_{input_name}_{window}", (input_value > 0).astype(float).rolling(window).mean()),
                        (f"crossomega_neg_rate_{symbol_key}_{input_name}_{window}", (input_value < 0).astype(float).rolling(window).mean()),
                        (f"crossomega_tail_spread_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.9) - input_value.rolling(window).quantile(0.1)),
                        (f"crossomega_iqr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                        (f"crossomega_main_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_return)),
                        (f"crossomega_related_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(related_return)),
                        (f"crossomega_volume_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_volume_change)),
                        (f"crossomega_amount_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_amount_change)),
                        (f"crossomega_range_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_range)),
                        (f"crossomega_location_corr_{symbol_key}_{input_name}_{window}", input_value.rolling(window).corr(main_location)),
                        (f"crossomega_main_beta_{symbol_key}_{input_name}_{window}", input_value.rolling(window).cov(main_return) / main_return.rolling(window).var().replace(0, np.nan)),
                        (f"crossomega_related_beta_{symbol_key}_{input_name}_{window}", input_value.rolling(window).cov(related_return) / related_return.rolling(window).var().replace(0, np.nan)),
                        (f"crossomega_diff_mean_{symbol_key}_{input_name}_{window}", input_value.diff().rolling(window).mean()),
                        (f"crossomega_diff_std_{symbol_key}_{input_name}_{window}", input_value.diff().rolling(window).std()),
                        (f"crossomega_accel_mean_{symbol_key}_{input_name}_{window}", input_value.diff().diff().rolling(window).mean()),
                        (f"crossomega_persistence_{symbol_key}_{input_name}_{window}", np.sign(input_value).rolling(window).sum() / window),
                        (f"crossomega_zero_cross_{symbol_key}_{input_name}_{window}", (np.sign(input_value).diff().abs() > 0).astype(float).rolling(window).mean()),
                    ]
                )

    factor_specs = factor_specs[:20000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=main_data.index,
    )

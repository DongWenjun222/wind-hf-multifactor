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

def add_complex_non_cross_factors(
    df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """追加生成更复杂的主品种自身因子。

    这批因子刻意放在 build_factors 的尾部，而不是插入 add_parametric_factors，
    目的是保持旧参数化因子和旧跨品种因子的编号稳定。
    """
    close = df["close"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    volume = df["volume"].replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    bar_return = df["bar_return_cc"]
    intrabar_return = df["bar_return_oc"]
    spread_pct = (high - low) / open_price
    body_pct = (close - open_price) / open_price
    upper_shadow_pct = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    lower_shadow_pct = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    close_location = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    typical_price = (high + low + close) / 3.0
    mid_price = (high + low) / 2.0
    gap_return = open_price / close.shift(1).replace(0, np.nan) - 1.0
    volume_change = volume.pct_change()
    amount_change = amount.pct_change()
    signed_volume_ratio = np.sign(bar_return).fillna(0.0) * volume / volume.rolling(20).mean().replace(0, np.nan)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_pct = true_range / open_price
    vwap_proxy = amount / volume
    vwap_gap = close / vwap_proxy.replace(0, np.nan) - 1.0
    price_pressure = close_location * spread_pct
    liquidity_pressure = bar_return.abs() / amount.replace(0, np.nan)
    body_shadow_balance = body_pct - (upper_shadow_pct - lower_shadow_pct)
    range_volume_product = spread_pct * volume_change
    return_volume_product = bar_return * volume_change
    trend_path_efficiency = bar_return / bar_return.abs().rolling(20).mean().replace(0, np.nan)
    close_to_typical = close / typical_price.replace(0, np.nan) - 1.0
    close_to_mid = close / mid_price.replace(0, np.nan) - 1.0

    windows = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233]
    factor_specs: list[tuple[str, pd.Series]] = []

    ultra_inputs = {
        "bar_return": bar_return,
        "intrabar_return": intrabar_return,
        "gap_return": gap_return,
        "volume_change": volume_change,
        "amount_change": amount_change,
        "spread_pct": spread_pct,
        "body_pct": body_pct,
        "upper_shadow_pct": upper_shadow_pct,
        "lower_shadow_pct": lower_shadow_pct,
        "close_location": close_location,
        "true_range_pct": true_range_pct,
        "vwap_gap": vwap_gap,
        "price_pressure": price_pressure,
        "liquidity_pressure": liquidity_pressure,
        "signed_volume_ratio": signed_volume_ratio,
        "body_shadow_balance": body_shadow_balance,
        "range_volume_product": range_volume_product,
        "return_volume_product": return_volume_product,
        "trend_path_efficiency": trend_path_efficiency,
        "close_to_typical": close_to_typical,
        "close_to_mid": close_to_mid,
        "high_return": high.pct_change(),
        "low_return": low.pct_change(),
        "typical_return": typical_price.pct_change(),
        "mid_return": mid_price.pct_change(),
    }

    for window in windows:
        half_window = max(2, window // 2)
        for input_name, input_value in ultra_inputs.items():
            rolling_mean = input_value.rolling(window).mean()
            rolling_std = input_value.rolling(window).std().replace(0, np.nan)
            rolling_abs = input_value.abs()
            rolling_abs_sum = rolling_abs.rolling(window).sum().replace(0, np.nan)
            half_mean = input_value.rolling(half_window).mean()
            half_std = input_value.rolling(half_window).std()
            ewm_mean = input_value.ewm(span=window, adjust=False, min_periods=half_window).mean()
            positive_sum = input_value.where(input_value > 0, 0.0).rolling(window).sum()
            negative_sum = input_value.where(input_value < 0, 0.0).abs().rolling(window).sum()
            factor_specs.extend(
                [
                    (f"ultra_mean_{input_name}_{window}", rolling_mean),
                    (f"ultra_std_{input_name}_{window}", rolling_std),
                    (f"ultra_skew_{input_name}_{window}", input_value.rolling(window).skew()),
                    (f"ultra_kurt_{input_name}_{window}", input_value.rolling(window).kurt()),
                    (f"ultra_rank_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                    (f"ultra_zdist_{input_name}_{window}", (input_value - rolling_mean) / rolling_std),
                    (f"ultra_abs_rank_{input_name}_{window}", rolling_abs.rolling(window).rank(pct=True) - 0.5),
                    (f"ultra_half_mean_gap_{input_name}_{window}", half_mean / rolling_mean.replace(0, np.nan) - 1.0),
                    (f"ultra_half_std_ratio_{input_name}_{window}", half_std / rolling_std - 1.0),
                    (f"ultra_ewm_gap_{input_name}_{window}", input_value - ewm_mean),
                    (f"ultra_diff_mean_{input_name}_{window}", input_value.diff().rolling(window).mean()),
                    (f"ultra_diff_std_{input_name}_{window}", input_value.diff().rolling(window).std()),
                    (f"ultra_accel_mean_{input_name}_{window}", input_value.diff().diff().rolling(window).mean()),
                    (f"ultra_positive_share_{input_name}_{window}", positive_sum / rolling_abs_sum),
                    (f"ultra_negative_share_{input_name}_{window}", negative_sum / rolling_abs_sum),
                    (f"ultra_sum_abs_share_{input_name}_{window}", input_value.rolling(window).sum() / rolling_abs_sum),
                    (f"ultra_quantile_spread_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                    (f"ultra_return_corr_{input_name}_{window}", input_value.rolling(window).corr(bar_return)),
                    (f"ultra_volume_corr_{input_name}_{window}", input_value.rolling(window).corr(volume_change)),
                    (f"ultra_range_corr_{input_name}_{window}", input_value.rolling(window).corr(spread_pct)),
                ]
            )

    factor_specs = factor_specs[:10000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=df.index,
    )

def add_hyper_non_cross_factors(
    df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """追加生成更高阶的主品种自身交互因子。"""
    close = df["close"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    volume = df["volume"].replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    bar_return = df["bar_return_cc"]
    intrabar_return = df["bar_return_oc"]
    spread_pct = (high - low) / open_price
    body_pct = (close - open_price) / open_price
    upper_shadow_pct = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    lower_shadow_pct = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    close_location = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    typical_price = (high + low + close) / 3.0
    mid_price = (high + low) / 2.0
    gap_return = open_price / close.shift(1).replace(0, np.nan) - 1.0
    volume_change = volume.pct_change()
    amount_change = amount.pct_change()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_pct = true_range / open_price
    signed_volume = np.sign(bar_return).fillna(0.0) * volume.fillna(0.0)
    signed_amount = np.sign(bar_return).fillna(0.0) * amount.fillna(0.0)
    vwap_proxy = amount / volume
    vwap_gap = close / vwap_proxy.replace(0, np.nan) - 1.0
    shadow_balance = lower_shadow_pct - upper_shadow_pct
    range_body_gap = spread_pct - body_pct.abs()
    pressure = close_location * spread_pct
    liquidity = bar_return.abs() / amount.replace(0, np.nan)
    trend_efficiency = bar_return.rolling(5).sum() / bar_return.abs().rolling(5).sum().replace(0, np.nan)
    volatility_pressure = spread_pct * volume_change
    money_pressure = signed_amount / amount.rolling(20).mean().replace(0, np.nan)

    windows = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233]
    factor_specs: list[tuple[str, pd.Series]] = []
    hyper_inputs = {
        "ret_x_vol": bar_return * volume_change,
        "ret_x_amt": bar_return * amount_change,
        "ret_x_range": bar_return * spread_pct,
        "intrabar_x_range": intrabar_return * spread_pct,
        "gap_x_intrabar": gap_return * intrabar_return,
        "body_x_location": body_pct * close_location,
        "shadow_x_location": shadow_balance * close_location,
        "range_body_gap": range_body_gap,
        "range_x_volume": spread_pct * volume_change,
        "range_x_amount": spread_pct * amount_change,
        "pressure": pressure,
        "liquidity": liquidity,
        "liquidity_x_range": liquidity * spread_pct,
        "signed_volume": signed_volume / volume.rolling(20).mean().replace(0, np.nan),
        "signed_amount": signed_amount / amount.rolling(20).mean().replace(0, np.nan),
        "vwap_gap": vwap_gap,
        "close_typical_gap": close / typical_price.replace(0, np.nan) - 1.0,
        "close_mid_gap": close / mid_price.replace(0, np.nan) - 1.0,
        "true_range_pct": true_range_pct,
        "trend_efficiency": trend_efficiency,
        "volatility_pressure": volatility_pressure,
        "money_pressure": money_pressure,
        "high_low_return_gap": high.pct_change() - low.pct_change(),
        "typical_mid_return_gap": typical_price.pct_change() - mid_price.pct_change(),
        "return_reversal_mix": bar_return - bar_return.rolling(5).mean(),
    }

    for window in windows:
        half_window = max(2, window // 2)
        double_window = min(240, max(window + 1, window * 2))
        for input_name, input_value in hyper_inputs.items():
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
                    (f"hyper_mean_{input_name}_{window}", rolling_mean),
                    (f"hyper_std_{input_name}_{window}", rolling_std),
                    (f"hyper_zdist_{input_name}_{window}", (input_value - rolling_mean) / rolling_std),
                    (f"hyper_rank_{input_name}_{window}", input_value.rolling(window).rank(pct=True) - 0.5),
                    (f"hyper_slow_zdist_{input_name}_{window}", (rolling_mean - slow_mean) / slow_std),
                    (f"hyper_fast_slow_gap_{input_name}_{window}", half_mean / slow_mean.replace(0, np.nan) - 1.0),
                    (f"hyper_ewm_cross_{input_name}_{window}", ewm_fast - ewm_slow),
                    (f"hyper_abs_share_{input_name}_{window}", input_value.rolling(window).sum() / abs_sum),
                    (f"hyper_pos_count_{input_name}_{window}", (input_value > 0).astype(float).rolling(window).mean()),
                    (f"hyper_neg_count_{input_name}_{window}", (input_value < 0).astype(float).rolling(window).mean()),
                    (f"hyper_tail_spread_{input_name}_{window}", input_value.rolling(window).quantile(0.9) - input_value.rolling(window).quantile(0.1)),
                    (f"hyper_iqr_{input_name}_{window}", input_value.rolling(window).quantile(0.75) - input_value.rolling(window).quantile(0.25)),
                    (f"hyper_skew_{input_name}_{window}", input_value.rolling(window).skew()),
                    (f"hyper_kurt_{input_name}_{window}", input_value.rolling(window).kurt()),
                    (f"hyper_ret_corr_{input_name}_{window}", input_value.rolling(window).corr(bar_return)),
                    (f"hyper_intrabar_corr_{input_name}_{window}", input_value.rolling(window).corr(intrabar_return)),
                    (f"hyper_vol_corr_{input_name}_{window}", input_value.rolling(window).corr(volume_change)),
                    (f"hyper_amount_corr_{input_name}_{window}", input_value.rolling(window).corr(amount_change)),
                    (f"hyper_range_corr_{input_name}_{window}", input_value.rolling(window).corr(spread_pct)),
                    (f"hyper_location_corr_{input_name}_{window}", input_value.rolling(window).corr(close_location)),
                ]
            )

    factor_specs = factor_specs[:10000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=df.index,
    )

def add_omega_non_cross_factors(
    df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """追加生成第四批主品种复杂因子，覆盖更高阶的量价形态交互。"""
    close = df["close"].replace(0, np.nan)
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    open_price = df["open"].replace(0, np.nan)
    volume = df["volume"].replace(0, np.nan)
    amount = df.get("amt", df.get("amount", close * volume)).replace(0, np.nan)
    bar_return = df.get("bar_return_cc", close.pct_change())
    intrabar_return = df.get("bar_return_oc", close / open_price - 1.0)
    spread_pct = (high - low) / open_price
    body_pct = (close - open_price) / open_price
    upper_shadow_pct = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / open_price
    lower_shadow_pct = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / open_price
    close_location = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    typical_price = (high + low + close) / 3.0
    mid_price = (high + low) / 2.0
    gap_return = open_price / close.shift(1).replace(0, np.nan) - 1.0
    volume_change = volume.pct_change()
    amount_change = amount.pct_change()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_pct = true_range / open_price
    signed_return = np.sign(bar_return).fillna(0.0)
    signed_volume = signed_return * volume.fillna(0.0)
    signed_amount = signed_return * amount.fillna(0.0)
    volume_base = volume.rolling(20).mean().replace(0, np.nan)
    amount_base = amount.rolling(20).mean().replace(0, np.nan)
    vwap_proxy = amount / volume
    shadow_balance = lower_shadow_pct - upper_shadow_pct
    price_pressure = close_location * spread_pct
    turnover_impact = bar_return.abs() / amount.replace(0, np.nan)
    wick_body_mix = shadow_balance * body_pct
    range_liquidity_mix = spread_pct * turnover_impact
    trend_quality = bar_return.rolling(8).sum() / bar_return.abs().rolling(8).sum().replace(0, np.nan)
    volume_accel = volume_change.diff()
    amount_accel = amount_change.diff()

    windows = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233]
    factor_specs: list[tuple[str, pd.Series]] = []
    omega_inputs = {
        "ret_pressure": bar_return * price_pressure,
        "intrabar_pressure": intrabar_return * price_pressure,
        "gap_pressure": gap_return * price_pressure,
        "body_pressure": body_pct * price_pressure,
        "shadow_pressure": shadow_balance * price_pressure,
        "range_pressure": spread_pct * price_pressure,
        "ret_volume_accel": bar_return * volume_accel,
        "ret_amount_accel": bar_return * amount_accel,
        "range_volume_accel": spread_pct * volume_accel,
        "range_amount_accel": spread_pct * amount_accel,
        "signed_volume_shock": signed_volume / volume_base,
        "signed_amount_shock": signed_amount / amount_base,
        "turnover_impact": turnover_impact,
        "range_liquidity_mix": range_liquidity_mix,
        "wick_body_mix": wick_body_mix,
        "vwap_gap": close / vwap_proxy.replace(0, np.nan) - 1.0,
        "typical_gap": close / typical_price.replace(0, np.nan) - 1.0,
        "mid_gap": close / mid_price.replace(0, np.nan) - 1.0,
        "true_range_pct": true_range_pct,
        "trend_quality": trend_quality,
        "return_path_noise": bar_return.abs() - bar_return.rolling(8).mean().abs(),
        "range_path_noise": spread_pct - spread_pct.rolling(8).mean(),
        "high_low_momentum_gap": high.pct_change(3) - low.pct_change(3),
        "open_close_momentum_gap": open_price.pct_change(3) - close.pct_change(3),
        "volume_amount_gap": volume_change - amount_change,
    }

    transforms = [
        ("mean", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: ma),
        ("std", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: std),
        ("zdist", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (value - ma) / std),
        ("rank", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).rank(pct=True) - 0.5),
        ("slow_zdist", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (ma - slow_ma) / slow_std),
        ("fast_slow_gap", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(half).mean() / slow_ma.replace(0, np.nan) - 1.0),
        ("ewm_cross", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.ewm(span=half, adjust=False, min_periods=half).mean() - value.ewm(span=window, adjust=False, min_periods=half).mean()),
        ("abs_share", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).sum() / abs_sum),
        ("pos_rate", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (value > 0).astype(float).rolling(window).mean()),
        ("neg_rate", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (value < 0).astype(float).rolling(window).mean()),
        ("tail_spread", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).quantile(0.9) - value.rolling(window).quantile(0.1)),
        ("iqr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).quantile(0.75) - value.rolling(window).quantile(0.25)),
        ("skew", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).skew()),
        ("kurt", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).kurt()),
        ("diff_mean", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.diff().rolling(window).mean()),
        ("diff_std", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.diff().rolling(window).std()),
        ("ret_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(bar_return)),
        ("intrabar_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(intrabar_return)),
        ("volume_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(volume_change)),
        ("range_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(spread_pct)),
        ("location_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(close_location)),
        ("amount_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(amount_change)),
        ("body_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(body_pct)),
        ("shadow_corr", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).corr(shadow_balance)),
        ("range_beta", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).cov(spread_pct) / spread_pct.rolling(window).var().replace(0, np.nan)),
        ("volume_beta", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).cov(volume_change) / volume_change.rolling(window).var().replace(0, np.nan)),
        ("amount_beta", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).cov(amount_change) / amount_change.rolling(window).var().replace(0, np.nan)),
        ("ret_beta", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).cov(bar_return) / bar_return.rolling(window).var().replace(0, np.nan)),
        ("max", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).max()),
        ("min", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(window).min()),
        ("last_mean_gap", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value / ma.replace(0, np.nan) - 1.0),
        ("mean_abs_ratio", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: ma / value.abs().rolling(window).mean().replace(0, np.nan)),
        ("accel_mean", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.diff().diff().rolling(window).mean()),
        ("accel_std", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.diff().diff().rolling(window).std()),
        ("zero_cross", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (np.sign(value).diff().abs() > 0).astype(float).rolling(window).mean()),
        ("persistence", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: np.sign(value).rolling(window).sum() / window),
        ("half_rank", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(half).rank(pct=True) - 0.5),
        ("slow_rank_gap", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: (value.rolling(window).rank(pct=True) - value.rolling(double).rank(pct=True))),
        ("vol_of_vol", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: value.rolling(half).std().rolling(window).std()),
        ("mean_slope", lambda value, window, half, double, ma, std, slow_ma, slow_std, abs_sum: ma.diff()),
    ]

    for window in windows:
        half_window = max(2, window // 2)
        double_window = min(240, max(window + 1, window * 2))
        for input_name, input_value in omega_inputs.items():
            rolling_mean = input_value.rolling(window).mean()
            rolling_std = input_value.rolling(window).std().replace(0, np.nan)
            slow_mean = input_value.rolling(double_window).mean()
            slow_std = input_value.rolling(double_window).std().replace(0, np.nan)
            abs_sum = input_value.abs().rolling(window).sum().replace(0, np.nan)
            for transform_name, transform_func in transforms:
                factor_specs.append(
                    (
                        f"omega_{transform_name}_{input_name}_{window}",
                        transform_func(
                            input_value,
                            window,
                            half_window,
                            double_window,
                            rolling_mean,
                            rolling_std,
                            slow_mean,
                            slow_std,
                            abs_sum,
                        ),
                    )
                )

    factor_specs = factor_specs[:20000]
    return pd.DataFrame(
        {
            factor_name: rolling_zscore(raw_factor, config.zscore_window)
            for factor_name, raw_factor in factor_specs
        },
        index=df.index,
    )

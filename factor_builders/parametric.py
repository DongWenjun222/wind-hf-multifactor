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

def add_parametric_factors(
    df: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """批量生成参数化因子。

    这里使用一组滚动窗口，把收益、波动、成交量、影线、缺口、流动性、
    价格位置、相关性等原始量价结构扩展成大量候选因子。

    返回值只包含参数化因子，不包含 momentum/reversal 等基础因子。
    每个原始因子都会再经过 rolling_zscore 标准化，便于统一阈值和模型输入。
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_price = df["open"].replace(0, np.nan)
    volume = df["volume"].replace(0, np.nan)
    bar_return = df.get("bar_return_cc", close.pct_change())
    intrabar_return = df.get("bar_return_oc", close / open_price - 1.0)
    spread = (high - low).replace(0, np.nan)
    body = close - df["open"]
    upper_shadow = high - pd.concat([df["open"], close], axis=1).max(axis=1)
    lower_shadow = pd.concat([df["open"], close], axis=1).min(axis=1) - low
    close_location = ((close - low) - (high - close)) / spread
    volume_change = volume.pct_change()
    signed_volume = np.sign(bar_return).fillna(0.0) * volume.fillna(0.0)
    gap_return = df["open"] / close.shift(1).replace(0, np.nan) - 1.0
    illiquidity = bar_return.abs() / volume.replace(0, np.nan)
    typical_price = (high + low + close) / 3.0
    mid_price = (high + low) / 2.0
    dollar_volume = close * volume
    dollar_volume_change = dollar_volume.pct_change()
    money_flow = typical_price * signed_volume
    obv = signed_volume.cumsum()
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    open_return = df["open"].pct_change()
    high_return = high.pct_change()
    low_return = low.pct_change()
    typical_return = typical_price.pct_change()
    positive_return = bar_return.where(bar_return > 0, 0.0)
    negative_return = bar_return.where(bar_return < 0, 0.0)
    wick_imbalance = (lower_shadow - upper_shadow) / spread
    body_direction = np.sign(body).replace(0, np.nan)
    body_pct = body / open_price
    upper_shadow_pct = upper_shadow / open_price
    lower_shadow_pct = lower_shadow / open_price
    range_pct = spread / open_price
    close_open_gap = close / open_price - 1.0
    high_open_gap = high / open_price - 1.0
    low_open_gap = low / open_price - 1.0
    typical_open_gap = typical_price / open_price - 1.0
    mid_open_gap = mid_price / open_price - 1.0
    close_prev_gap = close / prev_close.replace(0, np.nan) - 1.0
    volume_signed_return = np.sign(intrabar_return).fillna(0.0) * volume_change
    price_volume_product = bar_return * volume_change
    range_volume_product = range_pct * volume_change
    body_volume_product = body_pct * volume_change
    wick_balance_pct = (lower_shadow - upper_shadow) / open_price

    # 多个窗口代表不同时间尺度：短窗口更敏感，长窗口更稳定。
    windows = [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 68, 89, 110, 144, 178, 233]
    factor_specs = []
    additional_factor_specs = []
    new_factor_specs = []
    expansion_factor_specs = []

    for window in windows:
        half_window = max(2, window // 2)
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
        fast_vol = bar_return.rolling(half_window).std()
        slow_vol = bar_return.rolling(window).std().replace(0, np.nan)
        close_ma = close.rolling(window).mean()
        close_std = close.rolling(window).std().replace(0, np.nan)
        volume_ma = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        illiquidity_ma = illiquidity.rolling(window).mean()
        negative_abs_mean = negative_return.abs().rolling(window).mean().replace(0, np.nan)
        return_var = bar_return.rolling(window).var().replace(0, np.nan)
        volume_var = volume_change.rolling(window).var().replace(0, np.nan)
        positive_abs_sum = positive_return.rolling(window).sum().replace(0, np.nan)
        negative_abs_sum = negative_return.abs().rolling(window).sum().replace(0, np.nan)
        range_ma = spread.rolling(window).mean().replace(0, np.nan)
        body_ma = body.abs().rolling(window).mean().replace(0, np.nan)
        body_std = body.rolling(window).std().replace(0, np.nan)
        money_flow_ma = money_flow.rolling(window).mean()
        money_flow_std = money_flow.rolling(window).std().replace(0, np.nan)
        fast_return_sum = bar_return.rolling(half_window).sum()
        slow_return_sum = bar_return.rolling(window).sum().replace(0, np.nan)
        fast_volume_sum = volume.rolling(half_window).sum()
        slow_volume_sum = volume.rolling(window).sum().replace(0, np.nan)
        rolling_mid = mid_price.rolling(window).mean()
        close_rank = close.rolling(window).rank(pct=True)
        volume_rank = volume.rolling(window).rank(pct=True)
        range_rank = spread.rolling(window).rank(pct=True)
        typical_ma = typical_price.rolling(window).mean()
        typical_std = typical_price.rolling(window).std().replace(0, np.nan)
        close_ma_slope = close_ma.diff()
        volume_ma_slope = volume_ma.diff()
        range_ma_slope = range_ma.diff()
        abs_return = bar_return.abs()
        return_abs_ma = abs_return.rolling(window).mean().replace(0, np.nan)
        return_abs_std = abs_return.rolling(window).std().replace(0, np.nan)
        intrabar_abs = intrabar_return.abs()
        signed_range = np.sign(bar_return).fillna(0.0) * spread
        high_low_mid = (high + low) / 2.0
        close_to_high = close / high.replace(0, np.nan) - 1.0
        close_to_low = close / low.replace(0, np.nan) - 1.0
        open_to_high = df["open"] / high.replace(0, np.nan) - 1.0
        open_to_low = df["open"] / low.replace(0, np.nan) - 1.0
        return_rank = bar_return.rolling(window).rank(pct=True)
        abs_return_rank = abs_return.rolling(window).rank(pct=True)
        gap_rank = gap_return.rolling(window).rank(pct=True)
        illiquidity_rank = illiquidity.rolling(window).rank(pct=True)
        dollar_volume_rank = dollar_volume.rolling(window).rank(pct=True)
        true_range_rank = true_range.rolling(window).rank(pct=True)

        factor_specs.extend(
            [
                (f"ret_{window}", close.pct_change(window)),
                (f"ma_gap_{window}", close / close.rolling(window).mean() - 1.0),
                (f"volatility_{window}", bar_return.rolling(window).std()),
                (f"range_mean_{window}", (spread / open_price).rolling(window).mean()),
                (f"volume_ratio_{window}", volume / volume.rolling(window).mean() - 1.0),
                (f"vwap_gap_{window}", close / ((high + low + close) / 3.0).rolling(window).mean() - 1.0),
                (f"drawdown_{window}", close / close.rolling(window).max() - 1.0),
                (f"runup_{window}", close / close.rolling(window).min() - 1.0),
                (f"signed_volume_{window}", signed_volume.rolling(window).sum()),
                (f"intrabar_strength_{window}", intrabar_return.rolling(window).mean()),
                (f"return_mean_{window}", bar_return.rolling(window).mean()),
                (f"return_efficiency_{window}", close.pct_change(window) / bar_return.abs().rolling(window).sum().replace(0, np.nan)),
                (f"downside_vol_{window}", bar_return.where(bar_return < 0, 0.0).rolling(window).std()),
                (f"upside_vol_{window}", bar_return.where(bar_return > 0, 0.0).rolling(window).std()),
                (f"close_location_{window}", close_location.rolling(window).mean()),
                (f"body_range_{window}", (body.abs() / spread).rolling(window).mean()),
                (f"upper_shadow_{window}", (upper_shadow / spread).rolling(window).mean()),
                (f"lower_shadow_{window}", (lower_shadow / spread).rolling(window).mean()),
                (f"price_volume_corr_{window}", bar_return.rolling(window).corr(volume_change)),
                (f"volume_volatility_{window}", volume_change.rolling(window).std()),
                (f"return_skew_{window}", bar_return.rolling(window).skew()),
                (f"return_kurt_{window}", bar_return.rolling(window).kurt()),
                (f"range_position_{window}", (close - rolling_low) / rolling_range - 0.5),
                (f"gap_mean_{window}", gap_return.rolling(window).mean()),
                (f"gap_volatility_{window}", gap_return.rolling(window).std()),
                (f"illiquidity_{window}", illiquidity.rolling(window).mean()),
                (f"volume_pressure_{window}", signed_volume.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)),
                (f"return_acceleration_{window}", bar_return.diff().rolling(window).mean()),
                (f"volatility_ratio_{window}", fast_vol / slow_vol - 1.0),
                (f"max_min_return_gap_{window}", bar_return.rolling(window).max() + bar_return.rolling(window).min()),
                (f"true_range_mean_{window}", true_range.rolling(window).mean()),
                (f"true_range_ratio_{window}", true_range / true_range.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"atr_close_ratio_{window}", true_range.rolling(window).mean() / close.replace(0, np.nan)),
                (f"gap_abs_mean_{window}", gap_return.abs().rolling(window).mean()),
                (f"close_mid_gap_{window}", (close / mid_price.replace(0, np.nan) - 1.0).rolling(window).mean()),
                (f"mid_momentum_{window}", mid_price.pct_change(window)),
                (f"typical_ret_{window}", typical_price.pct_change(window)),
                (f"typical_ma_gap_{window}", typical_price / typical_price.rolling(window).mean() - 1.0),
                (f"dollar_volume_ratio_{window}", dollar_volume / dollar_volume.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"dollar_volume_vol_{window}", dollar_volume_change.rolling(window).std()),
                (f"money_flow_sum_{window}", money_flow.rolling(window).sum()),
                (f"obv_change_{window}", obv.diff(window)),
                (f"obv_ma_gap_{window}", obv / obv.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"price_dollar_corr_{window}", bar_return.rolling(window).corr(dollar_volume_change)),
                (f"return_dollar_cov_{window}", bar_return.rolling(window).cov(dollar_volume_change)),
                (f"high_break_dist_{window}", close / rolling_high - 1.0),
                (f"low_break_dist_{window}", close / rolling_low - 1.0),
                (f"range_expansion_{window}", spread / spread.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"body_direction_sum_{window}", body_direction.rolling(window).sum()),
                (f"wick_imbalance_{window}", wick_imbalance.rolling(window).mean()),
                (f"open_close_corr_{window}", open_return.rolling(window).corr(bar_return)),
                (f"high_low_corr_{window}", high_return.rolling(window).corr(low_return)),
                (f"return_volume_beta_{window}", bar_return.rolling(window).cov(volume_change) / volume_var),
                (f"return_autocorr_{window}", bar_return.rolling(window).corr(bar_return.shift(1))),
                (f"volume_autocorr_{window}", volume_change.rolling(window).corr(volume_change.shift(1))),
                (f"gain_loss_ratio_{window}", positive_return.rolling(window).mean() / negative_abs_mean),
                (f"positive_rate_{window}", (bar_return > 0).astype(float).rolling(window).mean()),
                (f"negative_rate_{window}", (bar_return < 0).astype(float).rolling(window).mean()),
                (f"max_return_{window}", bar_return.rolling(window).max()),
                (f"min_return_{window}", bar_return.rolling(window).min()),
                (f"return_quantile_spread_{window}", bar_return.rolling(window).quantile(0.75) - bar_return.rolling(window).quantile(0.25)),
                (f"volume_quantile_spread_{window}", volume_change.rolling(window).quantile(0.75) - volume_change.rolling(window).quantile(0.25)),
                (f"price_zdist_{window}", (close - close_ma) / close_std),
                (f"volume_zdist_{window}", (volume - volume_ma) / volume_std),
                (f"liquidity_shock_{window}", illiquidity / illiquidity_ma.replace(0, np.nan) - 1.0),
                (f"return_median_{window}", bar_return.rolling(window).median()),
                (f"range_std_{window}", (spread / open_price).rolling(window).std()),
                (f"volume_trend_gap_{window}", volume.rolling(half_window).mean() / volume_ma.replace(0, np.nan) - 1.0),
                (f"body_mean_{window}", (body / open_price).rolling(window).mean()),
                (f"typical_trend_gap_{window}", typical_price.rolling(half_window).mean() / typical_price.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"trend_quality_{window}", bar_return.rolling(window).sum() / bar_return.abs().rolling(window).sum().replace(0, np.nan)),
                (f"trend_persistence_{window}", np.sign(bar_return).rolling(window).sum() / window),
                (f"up_down_return_sum_ratio_{window}", positive_abs_sum / negative_abs_sum),
                (f"close_rank_{window}", close_rank - 0.5),
                (f"volume_rank_{window}", volume_rank - 0.5),
                (f"range_rank_{window}", range_rank - 0.5),
                (f"price_volume_divergence_{window}", close_rank - volume_rank),
                (f"price_range_divergence_{window}", close_rank - range_rank),
                (f"range_body_ratio_{window}", spread.rolling(window).mean() / body_ma),
                (f"body_expansion_{window}", body.abs() / body_ma - 1.0),
                (f"shadow_balance_std_{window}", wick_imbalance.rolling(window).std()),
                (f"upper_lower_shadow_ratio_{window}", upper_shadow.rolling(window).mean() / lower_shadow.rolling(window).mean().replace(0, np.nan)),
                (f"money_flow_zdist_{window}", (money_flow - money_flow_ma) / money_flow_std),
                (f"money_flow_acceleration_{window}", money_flow.diff().rolling(window).mean()),
                (f"obv_slope_{window}", obv.diff().rolling(window).mean()),
                (f"fast_slow_return_sum_ratio_{window}", fast_return_sum / slow_return_sum),
                (f"fast_slow_volume_sum_ratio_{window}", fast_volume_sum / slow_volume_sum - 1.0),
                (f"volatility_of_volatility_{window}", bar_return.rolling(half_window).std().rolling(window).std()),
                (f"range_volatility_ratio_{window}", spread.rolling(half_window).std() / spread.rolling(window).std().replace(0, np.nan) - 1.0),
                (f"gap_direction_consistency_{window}", np.sign(gap_return).rolling(window).sum() / window),
                (f"gap_return_corr_{window}", gap_return.rolling(window).corr(bar_return)),
                (f"open_to_mid_gap_{window}", (df["open"] / rolling_mid.replace(0, np.nan) - 1.0).rolling(window).mean()),
                (f"close_to_mid_zdist_{window}", (close - rolling_mid) / mid_price.rolling(window).std().replace(0, np.nan)),
                (f"liquidity_trend_gap_{window}", illiquidity.rolling(half_window).mean() / illiquidity_ma.replace(0, np.nan) - 1.0),
                (f"volume_return_asymmetry_{window}", signed_volume.rolling(window).mean() / volume_ma.replace(0, np.nan)),
            ]
        )

        additional_factor_specs.extend(
            [
                (f"close_ma_slope_{window}", close_ma_slope),
                (f"volume_ma_slope_{window}", volume_ma_slope / volume_ma.replace(0, np.nan)),
                (f"range_ma_slope_{window}", range_ma_slope / range_ma),
                (f"return_abs_zdist_{window}", (abs_return - return_abs_ma) / return_abs_std),
                (f"intrabar_abs_mean_{window}", intrabar_abs.rolling(window).mean()),
                (f"intrabar_abs_ratio_{window}", intrabar_abs / intrabar_abs.rolling(window).mean().replace(0, np.nan) - 1.0),
                (f"signed_range_sum_{window}", signed_range.rolling(window).sum()),
                (f"signed_range_pressure_{window}", signed_range.rolling(window).sum() / spread.rolling(window).sum().replace(0, np.nan)),
                (f"close_typical_zdist_{window}", (close - typical_ma) / typical_std),
                (f"typical_price_slope_{window}", typical_ma.diff() / typical_ma.replace(0, np.nan)),
                (f"body_zdist_{window}", (body - body.rolling(window).mean()) / body_std),
                (f"body_sign_consistency_{window}", np.sign(body).rolling(window).sum() / window),
                (f"upper_shadow_zdist_{window}", (upper_shadow - upper_shadow.rolling(window).mean()) / upper_shadow.rolling(window).std().replace(0, np.nan)),
                (f"lower_shadow_zdist_{window}", (lower_shadow - lower_shadow.rolling(window).mean()) / lower_shadow.rolling(window).std().replace(0, np.nan)),
                (f"shadow_reversal_pressure_{window}", (lower_shadow.rolling(window).sum() - upper_shadow.rolling(window).sum()) / spread.rolling(window).sum().replace(0, np.nan)),
                (f"price_volume_rank_product_{window}", (close_rank - 0.5) * (volume_rank - 0.5)),
                (f"price_range_rank_product_{window}", (close_rank - 0.5) * (range_rank - 0.5)),
                (f"volume_range_rank_product_{window}", (volume_rank - 0.5) * (range_rank - 0.5)),
                (f"volume_adjusted_momentum_{window}", close.pct_change(window) / volume_rank.replace(0, np.nan)),
                (f"range_adjusted_momentum_{window}", close.pct_change(window) / range_rank.replace(0, np.nan)),
                (f"illiquidity_zdist_{window}", (illiquidity - illiquidity_ma) / illiquidity.rolling(window).std().replace(0, np.nan)),
                (f"turnover_price_impact_{window}", abs_return.rolling(window).sum() / dollar_volume.rolling(window).sum().replace(0, np.nan)),
                (f"money_flow_rank_{window}", money_flow.rolling(window).rank(pct=True) - 0.5),
                (f"obv_rank_{window}", obv.rolling(window).rank(pct=True) - 0.5),
                (f"return_gap_interaction_{window}", bar_return.rolling(window).mean() * gap_return.rolling(window).mean()),
                (f"return_rank_{window}", return_rank - 0.5),
                (f"abs_return_rank_{window}", abs_return_rank - 0.5),
                (f"gap_rank_{window}", gap_rank - 0.5),
                (f"illiquidity_rank_{window}", illiquidity_rank - 0.5),
                (f"dollar_volume_rank_{window}", dollar_volume_rank - 0.5),
                (f"true_range_rank_{window}", true_range_rank - 0.5),
                (f"return_volume_rank_gap_{window}", return_rank - volume_rank),
                (f"return_range_rank_gap_{window}", return_rank - range_rank),
                (f"abs_return_volume_rank_gap_{window}", abs_return_rank - volume_rank),
                (f"gap_volume_rank_gap_{window}", gap_rank - volume_rank),
                (f"illiquidity_volume_rank_gap_{window}", illiquidity_rank - volume_rank),
                (f"dollar_volume_price_rank_gap_{window}", dollar_volume_rank - close_rank),
                (f"true_range_price_rank_gap_{window}", true_range_rank - close_rank),
                (f"close_to_high_mean_{window}", close_to_high.rolling(window).mean()),
                (f"close_to_low_mean_{window}", close_to_low.rolling(window).mean()),
                (f"open_to_high_mean_{window}", open_to_high.rolling(window).mean()),
                (f"open_to_low_mean_{window}", open_to_low.rolling(window).mean()),
                (f"high_low_mid_gap_{window}", (close / high_low_mid.replace(0, np.nan) - 1.0).rolling(window).mean()),
                (f"mid_price_zdist_{window}", (mid_price - mid_price.rolling(window).mean()) / mid_price.rolling(window).std().replace(0, np.nan)),
                (f"true_range_zdist_{window}", (true_range - true_range.rolling(window).mean()) / true_range.rolling(window).std().replace(0, np.nan)),
                (f"spread_zdist_{window}", (spread - range_ma) / spread.rolling(window).std().replace(0, np.nan)),
                (f"volume_change_zdist_{window}", (volume_change - volume_change.rolling(window).mean()) / volume_change.rolling(window).std().replace(0, np.nan)),
                (f"dollar_volume_zdist_{window}", (dollar_volume - dollar_volume.rolling(window).mean()) / dollar_volume.rolling(window).std().replace(0, np.nan)),
                (f"gap_zdist_{window}", (gap_return - gap_return.rolling(window).mean()) / gap_return.rolling(window).std().replace(0, np.nan)),
                (f"return_rank_volume_product_{window}", (return_rank - 0.5) * (volume_rank - 0.5)),
                (f"return_rank_range_product_{window}", (return_rank - 0.5) * (range_rank - 0.5)),
                (f"gap_rank_return_product_{window}", (gap_rank - 0.5) * (return_rank - 0.5)),
                (f"illiquidity_rank_return_product_{window}", (illiquidity_rank - 0.5) * (return_rank - 0.5)),
                (f"dollar_volume_rank_return_product_{window}", (dollar_volume_rank - 0.5) * (return_rank - 0.5)),
                (f"true_range_rank_return_product_{window}", (true_range_rank - 0.5) * (return_rank - 0.5)),
                (f"positive_volume_pressure_{window}", volume.where(bar_return > 0, 0.0).rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)),
                (f"negative_volume_pressure_{window}", volume.where(bar_return < 0, 0.0).rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)),
                (f"volume_pressure_balance_{window}", (volume.where(bar_return > 0, 0.0).rolling(window).sum() - volume.where(bar_return < 0, 0.0).rolling(window).sum()) / volume.rolling(window).sum().replace(0, np.nan)),
                (f"positive_money_flow_pressure_{window}", money_flow.where(bar_return > 0, 0.0).rolling(window).sum() / money_flow.abs().rolling(window).sum().replace(0, np.nan)),
                (f"negative_money_flow_pressure_{window}", money_flow.where(bar_return < 0, 0.0).rolling(window).sum() / money_flow.abs().rolling(window).sum().replace(0, np.nan)),
                (f"money_flow_pressure_balance_{window}", money_flow.rolling(window).sum() / money_flow.abs().rolling(window).sum().replace(0, np.nan)),
                (f"return_downside_share_{window}", negative_return.abs().rolling(window).sum() / bar_return.abs().rolling(window).sum().replace(0, np.nan)),
                (f"return_upside_share_{window}", positive_return.rolling(window).sum() / bar_return.abs().rolling(window).sum().replace(0, np.nan)),
                (f"range_downside_share_{window}", spread.where(bar_return < 0, 0.0).rolling(window).sum() / spread.rolling(window).sum().replace(0, np.nan)),
                (f"range_upside_share_{window}", spread.where(bar_return > 0, 0.0).rolling(window).sum() / spread.rolling(window).sum().replace(0, np.nan)),
                (f"gap_abs_zdist_{window}", (gap_return.abs() - gap_return.abs().rolling(window).mean()) / gap_return.abs().rolling(window).std().replace(0, np.nan)),
                (f"open_return_zdist_{window}", (open_return - open_return.rolling(window).mean()) / open_return.rolling(window).std().replace(0, np.nan)),
                (f"high_return_zdist_{window}", (high_return - high_return.rolling(window).mean()) / high_return.rolling(window).std().replace(0, np.nan)),
                (f"low_return_zdist_{window}", (low_return - low_return.rolling(window).mean()) / low_return.rolling(window).std().replace(0, np.nan)),
                (f"typical_return_zdist_{window}", (typical_return - typical_return.rolling(window).mean()) / typical_return.rolling(window).std().replace(0, np.nan)),
                (f"return_open_corr_{window}", bar_return.rolling(window).corr(open_return)),
                (f"return_high_corr_{window}", bar_return.rolling(window).corr(high_return)),
                (f"return_low_corr_{window}", bar_return.rolling(window).corr(low_return)),
                (f"typical_volume_corr_{window}", typical_return.rolling(window).corr(volume_change)),
                (f"illiquidity_return_corr_{window}", illiquidity.rolling(window).corr(bar_return.abs())),
                (f"range_volume_corr_{window}", spread.rolling(window).corr(volume_change)),
                (f"body_volume_corr_{window}", body.abs().rolling(window).corr(volume_change)),
            ]
        )

        structural_inputs = {
            "body_pct": body_pct,
            "upper_shadow_pct": upper_shadow_pct,
            "lower_shadow_pct": lower_shadow_pct,
            "range_pct": range_pct,
            "close_open_gap": close_open_gap,
            "high_open_gap": high_open_gap,
            "low_open_gap": low_open_gap,
            "typical_open_gap": typical_open_gap,
            "mid_open_gap": mid_open_gap,
            "close_prev_gap": close_prev_gap,
            "volume_signed_return": volume_signed_return,
            "price_volume_product": price_volume_product,
            "range_volume_product": range_volume_product,
            "body_volume_product": body_volume_product,
            "wick_balance_pct": wick_balance_pct,
        }
        for struct_name, struct_value in structural_inputs.items():
            struct_ma = struct_value.rolling(window).mean()
            struct_std = struct_value.rolling(window).std().replace(0, np.nan)
            struct_abs_sum = struct_value.abs().rolling(window).sum().replace(0, np.nan)
            struct_half_ma = struct_value.rolling(half_window).mean()
            struct_half_std = struct_value.rolling(half_window).std()
            new_factor_specs.extend(
                [
                    (f"struct_mean_{struct_name}_{window}", struct_ma),
                    (f"struct_std_{struct_name}_{window}", struct_std),
                    (f"struct_median_{struct_name}_{window}", struct_value.rolling(window).median()),
                    (f"struct_rank_{struct_name}_{window}", struct_value.rolling(window).rank(pct=True) - 0.5),
                    (f"struct_zdist_{struct_name}_{window}", (struct_value - struct_ma) / struct_std),
                    (f"struct_half_full_gap_{struct_name}_{window}", struct_half_ma / struct_ma.replace(0, np.nan) - 1.0),
                    (f"struct_last_mean_gap_{struct_name}_{window}", struct_value / struct_ma.replace(0, np.nan) - 1.0),
                    (f"struct_sum_abs_share_{struct_name}_{window}", struct_value.rolling(window).sum() / struct_abs_sum),
                    (f"struct_accel_{struct_name}_{window}", struct_value.diff().rolling(window).mean()),
                    (f"struct_vol_ratio_{struct_name}_{window}", struct_half_std / struct_std - 1.0),
                ]
            )

        expansion_inputs = {
            "bar_return": bar_return,
            "intrabar_return": intrabar_return,
            "gap_return": gap_return,
            "volume_change": volume_change,
            "dollar_volume_change": dollar_volume_change,
            "money_flow": money_flow / dollar_volume.replace(0, np.nan),
            "close_location": close_location,
            "wick_imbalance": wick_imbalance,
            "body_pct": body_pct,
            "range_pct": range_pct,
            "close_open_gap": close_open_gap,
            "typical_return": typical_return,
            "open_return": open_return,
            "high_return": high_return,
            "low_return": low_return,
            "abs_return": abs_return,
            "true_range_pct": true_range / open_price,
            "signed_volume_ratio": signed_volume / volume.replace(0, np.nan),
            "illiquidity": illiquidity,
            "obv_change": obv.diff() / volume.replace(0, np.nan),
            "dollar_volume": dollar_volume.pct_change(),
            "close_rank_centered": close_rank - 0.5,
            "volume_rank_centered": volume_rank - 0.5,
            "range_rank_centered": range_rank - 0.5,
            "price_volume_product": price_volume_product,
        }
        for expansion_name, expansion_value in expansion_inputs.items():
            expansion_ma = expansion_value.rolling(window).mean()
            expansion_std = expansion_value.rolling(window).std().replace(0, np.nan)
            expansion_abs = expansion_value.abs()
            expansion_abs_ma = expansion_abs.rolling(window).mean().replace(0, np.nan)
            expansion_abs_sum = expansion_abs.rolling(window).sum().replace(0, np.nan)
            expansion_half_ma = expansion_value.rolling(half_window).mean()
            expansion_half_std = expansion_value.rolling(half_window).std()
            expansion_ewm = expansion_value.ewm(span=window, adjust=False, min_periods=half_window).mean()
            expansion_positive_sum = expansion_value.where(expansion_value > 0, 0.0).rolling(window).sum()
            expansion_negative_sum = expansion_value.where(expansion_value < 0, 0.0).abs().rolling(window).sum()
            expansion_factor_specs.extend(
                [
                    (f"mega_mean_{expansion_name}_{window}", expansion_ma),
                    (f"mega_std_{expansion_name}_{window}", expansion_std),
                    (f"mega_median_{expansion_name}_{window}", expansion_value.rolling(window).median()),
                    (f"mega_rank_{expansion_name}_{window}", expansion_value.rolling(window).rank(pct=True) - 0.5),
                    (f"mega_zdist_{expansion_name}_{window}", (expansion_value - expansion_ma) / expansion_std),
                    (f"mega_abs_mean_{expansion_name}_{window}", expansion_abs_ma),
                    (f"mega_abs_zdist_{expansion_name}_{window}", (expansion_abs - expansion_abs_ma) / expansion_abs.rolling(window).std().replace(0, np.nan)),
                    (f"mega_half_full_gap_{expansion_name}_{window}", expansion_half_ma / expansion_ma.replace(0, np.nan) - 1.0),
                    (f"mega_half_std_ratio_{expansion_name}_{window}", expansion_half_std / expansion_std - 1.0),
                    (f"mega_ewm_gap_{expansion_name}_{window}", expansion_value - expansion_ewm),
                    (f"mega_diff_mean_{expansion_name}_{window}", expansion_value.diff().rolling(window).mean()),
                    (f"mega_diff_std_{expansion_name}_{window}", expansion_value.diff().rolling(window).std()),
                    (f"mega_accel_mean_{expansion_name}_{window}", expansion_value.diff().diff().rolling(window).mean()),
                    (f"mega_positive_share_{expansion_name}_{window}", expansion_positive_sum / expansion_abs_sum),
                    (f"mega_negative_share_{expansion_name}_{window}", expansion_negative_sum / expansion_abs_sum),
                    (f"mega_sum_abs_share_{expansion_name}_{window}", expansion_value.rolling(window).sum() / expansion_abs_sum),
                    (f"mega_max_{expansion_name}_{window}", expansion_value.rolling(window).max()),
                    (f"mega_min_{expansion_name}_{window}", expansion_value.rolling(window).min()),
                    (f"mega_quantile_spread_{expansion_name}_{window}", expansion_value.rolling(window).quantile(0.75) - expansion_value.rolling(window).quantile(0.25)),
                    (f"mega_return_corr_{expansion_name}_{window}", expansion_value.rolling(window).corr(bar_return)),
                ]
            )

    # 前 6400 个参数化因子保持历史编号稳定，本次再新增 10000 个 mega 结构因子用于增量测试。
    factor_specs = (
        factor_specs[:1900]
        + additional_factor_specs[:1500]
        + new_factor_specs[:3000]
        + expansion_factor_specs[:10000]
    )
    parametric_factors = {
        factor_name: rolling_zscore(raw_factor, config.zscore_window)
        for factor_name, raw_factor in factor_specs
    }
    return pd.DataFrame(parametric_factors, index=df.index)

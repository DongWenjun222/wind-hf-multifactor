from __future__ import annotations

"""轻量级项目冒烟测试。

这个脚本刻意保持轻量且非破坏性：
- 不更新因子库。
- 不覆盖单因子或综合因子回测输出。
- 只检查主要模块能否在一小段本地样本上正常运行。

建议在代码改动后、启动耗时较长的完整回测前运行：

    python smoke_test.py
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from composite_factor_backtest import (
    build_xgboost_rolling_signal,
    load_active_factor_pool,
    load_active_factor_names,
)
from config import BacktestConfig
from framework.factors import build_factors, fetch_intraday_data, get_factor_columns
from single_factor_backtest import (
    build_signal_from_score,
    choose_factor_direction,
    safe_run_backtest,
    split_train_validation_test_index,
)


def make_smoke_config() -> BacktestConfig:
    """创建一个快速、仅使用本地数据的冒烟测试配置。"""
    config = BacktestConfig()
    config.end_time = "2025-01-20 15:00:00"
    config.prefer_local_data = True
    config.enable_cross_asset_factors = False
    config.enable_experiment_run_dirs = False
    config.single_factor_plot_all = False
    config.single_factor_plot_top_n = 0
    config.zscore_window = 20
    config.xgboost_train_window = 40
    config.xgboost_min_train_samples = 20
    config.xgboost_retrain_every = 20
    config.xgboost_n_estimators = 5
    config.xgboost_best_top_n = 5
    config.xgboost_feature_mode = "signal"
    config.xgboost_target_horizon = 3
    config.xgboost_trade_use_market_filters = False
    config.xgboost_trade_use_confidence_rank_filter = False
    config.xgboost_use_dynamic_position_sizing = False
    config.xgboost_train_use_market_filters = False
    config.xgboost_train_min_directional_samples = 0
    config.xgboost_train_nonzero_class_weight = 1.0
    config.xgboost_train_neutral_class_weight = 1.0
    return config


def assert_non_empty_frame(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise AssertionError(f"{name} 为空")


def select_smoke_active_factors(config: BacktestConfig) -> list[str]:
    """读取少量 active 名称；空库由调用方使用非持久化代理继续检查链路。"""
    try:
        active_names = load_active_factor_names(config)
    except ValueError as exc:
        if "active 因子库为空" not in str(exc):
            raise
        active_names = []
    return list(dict.fromkeys(active_names))[:5]


def smoke_data_and_factors(
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("1/4 读取本地行情并按需构建少量 active 因子...")
    data = fetch_intraday_data(config)
    data = data.tail(300).copy()
    requested_factors = ["momentum", "reversal", "breakout", "volume_confirm"]
    factors = build_factors(data, config, requested_factors=requested_factors)
    usable_columns = [
        column
        for column in factors.columns
        if factors[column].replace([np.inf, -np.inf], np.nan).notna().sum() >= 30
        and factors[column].nunique(dropna=True) > 1
    ][:5]
    factors = factors[usable_columns].copy()
    assert_non_empty_frame(data, "data")
    assert_non_empty_frame(factors, "factors")
    print(
        f"    data={data.shape}, factors={factors.shape}, "
        f"requested={len(requested_factors)}"
    )
    return data, factors


def smoke_single_factor_backtest(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    print("2/4 检查前三个单因子回测...")
    split_time, _ = split_train_validation_test_index(
        data.index,
        config.auto_select_train_ratio,
        config.auto_select_validation_ratio,
    )
    successful_factors = []
    skipped_factors = []
    for factor_name in get_factor_columns(factors):
        direction, _ = choose_factor_direction(
            data,
            factors[factor_name],
            factor_name,
            config,
            split_time,
        )
        signal = build_signal_from_score(
            factors[factor_name],
            config.signal_threshold,
            factor_name,
            direction=direction,
        )
        # 冒烟测试验证函数链路，不承担样本外评估；短样本使用全段可避免
        # 验证/测试尾段过短导致所有滚动标准化信号都为空。
        backtest_df, metrics, error = safe_run_backtest(data, signal, config)
        if error is not None or backtest_df is None:
            skipped_factors.append(f"{factor_name}: {error}")
            continue
        if pd.isna(metrics.get("累计收益")):
            raise AssertionError(f"单因子指标无效: {factor_name}")
        successful_factors.append(factor_name)
        if len(successful_factors) >= 3:
            break

    if not successful_factors:
        raise AssertionError(
            "没有单因子通过冒烟回测；跳过原因: " + "; ".join(skipped_factors[:5])
        )
    print(
        f"    单因子回测 OK，通过={len(successful_factors)}，"
        f"短样本无信号跳过={len(skipped_factors)}"
    )


def smoke_active_factor_pool(
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, list[str]]:
    print("3/4 检查 active 因子池...")
    usable_source_columns = get_factor_columns(factors)
    if not usable_source_columns:
        raise AssertionError("没有可用于 active 代理列的基础因子")
    active_names = select_smoke_active_factors(config)
    using_library = bool(active_names)
    if not using_library:
        active_names = usable_source_columns[:5]
        print("    active 因子库为空，使用基础因子临时代理验证模型链路（不会写回因子库）。")

    # 冒烟测试只验证 active 硬边界和模型调用链，不重复计算庞大的参数网格。
    active_proxy = pd.DataFrame(index=factors.index)
    for position, active_name in enumerate(active_names):
        source_name = usable_source_columns[position % len(usable_source_columns)]
        active_proxy[active_name] = factors[source_name]

    active_factors = (
        load_active_factor_pool(active_proxy, config)
        if using_library
        else active_names
    )
    if not active_factors:
        raise AssertionError("active 因子池为空")
    print(f"    可用 active 因子数={len(active_factors)}")
    return active_proxy, active_factors


def smoke_optional_xgboost(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    active_factors: list[str],
    config: BacktestConfig,
) -> None:
    print("4/4 可选检查 XGBoost 小窗口滚动预测...")
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("    未安装 xgboost，跳过模型 smoke。")
        return

    _, validation_end_time = split_train_validation_test_index(
        data.index,
        config.auto_select_train_ratio,
        config.auto_select_validation_ratio,
    )
    predict_index = factors.loc[factors.index >= validation_end_time].index[:40]
    if len(predict_index) < config.xgboost_min_train_samples:
        print("    样本太短，跳过模型 smoke。")
        return

    model_config = replace(config)
    model_config.xgboost_feature_scope = "all"
    signal, feature_importance, _features, _selection = build_xgboost_rolling_signal(
        data,
        factors,
        active_factors[: model_config.xgboost_best_top_n],
        model_config,
        predict_index,
    )
    if signal["raw_signal"].dropna().empty:
        raise AssertionError("XGBoost 冒烟测试生成了空 raw_signal")
    if feature_importance.empty:
        raise AssertionError("XGBoost 冒烟测试生成了空 feature_importance")

    print("    XGBoost 小窗口 smoke OK")


def main() -> None:
    config = make_smoke_config()
    data, factors = smoke_data_and_factors(config)
    smoke_single_factor_backtest(data, factors, config)
    active_factors_frame, active_factors = smoke_active_factor_pool(factors, config)
    smoke_optional_xgboost(data, active_factors_frame, active_factors, config)
    print("\nsmoke_test 通过。")


if __name__ == "__main__":
    main()

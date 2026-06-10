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

import pandas as pd

from composite_factor_backtest import (
    build_xgboost_rolling_signal,
    load_active_factor_pool,
)
from config import BacktestConfig
from factors import build_factors, fetch_intraday_data, get_factor_columns
from single_factor_backtest import (
    build_signal_from_score,
    choose_factor_direction,
    safe_run_backtest,
    split_train_test_index,
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
    config.xgboost_train_window = 60
    config.xgboost_min_train_samples = 30
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


def smoke_data_and_factors(config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("1/4 读取本地行情并构建因子...")
    data = fetch_intraday_data(config)
    factors = build_factors(data, config)
    assert_non_empty_frame(data, "data")
    assert_non_empty_frame(factors, "factors")
    print(f"    data={data.shape}, factors={factors.shape}")
    return data, factors


def smoke_single_factor_backtest(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    print("2/4 检查前三个单因子回测...")
    split_time = split_train_test_index(data.index, config.auto_select_train_ratio)
    test_data = data.loc[data.index >= split_time]

    for factor_name in get_factor_columns(factors)[:3]:
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
        test_signal = signal.loc[signal.index >= split_time]
        backtest_df, metrics, error = safe_run_backtest(test_data, test_signal, config)
        if error is not None or backtest_df is None:
            raise AssertionError(f"单因子冒烟测试失败: {factor_name}, {error}")
        if pd.isna(metrics.get("累计收益")):
            raise AssertionError(f"单因子指标无效: {factor_name}")

    print("    单因子回测 OK")


def smoke_active_factor_pool(factors: pd.DataFrame, config: BacktestConfig) -> list[str]:
    print("3/4 检查 active 因子池...")
    active_factors = load_active_factor_pool(factors, config)
    if not active_factors:
        raise AssertionError("active 因子池为空")
    print(f"    可用 active 因子数={len(active_factors)}")
    return active_factors


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

    split_time = split_train_test_index(data.index, config.auto_select_train_ratio)
    predict_index = factors.loc[factors.index >= split_time].index[:40]
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
    active_factors = smoke_active_factor_pool(factors, config)
    smoke_optional_xgboost(data, factors, active_factors, config)
    print("\nsmoke_test 通过。")


if __name__ == "__main__":
    main()

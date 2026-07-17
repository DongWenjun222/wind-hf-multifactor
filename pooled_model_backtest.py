from __future__ import annotations

"""多品种共享信息建模入口。

本脚本在现有“逐品种独立建模”之外，增加一条板块/全市场 pooled model 流程：
- 每个品种仍然使用自己的行情、因子和 active 因子库。
- 在共享组内把多个品种的样本拼成 long-format 训练集。
- 训练时按时间滚动，只使用当前预测时点之前已经落地的标签。
- 预测结果再落回每个品种分别回测，便于和逐品种模型对比。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from composite_factor_backtest import (
    CLASS_TO_TARGET,
    apply_position_rules,
    build_dynamic_confidence_position_size,
    build_factor_signal_features,
    build_features_for_factors,
    build_training_sample_weights,
    calculate_future_horizon_return,
    calculate_next_bar_direction,
    calculate_prediction_metrics_for_segment,
    get_xgboost_target_horizon,
    predict_composite_probability,
    probabilities_to_trade_signal,
    train_composite_classifier,
)
from config import BacktestConfig, resolve_symbol_universe
from factors import build_factors, fetch_intraday_data, safe_symbol_name, stop_wind
from multi_symbol_backtest import (
    PORTFOLIO_METHODS,
    apply_cross_sectional_opportunity_selection,
    apply_group_risk_budget,
    apply_rolling_weights,
    build_cross_sectional_opportunity_scores,
    build_group_contribution_frame,
    build_group_weight_frame,
    build_portfolio_risk_multiplier,
    build_rolling_portfolio_weights,
    build_static_portfolio_weights,
    build_strategy_column_group_map,
    build_symbol_config,
    get_symbol_from_strategy_column,
    get_symbol_group,
    normalize_and_cap_weights,
    plot_multi_symbol_portfolio,
)
from runtime_utils import run_tracked
from single_factor_backtest import calculate_metrics, infer_annual_periods, run_backtest


PROBABILITY_COLUMNS = ["prob_down", "prob_flat", "prob_up"]


def get_pooled_output_dir(config: BacktestConfig) -> Path:
    """返回共享模型输出目录。"""
    output_dir = Path(config.output_dir) / str(getattr(config, "pooled_model_output_subdir", "pooled_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_pooled_symbols(config: BacktestConfig) -> list[str]:
    """读取共享模型要覆盖的品种列表。"""
    configured = getattr(config, "pooled_model_symbols", []) or getattr(config, "symbols", [])
    if isinstance(configured, str):
        return resolve_symbol_universe(configured)
    return resolve_symbol_universe([str(symbol) for symbol in configured])


def read_symbol_active_library(symbol_config: BacktestConfig) -> pd.DataFrame:
    """读取单个品种 active 因子库。"""
    active_path = Path(symbol_config.output_dir) / "factor_library" / "active_factors.csv"
    if not active_path.exists():
        return pd.DataFrame()
    try:
        active = pd.read_csv(active_path)
    except Exception:
        return pd.DataFrame()
    if "因子" not in active.columns:
        return pd.DataFrame()
    active = active.copy()
    active["因子"] = active["因子"].astype(str)
    return active[active["因子"].str.strip().ne("")]


def score_active_factor_rows(active_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总各品种 active 因子的出现次数和平均质量分。"""
    rows = []
    for symbol, active in active_by_symbol.items():
        if active.empty or "因子" not in active.columns:
            continue
        for _, row in active.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "factor": str(row["因子"]),
                    "predictive_score": pd.to_numeric(row.get("初筛预测能力评分"), errors="coerce"),
                    "research_score": pd.to_numeric(row.get("初筛科研综合评分"), errors="coerce"),
                    "sharpe": pd.to_numeric(row.get("初筛夏普"), errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame()
    all_rows = pd.DataFrame(rows)
    summary = (
        all_rows.groupby("factor", dropna=False)
        .agg(
            active_symbol_count=("symbol", "nunique"),
            avg_predictive_score=("predictive_score", "mean"),
            avg_research_score=("research_score", "mean"),
            avg_sharpe=("sharpe", "mean"),
            symbols=("symbol", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
        .sort_values(
            ["active_symbol_count", "avg_predictive_score", "avg_research_score", "avg_sharpe"],
            ascending=[False, False, False, False],
        )
    )
    return summary


def select_shared_factors(active_by_symbol: dict[str, pd.DataFrame], config: BacktestConfig) -> list[str]:
    """根据配置从各品种 active 因子库中选择共享基础因子。"""
    factor_sets = [
        set(active["因子"].dropna().astype(str))
        for active in active_by_symbol.values()
        if not active.empty and "因子" in active.columns
    ]
    if not factor_sets:
        return []

    mode = str(getattr(config, "pooled_model_feature_source", "active_union") or "active_union").lower()
    max_features = max(1, int(getattr(config, "pooled_model_max_features", 80) or 80))
    if mode == "active_intersection":
        selected = sorted(set.intersection(*factor_sets))
        return selected[:max_features]

    summary = score_active_factor_rows(active_by_symbol)
    if summary.empty:
        return []
    return summary["factor"].head(max_features).astype(str).tolist()


def add_pooled_identity_features(
    feature_frame: pd.DataFrame,
    symbol: str,
    group_name: str,
    all_symbols: list[str],
    all_groups: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """给 pooled 样本追加 symbol/group one-hot 特征。"""
    frame = feature_frame.copy()
    if bool(getattr(config, "pooled_model_include_symbol_features", True)):
        for candidate in all_symbols:
            frame[f"symbol_{safe_symbol_name(candidate).upper()}"] = 1.0 if candidate == symbol else 0.0
    if bool(getattr(config, "pooled_model_include_group_features", True)):
        for candidate in all_groups:
            frame[f"group_{safe_symbol_name(candidate)}"] = 1.0 if candidate == group_name else 0.0
    return frame


def build_symbol_pooled_dataset(
    symbol: str,
    symbol_config: BacktestConfig,
    selected_factors: list[str],
    all_symbols: list[str],
    all_groups: list[str],
    group_name: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """构建单个品种供 pooled model 使用的特征、标签和行情。"""
    print(f"构建 pooled 数据集: {symbol}，共享因子数={len(selected_factors)}")
    data = fetch_intraday_data(symbol_config)
    max_bars = int(getattr(config, "pooled_model_max_bars_per_symbol", 0) or 0)
    if max_bars > 0 and len(data) > max_bars:
        data = data.tail(max_bars).copy()

    factors = build_factors(data, symbol_config, requested_factors=selected_factors)
    available_factors = [factor for factor in selected_factors if factor in factors.columns]
    if not available_factors:
        return pd.DataFrame(), data, factors, []

    signal_features = build_factor_signal_features(factors, available_factors, symbol_config)
    features = build_features_for_factors(factors, signal_features, available_factors, symbol_config)
    features = add_pooled_identity_features(
        features,
        symbol=symbol,
        group_name=group_name,
        all_symbols=all_symbols,
        all_groups=all_groups,
        config=config,
    )
    target = calculate_next_bar_direction(data, features.index, symbol_config)
    future_return = calculate_future_horizon_return(data, features.index, symbol_config)
    dataset = features.copy()
    dataset["target"] = target
    dataset["future_horizon_return"] = future_return
    dataset["timestamp"] = dataset.index
    dataset["symbol"] = symbol
    dataset["group"] = group_name
    dataset["row_position"] = np.arange(len(dataset), dtype="int64")
    return dataset, data, factors, available_factors


def build_grouped_symbol_map(symbols: list[str], config: BacktestConfig) -> dict[str, list[str]]:
    """按 sector 或 market 模式把品种分组。"""
    scope = str(getattr(config, "pooled_model_scope", "sector") or "sector").lower()
    grouped: dict[str, list[str]] = {}
    for symbol in symbols:
        group_name = "全市场" if scope == "market" else get_symbol_group(symbol, config)
        grouped.setdefault(group_name, []).append(symbol)
    return grouped


def fit_predict_pooled_group(
    group_name: str,
    group_dataset: pd.DataFrame,
    feature_columns: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """对一个共享组做滚动训练和逐时点预测。"""
    model_name = str(getattr(config, "pooled_model_name", "xgboost") or "xgboost")
    horizon = get_xgboost_target_horizon(config)
    purge_delta = pd.Timedelta(minutes=int(config.bar_size) * horizon)
    retrain_every = max(1, int(getattr(config, "xgboost_retrain_every", 25) or 25))
    min_train_samples = max(1, int(getattr(config, "xgboost_min_train_samples", 600) or 600))
    train_window = max(1, int(getattr(config, "xgboost_train_window", 1200) or 1200))
    max_train_rows = max(0, int(getattr(config, "pooled_model_max_train_rows", 60000) or 0))

    dataset = group_dataset.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    unique_times = pd.Index(sorted(dataset["timestamp"].dropna().unique()))
    predictions = pd.DataFrame(
        np.nan,
        index=dataset.index,
        columns=PROBABILITY_COLUMNS + ["xgboost_predicted_direction"],
        dtype="float64",
    )
    model = None
    last_train_step = -10**9
    train_rows_used = 0

    for step, timestamp in enumerate(unique_times):
        current_mask = dataset["timestamp"] == timestamp
        if not current_mask.any():
            continue

        train_cutoff = pd.Timestamp(timestamp) - purge_delta
        train_frame = dataset.loc[
            dataset["timestamp"] <= train_cutoff,
            feature_columns + ["target", "timestamp"],
        ].replace([np.inf, -np.inf], np.nan)
        train_frame = train_frame.dropna(subset=["target"])
        if len(train_frame) > train_window:
            train_frame = train_frame.tail(train_window)
        if max_train_rows > 0 and len(train_frame) > max_train_rows:
            train_frame = train_frame.tail(max_train_rows)
        if len(train_frame) < min_train_samples or train_frame["target"].nunique(dropna=True) < 2:
            continue

        if model is None or step - last_train_step >= retrain_every:
            train_target = train_frame["target"].astype("float64")
            train_features = train_frame[feature_columns]
            sample_weight = build_training_sample_weights(train_target, config)
            model = train_composite_classifier(
                model_name,
                train_features,
                train_target,
                feature_columns,
                config,
                sample_weight=sample_weight,
            )
            last_train_step = step
            train_rows_used = len(train_frame)

        current_features = dataset.loc[current_mask, feature_columns].replace([np.inf, -np.inf], np.nan)
        probability = predict_composite_probability(model_name, model, current_features, feature_columns)
        current_index = dataset.index[current_mask]
        predictions.loc[current_index, PROBABILITY_COLUMNS] = probability
        predictions.loc[current_index, "xgboost_predicted_direction"] = [
            CLASS_TO_TARGET[int(class_id)] for class_id in np.argmax(probability, axis=1)
        ]

    result = dataset[["timestamp", "symbol", "group", "target", "future_horizon_return"]].copy()
    result = result.join(predictions)
    result["train_rows_used_last"] = train_rows_used
    return result


def build_symbol_signal_from_predictions(prediction_df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """把 pooled 模型概率预测转换成单品种回测信号。"""
    indexed = prediction_df.set_index("timestamp").sort_index()
    probabilities = indexed[PROBABILITY_COLUMNS].astype("float64")
    raw_signal = probabilities_to_trade_signal(probabilities, config).fillna(0.0)
    position_size = build_dynamic_confidence_position_size(
        probabilities,
        pd.Series(float(config.xgboost_trade_min_edge), index=probabilities.index),
        pd.Series(float(config.xgboost_trade_min_probability), index=probabilities.index),
        config,
    ).fillna(0.0)
    target_position_before_rules = raw_signal * position_size
    target_position = apply_position_rules(target_position_before_rules, config)
    signal = pd.DataFrame(index=probabilities.index)
    signal["composite_score"] = probabilities["prob_up"] - probabilities["prob_down"]
    signal["raw_signal"] = np.sign(target_position).astype("float64")
    signal["target_position"] = target_position
    signal["position"] = target_position.shift(1).fillna(0.0)
    signal["target_direction"] = indexed["target"]
    signal["future_horizon_return"] = indexed["future_horizon_return"]
    signal["xgboost_predicted_direction"] = indexed["xgboost_predicted_direction"]
    signal["calibrated_predicted_direction"] = indexed["xgboost_predicted_direction"]
    signal["calibrated_prob_edge"] = signal["composite_score"]
    signal = signal.join(probabilities)
    return signal


def build_pooled_portfolio_input(detail_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """把 pooled 单品种回测明细拼成多品种组合所需的宽表。"""
    wide_frames = []
    for detail in detail_frames:
        if detail.empty or "symbol" not in detail.columns:
            continue
        symbol = str(detail["symbol"].dropna().iloc[0]) if detail["symbol"].notna().any() else ""
        if not symbol:
            continue
        frame = detail.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[frame.index.notna()].sort_index()
        if frame.empty:
            continue

        output = pd.DataFrame(index=frame.index)
        column_map = {
            "strategy_net_return": f"{symbol}_strategy_return",
            "benchmark_return": f"{symbol}_benchmark_return",
            "position": f"{symbol}_position",
            "calibrated_prob_edge": f"{symbol}_calibrated_prob_edge",
            "directional_probability": f"{symbol}_directional_probability",
            "confidence_rank": f"{symbol}_confidence_rank",
        }
        for source_column, target_column in column_map.items():
            if source_column in frame.columns:
                output[target_column] = pd.to_numeric(frame[source_column], errors="coerce")
        if f"{symbol}_confidence_rank" not in output.columns and f"{symbol}_calibrated_prob_edge" in output.columns:
            # 横截面机会选择需要一个越大越好的置信度秩，缺失时用概率优势的历史分位近似。
            output[f"{symbol}_confidence_rank"] = (
                output[f"{symbol}_calibrated_prob_edge"].abs().rolling(240, min_periods=20).rank(pct=True)
            )
        wide_frames.append(output)

    if not wide_frames:
        return pd.DataFrame()
    return pd.concat(wide_frames, axis=1).sort_index()


def save_pooled_portfolio_outputs(
    detail_frames: list[pd.DataFrame],
    config: BacktestConfig,
    output_dir: Path,
) -> None:
    """基于 pooled 单品种结果生成多品种组合层明细、摘要和图表。"""
    portfolio = build_pooled_portfolio_input(detail_frames)
    if portfolio.empty:
        return

    strategy_columns = [column for column in portfolio.columns if column.endswith("_strategy_return")]
    benchmark_columns = [column for column in portfolio.columns if column.endswith("_benchmark_return")]
    position_columns = [column for column in portfolio.columns if column.endswith("_position")]
    if not strategy_columns:
        return

    annual_periods = infer_annual_periods(portfolio.index, config.annual_trading_days)
    portfolio["active_symbol_count"] = portfolio[strategy_columns].notna().sum(axis=1)
    strategy_returns = portfolio[strategy_columns]
    benchmark_returns = portfolio[benchmark_columns]
    abs_positions = portfolio[position_columns].abs()
    opportunity_scores = build_cross_sectional_opportunity_scores(portfolio, strategy_columns, config)
    strategy_column_group_map = build_strategy_column_group_map(strategy_columns, config)
    symbol_group_map = {
        get_symbol_from_strategy_column(column): group
        for column, group in strategy_column_group_map.items()
    }

    summary_rows = []
    weight_frames = []
    contribution_frames = []
    opportunity_selection_frames = []
    group_weight_frames = []
    group_contribution_frames = []
    use_rolling_weights = bool(getattr(config, "multi_symbol_use_rolling_portfolio_weights", True))
    weight_window = int(getattr(config, "multi_symbol_portfolio_weight_window", 480) or 480)
    min_weight_samples = int(getattr(config, "multi_symbol_portfolio_min_weight_samples", 120) or 120)
    max_symbol_weight = float(getattr(config, "multi_symbol_portfolio_max_symbol_weight", 1.0) or 1.0)

    for method, label in PORTFOLIO_METHODS.items():
        if use_rolling_weights:
            weights_by_time = build_rolling_portfolio_weights(
                strategy_returns,
                method,
                annual_periods,
                weight_window,
                min_weight_samples,
                max_symbol_weight,
            )
        else:
            weights = build_static_portfolio_weights(strategy_returns, method, annual_periods)
            weights = normalize_and_cap_weights(weights, max_symbol_weight)
            weights_by_time = pd.DataFrame(
                {column: float(weights.get(column, 0.0)) for column in strategy_returns.columns},
                index=portfolio.index,
            )
        if weights_by_time.empty:
            continue

        base_weights_by_time = weights_by_time.copy()
        weights_by_time, opportunity_selected = apply_cross_sectional_opportunity_selection(
            weights_by_time,
            opportunity_scores,
            max_symbol_weight,
            config,
        )
        weights_by_time = apply_group_risk_budget(weights_by_time, strategy_column_group_map, config)
        symbol_weight_columns = {
            column: get_symbol_from_strategy_column(column) for column in weights_by_time.columns
        }
        benchmark_weights_by_time = weights_by_time.rename(
            columns={
                strategy_column: f"{symbol}_benchmark_return"
                for strategy_column, symbol in symbol_weight_columns.items()
            }
        )
        position_weights_by_time = weights_by_time.rename(
            columns={
                strategy_column: f"{symbol}_position"
                for strategy_column, symbol in symbol_weight_columns.items()
            }
        )

        portfolio[f"{method}_strategy_return"] = apply_rolling_weights(strategy_returns, weights_by_time)
        portfolio[f"{method}_benchmark_return"] = apply_rolling_weights(
            benchmark_returns,
            benchmark_weights_by_time,
        )
        portfolio[f"{method}_avg_abs_position"] = apply_rolling_weights(abs_positions, position_weights_by_time)
        portfolio[f"{method}_raw_strategy_return"] = portfolio[f"{method}_strategy_return"]
        portfolio[f"{method}_raw_avg_abs_position"] = portfolio[f"{method}_avg_abs_position"]

        risk_state = build_portfolio_risk_multiplier(
            portfolio[f"{method}_raw_strategy_return"],
            annual_periods,
            config,
        )
        portfolio[f"{method}_risk_multiplier"] = risk_state["risk_multiplier"]
        portfolio[f"{method}_realized_annual_vol"] = risk_state["realized_annual_vol"]
        portfolio[f"{method}_vol_target_multiplier"] = risk_state["vol_target_multiplier"]
        portfolio[f"{method}_risk_drawdown"] = risk_state["drawdown"]
        portfolio[f"{method}_drawdown_multiplier"] = risk_state["drawdown_multiplier"]
        portfolio[f"{method}_strategy_return"] = (
            portfolio[f"{method}_raw_strategy_return"] * portfolio[f"{method}_risk_multiplier"]
        )
        portfolio[f"{method}_avg_abs_position"] = (
            portfolio[f"{method}_raw_avg_abs_position"] * portfolio[f"{method}_risk_multiplier"]
        )
        portfolio[f"{method}_nav"] = (1.0 + portfolio[f"{method}_strategy_return"]).cumprod()
        portfolio[f"{method}_benchmark_nav"] = (1.0 + portfolio[f"{method}_benchmark_return"]).cumprod()
        portfolio[f"{method}_drawdown"] = portfolio[f"{method}_nav"] / portfolio[f"{method}_nav"].cummax() - 1.0

        renamed_weights = weights_by_time.rename(columns=symbol_weight_columns)
        weight_frames.append(renamed_weights.add_prefix(f"{method}_"))
        opportunity_selection_frames.append(
            opportunity_selected.rename(columns=symbol_weight_columns).add_prefix(f"{method}_")
        )
        contribution_frame = (
            strategy_returns.fillna(0.0)
            .mul(weights_by_time, axis=0)
            .mul(portfolio[f"{method}_risk_multiplier"], axis=0)
            .rename(columns=symbol_weight_columns)
            .add_prefix(f"{method}_")
        )
        contribution_frames.append(contribution_frame)
        group_weights = build_group_weight_frame(weights_by_time, strategy_column_group_map)
        if not group_weights.empty:
            group_weight_frames.append(group_weights.add_prefix(f"{method}_"))
        method_symbol_group_map = {
            f"{method}_{symbol}": group
            for symbol, group in symbol_group_map.items()
        }
        group_contribution = build_group_contribution_frame(contribution_frame, method_symbol_group_map)
        if not group_contribution.empty:
            group_contribution_frames.append(group_contribution.add_prefix(f"{method}_"))

        metrics = calculate_metrics(
            portfolio[f"{method}_strategy_return"],
            portfolio[f"{method}_benchmark_return"],
            portfolio[f"{method}_avg_abs_position"],
            annual_periods,
        )
        avg_symbol_weights = renamed_weights.mean()
        metrics["组合方式"] = label
        metrics["来源"] = "pooled共享模型"
        metrics["权重方式"] = "滚动历史权重" if use_rolling_weights else "全样本静态权重"
        metrics["权重窗口"] = float(weight_window if use_rolling_weights else 0)
        metrics["单品种最大权重"] = float(max_symbol_weight)
        metrics["参与品种数"] = float(len(avg_symbol_weights))
        metrics["平均每根K线活跃品种数"] = float(portfolio["active_symbol_count"].mean())
        metrics["机会选择前平均入选品种数"] = float((base_weights_by_time > 0).sum(axis=1).mean())
        metrics["机会选择后平均入选品种数"] = float((weights_by_time > 0).sum(axis=1).mean())
        metrics["平均风险乘数"] = float(portfolio[f"{method}_risk_multiplier"].mean())
        metrics["最低风险乘数"] = float(portfolio[f"{method}_risk_multiplier"].min())
        metrics["平均绝对仓位"] = float(portfolio[f"{method}_avg_abs_position"].mean())
        metrics["品种权重"] = ",".join(f"{symbol}:{weight:.4f}" for symbol, weight in avg_symbol_weights.items())
        if not group_weights.empty:
            metrics["板块权重"] = ",".join(
                f"{group}:{weight:.4f}" for group, weight in group_weights.mean().sort_values(ascending=False).items()
            )
        summary_rows.append(metrics)

    portfolio.to_csv(output_dir / "pooled_portfolio_detail.csv", encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output_dir / "pooled_portfolio_summary.csv", index=False, encoding="utf-8-sig")
    if weight_frames:
        pd.concat(weight_frames, axis=1).to_csv(output_dir / "pooled_portfolio_weights.csv", encoding="utf-8-sig")
    if not opportunity_scores.empty:
        opportunity_scores.rename(
            columns={column: get_symbol_from_strategy_column(column) for column in opportunity_scores.columns}
        ).to_csv(output_dir / "pooled_opportunity_scores.csv", encoding="utf-8-sig")
    if opportunity_selection_frames:
        pd.concat(opportunity_selection_frames, axis=1).to_csv(
            output_dir / "pooled_opportunity_selection.csv",
            encoding="utf-8-sig",
        )
    if contribution_frames:
        pd.concat(contribution_frames, axis=1).to_csv(
            output_dir / "pooled_portfolio_contribution.csv",
            encoding="utf-8-sig",
        )
    if group_weight_frames:
        pd.concat(group_weight_frames, axis=1).to_csv(output_dir / "pooled_group_weights.csv", encoding="utf-8-sig")
    if group_contribution_frames:
        pd.concat(group_contribution_frames, axis=1).to_csv(
            output_dir / "pooled_group_contribution.csv",
            encoding="utf-8-sig",
        )
    strategy_returns.rename(
        columns={column: get_symbol_from_strategy_column(column) for column in strategy_columns}
    ).corr().to_csv(output_dir / "pooled_strategy_return_corr.csv", encoding="utf-8-sig")
    plot_path = output_dir / "pooled_portfolio_report.png"
    plot_multi_symbol_portfolio(portfolio, PORTFOLIO_METHODS, plot_path)
    print(f"pooled 组合层明细已保存: {output_dir / 'pooled_portfolio_detail.csv'}")
    print(f"pooled 组合层摘要已保存: {output_dir / 'pooled_portfolio_summary.csv'}")
    print(f"pooled 组合层图表已保存: {plot_path}")


def run_pooled_model_backtest(config: BacktestConfig) -> pd.DataFrame:
    """运行多品种共享信息模型，并保存结果。"""
    output_dir = get_pooled_output_dir(config)
    symbols = get_pooled_symbols(config)
    if not symbols:
        raise ValueError("pooled_model_symbols 和 symbols 均为空，无法运行共享模型。")

    symbol_configs = {symbol: build_symbol_config(config, symbol) for symbol in symbols}
    active_by_symbol = {
        symbol: read_symbol_active_library(symbol_config)
        for symbol, symbol_config in symbol_configs.items()
    }
    active_by_symbol = {symbol: active for symbol, active in active_by_symbol.items() if not active.empty}
    if not active_by_symbol:
        raise ValueError("没有找到任何品种的 active_factors.csv，请先运行 multi/single 更新因子库。")

    selected_factors = select_shared_factors(active_by_symbol, config)
    if not selected_factors:
        raise ValueError("共享模型没有选出可用因子，请检查各品种 active 因子库。")

    factor_summary = score_active_factor_rows(active_by_symbol)
    factor_summary.to_csv(output_dir / "pooled_shared_factor_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": selected_factors}).to_csv(
        output_dir / "pooled_selected_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grouped_symbols = build_grouped_symbol_map(list(active_by_symbol.keys()), config)
    min_symbols = max(1, int(getattr(config, "pooled_model_min_symbols_per_group", 2) or 2))
    all_groups = sorted(grouped_symbols.keys())
    symbol_datasets: dict[str, pd.DataFrame] = {}
    symbol_data: dict[str, pd.DataFrame] = {}
    available_by_symbol: dict[str, list[str]] = {}

    for group_name, group_symbols in grouped_symbols.items():
        if len(group_symbols) < min_symbols:
            print(f"跳过 pooled 组 {group_name}: 品种数 {len(group_symbols)} < {min_symbols}")
            continue
        for symbol in group_symbols:
            dataset, data, _factors, available_factors = build_symbol_pooled_dataset(
                symbol,
                symbol_configs[symbol],
                selected_factors,
                all_symbols=sorted(active_by_symbol.keys()),
                all_groups=all_groups,
                group_name=group_name,
                config=config,
            )
            if dataset.empty:
                print(f"跳过 {symbol}: 没有可用共享因子。")
                continue
            symbol_datasets[symbol] = dataset
            symbol_data[symbol] = data
            available_by_symbol[symbol] = available_factors

    if not symbol_datasets:
        raise ValueError("没有任何品种成功构建 pooled 数据集。")

    availability = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "group": dataset["group"].iloc[0],
                "rows": len(dataset),
                "available_factor_count": len(available_by_symbol.get(symbol, [])),
            }
            for symbol, dataset in symbol_datasets.items()
        ]
    )
    availability.to_csv(output_dir / "pooled_symbol_data_coverage.csv", index=False, encoding="utf-8-sig")

    prediction_frames = []
    detail_frames = []
    summary_rows = []
    for group_name, group_symbols in grouped_symbols.items():
        usable_symbols = [symbol for symbol in group_symbols if symbol in symbol_datasets]
        if len(usable_symbols) < min_symbols:
            continue
        group_dataset = pd.concat([symbol_datasets[symbol] for symbol in usable_symbols], axis=0, ignore_index=True)
        metadata_columns = {"target", "future_horizon_return", "timestamp", "symbol", "group", "row_position"}
        feature_columns = [column for column in group_dataset.columns if column not in metadata_columns]
        print(f"训练 pooled 组 {group_name}: 品种数={len(usable_symbols)}, 样本数={len(group_dataset)}, 特征数={len(feature_columns)}")
        group_prediction = fit_predict_pooled_group(group_name, group_dataset, feature_columns, config)
        prediction_frames.append(group_prediction)

        for symbol in usable_symbols:
            symbol_prediction = group_prediction[group_prediction["symbol"] == symbol].copy()
            if symbol_prediction[PROBABILITY_COLUMNS].dropna(how="all").empty:
                continue
            signal = build_symbol_signal_from_predictions(symbol_prediction, config)
            backtest_df, metrics = run_backtest(symbol_data[symbol], signal, config)
            backtest_df.insert(0, "symbol", symbol)
            backtest_df.insert(1, "group", group_name)
            detail_frames.append(backtest_df)
            prediction_metrics = calculate_prediction_metrics_for_segment(symbol, backtest_df) or {}
            summary_rows.append(
                {
                    "symbol": symbol,
                    "group": group_name,
                    "pooled_model": getattr(config, "pooled_model_name", "xgboost"),
                    "feature_count": len(feature_columns),
                    "prediction_rows": int(symbol_prediction[PROBABILITY_COLUMNS].notna().all(axis=1).sum()),
                    **metrics,
                    **{f"预测_{key}": value for key, value in prediction_metrics.items() if key != "样本段"},
                }
            )

    if prediction_frames:
        pd.concat(prediction_frames, axis=0, ignore_index=True).to_csv(
            output_dir / "pooled_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if detail_frames:
        pd.concat(detail_frames, axis=0).to_csv(
            output_dir / "pooled_symbol_detail.csv",
            encoding="utf-8-sig",
        )
        save_pooled_portfolio_outputs(detail_frames, config, output_dir)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(output_dir / "pooled_symbol_summary.csv", index=False, encoding="utf-8-sig")
        group_summary = (
            summary.groupby("group", dropna=False)
            .agg(
                品种数=("symbol", "nunique"),
                平均累计收益=("累计收益", "mean"),
                平均夏普=("夏普比率", "mean"),
                平均最大回撤=("最大回撤", "mean"),
                平均预测方向准确率=("预测_方向目标准确率", "mean"),
                总预测样本数=("prediction_rows", "sum"),
            )
            .reset_index()
        )
        group_summary.to_csv(output_dir / "pooled_group_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    """脚本入口。"""
    config = BacktestConfig()
    try:
        run_tracked(config, "pooled", lambda: run_pooled_model_backtest(config))
    finally:
        stop_wind()


if __name__ == "__main__":
    main()

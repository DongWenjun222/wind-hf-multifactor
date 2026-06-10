from __future__ import annotations

"""多品种批量回测入口。

这个脚本不会改变原有单品种脚本的行为。它会读取 config.symbols，
为每个品种创建独立输出目录，并按配置运行单因子流程和综合因子流程。
"""

from dataclasses import asdict, replace
import datetime as dt
import json
from pathlib import Path
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from composite_factor_backtest import run_composite_backtest
from config import BacktestConfig
from factors import build_single_factor_matrix, fetch_intraday_data, get_factor_prune_list_path, safe_symbol_name, stop_wind
from single_factor_backtest import calculate_metrics, infer_annual_periods, run_single_factor_backtests
from runtime_utils import run_tracked


PORTFOLIO_METHODS = {
    "equal_weight": "等权组合",
    "inverse_vol": "波动率倒数加权",
    "positive_sharpe": "夏普正向加权",
}


def get_symbol_output_dir(base_output_dir: str, subdir: str, symbol: str) -> str:
    """返回某个品种的独立输出目录。"""
    symbol_dir = safe_symbol_name(symbol).upper()
    return str(Path(base_output_dir) / subdir / symbol_dir)


def get_symbol_active_library_path(config: BacktestConfig) -> Path:
    """返回单个品种输出目录下的 active 因子库路径。"""
    return Path(config.output_dir) / "factor_library" / "active_factors.csv"


def get_symbol_composite_detail_path(config: BacktestConfig) -> Path:
    """返回单个品种综合回测明细路径。"""
    return Path(config.output_dir) / "composite_factor" / "composite_detail.csv"


def should_skip_single_factor_pipeline(config: BacktestConfig) -> bool:
    """判断是否可以复用已存在的单因子库结果。"""
    if not bool(getattr(config, "multi_symbol_skip_existing", False)):
        return False

    # single_factor_scope="new" 的含义是继续测试增量因子并更新该品种因子库。
    # 即使已有 active_factors.csv，也不能直接跳过，否则多品种流程会永远复用旧库。
    if str(getattr(config, "single_factor_scope", "")).lower() == "new":
        return False

    return get_symbol_active_library_path(config).exists()


def should_skip_composite_pipeline(config: BacktestConfig) -> bool:
    """判断是否可以复用已存在的综合因子结果。"""
    return bool(getattr(config, "multi_symbol_skip_existing", False)) and get_symbol_composite_detail_path(
        config
    ).exists()


def load_existing_active_library(config: BacktestConfig) -> pd.DataFrame | None:
    """读取已存在的 active 因子库，用于断点续跑时补充汇总信息。"""
    active_path = get_symbol_active_library_path(config)
    if not active_path.exists():
        return None
    try:
        return pd.read_csv(active_path)
    except Exception:
        return None


def build_symbol_config(base_config: BacktestConfig, symbol: str) -> BacktestConfig:
    """基于全局配置创建单个品种的配置副本。"""
    symbol_config = replace(base_config, symbol=symbol)
    prune_path = Path(getattr(base_config, "factor_prune_list_path", "factor_prune_list.csv"))
    if not prune_path.is_absolute():
        symbol_config.factor_prune_list_path = str(Path(base_config.output_dir) / prune_path)
    if base_config.multi_symbol_separate_output_dirs:
        symbol_config.output_dir = get_symbol_output_dir(
            base_config.output_dir,
            base_config.multi_symbol_output_subdir,
            symbol,
        )
    return symbol_config


def summarize_active_library(active_library: pd.DataFrame | None) -> dict[str, Any]:
    """从 active 因子库中提取跨品种汇总字段。"""
    if active_library is None or active_library.empty:
        return {
            "active因子数": 0,
            "active平均入库夏普": np.nan,
            "active最高入库夏普": np.nan,
        }
    sharpe = pd.to_numeric(active_library.get("初筛夏普"), errors="coerce")
    if sharpe.isna().all():
        fallback_column = "验证夏普比率" if "验证夏普比率" in active_library.columns else "测试夏普比率"
        sharpe = pd.to_numeric(active_library.get(fallback_column), errors="coerce")
    return {
        "active因子数": int(len(active_library)),
        "active平均入库夏普": float(sharpe.mean()) if sharpe.notna().any() else np.nan,
        "active最高入库夏普": float(sharpe.max()) if sharpe.notna().any() else np.nan,
    }


def run_single_symbol_pipeline(config: BacktestConfig) -> dict[str, Any]:
    """运行单个品种的单因子和综合因子流程。"""
    row: dict[str, Any] = {
        "品种": config.symbol,
        "输出目录": config.output_dir,
        "单因子状态": "未运行",
        "综合因子状态": "未运行",
    }

    active_library: pd.DataFrame | None = None
    if config.multi_symbol_run_single_factor:
        if should_skip_single_factor_pipeline(config):
            print(f"\n========== {config.symbol} 单因子流程已存在，跳过 ==========")
            active_library = load_existing_active_library(config)
            row["单因子状态"] = "复用已有结果"
        else:
            print(f"\n========== {config.symbol} 单因子流程 ==========")
            data = fetch_intraday_data(config)
            factors = build_single_factor_matrix(data, config)
            active_library = run_single_factor_backtests(data, factors, config)
            row["单因子状态"] = "完成"
        row.update(summarize_active_library(active_library))
    else:
        row.update(summarize_active_library(None))

    if config.multi_symbol_run_composite:
        if should_skip_composite_pipeline(config):
            print(f"\n========== {config.symbol} 综合因子流程已存在，跳过 ==========")
            metrics = {}
            row["综合因子状态"] = "复用已有结果"
        else:
            print(f"\n========== {config.symbol} 综合因子流程 ==========")
            metrics = run_composite_backtest(config)
            row["综合因子状态"] = "完成"
        for key, value in metrics.items():
            row[f"综合_{key}"] = value

    return row


def load_symbol_composite_detail(symbol: str, output_dir: str) -> pd.DataFrame | None:
    """读取单个品种的综合因子最终测试集明细。"""
    detail_path = Path(output_dir) / "composite_factor" / "composite_detail.csv"
    if not detail_path.exists():
        return None
    detail = pd.read_csv(detail_path, index_col=0, parse_dates=True)
    required = {"strategy_net_return", "benchmark_return", "position"}
    if not required.issubset(detail.columns):
        return None
    optional_columns = [
        "calibrated_prob_edge",
        "directional_probability",
        "confidence_rank",
        "position_size",
        "raw_signal",
    ]
    selected_columns = ["strategy_net_return", "benchmark_return", "position"] + [
        column for column in optional_columns if column in detail.columns
    ]
    detail = detail[selected_columns].copy()
    rename_map = {
        "strategy_net_return": f"{symbol}_strategy_return",
        "benchmark_return": f"{symbol}_benchmark_return",
        "position": f"{symbol}_position",
        "calibrated_prob_edge": f"{symbol}_calibrated_prob_edge",
        "directional_probability": f"{symbol}_directional_probability",
        "confidence_rank": f"{symbol}_confidence_rank",
        "position_size": f"{symbol}_position_size",
        "raw_signal": f"{symbol}_raw_signal",
    }
    detail = detail.rename(columns={column: rename_map[column] for column in selected_columns})
    return detail


def read_factor_name_set(path: Path) -> set[str]:
    """从因子库 CSV 中读取因子名称集合。"""
    if not path.exists():
        return set()
    try:
        table = pd.read_csv(path)
    except Exception:
        return set()
    if "因子" not in table.columns:
        return set()
    return set(table["因子"].dropna().astype(str))


def read_single_factor_summary_for_pruning(symbol: str, output_dir: str) -> pd.DataFrame:
    """读取某个品种的单因子全量汇总，用于跨品种淘汰判断。"""
    summary_path = Path(output_dir) / "single_factor" / "single_factor_all_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()
    if "因子" not in summary.columns:
        return pd.DataFrame()

    sharpe_column = "初筛夏普" if "初筛夏普" in summary.columns else "测试夏普比率"
    return_column = "初筛累计收益" if "初筛累计收益" in summary.columns else "测试累计收益"
    summary = summary[["因子", sharpe_column, return_column]].copy()
    summary.columns = ["因子", "初筛夏普", "初筛累计收益"]
    summary["品种"] = symbol
    summary["初筛夏普"] = pd.to_numeric(summary["初筛夏普"], errors="coerce")
    summary["初筛累计收益"] = pd.to_numeric(summary["初筛累计收益"], errors="coerce")
    return summary


def update_factor_prune_list(
    config: BacktestConfig,
    summary: pd.DataFrame,
    summary_dir: Path,
) -> pd.DataFrame:
    """根据多品种单因子表现更新全局因子淘汰清单。"""
    if not getattr(config, "enable_factor_pruning", False):
        return pd.DataFrame()

    min_tested_symbols = max(1, int(getattr(config, "factor_pruning_min_tested_symbols", 1) or 1))
    keep_min_sharpe = float(getattr(config, "factor_pruning_keep_min_sharpe", 0.0) or 0.0)
    keep_min_total_return = float(getattr(config, "factor_pruning_keep_min_total_return", 0.0) or 0.0)
    keep_if_active = bool(getattr(config, "factor_pruning_keep_if_active_any_symbol", True))

    all_summary_rows = []
    active_by_factor: dict[str, set[str]] = {}
    for _, row in summary.iterrows():
        symbol = str(row.get("品种", "")).strip()
        output_dir = str(row.get("输出目录", "")).strip()
        if not symbol or not output_dir:
            continue

        factor_summary = read_single_factor_summary_for_pruning(symbol, output_dir)
        if not factor_summary.empty:
            all_summary_rows.append(factor_summary)

        active_path = Path(output_dir) / "factor_library" / "active_factors.csv"
        for factor_name in read_factor_name_set(active_path):
            active_by_factor.setdefault(factor_name, set()).add(symbol)

    if not all_summary_rows:
        return pd.DataFrame()

    factor_summary_all = pd.concat(all_summary_rows, ignore_index=True)
    grouped = factor_summary_all.groupby("因子", dropna=True)
    prune_rows = []
    for factor_name, factor_rows in grouped:
        tested_symbols = sorted(set(factor_rows["品种"].dropna().astype(str)))
        active_symbols = sorted(active_by_factor.get(str(factor_name), set()))
        best_sharpe = pd.to_numeric(factor_rows["初筛夏普"], errors="coerce").max()
        best_total_return = pd.to_numeric(factor_rows["初筛累计收益"], errors="coerce").max()
        has_positive_evidence = (
            pd.notna(best_sharpe)
            and best_sharpe >= keep_min_sharpe
        ) or (
            pd.notna(best_total_return)
            and best_total_return >= keep_min_total_return
        )
        protected_by_active = keep_if_active and bool(active_symbols)
        should_prune = (
            len(tested_symbols) >= min_tested_symbols
            and not protected_by_active
            and not has_positive_evidence
        )
        if should_prune:
            prune_rows.append(
                {
                    "因子": factor_name,
                    "测试品种数": len(tested_symbols),
                    "active品种数": len(active_symbols),
                    "最佳初筛夏普": best_sharpe,
                    "最佳初筛累计收益": best_total_return,
                    "测试品种": ",".join(tested_symbols),
                    "active品种": ",".join(active_symbols),
                    "淘汰原因": "all_tested_symbols_below_keep_threshold",
                }
            )

    new_prune_table = pd.DataFrame(prune_rows)
    prune_path = get_factor_prune_list_path(config)
    prune_path.parent.mkdir(parents=True, exist_ok=True)
    if prune_path.exists():
        try:
            existing_prune_table = pd.read_csv(prune_path)
        except Exception:
            existing_prune_table = pd.DataFrame()
    else:
        existing_prune_table = pd.DataFrame()

    combined_prune_table = pd.concat(
        [existing_prune_table, new_prune_table],
        ignore_index=True,
        sort=False,
    )
    if not combined_prune_table.empty and "因子" in combined_prune_table.columns:
        combined_prune_table = combined_prune_table.drop_duplicates(subset=["因子"], keep="last")
    combined_prune_table.to_csv(prune_path, index=False, encoding="utf-8-sig")
    new_prune_table.to_csv(summary_dir / "factor_pruning_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"因子淘汰清单已更新: {prune_path}，本轮新增淘汰候选 {len(new_prune_table)} 个")
    return new_prune_table


def get_symbol_from_strategy_column(column: str) -> str:
    """从组合收益列名还原品种代码。"""
    return column.removesuffix("_strategy_return")


def build_static_portfolio_weights(
    returns: pd.DataFrame,
    method: str,
    annual_periods: int,
) -> pd.Series:
    """根据指定方法构造静态品种权重。"""
    valid_counts = returns.notna().sum()
    valid_columns = valid_counts[valid_counts > 0].index.tolist()
    if not valid_columns:
        return pd.Series(dtype="float64")

    clean_returns = returns[valid_columns]
    if method == "equal_weight":
        raw_weights = pd.Series(1.0, index=valid_columns)
    elif method == "inverse_vol":
        vol = clean_returns.std(ddof=0).replace(0, np.nan)
        raw_weights = 1.0 / vol
    elif method == "positive_sharpe":
        mean_return = clean_returns.mean()
        vol = clean_returns.std(ddof=0).replace(0, np.nan)
        sharpe_proxy = mean_return / vol * np.sqrt(max(1, annual_periods))
        raw_weights = sharpe_proxy.clip(lower=0.0)
        if raw_weights.fillna(0.0).sum() <= 0:
            raw_weights = pd.Series(1.0, index=valid_columns)
    else:
        raise ValueError(f"未知组合方法: {method}")

    raw_weights = raw_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw_weights.sum() <= 0:
        raw_weights = pd.Series(1.0, index=valid_columns)
    return raw_weights / raw_weights.sum()


def apply_static_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """按静态权重计算组合收益，自动跳过当期缺失品种并重归一。"""
    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    available_weight = returns.notna().mul(aligned_weights, axis=1).sum(axis=1)
    weighted_return = returns.fillna(0.0).mul(aligned_weights, axis=1).sum(axis=1)
    return (weighted_return / available_weight.replace(0, np.nan)).fillna(0.0)


def normalize_and_cap_weights(raw_weights: pd.Series, max_weight: float) -> pd.Series:
    """归一化权重，并可选限制单品种最大权重。"""
    weights = raw_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=raw_weights.index, dtype="float64")
    weights = weights / weights.sum()
    max_weight = float(max_weight)
    if max_weight <= 0 or max_weight >= 1 or len(weights) <= 1:
        return weights

    capped = pd.Series(0.0, index=weights.index, dtype="float64")
    remaining = weights.copy()
    remaining_budget = 1.0
    while not remaining.empty:
        if remaining.sum() <= 0:
            capped.loc[remaining.index] = remaining_budget / len(remaining)
            break
        scaled = remaining / remaining.sum() * remaining_budget
        over_limit = scaled > max_weight
        if not over_limit.any():
            capped.loc[scaled.index] = scaled
            break
        capped.loc[scaled[over_limit].index] = max_weight
        remaining_budget -= max_weight * int(over_limit.sum())
        remaining = remaining.loc[~over_limit]
        if remaining_budget <= 0:
            break
    if capped.sum() <= 0:
        return weights
    return capped / capped.sum()


def build_portfolio_weights_from_history(
    history_returns: pd.DataFrame,
    current_available: pd.Series,
    method: str,
    annual_periods: int,
    min_samples: int,
    max_weight: float,
) -> pd.Series:
    """仅使用当前时点之前的历史收益估计组合权重。"""
    available_columns = current_available[current_available].index.tolist()
    if not available_columns:
        return pd.Series(dtype="float64")
    history = history_returns[available_columns]
    valid_counts = history.notna().sum()
    valid_columns = valid_counts[valid_counts >= min_samples].index.tolist()

    if method == "equal_weight" or not valid_columns:
        raw_weights = pd.Series(1.0, index=available_columns, dtype="float64")
        return normalize_and_cap_weights(raw_weights, max_weight)

    history = history[valid_columns]
    if method == "inverse_vol":
        vol = history.std(ddof=0).replace(0, np.nan)
        raw_weights = 1.0 / vol
    elif method == "positive_sharpe":
        mean_return = history.mean()
        vol = history.std(ddof=0).replace(0, np.nan)
        raw_weights = (mean_return / vol * np.sqrt(max(1, annual_periods))).clip(lower=0.0)
        if raw_weights.fillna(0.0).sum() <= 0:
            raw_weights = pd.Series(1.0, index=valid_columns, dtype="float64")
    else:
        raise ValueError(f"未知组合方法: {method}")

    return normalize_and_cap_weights(raw_weights.reindex(available_columns).fillna(0.0), max_weight)


def build_rolling_portfolio_weights(
    returns: pd.DataFrame,
    method: str,
    annual_periods: int,
    window: int,
    min_samples: int,
    max_weight: float,
) -> pd.DataFrame:
    """生成逐 K 线滚动组合权重，权重只使用当前行之前的历史数据。"""
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns, dtype="float64")
    window = max(1, int(window))
    min_samples = max(1, int(min_samples))
    for position, timestamp in enumerate(returns.index):
        history_start = max(0, position - window)
        history = returns.iloc[history_start:position]
        current_available = returns.iloc[position].notna()
        row_weights = build_portfolio_weights_from_history(
            history,
            current_available,
            method,
            annual_periods,
            min_samples,
            max_weight,
        )
        if not row_weights.empty:
            weights.loc[timestamp, row_weights.index] = row_weights
    return weights


def apply_rolling_weights(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """按逐时点权重计算组合收益。"""
    aligned_weights = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    weighted_return = returns.fillna(0.0).mul(aligned_weights, axis=0).sum(axis=1)
    active_weight = returns.notna().mul(aligned_weights, axis=0).sum(axis=1)
    return (weighted_return / active_weight.replace(0, np.nan)).fillna(0.0)


def build_cross_sectional_opportunity_scores(
    portfolio: pd.DataFrame,
    strategy_columns: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """根据各品种上一根模型置信度构造横截面机会评分。"""
    mode = str(getattr(config, "multi_symbol_opportunity_score_mode", "edge")).lower()
    score_frame = pd.DataFrame(index=portfolio.index)
    for strategy_column in strategy_columns:
        symbol = get_symbol_from_strategy_column(strategy_column)
        edge = portfolio.get(f"{symbol}_calibrated_prob_edge")
        probability = portfolio.get(f"{symbol}_directional_probability")
        confidence_rank = portfolio.get(f"{symbol}_confidence_rank")
        position = portfolio.get(f"{symbol}_position")

        if edge is not None and mode in {"edge", "edge_probability", "edge_rank"}:
            # 概率优势在当前K线结束后才知道，组合当前收益只能使用上一根的评分。
            score = edge.abs().shift(1)
            if mode == "edge_probability" and probability is not None:
                score = score * probability.shift(1).clip(lower=0.0, upper=1.0)
            elif mode == "edge_rank" and confidence_rank is not None:
                score = score * confidence_rank.shift(1).clip(lower=0.0, upper=1.0)
        elif position is not None:
            score = position.abs()
        else:
            score = portfolio[strategy_column].notna().astype("float64")

        score_frame[strategy_column] = score.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    return score_frame


def apply_cross_sectional_opportunity_selection(
    weights: pd.DataFrame,
    opportunity_scores: pd.DataFrame,
    max_weight: float,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用横截面机会评分过滤并重分配组合权重。"""
    if not bool(getattr(config, "multi_symbol_use_opportunity_selection", False)):
        selected_mask = weights.notna() & (weights > 0)
        return weights, selected_mask.astype("float64")

    top_n = int(getattr(config, "multi_symbol_opportunity_top_n", 0) or 0)
    min_score = max(0.0, float(getattr(config, "multi_symbol_opportunity_min_score", 0.0) or 0.0))
    power = max(0.0, float(getattr(config, "multi_symbol_opportunity_weight_power", 1.0) or 1.0))
    scores = opportunity_scores.reindex(index=weights.index, columns=weights.columns)
    adjusted = pd.DataFrame(0.0, index=weights.index, columns=weights.columns, dtype="float64")
    selected_mask = pd.DataFrame(0.0, index=weights.index, columns=weights.columns, dtype="float64")

    for timestamp in weights.index:
        row_weights = weights.loc[timestamp].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        row_scores = scores.loc[timestamp].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        candidates = row_weights[row_weights > 0].index
        if len(candidates) == 0:
            continue

        candidate_scores = row_scores.reindex(candidates).fillna(0.0)
        if min_score <= 0:
            candidate_scores = candidate_scores[candidate_scores > 0]
        else:
            candidate_scores = candidate_scores[candidate_scores >= min_score]
        if candidate_scores.empty:
            continue
        if top_n > 0 and len(candidate_scores) > top_n:
            candidate_scores = candidate_scores.sort_values(ascending=False).head(top_n)

        selected_columns = candidate_scores.index
        opportunity_multiplier = candidate_scores.pow(power).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if opportunity_multiplier.sum() <= 0:
            opportunity_multiplier = pd.Series(1.0, index=selected_columns, dtype="float64")
        raw_weights = row_weights.reindex(selected_columns).fillna(0.0) * opportunity_multiplier
        adjusted_weights = normalize_and_cap_weights(raw_weights, max_weight)
        adjusted.loc[timestamp, adjusted_weights.index] = adjusted_weights
        selected_mask.loc[timestamp, adjusted_weights.index] = 1.0

    return adjusted, selected_mask


def build_portfolio_risk_multiplier(
    raw_returns: pd.Series,
    annual_periods: int,
    config: BacktestConfig,
) -> pd.DataFrame:
    """根据历史波动和历史回撤生成组合层风险乘数。"""
    raw_returns = raw_returns.fillna(0.0)
    multiplier = pd.Series(1.0, index=raw_returns.index, dtype="float64")

    if bool(getattr(config, "multi_symbol_use_vol_target", False)):
        window = max(2, int(getattr(config, "multi_symbol_vol_target_window", 240) or 240))
        target_vol = max(0.0, float(getattr(config, "multi_symbol_target_annual_vol", 0.0) or 0.0))
        max_leverage = max(
            0.0,
            float(getattr(config, "multi_symbol_max_portfolio_leverage", 1.0) or 1.0),
        )
        realized_vol = raw_returns.rolling(window, min_periods=max(2, window // 4)).std(ddof=0)
        realized_vol = realized_vol.shift(1) * np.sqrt(max(1, annual_periods))
        vol_multiplier = (target_vol / realized_vol.replace(0, np.nan)).clip(
            lower=0.0,
            upper=max_leverage,
        )
        multiplier = multiplier * vol_multiplier.fillna(1.0)
    else:
        realized_vol = pd.Series(np.nan, index=raw_returns.index, dtype="float64")
        vol_multiplier = pd.Series(1.0, index=raw_returns.index, dtype="float64")

    if bool(getattr(config, "multi_symbol_use_drawdown_control", False)):
        reduce_start = float(getattr(config, "multi_symbol_drawdown_reduce_start", -1.0) or -1.0)
        stop_level = float(getattr(config, "multi_symbol_drawdown_stop", -1.0) or -1.0)
        if stop_level > reduce_start:
            reduce_start, stop_level = stop_level, reduce_start
        raw_nav = (1.0 + raw_returns).cumprod()
        drawdown = raw_nav / raw_nav.cummax() - 1.0
        prior_drawdown = drawdown.shift(1).fillna(0.0)
        drawdown_multiplier = pd.Series(1.0, index=raw_returns.index, dtype="float64")
        drawdown_multiplier[prior_drawdown <= stop_level] = 0.0
        transition = (prior_drawdown < reduce_start) & (prior_drawdown > stop_level)
        if transition.any() and reduce_start != stop_level:
            drawdown_multiplier.loc[transition] = (
                (prior_drawdown.loc[transition] - stop_level) / (reduce_start - stop_level)
            ).clip(0.0, 1.0)
        multiplier = multiplier * drawdown_multiplier
    else:
        drawdown = pd.Series(np.nan, index=raw_returns.index, dtype="float64")
        drawdown_multiplier = pd.Series(1.0, index=raw_returns.index, dtype="float64")

    multiplier = multiplier.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pd.DataFrame(
        {
            "risk_multiplier": multiplier,
            "realized_annual_vol": realized_vol,
            "vol_target_multiplier": vol_multiplier,
            "drawdown": drawdown,
            "drawdown_multiplier": drawdown_multiplier,
        },
        index=raw_returns.index,
    )


def plot_multi_symbol_portfolio(
    portfolio: pd.DataFrame,
    method_names: dict[str, str],
    output_path: Path,
) -> Path:
    """绘制多组合方法的净值、回撤、收益和平均仓位对比图。"""
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(18, 18),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.0]},
    )
    fig.suptitle("多品种组合层回测对比", fontsize=16)

    total_returns = {}
    avg_positions = {}
    for method, label in method_names.items():
        nav_column = f"{method}_nav"
        benchmark_column = f"{method}_benchmark_nav"
        drawdown_column = f"{method}_drawdown"
        return_column = f"{method}_strategy_return"
        position_column = f"{method}_avg_abs_position"
        if nav_column not in portfolio.columns:
            continue

        axes[0].plot(portfolio.index, portfolio[nav_column], label=label, linewidth=1.6)
        if benchmark_column in portfolio.columns:
            axes[0].plot(
                portfolio.index,
                portfolio[benchmark_column],
                label=f"{label}基准",
                linewidth=1.0,
                alpha=0.45,
            )
        axes[1].plot(portfolio.index, portfolio[drawdown_column], label=label, linewidth=1.2)
        total_returns[label] = portfolio[nav_column].dropna().iloc[-1] - 1.0
        avg_positions[label] = portfolio[position_column].mean()
        axes[3].plot(portfolio.index, portfolio[position_column], label=label, linewidth=1.1)

    axes[0].set_ylabel("净值")
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.7)
    axes[1].set_ylabel("回撤")
    axes[1].legend(loc="lower left", ncol=2)
    axes[1].grid(alpha=0.3)

    return_series = pd.Series(total_returns).sort_values(ascending=False)
    colors = np.where(return_series >= 0, "#2ca02c", "#d62728")
    axes[2].bar(return_series.index, return_series.values, color=colors, alpha=0.85)
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("累计收益")
    axes[2].grid(axis="y", alpha=0.3)

    axes[3].set_ylabel("平均绝对仓位")
    axes[3].legend(loc="upper left", ncol=2)
    axes[3].grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_multi_symbol_portfolio(
    summary: pd.DataFrame,
    config: BacktestConfig,
    summary_dir: Path,
) -> None:
    """基于各品种最终测试集收益生成多种组合明细、摘要和图表。"""
    detail_frames = []
    for _, row in summary.iterrows():
        if row.get("综合因子状态") not in {"完成", "复用已有结果"}:
            continue
        symbol = str(row.get("品种", ""))
        output_dir = str(row.get("输出目录", ""))
        detail = load_symbol_composite_detail(symbol, output_dir)
        if detail is not None and not detail.empty:
            detail_frames.append(detail)

    if not detail_frames:
        return

    portfolio = pd.concat(detail_frames, axis=1).sort_index()
    strategy_columns = [column for column in portfolio.columns if column.endswith("_strategy_return")]
    benchmark_columns = [column for column in portfolio.columns if column.endswith("_benchmark_return")]
    position_columns = [column for column in portfolio.columns if column.endswith("_position")]
    annual_periods = infer_annual_periods(portfolio.index, config.annual_trading_days)
    portfolio["active_symbol_count"] = portfolio[strategy_columns].notna().sum(axis=1)
    strategy_returns = portfolio[strategy_columns]
    benchmark_returns = portfolio[benchmark_columns]
    abs_positions = portfolio[position_columns].abs()
    opportunity_scores = build_cross_sectional_opportunity_scores(portfolio, strategy_columns, config)
    summary_rows = []
    weight_frames = []
    contribution_frames = []
    opportunity_selection_frames = []
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
            if weights_by_time.empty:
                continue
            base_weights_by_time = weights_by_time.copy()
            weights_by_time, opportunity_selected = apply_cross_sectional_opportunity_selection(
                weights_by_time,
                opportunity_scores,
                max_symbol_weight,
                config,
            )
            symbol_weight_columns = {
                column: get_symbol_from_strategy_column(column) for column in weights_by_time.columns
            }
            avg_symbol_weights = weights_by_time.mean().rename(index=symbol_weight_columns)
            portfolio[f"{method}_strategy_return"] = apply_rolling_weights(
                strategy_returns,
                weights_by_time,
            )
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
            portfolio[f"{method}_benchmark_return"] = apply_rolling_weights(
                benchmark_returns,
                benchmark_weights_by_time,
            )
            portfolio[f"{method}_avg_abs_position"] = apply_rolling_weights(
                abs_positions,
                position_weights_by_time,
            )
            weight_frames.append(
                weights_by_time.rename(columns=symbol_weight_columns).add_prefix(f"{method}_")
            )
            opportunity_selection_frames.append(
                opportunity_selected.rename(columns=symbol_weight_columns).add_prefix(f"{method}_")
            )
            contribution_frame = (
                strategy_returns.fillna(0.0)
                .mul(weights_by_time, axis=0)
                .rename(columns=symbol_weight_columns)
                .add_prefix(f"{method}_")
            )
            symbol_weights_for_summary = avg_symbol_weights
        else:
            weights = build_static_portfolio_weights(strategy_returns, method, annual_periods)
            if weights.empty:
                continue
            weights = normalize_and_cap_weights(weights, max_symbol_weight)
            weights_by_time = pd.DataFrame(
                {column: float(weights.get(column, 0.0)) for column in strategy_returns.columns},
                index=portfolio.index,
            )
            base_weights_by_time = weights_by_time.copy()
            weights_by_time, opportunity_selected = apply_cross_sectional_opportunity_selection(
                weights_by_time,
                opportunity_scores,
                max_symbol_weight,
                config,
            )
            symbol_weight_columns = {
                column: get_symbol_from_strategy_column(column) for column in weights_by_time.columns
            }
            avg_symbol_weights = weights_by_time.mean().rename(index=symbol_weight_columns)
            portfolio[f"{method}_strategy_return"] = apply_rolling_weights(
                strategy_returns,
                weights_by_time,
            )
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
            portfolio[f"{method}_benchmark_return"] = apply_rolling_weights(
                benchmark_returns,
                benchmark_weights_by_time,
            )
            portfolio[f"{method}_avg_abs_position"] = apply_rolling_weights(
                abs_positions,
                position_weights_by_time,
            )
            weight_frames.append(
                weights_by_time.rename(columns=symbol_weight_columns).add_prefix(f"{method}_")
            )
            opportunity_selection_frames.append(
                opportunity_selected.rename(columns=symbol_weight_columns).add_prefix(f"{method}_")
            )
            contribution_frame = (
                strategy_returns.fillna(0.0)
                .mul(weights_by_time, axis=0)
                .rename(columns={col: get_symbol_from_strategy_column(col) for col in strategy_columns})
                .add_prefix(f"{method}_")
            )
            symbol_weights_for_summary = avg_symbol_weights
        base_avg_selected_count = float((base_weights_by_time > 0).sum(axis=1).mean())
        avg_selected_count = float((weights_by_time > 0).sum(axis=1).mean())
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
        contribution_frames.append(
            contribution_frame.mul(portfolio[f"{method}_risk_multiplier"], axis=0)
        )
        portfolio[f"{method}_nav"] = (1.0 + portfolio[f"{method}_strategy_return"]).cumprod()
        portfolio[f"{method}_benchmark_nav"] = (
            1.0 + portfolio[f"{method}_benchmark_return"]
        ).cumprod()
        portfolio[f"{method}_drawdown"] = (
            portfolio[f"{method}_nav"] / portfolio[f"{method}_nav"].cummax() - 1.0
        )

        metrics = calculate_metrics(
            portfolio[f"{method}_strategy_return"],
            portfolio[f"{method}_benchmark_return"],
            portfolio[f"{method}_avg_abs_position"],
            annual_periods,
        )
        metrics["组合方式"] = label
        metrics["权重方式"] = "滚动历史权重" if use_rolling_weights else "全样本静态权重"
        metrics["权重窗口"] = float(weight_window if use_rolling_weights else 0)
        metrics["单品种最大权重"] = float(max_symbol_weight)
        metrics["启用波动率目标"] = bool(getattr(config, "multi_symbol_use_vol_target", False))
        metrics["目标年化波动率"] = float(getattr(config, "multi_symbol_target_annual_vol", 0.0) or 0.0)
        metrics["最大组合杠杆"] = float(getattr(config, "multi_symbol_max_portfolio_leverage", 1.0) or 1.0)
        metrics["启用回撤降仓"] = bool(getattr(config, "multi_symbol_use_drawdown_control", False))
        metrics["平均风险乘数"] = float(portfolio[f"{method}_risk_multiplier"].mean())
        metrics["最低风险乘数"] = float(portfolio[f"{method}_risk_multiplier"].min())
        metrics["参与品种数"] = float(len(symbol_weights_for_summary))
        metrics["平均每根K线活跃品种数"] = float(portfolio["active_symbol_count"].mean())
        metrics["启用横截面机会选择"] = bool(getattr(config, "multi_symbol_use_opportunity_selection", False))
        metrics["机会选择TopN"] = float(getattr(config, "multi_symbol_opportunity_top_n", 0) or 0)
        metrics["机会评分模式"] = str(getattr(config, "multi_symbol_opportunity_score_mode", "edge"))
        metrics["机会评分最低阈值"] = float(getattr(config, "multi_symbol_opportunity_min_score", 0.0) or 0.0)
        metrics["机会权重幂次"] = float(getattr(config, "multi_symbol_opportunity_weight_power", 1.0) or 1.0)
        metrics["机会选择前平均入选品种数"] = base_avg_selected_count
        metrics["机会选择后平均入选品种数"] = avg_selected_count
        metrics["平均绝对仓位"] = float(portfolio[f"{method}_avg_abs_position"].mean())
        metrics["品种权重"] = ",".join(
            f"{symbol}:{weight:.4f}" for symbol, weight in symbol_weights_for_summary.items()
        )
        summary_rows.append(metrics)

    detail_path = summary_dir / "multi_symbol_portfolio_detail.csv"
    portfolio.to_csv(detail_path, encoding="utf-8-sig")

    summary_path = summary_dir / "multi_symbol_portfolio_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    if weight_frames:
        weights_path = summary_dir / "multi_symbol_portfolio_weights.csv"
        pd.concat(weight_frames, axis=1).to_csv(weights_path, encoding="utf-8-sig")
    if not opportunity_scores.empty:
        opportunity_scores.rename(
            columns={col: get_symbol_from_strategy_column(col) for col in opportunity_scores.columns}
        ).to_csv(
            summary_dir / "multi_symbol_opportunity_scores.csv",
            encoding="utf-8-sig",
        )
    if opportunity_selection_frames:
        selection_path = summary_dir / "multi_symbol_opportunity_selection.csv"
        pd.concat(opportunity_selection_frames, axis=1).to_csv(selection_path, encoding="utf-8-sig")
    if contribution_frames:
        contribution_path = summary_dir / "multi_symbol_portfolio_contribution.csv"
        pd.concat(contribution_frames, axis=1).to_csv(contribution_path, encoding="utf-8-sig")
    corr_path = summary_dir / "multi_symbol_strategy_return_corr.csv"
    strategy_returns.rename(columns={col: get_symbol_from_strategy_column(col) for col in strategy_columns}).corr().to_csv(
        corr_path,
        encoding="utf-8-sig",
    )
    plot_path = summary_dir / "multi_symbol_portfolio_report.png"
    plot_multi_symbol_portfolio(portfolio, PORTFOLIO_METHODS, plot_path)
    print(f"多品种组合明细已保存: {detail_path}")
    print(f"多品种组合摘要已保存: {summary_path}")
    print(f"多品种组合图表已保存: {plot_path}")


def write_multi_symbol_run_manifest(
    summary: pd.DataFrame,
    config: BacktestConfig,
    summary_dir: Path,
    error_rows: list[dict[str, Any]],
) -> None:
    """保存多品种批量运行清单，方便复现、排错和生产化监控。"""
    output_files = []
    for path in sorted(summary_dir.rglob("*")):
        if path.is_file():
            output_files.append(
                {
                    "path": str(path.relative_to(summary_dir)),
                    "size_bytes": int(path.stat().st_size),
                    "modified_time": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "symbols": [str(symbol) for symbol in config.symbols],
        "bar_size": int(config.bar_size),
        "start_time": config.start_time,
        "end_time": config.end_time,
        "output_dir": str(summary_dir),
        "config": asdict(config),
        "symbol_status": summary.to_dict("records"),
        "error_count": int(len(error_rows)),
        "errors": error_rows,
        "output_files": output_files,
    }
    with (summary_dir / "multi_symbol_run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)


def run_multi_symbol_backtest(config: BacktestConfig) -> pd.DataFrame:
    """按 symbols 批量运行多品种回测，并保存跨品种汇总。"""
    symbols = [str(symbol).strip() for symbol in config.symbols if str(symbol).strip()]
    if not symbols:
        raise ValueError("config.symbols 为空，无法运行多品种回测。")

    summary_rows = []
    error_rows = []
    for symbol in symbols:
        symbol_config = build_symbol_config(config, symbol)
        try:
            summary_rows.append(run_single_symbol_pipeline(symbol_config))
        except Exception as exc:
            error_traceback = traceback.format_exc()
            summary_rows.append(
                {
                    "品种": symbol,
                    "输出目录": symbol_config.output_dir,
                    "单因子状态": "失败",
                    "综合因子状态": "失败",
                    "错误": str(exc),
                    "错误堆栈": error_traceback,
                }
            )
            error_rows.append(
                {
                    "品种": symbol,
                    "输出目录": symbol_config.output_dir,
                    "错误": str(exc),
                    "错误堆栈": error_traceback,
                }
            )
            print(f"{symbol} 批量回测失败: {exc}")

    summary = pd.DataFrame(summary_rows)
    summary_dir = Path(config.output_dir) / config.multi_symbol_output_subdir
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "multi_symbol_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if error_rows:
        pd.DataFrame(error_rows).to_csv(
            summary_dir / "multi_symbol_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"\n多品种汇总已保存: {summary_path}")
    update_factor_prune_list(config, summary, summary_dir)
    save_multi_symbol_portfolio(summary, config, summary_dir)
    write_multi_symbol_run_manifest(summary, config, summary_dir, error_rows)
    return summary


def main() -> None:
    """多品种批量回测脚本入口。"""
    config = BacktestConfig()

    def action() -> pd.DataFrame:
        try:
            return run_multi_symbol_backtest(config)
        finally:
            stop_wind()

    run_tracked(config, "multi", action)


if __name__ == "__main__":
    main()

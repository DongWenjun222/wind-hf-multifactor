from __future__ import annotations

"""因子库管理模块。

本文件负责把单因子回测结果沉淀成可长期维护的因子库：
- 同时参考训练集和验证集表现做入库筛选，最终测试集只作为留存评估。
- 合并历史因子库，避免每次回测覆盖已有记录。
- 对候选因子做收益门槛和相关性去重，只保留表现较好且差异足够大的因子。
- 输出 active / all / rejected 三类 CSV，供后续 XGBoost 综合因子和人工复盘使用。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig
from factor_taxonomy import get_factor_family
from factors import score_to_raw_signal


def get_family_quota_limit(config: BacktestConfig, family: str) -> int | None:
    """读取某个因子家族的 active 数量上限。"""
    if not bool(getattr(config, "factor_library_enable_family_quota", False)):
        return None
    quota = getattr(config, "factor_library_family_max_counts", {}) or {}
    value = quota.get(str(family))
    if value is None:
        return None
    return max(0, int(value))


def rank_single_factor_summary(
    summary: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按照单因子训练集和验证集表现做初筛排序。

    排序逻辑：
    1. 优先使用验证集与训练集表现的较弱值排序，避免单段偶然表现决定入库。
    2. 如果没有验证列，则只使用训练集；如果训练列也不存在，再回退使用旧测试列。
    3. 仅把前 single_factor_keep_top_n 个标记为 active，其余先标记为 rejected。

    这里还没有做相关性去重，相关性过滤会在 build_factor_library 中完成。
    """
    ranked = summary.copy()
    has_train = {"训练夏普比率", "训练累计收益"}.issubset(ranked.columns)
    has_validation = {"验证夏普比率", "验证累计收益"}.issubset(ranked.columns)
    has_test = {"测试夏普比率", "测试累计收益"}.issubset(ranked.columns)
    has_train_validation_rank_ic = {"训练RankIC", "验证RankIC"}.issubset(ranked.columns)
    has_train_validation_monotonicity = {"训练分组单调性", "验证分组单调性"}.issubset(ranked.columns)
    if has_train and has_validation:
        train_sharpe = ranked["训练夏普比率"].replace([np.inf, -np.inf], np.nan)
        validation_sharpe = ranked["验证夏普比率"].replace([np.inf, -np.inf], np.nan)
        train_return = ranked["训练累计收益"].replace([np.inf, -np.inf], np.nan)
        validation_return = ranked["验证累计收益"].replace([np.inf, -np.inf], np.nan)
        sharpe = pd.concat([train_sharpe, validation_sharpe], axis=1).min(axis=1)
        total_return = pd.concat([train_return, validation_return], axis=1).min(axis=1)
        selection_sample = "训练+验证"
    elif has_train:
        sharpe = ranked["训练夏普比率"].replace([np.inf, -np.inf], np.nan)
        total_return = ranked["训练累计收益"].replace([np.inf, -np.inf], np.nan)
        selection_sample = "训练"
    elif has_test:
        sharpe = ranked["测试夏普比率"].replace([np.inf, -np.inf], np.nan)
        total_return = ranked["测试累计收益"].replace([np.inf, -np.inf], np.nan)
        selection_sample = "测试"
    else:
        raise KeyError("单因子汇总缺少初筛排序字段，需要训练、验证或测试收益/夏普列。")

    if has_train_validation_rank_ic:
        train_rank_ic = ranked["训练RankIC"].replace([np.inf, -np.inf], np.nan)
        validation_rank_ic = ranked["验证RankIC"].replace([np.inf, -np.inf], np.nan)
        selection_rank_ic = pd.concat([train_rank_ic, validation_rank_ic], axis=1).min(axis=1)
    elif "训练RankIC" in ranked.columns:
        selection_rank_ic = ranked["训练RankIC"].replace([np.inf, -np.inf], np.nan)
    elif "测试RankIC" in ranked.columns:
        selection_rank_ic = ranked["测试RankIC"].replace([np.inf, -np.inf], np.nan)
    else:
        selection_rank_ic = pd.Series(np.nan, index=ranked.index)

    if has_train_validation_monotonicity:
        train_monotonicity = ranked["训练分组单调性"].replace([np.inf, -np.inf], np.nan)
        validation_monotonicity = ranked["验证分组单调性"].replace([np.inf, -np.inf], np.nan)
        selection_monotonicity = pd.concat([train_monotonicity, validation_monotonicity], axis=1).min(axis=1)
    elif "训练分组单调性" in ranked.columns:
        selection_monotonicity = ranked["训练分组单调性"].replace([np.inf, -np.inf], np.nan)
    elif "测试分组单调性" in ranked.columns:
        selection_monotonicity = ranked["测试分组单调性"].replace([np.inf, -np.inf], np.nan)
    else:
        selection_monotonicity = pd.Series(np.nan, index=ranked.index)

    ranked["初筛夏普"] = sharpe
    ranked["初筛累计收益"] = total_return
    ranked["初筛RankIC"] = selection_rank_ic
    ranked["初筛分组单调性"] = selection_monotonicity
    ranked["初筛样本"] = selection_sample
    ranked["因子家族"] = ranked["因子"].map(get_factor_family)
    ranked["初筛有效"] = sharpe.notna() & total_return.notna()
    ranked = ranked.sort_values(
        ["初筛有效", "初筛夏普", "初筛RankIC", "初筛分组单调性", "初筛累计收益"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )

    keep_top_n = max(1, int(config.single_factor_keep_top_n))
    selected = ranked.head(keep_top_n).copy()
    selected["因子库状态"] = "active"

    full_ranked = ranked.copy()
    full_ranked["因子库状态"] = "rejected"
    full_ranked.loc[selected.index, "因子库状态"] = "active"
    return selected, full_ranked


def get_factor_library_dir(config: BacktestConfig) -> Path:
    """返回因子库目录，并确保目录存在。

    config.factor_library_dir 可以是绝对路径，也可以是相对路径。
    如果是相对路径，会自动挂到 config.output_dir 下面。
    """
    library_dir = Path(config.factor_library_dir)
    if not library_dir.is_absolute():
        library_dir = Path(config.output_dir) / library_dir
    library_dir.mkdir(parents=True, exist_ok=True)
    return library_dir


def load_existing_factor_library(config: BacktestConfig) -> pd.DataFrame:
    """读取已有的全量因子库文件。

    如果文件不存在，返回空 DataFrame，方便首次运行时直接创建新因子库。
    """
    library_path = get_factor_library_dir(config) / "factor_library_all.csv"
    if not library_path.exists():
        return pd.DataFrame()
    return pd.read_csv(library_path)


def build_signal_corr_frame(
    factors: pd.DataFrame,
    factor_names: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """把因子连续值转换成 -1/0/1 交易信号矩阵，用于计算信号相关性。

    连续值相关性关注因子数值形态是否相似；
    信号相关性关注这些因子真正落到交易决策上是否相似。
    两者配合可以减少因子库中“看起来不同、交易起来一样”的冗余因子。
    """
    signal_data = {
        factor_name: score_to_raw_signal(
            factors[factor_name].replace([np.inf, -np.inf], np.nan),
            config.signal_threshold,
        )
        for factor_name in factor_names
    }
    return pd.DataFrame(signal_data, index=factors.index)


def get_numeric_value(row: pd.Series, column: str, default: float = np.nan) -> float:
    """从一行记录中读取数值，同时兼容旧版因子库缺失字段的情况。"""
    if column not in row.index:
        return default
    return pd.to_numeric(row.get(column), errors="coerce")


def choose_metric_column(
    frame: pd.DataFrame,
    preferred_column: str,
    fallback_column: str,
) -> str:
    """优先选择有有效数据的指标列，否则回退到兼容列。"""
    if preferred_column in frame.columns and pd.to_numeric(frame[preferred_column], errors="coerce").notna().any():
        return preferred_column
    return fallback_column


def build_factor_library(
    full_summary: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构建并更新长期因子库。

    输入：
    - full_summary：本轮单因子回测的全量结果。
    - factors：当前可用的因子值矩阵，用于检查因子是否仍存在并计算相关性。
    - config：入库门槛、保留数量、相关性阈值等配置。

    输出：
    - active_library：当前仍然保留在库中的有效因子。
    - library_all：合并历史与本轮后的完整因子记录。
    - rejected_library：被拒绝或退役的因子，包含拒绝原因。
    """
    existing = load_existing_factor_library(config)
    combined = pd.concat([existing, full_summary], ignore_index=True, sort=False)
    if combined.empty:
        return combined, combined, pd.DataFrame()

    combined = combined.drop_duplicates(subset=["因子"], keep="last")
    combined["因子家族"] = combined["因子"].map(get_factor_family)
    combined["初筛有效"] = combined["初筛有效"].map(
        lambda value: str(value).lower() == "true" if pd.notna(value) else False
    )
    numeric_columns = [
        "初筛夏普",
        "初筛累计收益",
        "初筛RankIC",
        "初筛分组单调性",
        "训练夏普比率",
        "训练累计收益",
        "训练IC",
        "训练RankIC",
        "训练ICIR",
        "训练RankICIR",
        "训练IC胜率",
        "训练分组单调性",
        "训练分组收益差",
        "训练胜率",
        "训练交易次数",
        "训练信号覆盖率",
        "训练最大回撤",
        "验证IC",
        "验证RankIC",
        "验证ICIR",
        "验证RankICIR",
        "验证IC胜率",
        "验证分组单调性",
        "验证分组收益差",
        "验证夏普比率",
        "验证累计收益",
        "验证胜率",
        "验证交易次数",
        "验证信号覆盖率",
        "验证最大回撤",
        "测试IC",
        "测试RankIC",
        "测试ICIR",
        "测试RankICIR",
        "测试IC胜率",
        "测试分组单调性",
        "测试分组收益差",
        "测试夏普比率",
        "测试累计收益",
        "测试胜率",
        "测试交易次数",
        "测试信号覆盖率",
        "测试最大回撤",
    ]
    for numeric_column in numeric_columns:
        if numeric_column not in combined.columns:
            combined[numeric_column] = np.nan
        combined[numeric_column] = pd.to_numeric(combined[numeric_column], errors="coerce")
    combined = combined.sort_values(
        ["初筛有效", "初筛夏普", "初筛RankIC", "初筛分组单调性", "初筛累计收益"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    available = set(factors.columns)
    preserve_existing_active = (
        str(getattr(config, "single_factor_scope", "")).lower() == "new"
        and bool(getattr(factors, "attrs", {}).get("factor_id_map"))
    )
    existing_active_factors: set[str] = set()
    if preserve_existing_active and not existing.empty and {"因子", "因子库状态"}.issubset(existing.columns):
        existing_active_factors = set(
            existing.loc[existing["因子库状态"] == "active", "因子"].dropna().astype(str)
        )
    # 只有“表现有效 + 当前代码仍能生成 + 满足收益门槛”的因子，才进入相关性去重候选池。
    min_train_sharpe = float(getattr(config, "factor_library_min_train_sharpe", -np.inf))
    min_train_total_return = float(getattr(config, "factor_library_min_train_total_return", -np.inf))
    min_train_win_rate = float(getattr(config, "factor_library_min_train_win_rate", -np.inf))
    min_test_win_rate = float(getattr(config, "factor_library_min_test_win_rate", -np.inf))
    min_test_trades = max(0, int(getattr(config, "factor_library_min_test_trades", 0) or 0))
    min_train_trades = max(0, int(getattr(config, "factor_library_min_train_trades", 0) or 0))
    min_test_signal_coverage = max(
        0.0,
        float(getattr(config, "factor_library_min_test_signal_coverage", 0.0) or 0.0),
    )
    min_train_signal_coverage = max(
        0.0,
        float(getattr(config, "factor_library_min_train_signal_coverage", 0.0) or 0.0),
    )
    max_test_drawdown = getattr(config, "factor_library_max_test_drawdown", None)
    max_train_drawdown = getattr(config, "factor_library_max_train_drawdown", None)
    min_selection_rank_ic = getattr(config, "factor_library_min_selection_rank_ic", None)
    min_selection_monotonicity = getattr(config, "factor_library_min_selection_monotonicity", None)
    selection_win_rate_column = choose_metric_column(combined, "验证胜率", "测试胜率")
    selection_trade_column = choose_metric_column(combined, "验证交易次数", "测试交易次数")
    selection_coverage_column = choose_metric_column(combined, "验证信号覆盖率", "测试信号覆盖率")
    selection_drawdown_column = choose_metric_column(combined, "验证最大回撤", "测试最大回撤")

    eligible_mask = (
        combined["初筛有效"].fillna(False)
        & combined["因子"].isin(available)
        & (combined["初筛夏普"] >= config.factor_library_min_sharpe)
        & (combined["初筛累计收益"] >= config.factor_library_min_total_return)
        & (combined["训练夏普比率"] >= min_train_sharpe)
        & (combined["训练累计收益"] >= min_train_total_return)
        & (combined["训练胜率"] > min_train_win_rate)
        & (combined[selection_win_rate_column] > min_test_win_rate)
    )
    if min_selection_rank_ic is not None:
        eligible_mask &= combined["初筛RankIC"].fillna(-np.inf) >= float(min_selection_rank_ic)
    if min_selection_monotonicity is not None:
        eligible_mask &= combined["初筛分组单调性"].fillna(-np.inf) >= float(min_selection_monotonicity)
    if min_test_trades > 0:
        eligible_mask &= combined[selection_trade_column].fillna(0.0) >= min_test_trades
    if min_train_trades > 0:
        eligible_mask &= combined["训练交易次数"].fillna(0.0) >= min_train_trades
    if min_test_signal_coverage > 0:
        eligible_mask &= combined[selection_coverage_column].fillna(0.0) >= min_test_signal_coverage
    if min_train_signal_coverage > 0:
        eligible_mask &= combined["训练信号覆盖率"].fillna(0.0) >= min_train_signal_coverage
    if max_test_drawdown is not None:
        eligible_mask &= combined[selection_drawdown_column].fillna(-np.inf) >= float(max_test_drawdown)
    if max_train_drawdown is not None:
        eligible_mask &= combined["训练最大回撤"].fillna(-np.inf) >= float(max_train_drawdown)
    eligible_factors = combined.loc[eligible_mask, "因子"].tolist()

    value_corr = pd.DataFrame()
    signal_corr = pd.DataFrame()
    if eligible_factors and config.factor_library_use_value_corr:
        value_corr = factors[eligible_factors].replace([np.inf, -np.inf], np.nan).corr().abs()
    if eligible_factors and config.factor_library_use_signal_corr:
        signal_corr = build_signal_corr_frame(factors, eligible_factors, config).corr().abs()

    selected: list[str] = []
    selected_set: set[str] = set()
    selected_family_counts: dict[str, int] = {}
    library_rows = []
    max_corr_limit = float(config.factor_library_max_corr)
    keep_top_n = max(1, int(config.single_factor_keep_top_n))

    # 从表现最好的因子开始贪心入库：先占坑的强因子会成为后续候选的相关性参照。
    for _, row in combined.iterrows():
        factor_name = row["因子"]
        status = "rejected"
        reject_reason = ""
        max_value_corr = np.nan
        max_signal_corr = np.nan
        max_library_corr = np.nan
        family = str(row.get("因子家族", get_factor_family(factor_name)))
        family_quota_limit = get_family_quota_limit(config, family)
        family_count = selected_family_counts.get(family, 0)

        if factor_name not in available and factor_name in existing_active_factors and len(selected) < keep_top_n:
            if family_quota_limit is not None and family_count >= family_quota_limit:
                status = "retired"
                reject_reason = "family_quota_exceeded"
            else:
                status = "active"
                reject_reason = "preserved_existing_active_incremental_run"
                selected.append(factor_name)
                selected_set.add(factor_name)
                selected_family_counts[family] = family_count + 1
        elif factor_name not in available:
            reject_reason = "factor_not_available"
        elif not bool(row.get("初筛有效", False)):
            reject_reason = "invalid_score"
        elif row.get("初筛夏普", np.nan) < config.factor_library_min_sharpe:
            reject_reason = "low_selection_sharpe"
        elif row.get("初筛累计收益", np.nan) < config.factor_library_min_total_return:
            reject_reason = "low_selection_total_return"
        elif get_numeric_value(row, "训练夏普比率", -np.inf) < min_train_sharpe:
            reject_reason = "low_train_sharpe"
        elif get_numeric_value(row, "训练累计收益", -np.inf) < min_train_total_return:
            reject_reason = "low_train_total_return"
        elif get_numeric_value(row, "训练胜率", -np.inf) <= min_train_win_rate:
            reject_reason = "low_train_win_rate"
        elif get_numeric_value(row, selection_win_rate_column, -np.inf) <= min_test_win_rate:
            reject_reason = "low_selection_win_rate"
        elif min_selection_rank_ic is not None and get_numeric_value(row, "初筛RankIC", -np.inf) < float(min_selection_rank_ic):
            reject_reason = "low_selection_rank_ic"
        elif min_selection_monotonicity is not None and get_numeric_value(row, "初筛分组单调性", -np.inf) < float(min_selection_monotonicity):
            reject_reason = "low_selection_monotonicity"
        elif get_numeric_value(row, selection_trade_column, 0.0) < min_test_trades:
            reject_reason = "low_selection_trade_count"
        elif get_numeric_value(row, "训练交易次数", 0.0) < min_train_trades:
            reject_reason = "low_train_trade_count"
        elif get_numeric_value(row, selection_coverage_column, 0.0) < min_test_signal_coverage:
            reject_reason = "low_selection_signal_coverage"
        elif get_numeric_value(row, "训练信号覆盖率", 0.0) < min_train_signal_coverage:
            reject_reason = "low_train_signal_coverage"
        elif max_test_drawdown is not None and get_numeric_value(row, selection_drawdown_column, -np.inf) < float(max_test_drawdown):
            reject_reason = "high_selection_drawdown"
        elif max_train_drawdown is not None and get_numeric_value(row, "训练最大回撤", -np.inf) < float(max_train_drawdown):
            reject_reason = "high_train_drawdown"
        elif family_quota_limit is not None and family_count >= family_quota_limit:
            status = "retired"
            reject_reason = "family_quota_exceeded"
        elif len(selected) >= keep_top_n:
            status = "retired"
            reject_reason = "rank_outside_top_n"
        else:
            if selected:
                if not value_corr.empty and factor_name in value_corr.index:
                    selected_value_factors = [
                        selected_factor
                        for selected_factor in selected
                        if selected_factor in value_corr.columns
                    ]
                    if selected_value_factors:
                        max_value_corr = value_corr.loc[factor_name, selected_value_factors].max()
                if not signal_corr.empty and factor_name in signal_corr.index:
                    selected_signal_factors = [
                        selected_factor
                        for selected_factor in selected
                        if selected_factor in signal_corr.columns
                    ]
                    if selected_signal_factors:
                        max_signal_corr = signal_corr.loc[factor_name, selected_signal_factors].max()
                corr_values = [
                    value
                    for value in [max_value_corr, max_signal_corr]
                    if pd.notna(value)
                ]
                max_library_corr = max(corr_values) if corr_values else np.nan

            if pd.notna(max_library_corr) and max_library_corr >= max_corr_limit:
                reject_reason = "high_corr"
            else:
                status = "active"
                reject_reason = ""
                selected.append(factor_name)
                selected_set.add(factor_name)
                selected_family_counts[family] = family_count + 1

        library_row = row.to_dict()
        library_row["因子家族"] = family
        library_row["家族已选数量"] = selected_family_counts.get(family, family_count)
        library_row["家族数量上限"] = family_quota_limit if family_quota_limit is not None else np.nan
        library_row["因子库状态"] = status
        library_row["拒绝原因"] = reject_reason
        library_row["最大因子值相关性"] = max_value_corr
        library_row["最大信号相关性"] = max_signal_corr
        library_row["最大库内相关性"] = max_library_corr
        library_rows.append(library_row)

    library_all = pd.DataFrame(library_rows)
    active_library = library_all[library_all["因子"].isin(selected_set)].copy()
    rejected_library = library_all[library_all["因子库状态"] != "active"].copy()
    return active_library, library_all, rejected_library


def save_factor_library(
    active_library: pd.DataFrame,
    library_all: pd.DataFrame,
    rejected_library: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """保存因子库三张核心表。

    active_factors.csv：当前用于综合模型候选池的因子。
    factor_library_all.csv：所有历史出现过的因子及其最新状态。
    rejected_factors.csv：被拒绝或退役的因子，便于复盘原因。
    """
    library_dir = get_factor_library_dir(config)
    active_library.to_csv(
        library_dir / "active_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    library_all.to_csv(
        library_dir / "factor_library_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rejected_library.to_csv(
        library_dir / "rejected_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if "因子家族" in active_library.columns:
        family_summary = (
            active_library.groupby("因子家族", dropna=False)
            .agg(
                active因子数=("因子", "count"),
                平均初筛夏普=("初筛夏普", "mean"),
                平均初筛RankIC=("初筛RankIC", "mean"),
                平均分组单调性=("初筛分组单调性", "mean"),
            )
            .reset_index()
            .sort_values("active因子数", ascending=False)
        )
        family_summary.to_csv(
            library_dir / "active_factor_family_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

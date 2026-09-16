from __future__ import annotations

"""因子库管理模块。

本文件负责把单因子回测结果沉淀成可长期维护的因子库：
- 新结果优先使用研究期内多折 Walk-Forward 样本外表现；最终测试集默认只作
  留存评估，也可通过显式配置把它作为不参与排序的确认门槛。
- 合并历史因子库，避免每次回测覆盖已有记录。
- 对候选因子做收益门槛和相关性去重，只保留表现较好且差异足够大的因子。
- 输出 active CSV、压缩全量主库和精简 rejected CSV，兼顾程序更新与人工复盘。
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig
from .output_layout import get_frequency_scoped_dir, get_research_output_dir
from .factor_taxonomy import get_factor_family
from .factors import score_to_raw_signal


def get_family_quota_limit(config: BacktestConfig, family: str) -> int | None:
    """读取某个因子家族的 active 数量上限。"""
    if not bool(getattr(config, "factor_library_enable_family_quota", False)):
        return None
    quota = getattr(config, "factor_library_family_max_counts", {}) or {}
    value = quota.get(str(family))
    if value is None:
        return None
    return max(0, int(value))


def normalize_score_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """把不同量纲的指标转换成 0-1 横截面分位分数。"""
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    rank = clean.rank(pct=True, ascending=higher_is_better)
    return rank.fillna(0.0).astype("float64")


def conservative_pair(
    frame: pd.DataFrame,
    train_column: str,
    validation_column: str,
) -> pd.Series:
    """取训练/验证较弱值；验证缺失时用训练，训练缺失时保持无效。"""
    train = (
        pd.to_numeric(frame[train_column], errors="coerce")
        if train_column in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    validation = (
        pd.to_numeric(frame[validation_column], errors="coerce")
        if validation_column in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    result = train.replace([np.inf, -np.inf], np.nan).copy()
    validation = validation.replace([np.inf, -np.inf], np.nan)
    both_valid = result.notna() & validation.notna()
    result.loc[both_valid] = pd.concat(
        [result.loc[both_valid], validation.loc[both_valid]],
        axis=1,
    ).min(axis=1)
    return result.astype("float64")


def consistency_score(frame: pd.DataFrame, train_column: str, validation_column: str) -> pd.Series:
    """计算训练/验证一致性分数，越接近 1 说明两段表现越接近。"""
    if train_column not in frame.columns or validation_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    train = pd.to_numeric(frame[train_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    validation = pd.to_numeric(frame[validation_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    denominator = train.abs() + validation.abs()
    score = 1.0 - (train - validation).abs() / denominator.replace(0.0, np.nan)
    return score.clip(lower=0.0, upper=1.0)


def get_walk_forward_selection_masks(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    """区分需按新口径筛选的记录，以及其中真正可用的 Walk-Forward 记录。"""
    disabled = pd.Series(False, index=frame.index, dtype="bool")
    if (
        not bool(getattr(config, "single_factor_walk_forward_enabled", True))
        or "WalkForward状态" not in frame.columns
    ):
        return disabled, disabled.copy()

    status = frame["WalkForward状态"].fillna("").astype(str).str.strip()
    expected = status.ne("") & status.ne("已关闭")
    fold_count = pd.to_numeric(
        frame.get(
            "WalkForward有效折数",
            pd.Series(0.0, index=frame.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    min_folds = max(
        1,
        int(getattr(config, "factor_library_min_walk_forward_folds", 3) or 3),
    )
    sharpe = pd.to_numeric(
        frame.get("WalkForward夏普比率", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    total_return = pd.to_numeric(
        frame.get("WalkForward累计收益", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    valid = (
        expected
        & status.eq("已计算")
        & fold_count.ge(min_folds)
        & sharpe.replace([np.inf, -np.inf], np.nan).notna()
        & total_return.replace([np.inf, -np.inf], np.nan).notna()
    )
    return expected.astype("bool"), valid.astype("bool")


def apply_walk_forward_primary_metrics(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """让新结果优先使用多折样本外指标，同时保留旧主库的兼容口径。"""
    expected, valid = get_walk_forward_selection_masks(frame, config)
    if not expected.any():
        return frame, expected, valid

    frame.loc[expected, ["初筛夏普", "初筛累计收益", "初筛RankIC"]] = np.nan
    frame.loc[expected, "初筛样本"] = "WalkForward无效"
    frame.loc[expected, "初筛有效"] = False
    if valid.any():
        walk_forward_sharpe = pd.to_numeric(
            frame.get("WalkForward夏普比率", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        walk_forward_return = pd.to_numeric(
            frame.get("WalkForward累计收益", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        walk_forward_rank_ic = pd.to_numeric(
            frame.get("WalkForwardRankIC中位数", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        frame.loc[valid, "初筛夏普"] = walk_forward_sharpe.loc[valid]
        frame.loc[valid, "初筛累计收益"] = walk_forward_return.loc[valid]
        frame.loc[valid, "初筛RankIC"] = walk_forward_rank_ic.loc[valid]
        frame.loc[valid, "初筛样本"] = "WalkForward样本外"
        frame.loc[valid, "初筛有效"] = True
    return frame, expected, valid


def add_factor_research_scores(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """为因子增加科研化综合评分。

    新记录优先使用研究期内多折 Walk-Forward 指标；历史记录尚未重测时，
    兼容使用训练集和验证集表现。最终测试集不参与主排序，避免测试集污染。
    """
    scored = frame.copy()

    performance_score = (
        0.65 * normalize_score_series(conservative_pair(scored, "训练夏普比率", "验证夏普比率"))
        + 0.35 * normalize_score_series(conservative_pair(scored, "训练累计收益", "验证累计收益"))
    )

    predictive_score = (
        0.30 * normalize_score_series(conservative_pair(scored, "训练RankIC", "验证RankIC"))
        + 0.15 * normalize_score_series(conservative_pair(scored, "训练RankICIR", "验证RankICIR"))
        + 0.15 * normalize_score_series(conservative_pair(scored, "训练IC胜率", "验证IC胜率"))
        + 0.15 * normalize_score_series(conservative_pair(scored, "训练方向命中率", "验证方向命中率"))
        + 0.15 * normalize_score_series(conservative_pair(scored, "训练分组单调性", "验证分组单调性"))
        + 0.10 * normalize_score_series(conservative_pair(scored, "训练分组收益差", "验证分组收益差"))
    )

    consistency = pd.concat(
        [
            consistency_score(scored, "训练夏普比率", "验证夏普比率"),
            consistency_score(scored, "训练RankIC", "验证RankIC"),
            consistency_score(scored, "训练分组单调性", "验证分组单调性"),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.0)

    profit_month_share = conservative_pair(scored, "训练盈利月份占比", "验证盈利月份占比")
    concentration = conservative_pair(scored, "训练月度收益集中度", "验证月度收益集中度")
    stability_score = (
        0.60 * pd.to_numeric(profit_month_share, errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
        + 0.40 * (1.0 - pd.to_numeric(concentration, errors="coerce").clip(lower=0.0, upper=1.0)).fillna(0.0)
    )

    walk_forward_expected, walk_forward_valid = get_walk_forward_selection_masks(
        scored,
        config,
    )
    if walk_forward_expected.any():
        performance_score.loc[walk_forward_expected] = 0.0
        predictive_score.loc[walk_forward_expected] = 0.0
        consistency.loc[walk_forward_expected] = 0.0
        stability_score.loc[walk_forward_expected] = 0.0
    if walk_forward_valid.any():
        def walk_forward_numeric(column: str) -> pd.Series:
            return pd.to_numeric(
                scored.get(column, pd.Series(np.nan, index=scored.index)),
                errors="coerce",
            )

        wf_sharpe = walk_forward_numeric("WalkForward夏普比率")
        wf_return = walk_forward_numeric("WalkForward累计收益")
        wf_rank_ic = walk_forward_numeric("WalkForwardRankIC中位数")
        wf_hit_rate = walk_forward_numeric("WalkForward方向命中率")
        wf_positive_share = pd.to_numeric(
            walk_forward_numeric("WalkForward盈利折占比"),
            errors="coerce",
        ).clip(0.0, 1.0)
        wf_rank_ic_positive_share = pd.to_numeric(
            walk_forward_numeric("WalkForwardRankIC正向折占比"),
            errors="coerce",
        ).clip(0.0, 1.0)
        wf_direction_consistency = pd.to_numeric(
            walk_forward_numeric("WalkForward方向一致率"),
            errors="coerce",
        ).clip(0.0, 1.0)
        wf_median_sharpe = walk_forward_numeric("WalkForward夏普中位数")
        wf_worst_sharpe = walk_forward_numeric("WalkForward夏普最差值")

        wf_performance_score = (
            0.65 * normalize_score_series(wf_sharpe)
            + 0.35 * normalize_score_series(wf_return)
        )
        wf_predictive_score = (
            0.45 * normalize_score_series(wf_rank_ic)
            + 0.25 * normalize_score_series(wf_hit_rate)
            + 0.15 * wf_rank_ic_positive_share.fillna(0.0)
            + 0.15 * wf_positive_share.fillna(0.0)
        )
        wf_consistency_score = (
            0.60 * wf_direction_consistency.fillna(0.0)
            + 0.40 * wf_positive_share.fillna(0.0)
        )
        wf_stability_score = (
            0.50 * wf_positive_share.fillna(0.0)
            + 0.25 * normalize_score_series(wf_median_sharpe)
            + 0.25 * normalize_score_series(wf_worst_sharpe)
        )
        performance_score.loc[walk_forward_valid] = wf_performance_score.loc[
            walk_forward_valid
        ]
        predictive_score.loc[walk_forward_valid] = wf_predictive_score.loc[
            walk_forward_valid
        ]
        consistency.loc[walk_forward_valid] = wf_consistency_score.loc[
            walk_forward_valid
        ]
        stability_score.loc[walk_forward_valid] = wf_stability_score.loc[
            walk_forward_valid
        ]

    weights = {
        "performance": max(0.0, float(getattr(config, "factor_library_score_weight_performance", 0.40) or 0.0)),
        "predictive": max(0.0, float(getattr(config, "factor_library_score_weight_predictive", 0.30) or 0.0)),
        "consistency": max(0.0, float(getattr(config, "factor_library_score_weight_consistency", 0.20) or 0.0)),
        "stability": max(0.0, float(getattr(config, "factor_library_score_weight_stability", 0.10) or 0.0)),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {"performance": 1.0, "predictive": 0.0, "consistency": 0.0, "stability": 0.0}
        total_weight = 1.0

    scored["初筛交易表现评分"] = performance_score.astype("float64")
    scored["初筛预测能力评分"] = predictive_score.astype("float64")
    scored["初筛一致性评分"] = consistency.astype("float64")
    scored["初筛稳定性评分"] = stability_score.astype("float64")
    scored["初筛科研综合评分"] = (
        weights["performance"] * scored["初筛交易表现评分"]
        + weights["predictive"] * scored["初筛预测能力评分"]
        + weights["consistency"] * scored["初筛一致性评分"]
        + weights["stability"] * scored["初筛稳定性评分"]
    ) / total_weight
    return scored


def rank_single_factor_summary(
    summary: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """优先按照多折 Walk-Forward 样本外表现做初筛排序。

    排序逻辑：
    1. 新回测记录使用最终测试集之前的多折 Walk-Forward 汇总指标。
    2. 历史记录尚未按新口径重测时，兼容使用训练/验证较弱值。
       最终测试集永不参与排序。
    3. 仅把前 single_factor_keep_top_n 个标记为 active，其余先标记为 rejected。

    这里还没有做相关性去重，相关性过滤会在 build_factor_library 中完成。
    """
    ranked = summary.copy()
    has_train = (
        {"训练夏普比率", "训练累计收益"}.issubset(ranked.columns)
        and pd.to_numeric(ranked["训练夏普比率"], errors="coerce").notna().any()
        and pd.to_numeric(ranked["训练累计收益"], errors="coerce").notna().any()
    )
    has_validation = (
        {"验证夏普比率", "验证累计收益"}.issubset(ranked.columns)
        and pd.to_numeric(ranked["验证夏普比率"], errors="coerce").notna().any()
        and pd.to_numeric(ranked["验证累计收益"], errors="coerce").notna().any()
    )
    has_train_validation_rank_ic = (
        {"训练RankIC", "验证RankIC"}.issubset(ranked.columns)
        and pd.to_numeric(ranked["验证RankIC"], errors="coerce").notna().any()
    )
    has_train_validation_monotonicity = (
        {"训练分组单调性", "验证分组单调性"}.issubset(ranked.columns)
        and pd.to_numeric(ranked["验证分组单调性"], errors="coerce").notna().any()
    )
    if not has_train:
        raise KeyError("单因子汇总缺少训练收益/夏普列，不能使用最终测试集替代入库筛选。")
    sharpe = conservative_pair(ranked, "训练夏普比率", "验证夏普比率")
    total_return = conservative_pair(ranked, "训练累计收益", "验证累计收益")
    valid_train = (
        pd.to_numeric(ranked["训练夏普比率"], errors="coerce").notna()
        & pd.to_numeric(ranked["训练累计收益"], errors="coerce").notna()
    )
    valid_validation = (
        pd.to_numeric(ranked.get("验证夏普比率"), errors="coerce").notna()
        & pd.to_numeric(ranked.get("验证累计收益"), errors="coerce").notna()
        if has_validation
        else pd.Series(False, index=ranked.index)
    )
    selection_sample = np.select(
        [valid_train & valid_validation, valid_train],
        ["训练+验证", "训练"],
        default="不可追溯",
    )

    if has_train_validation_rank_ic:
        selection_rank_ic = conservative_pair(ranked, "训练RankIC", "验证RankIC")
    elif "训练RankIC" in ranked.columns:
        selection_rank_ic = ranked["训练RankIC"].replace([np.inf, -np.inf], np.nan)
    else:
        selection_rank_ic = pd.Series(np.nan, index=ranked.index)

    if has_train_validation_monotonicity:
        selection_monotonicity = conservative_pair(
            ranked,
            "训练分组单调性",
            "验证分组单调性",
        )
    elif "训练分组单调性" in ranked.columns:
        selection_monotonicity = ranked["训练分组单调性"].replace([np.inf, -np.inf], np.nan)
    else:
        selection_monotonicity = pd.Series(np.nan, index=ranked.index)

    ranked["初筛夏普"] = sharpe
    ranked["初筛累计收益"] = total_return
    ranked["初筛RankIC"] = selection_rank_ic
    ranked["初筛分组单调性"] = selection_monotonicity
    ranked["初筛样本"] = selection_sample
    ranked["因子家族"] = ranked["因子"].map(get_factor_family)
    ranked["初筛有效"] = sharpe.notna() & total_return.notna()
    ranked, _, _ = apply_walk_forward_primary_metrics(ranked, config)
    ranked = add_factor_research_scores(ranked, config)
    ranked = ranked.sort_values(
        [
            "初筛有效",
            "初筛预测能力评分",
            "初筛科研综合评分",
            "初筛一致性评分",
            "初筛夏普",
            "初筛RankIC",
            "初筛累计收益",
        ],
        ascending=[False, False, False, False, False, False, False],
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
    if library_dir.is_absolute():
        library_dir = get_frequency_scoped_dir(library_dir, config)
    else:
        library_dir = get_research_output_dir(config, str(library_dir))
    library_dir.mkdir(parents=True, exist_ok=True)
    return library_dir


def get_manual_factor_exclusions_path(config: BacktestConfig) -> Path:
    """返回当前品种、当前频率的人工排除清单路径。"""
    return get_factor_library_dir(config) / "manual_factor_exclusions.csv"


def get_manual_factor_approvals_path(config: BacktestConfig) -> Path:
    """返回当前品种、当前频率的人工审批清单路径。"""
    return get_factor_library_dir(config) / "manual_factor_approvals.csv"


def get_manual_factor_protections_path(config: BacktestConfig) -> Path:
    """返回当前品种、当前频率的手工保护清单路径。"""
    return get_factor_library_dir(config) / "manual_factor_protections.csv"


def load_manual_factor_approvals(
    config: BacktestConfig,
    *,
    bootstrap_existing_active: bool = True,
) -> pd.DataFrame:
    """读取人工审批清单，并可把升级前的 active 库登记为存量审批。"""
    columns = ["因子编号", "因子标签", "因子", "人工审批原因", "人工审批时间", "审批来源"]
    path = get_manual_factor_approvals_path(config)
    if not path.exists() and bootstrap_existing_active:
        active_path = get_factor_library_dir(config) / "active_factors.csv"
        if active_path.exists():
            try:
                existing_active = pd.read_csv(active_path, encoding="utf-8-sig")
            except Exception:
                existing_active = pd.DataFrame()
            if "因子" in existing_active.columns and not existing_active.empty:
                now = pd.Timestamp.now().isoformat(timespec="seconds")
                migrated = existing_active.copy()
                for identity_column in ("因子编号", "因子标签"):
                    if identity_column not in migrated.columns:
                        migrated[identity_column] = ""
                migrated["因子"] = migrated["因子"].fillna("").astype(str).str.strip()
                migrated["人工审批原因"] = "升级前已有active，自动迁移为存量审批"
                migrated["人工审批时间"] = now
                migrated["审批来源"] = "legacy_active_migration"
                migrated = migrated[migrated["因子"].ne("")]
                save_manual_factor_approvals(migrated, config)
                return migrated[columns].reset_index(drop=True)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        approvals = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in approvals.columns:
            approvals[column] = ""
    approvals["因子"] = approvals["因子"].fillna("").astype(str).str.strip()
    approvals = approvals[approvals["因子"].ne("")]
    return approvals[columns].drop_duplicates("因子", keep="last").reset_index(drop=True)


def save_manual_factor_approvals(
    approvals: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    """原子保存人工审批清单。"""
    path = get_manual_factor_approvals_path(config)
    columns = ["因子编号", "因子标签", "因子", "人工审批原因", "人工审批时间", "审批来源"]
    output = approvals.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    output = output[columns].drop_duplicates("因子", keep="last")
    temp_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)
    return path


def load_manual_factor_exclusions(config: BacktestConfig) -> pd.DataFrame:
    """读取人工排除清单；文件不存在时返回结构完整的空表。"""
    columns = ["因子编号", "因子标签", "因子", "人工排除原因", "人工排除时间"]
    path = get_manual_factor_exclusions_path(config)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        exclusions = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in exclusions.columns:
            exclusions[column] = ""
    exclusions["因子"] = exclusions["因子"].fillna("").astype(str).str.strip()
    exclusions = exclusions[exclusions["因子"].ne("")]
    return exclusions[columns].drop_duplicates("因子", keep="last").reset_index(drop=True)


def save_manual_factor_exclusions(
    exclusions: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    """原子保存人工排除清单，避免维护过程中留下半写入文件。"""
    path = get_manual_factor_exclusions_path(config)
    columns = ["因子编号", "因子标签", "因子", "人工排除原因", "人工排除时间"]
    output = exclusions.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    output = output[columns].drop_duplicates("因子", keep="last")
    temp_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)
    return path


def load_manual_factor_protections(config: BacktestConfig) -> pd.DataFrame:
    """读取不会被自动治理流程剔除的 active 因子清单。"""
    columns = ["因子编号", "因子标签", "因子", "手工保护原因", "手工保护时间"]
    path = get_manual_factor_protections_path(config)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        protections = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in protections.columns:
            protections[column] = ""
    protections["因子"] = protections["因子"].fillna("").astype(str).str.strip()
    protections = protections[protections["因子"].ne("")]
    return protections[columns].drop_duplicates("因子", keep="last").reset_index(drop=True)


def save_manual_factor_protections(
    protections: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    """原子保存 active 因子手工保护清单。"""
    path = get_manual_factor_protections_path(config)
    columns = ["因子编号", "因子标签", "因子", "手工保护原因", "手工保护时间"]
    output = protections.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    output = output[columns].drop_duplicates("因子", keep="last")
    temp_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)
    return path


def get_manual_excluded_factor_names(config: BacktestConfig) -> set[str]:
    """返回人工禁止进入 active 库的因子名集合。"""
    if not bool(getattr(config, "factor_library_respect_manual_exclusions", True)):
        return set()
    exclusions = load_manual_factor_exclusions(config)
    return set(exclusions["因子"].astype(str))


def load_existing_factor_library(config: BacktestConfig) -> pd.DataFrame:
    """读取已有的全量因子库，优先压缩格式并兼容旧 CSV。

    如果文件不存在，返回空 DataFrame，方便首次运行时直接创建新因子库。
    """
    library_dir = get_factor_library_dir(config)
    candidates = [
        (library_dir / "factor_library_all.parquet", "parquet"),
        (library_dir / "factor_library_all.pkl.gz", "pickle"),
        (library_dir / "factor_library_all.csv", "csv"),
    ]
    candidates.sort(
        key=lambda item: item[0].stat().st_mtime if item[0].exists() else -1.0,
        reverse=True,
    )
    for library_path, storage_format in candidates:
        if not library_path.exists():
            continue
        try:
            if storage_format == "parquet":
                return pd.read_parquet(library_path)
            if storage_format == "pickle":
                return pd.read_pickle(library_path, compression="gzip")
            return pd.read_csv(library_path, low_memory=False)
        except Exception as exc:
            concise_reason = str(exc).splitlines()[0]
            print(
                f"因子主库读取失败，尝试下一兼容格式: {library_path}，"
                f"原因: {type(exc).__name__}: {concise_reason}"
            )

    # 主库存在但当前环境缺少读取引擎时，利用三张可移植 CSV 视图恢复维护能力。
    # 该视图可能缺少完整诊断列，因此用 attrs 标记为部分主库，管理脚本不会反写覆盖原文件。
    view_specs = [
        (library_dir / "rejected_factors.csv", None),
        (library_dir / "pre_active_factors.csv", "pre_active"),
        (library_dir / "active_factors.csv", "active"),
    ]
    view_frames: list[pd.DataFrame] = []
    for view_path, forced_status in view_specs:
        if not view_path.exists():
            continue
        try:
            frame = pd.read_csv(view_path, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        if "因子" not in frame.columns:
            continue
        if forced_status is not None:
            frame["因子库状态"] = forced_status
        view_frames.append(frame)
    if view_frames:
        restored = pd.concat(view_frames, ignore_index=True, sort=False).copy()
        restored = restored.dropna(subset=["因子"]).drop_duplicates("因子", keep="last")
        restored.attrs["factor_library_partial"] = True
        print(
            "完整主库不可读，已从 active/pre_active/rejected CSV 恢复兼容视图；"
            "原主库不会被人工维护命令覆盖。"
        )
        return restored.reset_index(drop=True)
    return pd.DataFrame()


def get_existing_factor_library_storage_path(config: BacktestConfig) -> Path | None:
    """返回当前实际存在的全量因子主库路径。"""
    library_dir = get_factor_library_dir(config)
    for filename in (
        "factor_library_all.parquet",
        "factor_library_all.pkl.gz",
        "factor_library_all.csv",
    ):
        path = library_dir / filename
        if path.exists():
            return path
    return None


def _write_factor_library_master(
    library_all: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    """原子写入全量主库；auto 模式在 Parquet 不可用时回退 gzip Pickle。"""
    library_dir = get_factor_library_dir(config)
    requested_format = str(
        getattr(config, "factor_library_storage_format", "auto") or "auto"
    ).strip().lower()
    if requested_format not in {"auto", "parquet", "pickle", "csv"}:
        raise ValueError("factor_library_storage_format 只能是 auto/parquet/pickle/csv。")

    errors: list[str] = []
    formats = [requested_format] if requested_format != "auto" else ["parquet", "pickle"]
    for storage_format in formats:
        if storage_format == "parquet":
            target = library_dir / "factor_library_all.parquet"
        elif storage_format == "pickle":
            target = library_dir / "factor_library_all.pkl.gz"
        else:
            target = library_dir / "factor_library_all.csv"
        temp_path = target.with_name(f".{target.name}.tmp")
        try:
            if storage_format == "parquet":
                parquet_frame = library_all.copy()
                for column in parquet_frame.select_dtypes(include=["object"]).columns:
                    parquet_frame[column] = parquet_frame[column].map(
                        lambda value: None if pd.isna(value) else str(value)
                    )
                parquet_frame.to_parquet(temp_path, index=False, compression="zstd")
            elif storage_format == "pickle":
                library_all.to_pickle(temp_path, compression="gzip")
            else:
                library_all.to_csv(temp_path, index=False, encoding="utf-8-sig")
            temp_path.replace(target)

            # Parquet 依赖可选引擎；同时保留 Pickle 副本，确保换解释器后仍可维护。
            if storage_format == "parquet" and bool(
                getattr(config, "factor_library_write_pickle_fallback", True)
            ):
                fallback_target = library_dir / "factor_library_all.pkl.gz"
                fallback_temp = fallback_target.with_name(f".{fallback_target.name}.tmp")
                try:
                    library_all.to_pickle(fallback_temp, compression="gzip")
                    fallback_temp.replace(fallback_target)
                finally:
                    if fallback_temp.exists():
                        fallback_temp.unlink()

            if bool(
                getattr(config, "factor_library_remove_legacy_csv_after_migration", True)
            ) and storage_format != "csv":
                legacy_csv = library_dir / "factor_library_all.csv"
                if legacy_csv.exists():
                    legacy_csv.unlink()
            for stale_name in ("factor_library_all.parquet", "factor_library_all.pkl.gz"):
                stale_path = library_dir / stale_name
                preserve_compatible_master = bool(
                    getattr(config, "factor_library_write_pickle_fallback", True)
                )
                if (
                    stale_path != target
                    and stale_path.exists()
                    and not preserve_compatible_master
                ):
                    stale_path.unlink()
            return target
        except Exception as exc:
            errors.append(f"{storage_format}: {exc}")
        finally:
            if temp_path.exists():
                temp_path.unlink()
    raise RuntimeError("全量因子主库写入失败: " + " | ".join(errors))


def build_compact_rejected_library(rejected_library: pd.DataFrame) -> pd.DataFrame:
    """生成可人工查看、但不重复保存全部 135 列的拒绝因子表。"""
    compact_columns = [
        "因子编号", "因子", "因子家族", "初筛样本", "初筛夏普", "初筛累计收益",
        "初筛RankIC", "初筛分组单调性", "初筛预测能力评分", "初筛一致性评分",
        "初筛科研综合评分", "训练夏普比率", "验证夏普比率",
        "测试夏普比率", "测试累计收益", "训练胜率", "验证胜率", "测试胜率",
        "训练交易次数", "验证交易次数", "测试交易次数", "因子库状态",
        "WalkForward状态", "WalkForward有效折数", "WalkForward夏普比率",
        "WalkForward夏普中位数", "WalkForward夏普最差值",
        "WalkForward累计收益", "WalkForward盈利折占比",
        "WalkForward方向一致率", "WalkForwardRankIC中位数",
        "WalkForward方向命中率", "WalkForward交易次数",
        "最终测试入库要求生效", "最终测试入库表现通过", "最终测试入库拒绝原因",
        "拒绝原因", "详细诊断状态", "人工排除原因", "人工排除时间",
        "最大库内相关性", "家族数量上限", "错误",
    ]
    available_columns = [column for column in compact_columns if column in rejected_library.columns]
    return rejected_library[available_columns].copy()


def slice_factor_selection_covariates(
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """截取允许参与因子筛选的协变量样本，始终隔离最终测试集。

    因子相关性虽然不读取未来收益标签，但如果使用最终测试期的因子分布，
    仍会让测试集参与决定候选集合。这里按与单因子回测相同的时间比例做
    位置切分，只返回训练集或训练集加验证集。
    """
    if factors.empty:
        return factors

    sample_count = len(factors)
    if sample_count < 3:
        # 极短输入无法形成完整三段；至少保留最后一行作为未使用留存样本。
        return factors.iloc[: max(0, sample_count - 1)]

    train_ratio = float(getattr(config, "auto_select_train_ratio", 0.7))
    validation_ratio = float(getattr(config, "auto_select_validation_ratio", 0.15))
    train_end = min(max(int(sample_count * train_ratio), 1), sample_count - 2)
    validation_end = int(sample_count * (train_ratio + validation_ratio))
    validation_end = min(max(validation_end, train_end + 1), sample_count - 1)

    scope = str(
        getattr(config, "factor_selection_covariate_scope", "train_validation")
        or "train_validation"
    ).strip().lower()
    end_position = train_end if scope == "train" else validation_end
    return factors.iloc[:end_position]


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


def get_selection_config_value(
    config: BacktestConfig,
    canonical_name: str,
    legacy_name: str,
) -> Any:
    """读取入库选择参数；显式旧字段仅作为迁移期兼容覆盖。"""
    legacy_value = getattr(config, legacy_name, None)
    if legacy_value is not None:
        return legacy_value
    return getattr(config, canonical_name)


def build_final_test_performance_reject_reasons(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    """向量化检查最终测试表现；空字符串表示通过或未启用。"""
    reasons = pd.Series("", index=frame.index, dtype="object")
    if not bool(getattr(config, "factor_library_require_test_performance", False)):
        return reasons

    def numeric(column: str) -> pd.Series:
        values = frame.get(column, pd.Series(np.nan, index=frame.index))
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def reject(mask: pd.Series, reason: str) -> None:
        reasons.loc[reasons.eq("") & mask.fillna(False)] = reason

    test_sharpe = numeric("测试夏普比率")
    test_return = numeric("测试累计收益")
    reject(test_sharpe.isna() | test_return.isna(), "missing_final_test_metrics")
    reject(test_sharpe < float(config.factor_library_min_sharpe), "low_test_sharpe")
    reject(
        test_return < float(config.factor_library_min_total_return),
        "low_test_total_return",
    )

    min_win_rate = get_selection_config_value(
        config,
        "factor_library_min_selection_win_rate",
        "factor_library_min_test_win_rate",
    )
    if min_win_rate is not None:
        test_win_rate = numeric("测试胜率")
        reject(test_win_rate.isna(), "missing_final_test_metrics")
        reject(test_win_rate <= float(min_win_rate), "low_test_win_rate")

    min_rank_ic = getattr(config, "factor_library_min_selection_rank_ic", None)
    if min_rank_ic is not None:
        test_rank_ic = numeric("测试RankIC")
        reject(test_rank_ic.isna(), "missing_final_test_predictive_metrics")
        reject(test_rank_ic < float(min_rank_ic), "low_test_rank_ic")

    min_monotonicity = getattr(
        config,
        "factor_library_min_selection_monotonicity",
        None,
    )
    if min_monotonicity is not None:
        test_monotonicity = numeric("测试分组单调性")
        reject(test_monotonicity.isna(), "missing_final_test_predictive_metrics")
        reject(
            test_monotonicity < float(min_monotonicity),
            "low_test_monotonicity",
        )

    min_trades = max(
        0,
        int(
            get_selection_config_value(
                config,
                "factor_library_min_selection_trades",
                "factor_library_min_test_trades",
            )
            or 0
        ),
    )
    if min_trades > 0:
        test_trades = numeric("测试交易次数")
        reject(test_trades.isna(), "missing_final_test_metrics")
        reject(test_trades < min_trades, "low_test_trade_count")

    min_coverage = max(
        0.0,
        float(
            get_selection_config_value(
                config,
                "factor_library_min_selection_signal_coverage",
                "factor_library_min_test_signal_coverage",
            )
            or 0.0
        ),
    )
    if min_coverage > 0:
        test_coverage = numeric("测试信号覆盖率")
        reject(test_coverage.isna(), "missing_final_test_metrics")
        reject(test_coverage < min_coverage, "low_test_signal_coverage")

    max_drawdown = get_selection_config_value(
        config,
        "factor_library_max_selection_drawdown",
        "factor_library_max_test_drawdown",
    )
    if max_drawdown is not None:
        test_drawdown = numeric("测试最大回撤")
        reject(test_drawdown.isna(), "missing_final_test_metrics")
        reject(test_drawdown < float(max_drawdown), "high_test_drawdown")
    return reasons


def get_final_test_performance_reject_reason(
    row: pd.Series,
    config: BacktestConfig,
) -> str:
    """检查单条最终测试记录，主要供测试和外部诊断调用。"""
    result = build_final_test_performance_reject_reasons(row.to_frame().T, config)
    return str(result.iloc[0])


def build_selection_metric(
    frame: pd.DataFrame,
    validation_column: str,
    train_column: str,
) -> tuple[pd.Series, pd.Series]:
    """逐因子选择验证指标，单行验证缺失时回退同一行训练指标。"""
    validation = (
        pd.to_numeric(frame[validation_column], errors="coerce")
        if validation_column in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    train = (
        pd.to_numeric(frame[train_column], errors="coerce")
        if train_column in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    validation = validation.replace([np.inf, -np.inf], np.nan)
    train = train.replace([np.inf, -np.inf], np.nan)
    selected = validation.combine_first(train).astype("float64")
    source = pd.Series(
        np.select(
            [validation.notna(), train.notna()],
            [validation_column, train_column],
            default="不可用",
        ),
        index=frame.index,
        dtype="object",
    )
    return selected, source


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
    manual_exclusions = load_manual_factor_exclusions(config)
    manual_excluded = (
        set(manual_exclusions["因子"].astype(str))
        if bool(getattr(config, "factor_library_respect_manual_exclusions", True))
        else set()
    )
    manual_reason_map = manual_exclusions.set_index("因子")["人工排除原因"].to_dict()
    manual_time_map = manual_exclusions.set_index("因子")["人工排除时间"].to_dict()
    manual_protections = load_manual_factor_protections(config)
    protected_names = set(manual_protections["因子"].astype(str))
    protection_reason_map = manual_protections.set_index("因子")["手工保护原因"].to_dict()
    protection_time_map = manual_protections.set_index("因子")["手工保护时间"].to_dict()
    combined["人工排除原因"] = combined["因子"].map(manual_reason_map).fillna("")
    combined["人工排除时间"] = combined["因子"].map(manual_time_map).fillna("")
    combined["因子家族"] = combined["因子"].map(get_factor_family)
    if "初筛有效" not in combined.columns:
        combined["初筛有效"] = False
    else:
        combined["初筛有效"] = combined["初筛有效"].map(
            lambda value: str(value).lower() == "true" if pd.notna(value) else False
        )
    numeric_columns = [
        "初筛夏普",
        "初筛累计收益",
        "初筛RankIC",
        "初筛分组单调性",
        "初筛交易表现评分",
        "初筛预测能力评分",
        "初筛一致性评分",
        "初筛稳定性评分",
        "初筛科研综合评分",
        "训练夏普比率",
        "训练累计收益",
        "训练IC",
        "训练RankIC",
        "训练ICIR",
        "训练RankICIR",
        "训练IC胜率",
        "训练方向命中率",
        "训练多空方向命中率",
        "训练有效方向样本数",
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
        "验证方向命中率",
        "验证多空方向命中率",
        "验证有效方向样本数",
        "验证分组单调性",
        "验证分组收益差",
        "验证夏普比率",
        "验证累计收益",
        "验证胜率",
        "验证交易次数",
        "验证信号覆盖率",
        "验证最大回撤",
        "WalkForward计划折数",
        "WalkForward有效折数",
        "WalkForward累计收益",
        "WalkForward年化收益",
        "WalkForward夏普比率",
        "WalkForward最大回撤",
        "WalkForward胜率",
        "WalkForward胜率Wilson下限",
        "WalkForward胜率Wilson上限",
        "WalkForward交易次数",
        "WalkForward样本K线数",
        "WalkForward信号覆盖率",
        "WalkForward持仓覆盖率",
        "WalkForward夏普中位数",
        "WalkForward夏普最差值",
        "WalkForward盈利折占比",
        "WalkForward方向一致率",
        "WalkForwardRankIC中位数",
        "WalkForwardRankIC正向折占比",
        "WalkForward方向命中率",
        "测试IC",
        "测试RankIC",
        "测试ICIR",
        "测试RankICIR",
        "测试IC胜率",
        "测试方向命中率",
        "测试多空方向命中率",
        "测试有效方向样本数",
        "测试分组单调性",
        "测试分组收益差",
        "测试夏普比率",
        "测试累计收益",
        "测试胜率",
        "测试交易次数",
        "测试信号覆盖率",
        "测试最大回撤",
    ]
    missing_numeric_columns = [
        column for column in numeric_columns if column not in combined.columns
    ]
    if missing_numeric_columns:
        combined = pd.concat(
            [
                combined,
                pd.DataFrame(
                    np.nan,
                    index=combined.index,
                    columns=missing_numeric_columns,
                ),
            ],
            axis=1,
        )
    numeric_frame = combined[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    combined = pd.concat(
        [combined.drop(columns=numeric_columns), numeric_frame],
        axis=1,
    ).copy()

    # 历史初筛列可能由旧版本口径生成，统一从可追溯的训练/验证指标重建。
    combined["初筛夏普"] = conservative_pair(
        combined,
        "训练夏普比率",
        "验证夏普比率",
    )
    combined["初筛累计收益"] = conservative_pair(
        combined,
        "训练累计收益",
        "验证累计收益",
    )
    combined["初筛RankIC"] = conservative_pair(
        combined,
        "训练RankIC",
        "验证RankIC",
    )
    combined["初筛分组单调性"] = conservative_pair(
        combined,
        "训练分组单调性",
        "验证分组单调性",
    )
    has_traceable_train = (
        combined["训练夏普比率"].notna()
        & combined["训练累计收益"].notna()
    )
    has_traceable_validation = (
        combined["验证夏普比率"].notna()
        & combined["验证累计收益"].notna()
    )
    combined["入库数据可追溯"] = has_traceable_train
    combined["初筛有效"] = (
        has_traceable_train
        & combined["初筛夏普"].notna()
        & combined["初筛累计收益"].notna()
    )
    combined["初筛样本"] = np.select(
        [has_traceable_train & has_traceable_validation, has_traceable_train],
        ["训练+验证", "训练"],
        default="不可追溯",
    )
    combined, walk_forward_expected, walk_forward_valid = (
        apply_walk_forward_primary_metrics(combined, config)
    )
    combined["WalkForward要求生效"] = walk_forward_expected
    combined["WalkForward有效"] = walk_forward_valid
    combined = add_factor_research_scores(combined, config)
    combined = combined.sort_values(
        [
            "初筛有效",
            "初筛预测能力评分",
            "初筛科研综合评分",
            "初筛一致性评分",
            "初筛夏普",
            "初筛RankIC",
            "初筛累计收益",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    if protected_names:
        # 保护因子优先占位，确保不会被普通候选的相关性、家族配额或全库上限挤出。
        protected_mask = combined["因子"].astype(str).isin(protected_names)
        combined = pd.concat(
            [combined.loc[protected_mask], combined.loc[~protected_mask]],
            ignore_index=True,
        )

    # 两列会随因子记录一起排序；从排序后的列恢复掩码，彻底隔离旧 Series 索引。
    walk_forward_expected = combined["WalkForward要求生效"].fillna(False).astype("bool")
    walk_forward_valid = combined["WalkForward有效"].fillna(False).astype("bool")
    test_reject_reasons = build_final_test_performance_reject_reasons(combined, config)
    combined["最终测试入库要求生效"] = bool(
        getattr(config, "factor_library_require_test_performance", False)
    )
    combined["最终测试入库表现通过"] = test_reject_reasons.eq("")
    combined["最终测试入库拒绝原因"] = test_reject_reasons

    available = set(factors.columns)
    # 只有“表现有效 + 当前代码仍能生成 + 满足收益门槛”的因子，才进入相关性去重候选池。
    min_train_sharpe = float(getattr(config, "factor_library_min_train_sharpe", -np.inf))
    min_train_total_return = float(getattr(config, "factor_library_min_train_total_return", -np.inf))
    raw_train_win_rate = getattr(config, "factor_library_min_train_win_rate", None)
    min_train_win_rate = (
        None if raw_train_win_rate is None else float(raw_train_win_rate)
    )
    raw_selection_win_rate = get_selection_config_value(
        config,
        "factor_library_min_selection_win_rate",
        "factor_library_min_test_win_rate",
    )
    min_selection_win_rate = (
        None if raw_selection_win_rate is None else float(raw_selection_win_rate)
    )
    min_selection_trades = max(
        0,
        int(
            get_selection_config_value(
                config,
                "factor_library_min_selection_trades",
                "factor_library_min_test_trades",
            )
            or 0
        ),
    )
    min_train_trades = max(0, int(getattr(config, "factor_library_min_train_trades", 0) or 0))
    min_selection_signal_coverage = max(
        0.0,
        float(
            get_selection_config_value(
                config,
                "factor_library_min_selection_signal_coverage",
                "factor_library_min_test_signal_coverage",
            )
            or 0.0
        ),
    )
    min_train_signal_coverage = max(
        0.0,
        float(getattr(config, "factor_library_min_train_signal_coverage", 0.0) or 0.0),
    )
    max_selection_drawdown = get_selection_config_value(
        config,
        "factor_library_max_selection_drawdown",
        "factor_library_max_test_drawdown",
    )
    max_train_drawdown = getattr(config, "factor_library_max_train_drawdown", None)
    min_selection_rank_ic = getattr(config, "factor_library_min_selection_rank_ic", None)
    min_selection_monotonicity = getattr(config, "factor_library_min_selection_monotonicity", None)
    min_research_score = getattr(config, "factor_library_min_research_score", None)
    min_predictive_score = getattr(config, "factor_library_min_predictive_score", None)
    min_walk_forward_folds = max(
        1,
        int(getattr(config, "factor_library_min_walk_forward_folds", 3) or 3),
    )
    min_walk_forward_positive_share = float(
        getattr(config, "factor_library_min_walk_forward_positive_fold_ratio", 0.60)
    )
    min_walk_forward_direction_consistency = float(
        getattr(config, "factor_library_min_walk_forward_direction_consistency", 0.60)
    )
    min_walk_forward_median_sharpe = float(
        getattr(config, "factor_library_min_walk_forward_median_sharpe", 0.0)
    )
    min_walk_forward_trades = max(
        0,
        int(getattr(config, "factor_library_min_walk_forward_trades", 30) or 0),
    )
    threshold_epsilon = 1e-12
    selection_win_rate, selection_win_rate_source = build_selection_metric(
        combined,
        "验证胜率",
        "训练胜率",
    )
    selection_trades, selection_trade_source = build_selection_metric(
        combined,
        "验证交易次数",
        "训练交易次数",
    )
    selection_coverage, selection_coverage_source = build_selection_metric(
        combined,
        "验证信号覆盖率",
        "训练信号覆盖率",
    )
    selection_drawdown, selection_drawdown_source = build_selection_metric(
        combined,
        "验证最大回撤",
        "训练最大回撤",
    )
    if walk_forward_expected.any():
        for metric, source in (
            (selection_win_rate, selection_win_rate_source),
            (selection_trades, selection_trade_source),
            (selection_coverage, selection_coverage_source),
            (selection_drawdown, selection_drawdown_source),
        ):
            metric.loc[walk_forward_expected] = np.nan
            source.loc[walk_forward_expected] = "WalkForward不可用"
    if walk_forward_valid.any():
        selection_win_rate.loc[walk_forward_valid] = combined.loc[
            walk_forward_valid,
            "WalkForward胜率",
        ]
        selection_trades.loc[walk_forward_valid] = combined.loc[
            walk_forward_valid,
            "WalkForward交易次数",
        ]
        selection_coverage.loc[walk_forward_valid] = combined.loc[
            walk_forward_valid,
            "WalkForward信号覆盖率",
        ]
        selection_drawdown.loc[walk_forward_valid] = combined.loc[
            walk_forward_valid,
            "WalkForward最大回撤",
        ]
        for source in (
            selection_win_rate_source,
            selection_trade_source,
            selection_coverage_source,
            selection_drawdown_source,
        ):
            source.loc[walk_forward_valid] = "WalkForward样本外"
    combined["入库胜率值"] = selection_win_rate
    combined["入库交易次数值"] = selection_trades
    combined["入库信号覆盖率值"] = selection_coverage
    combined["入库回撤值"] = selection_drawdown
    combined["入库胜率口径"] = selection_win_rate_source
    combined["入库交易次数口径"] = selection_trade_source
    combined["入库信号覆盖率口径"] = selection_coverage_source
    combined["入库回撤口径"] = selection_drawdown_source

    eligible_mask = (
        combined["初筛有效"].fillna(False)
        & combined["因子"].isin(available)
        & ~combined["因子"].isin(manual_excluded)
        & combined["最终测试入库表现通过"].fillna(False)
        & (combined["初筛夏普"] >= config.factor_library_min_sharpe)
        & (combined["初筛累计收益"] >= config.factor_library_min_total_return)
        & (combined["训练夏普比率"] >= min_train_sharpe)
        & (combined["训练累计收益"] >= min_train_total_return)
    )
    walk_forward_gate = ~walk_forward_expected | (
        walk_forward_valid
        & combined["WalkForward有效折数"].fillna(0.0).ge(min_walk_forward_folds)
        & combined["WalkForward盈利折占比"]
        .fillna(-np.inf)
        .ge(min_walk_forward_positive_share)
        & combined["WalkForward方向一致率"]
        .fillna(-np.inf)
        .ge(min_walk_forward_direction_consistency)
        & combined["WalkForward夏普中位数"]
        .fillna(-np.inf)
        .ge(min_walk_forward_median_sharpe)
        & combined["WalkForward交易次数"].fillna(0.0).ge(min_walk_forward_trades)
    )
    eligible_mask &= walk_forward_gate
    if min_train_win_rate is not None:
        eligible_mask &= combined["训练胜率"] > min_train_win_rate
    if min_selection_win_rate is not None:
        eligible_mask &= combined["入库胜率值"] > min_selection_win_rate
    if min_selection_rank_ic is not None:
        eligible_mask &= combined["初筛RankIC"].fillna(-np.inf) >= float(min_selection_rank_ic)
    if min_selection_monotonicity is not None:
        eligible_mask &= combined["初筛分组单调性"].fillna(-np.inf) >= float(min_selection_monotonicity)
    if min_research_score is not None:
        eligible_mask &= combined["初筛科研综合评分"].fillna(-np.inf) + threshold_epsilon >= float(min_research_score)
    if min_predictive_score is not None:
        eligible_mask &= combined["初筛预测能力评分"].fillna(-np.inf) + threshold_epsilon >= float(min_predictive_score)
    if min_selection_trades > 0:
        eligible_mask &= (
            combined["入库交易次数值"].fillna(0.0) >= min_selection_trades
        )
    if min_train_trades > 0:
        eligible_mask &= combined["训练交易次数"].fillna(0.0) >= min_train_trades
    if min_selection_signal_coverage > 0:
        eligible_mask &= (
            combined["入库信号覆盖率值"].fillna(0.0)
            >= min_selection_signal_coverage
        )
    if min_train_signal_coverage > 0:
        eligible_mask &= combined["训练信号覆盖率"].fillna(0.0) >= min_train_signal_coverage
    if max_selection_drawdown is not None:
        eligible_mask &= (
            combined["入库回撤值"].fillna(-np.inf)
            >= float(max_selection_drawdown)
        )
    if max_train_drawdown is not None:
        eligible_mask &= combined["训练最大回撤"].fillna(-np.inf) >= float(max_train_drawdown)
    eligible_factors = combined.loc[eligible_mask, "因子"].tolist()
    protected_available = [
        factor_name
        for factor_name in combined["因子"].astype(str)
        if factor_name in protected_names and factor_name in available
    ]
    eligible_factors = list(dict.fromkeys([*protected_available, *eligible_factors]))

    correlation_factors = slice_factor_selection_covariates(factors, config)
    correlation_scope = str(
        getattr(config, "factor_selection_covariate_scope", "train_validation")
        or "train_validation"
    ).strip().lower()
    correlation_cutoff = (
        correlation_factors.index[-1] if not correlation_factors.empty else ""
    )
    value_corr = pd.DataFrame()
    signal_corr = pd.DataFrame()
    if eligible_factors and config.factor_library_use_value_corr:
        value_corr = (
            correlation_factors[eligible_factors]
            .replace([np.inf, -np.inf], np.nan)
            .corr()
            .abs()
        )
    if eligible_factors and config.factor_library_use_signal_corr:
        signal_corr = (
            build_signal_corr_frame(correlation_factors, eligible_factors, config)
            .corr()
            .abs()
        )

    selected: list[str] = []
    selected_set: set[str] = set()
    require_manual_approval = bool(
        getattr(config, "factor_library_require_manual_approval", False)
    )
    approvals = load_manual_factor_approvals(
        config,
        bootstrap_existing_active=require_manual_approval,
    )
    approved_names = set(approvals["因子"].astype(str))
    approval_reason_map = approvals.set_index("因子")["人工审批原因"].to_dict()
    approval_time_map = approvals.set_index("因子")["人工审批时间"].to_dict()
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

        if factor_name in manual_excluded:
            status = "retired"
            reject_reason = "manual_exclusion"
        elif factor_name not in available:
            reject_reason = "factor_not_available"
        elif factor_name in protected_names:
            status = "active"
            selected.append(factor_name)
            selected_set.add(factor_name)
            selected_family_counts[family] = family_count + 1
        elif bool(row.get("WalkForward要求生效", False)) and str(
            row.get("WalkForward状态", "")
        ) != "已计算":
            reject_reason = "walk_forward_unavailable"
        elif bool(row.get("WalkForward要求生效", False)) and get_numeric_value(
            row,
            "WalkForward有效折数",
            0.0,
        ) < min_walk_forward_folds:
            reject_reason = "low_walk_forward_fold_count"
        elif bool(row.get("WalkForward要求生效", False)) and get_numeric_value(
            row,
            "WalkForward盈利折占比",
            -np.inf,
        ) < min_walk_forward_positive_share:
            reject_reason = "low_walk_forward_positive_fold_ratio"
        elif bool(row.get("WalkForward要求生效", False)) and get_numeric_value(
            row,
            "WalkForward方向一致率",
            -np.inf,
        ) < min_walk_forward_direction_consistency:
            reject_reason = "low_walk_forward_direction_consistency"
        elif bool(row.get("WalkForward要求生效", False)) and get_numeric_value(
            row,
            "WalkForward夏普中位数",
            -np.inf,
        ) < min_walk_forward_median_sharpe:
            reject_reason = "low_walk_forward_median_sharpe"
        elif bool(row.get("WalkForward要求生效", False)) and get_numeric_value(
            row,
            "WalkForward交易次数",
            0.0,
        ) < min_walk_forward_trades:
            reject_reason = "low_walk_forward_trade_count"
        elif not bool(row.get("入库数据可追溯", False)):
            reject_reason = "missing_traceable_train_metrics"
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
        elif (
            min_train_win_rate is not None
            and get_numeric_value(row, "训练胜率", -np.inf) <= min_train_win_rate
        ):
            reject_reason = "low_train_win_rate"
        elif (
            min_selection_win_rate is not None
            and
            get_numeric_value(row, "入库胜率值", -np.inf)
            <= min_selection_win_rate
        ):
            reject_reason = "low_selection_win_rate"
        elif min_selection_rank_ic is not None and get_numeric_value(row, "初筛RankIC", -np.inf) < float(min_selection_rank_ic):
            reject_reason = "low_selection_rank_ic"
        elif min_selection_monotonicity is not None and get_numeric_value(row, "初筛分组单调性", -np.inf) < float(min_selection_monotonicity):
            reject_reason = "low_selection_monotonicity"
        elif min_research_score is not None and get_numeric_value(row, "初筛科研综合评分", -np.inf) + threshold_epsilon < float(min_research_score):
            reject_reason = "low_research_score"
        elif min_predictive_score is not None and get_numeric_value(row, "初筛预测能力评分", -np.inf) + threshold_epsilon < float(min_predictive_score):
            reject_reason = "low_predictive_score"
        elif (
            get_numeric_value(row, "入库交易次数值", 0.0)
            < min_selection_trades
        ):
            reject_reason = "low_selection_trade_count"
        elif get_numeric_value(row, "训练交易次数", 0.0) < min_train_trades:
            reject_reason = "low_train_trade_count"
        elif (
            get_numeric_value(row, "入库信号覆盖率值", 0.0)
            < min_selection_signal_coverage
        ):
            reject_reason = "low_selection_signal_coverage"
        elif get_numeric_value(row, "训练信号覆盖率", 0.0) < min_train_signal_coverage:
            reject_reason = "low_train_signal_coverage"
        elif (
            max_selection_drawdown is not None
            and get_numeric_value(row, "入库回撤值", -np.inf)
            < float(max_selection_drawdown)
        ):
            reject_reason = "high_selection_drawdown"
        elif max_train_drawdown is not None and get_numeric_value(row, "训练最大回撤", -np.inf) < float(max_train_drawdown):
            reject_reason = "high_train_drawdown"
        elif str(row.get("最终测试入库拒绝原因", "")).strip():
            reject_reason = str(row["最终测试入库拒绝原因"])
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
                status = (
                    "active"
                    if not require_manual_approval or factor_name in approved_names
                    else "pre_active"
                )
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
        library_row["人工审批原因"] = approval_reason_map.get(factor_name, "")
        library_row["人工审批时间"] = approval_time_map.get(factor_name, "")
        library_row["是否手工保护"] = factor_name in protected_names
        library_row["手工保护原因"] = protection_reason_map.get(factor_name, "")
        library_row["手工保护时间"] = protection_time_map.get(factor_name, "")
        library_row["最大因子值相关性"] = max_value_corr
        library_row["最大信号相关性"] = max_signal_corr
        library_row["最大库内相关性"] = max_library_corr
        library_row["相关性样本范围"] = correlation_scope
        library_row["相关性样本数"] = len(correlation_factors)
        library_row["相关性样本截止"] = str(correlation_cutoff)
        library_rows.append(library_row)

    library_all = pd.DataFrame(library_rows)
    # active 逻辑审查是独立持久档案。每次重建因子库时重新附加仍与当前
    # 构造器指纹一致的结论，避免 active_factors.csv 的审查列被覆盖。
    from .factor_logic_review import load_factor_logic_reviews, merge_factor_logic_reviews

    library_all = merge_factor_logic_reviews(
        library_all,
        load_factor_logic_reviews(config),
    )
    active_library = library_all[library_all["因子库状态"].eq("active")].copy()
    rejected_library = library_all[
        ~library_all["因子库状态"].isin(["active", "pre_active"])
    ].copy()
    return active_library, library_all, rejected_library


def save_factor_library(
    active_library: pd.DataFrame,
    library_all: pd.DataFrame,
    rejected_library: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """保存因子库核心表。

    active_factors.csv：当前用于综合模型候选池的因子。
    pre_active_factors.csv：规则通过、等待人工确认的候选因子。
    factor_library_all.parquet/pkl.gz：全部因子的最新完整指标主库。
    rejected_factors.csv：被拒绝或退役因子的精简人工诊断视图。
    """
    library_dir = get_factor_library_dir(config)
    active_library.to_csv(
        library_dir / "active_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pre_active_library = (
        library_all[library_all["因子库状态"].eq("pre_active")].copy()
        if "因子库状态" in library_all.columns
        else library_all.iloc[0:0].copy()
    )
    pre_active_library.to_csv(
        library_dir / "pre_active_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    master_path = _write_factor_library_master(library_all, config)
    rejected_output = (
        build_compact_rejected_library(rejected_library)
        if bool(getattr(config, "factor_library_compact_rejected_output", True))
        else rejected_library
    )
    rejected_output.to_csv(
        library_dir / "rejected_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if "拒绝原因" in rejected_library.columns:
        rejected_library["拒绝原因"].fillna("未标注").value_counts(dropna=False).rename_axis(
            "拒绝原因"
        ).reset_index(name="因子数量").to_csv(
            library_dir / "rejected_reason_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"全量因子主库已保存: {master_path}")
    if "因子家族" in active_library.columns:
        family_summary = (
            active_library.groupby("因子家族", dropna=False)
            .agg(
                active因子数=("因子", "count"),
                平均初筛夏普=("初筛夏普", "mean"),
                平均初筛RankIC=("初筛RankIC", "mean"),
                平均分组单调性=("初筛分组单调性", "mean"),
                平均预测能力评分=("初筛预测能力评分", "mean"),
                平均科研综合评分=("初筛科研综合评分", "mean"),
            )
            .reset_index()
            .sort_values("active因子数", ascending=False)
        )
        family_summary.to_csv(
            library_dir / "active_factor_family_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

from __future__ import annotations

"""行情读取、数据清洗和因子生成模块。

这个文件是整个多因子框架的“数据入口 + 因子工厂”：
- 优先读取本地缓存行情，必要时从 Wind 拉取并缓存。
- 对行情字段做标准化、缺失值处理和周期校验。
- 构造基础因子与大批量参数化因子。
- 提供因子编号、因子标签、单因子测试范围选择等辅助函数。

注意：本模块只负责生成因子值，不负责判断因子方向、回测绩效或模型训练。
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_loader import (
    ensure_wind_started as data_ensure_wind_started,
    fetch_intraday_data as data_fetch_intraday_data,
    fetch_macro_state_data as data_fetch_macro_state_data,
    fetch_related_intraday_data as data_fetch_related_intraday_data,
    get_data_cache_path as data_get_data_cache_path,
    get_local_data_candidates as data_get_local_data_candidates,
    get_macro_data_cache_path as data_get_macro_data_cache_path,
    load_local_intraday_data as data_load_local_intraday_data,
    normalize_intraday_data as data_normalize_intraday_data,
    normalize_macro_daily_data as data_normalize_macro_daily_data,
    safe_symbol_name as data_safe_symbol_name,
    stop_wind as data_stop_wind,
)
from factor_builders import (
    add_basic_factors,
    add_calendar_seasonality_factors,
    add_complex_cross_asset_factors,
    add_complex_non_cross_factors,
    add_cross_asset_factors,
    add_hyper_cross_asset_factors,
    add_hyper_non_cross_factors,
    add_macro_state_factors,
    add_omega_cross_asset_factors,
    add_omega_non_cross_factors,
    add_parametric_factors,
)
from factor_builders.cross_asset import get_last_related_data_coverage as get_cross_asset_data_coverage


LAST_RELATED_DATA_COVERAGE = pd.DataFrame()
CALENDAR_FACTOR_START_INDEX = 106885
MACRO_FACTOR_START_INDEX = 126978


# 数据读取逻辑已经拆到 data_loader.py；这里保留同名入口，兼容旧脚本导入。
ensure_wind_started = data_ensure_wind_started
stop_wind = data_stop_wind
safe_symbol_name = data_safe_symbol_name
get_data_cache_path = data_get_data_cache_path
get_local_data_candidates = data_get_local_data_candidates
normalize_intraday_data = data_normalize_intraday_data
load_local_intraday_data = data_load_local_intraday_data
fetch_intraday_data = data_fetch_intraday_data
get_macro_data_cache_path = data_get_macro_data_cache_path
normalize_macro_daily_data = data_normalize_macro_daily_data
fetch_macro_state_data = data_fetch_macro_state_data
fetch_related_intraday_data = data_fetch_related_intraday_data

def align_related_data_to_main(
    related_data: pd.DataFrame,
    main_index: pd.Index,
    max_ffill_bars: int,
) -> pd.DataFrame:
    """把相关品种行情对齐到主标的时间轴，且不使用向后填充。"""
    aligned = related_data.reindex(main_index)
    if max_ffill_bars > 0:
        aligned = aligned.ffill(limit=max_ffill_bars)
    return aligned


def calculate_related_data_coverage(
    symbol: str,
    related_data: pd.DataFrame,
    main_index: pd.Index,
    max_ffill_bars: int,
) -> dict[str, Any]:
    """计算单个相关品种的数据覆盖率诊断。"""
    before_ffill = related_data.reindex(main_index)
    after_ffill = align_related_data_to_main(related_data, main_index, max_ffill_bars)
    before_valid = before_ffill["close"].notna() if "close" in before_ffill.columns else pd.Series(False, index=main_index)
    after_valid = after_ffill["close"].notna() if "close" in after_ffill.columns else pd.Series(False, index=main_index)
    ffill_added = after_valid & ~before_valid
    missing_after_ffill = ~after_valid

    if len(missing_after_ffill):
        missing_groups = missing_after_ffill.ne(missing_after_ffill.shift(fill_value=False)).cumsum()
        max_consecutive_missing = int(
            missing_after_ffill.groupby(missing_groups).sum().max()
        )
    else:
        max_consecutive_missing = 0

    clean_related_index = pd.DatetimeIndex(related_data.index).sort_values()
    main_datetime_index = pd.DatetimeIndex(main_index)
    if len(clean_related_index) and len(main_datetime_index):
        matched_positions = clean_related_index.searchsorted(main_datetime_index, side="right") - 1
        valid_match = matched_positions >= 0
        lag_minutes = pd.Series(np.nan, index=main_index, dtype="float64")
        if valid_match.any():
            matched_times = clean_related_index[matched_positions[valid_match]]
            lag_minutes.loc[valid_match] = (
                main_datetime_index[valid_match] - matched_times
            ).total_seconds() / 60.0
        usable_lag = lag_minutes.loc[after_valid]
    else:
        usable_lag = pd.Series(dtype="float64")

    return {
        "symbol": symbol,
        "raw_rows": int(len(related_data)),
        "raw_start": str(related_data.index.min()) if len(related_data) else "",
        "raw_end": str(related_data.index.max()) if len(related_data) else "",
        "raw_duplicate_timestamps": int(pd.Index(related_data.index).duplicated().sum()),
        "main_rows": int(len(main_index)),
        "direct_aligned_rows": int(before_valid.sum()),
        "ffill_added_rows": int(ffill_added.sum()),
        "usable_rows": int(after_valid.sum()),
        "missing_rows_after_ffill": int(missing_after_ffill.sum()),
        "max_consecutive_missing_after_ffill": max_consecutive_missing,
        "direct_coverage_rate": float(before_valid.mean()) if len(before_valid) else np.nan,
        "usable_coverage_rate": float(after_valid.mean()) if len(after_valid) else np.nan,
        "missing_rate_after_ffill": float(missing_after_ffill.mean()) if len(missing_after_ffill) else np.nan,
        "ffill_share_in_usable": float(ffill_added.sum() / after_valid.sum()) if after_valid.sum() else np.nan,
        "max_alignment_lag_minutes": float(usable_lag.max()) if not usable_lag.empty else np.nan,
        "avg_alignment_lag_minutes": float(usable_lag.mean()) if not usable_lag.empty else np.nan,
    }


def get_last_related_data_coverage() -> pd.DataFrame:
    """返回最近一次 build_factors 调用产生的数据覆盖率诊断。"""
    return get_cross_asset_data_coverage()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 zscore 标准化，并把极端值截断到 [-3, 3]。

    高频因子中异常值比较常见，截断可以降低极端点对模型和单因子信号的影响。
    """
    min_periods = max(20, window // 3)
    mean_ = series.rolling(window=window, min_periods=min_periods).mean()
    std_ = series.rolling(window=window, min_periods=min_periods).std()
    zscore = (series - mean_) / std_.replace(0, np.nan)
    return zscore.clip(-3, 3)


def score_to_raw_signal(score: pd.Series, threshold: float) -> pd.Series:
    """把连续因子分数转换成 -1/0/1 原始交易信号。

    score > threshold 生成 1；
    score < -threshold 生成 -1；
    其余区间生成 0。
    """
    raw_signal = np.where(
        score > threshold,
        1.0,
        np.where(score < -threshold, -1.0, 0.0),
    )
    return pd.Series(raw_signal, index=score.index, dtype="float64")


def get_factor_columns(factors: pd.DataFrame) -> list[str]:
    """返回因子列名列表，保持 DataFrame 当前列顺序。"""
    return list(factors.columns)


def get_factor_id_map(factors: pd.DataFrame) -> dict[str, int]:
    """为每个因子分配从 1 开始的稳定编号。

    编号跟随当前 factors 的列顺序，输出报表里用它来快速定位因子。
    """
    return {
        factor_name: factor_id
        for factor_id, factor_name in enumerate(get_factor_columns(factors), start=1)
    }


def format_factor_label(factor_name: str, factor_id: int) -> str:
    """生成带编号的因子展示名，例如 127_return_volume_beta_3。"""
    return f"{factor_id}_{factor_name}"


def get_factor_label_map(factors: pd.DataFrame) -> dict[str, str]:
    """返回 {因子列名: 带编号因子标签} 的映射。"""
    factor_id_map = get_factor_id_map(factors)
    return {
        factor_name: format_factor_label(factor_name, factor_id_map[factor_name])
        for factor_name in get_factor_columns(factors)
    }


def normalize_factor_symbol_key(symbol: Any) -> str:
    """把期货品种代码归一化成稳定 key，便于匹配每个品种自己的增量因子起点。"""
    return str(symbol).strip().upper().replace(".", "_").replace("/", "_").replace("-", "_")


def get_single_factor_new_factor_start_index(config: Any) -> int:
    """返回当前品种实际使用的新因子起始编号。"""
    default_start_index = max(1, int(getattr(config, "single_factor_new_factor_start_index", 1)))
    symbol_start_index_map = getattr(config, "single_factor_new_factor_start_index_by_symbol", {}) or {}
    if not symbol_start_index_map:
        return default_start_index

    current_symbol_key = normalize_factor_symbol_key(getattr(config, "symbol", ""))
    for symbol_key, start_index in symbol_start_index_map.items():
        if normalize_factor_symbol_key(symbol_key) == current_symbol_key:
            return max(1, int(start_index))
    return default_start_index


def get_factor_prune_list_path(config: Any) -> Path:
    """返回全局因子淘汰清单路径。"""
    prune_path = Path(getattr(config, "factor_prune_list_path", "factor_prune_list.csv"))
    if prune_path.is_absolute():
        return prune_path
    return Path(getattr(config, "output_dir", ".")) / prune_path


def load_pruned_factor_names(config: Any) -> set[str]:
    """读取需要从候选矩阵中排除的因子名称。"""
    if not getattr(config, "enable_factor_pruning", False):
        return set()
    if not getattr(config, "factor_pruning_apply_to_build", True):
        return set()

    prune_path = get_factor_prune_list_path(config)
    if not prune_path.exists():
        return set()
    try:
        prune_table = pd.read_csv(prune_path)
    except Exception as exc:
        print(f"因子淘汰清单读取失败，暂不应用: {prune_path}, 原因: {exc}")
        return set()

    factor_column = "因子" if "因子" in prune_table.columns else "factor"
    if factor_column not in prune_table.columns:
        return set()
    return set(prune_table[factor_column].dropna().astype(str))


def select_single_factor_columns(factors: pd.DataFrame, config: Any) -> list[str]:
    """根据配置选择本轮需要做单因子回测的因子。

    支持三种模式：
    - all：测试全部因子。
    - new：从指定编号之后开始测试，适合增量测试 AI 新生成的因子。
    - selected：只测试手工指定的因子列表。
    """
    factor_columns = get_factor_columns(factors)
    scope = config.single_factor_scope.lower()

    if scope == "all":
        return factor_columns

    if scope == "new":
        start_factor_id = get_single_factor_new_factor_start_index(config)
        # 因子编号从 1 开始，Python 列表切片从 0 开始；这里选择编号 >= start_factor_id 的因子。
        return factor_columns[start_factor_id - 1 :]

    if scope == "selected":
        selected = config.single_factor_selected_factors or []
        if not selected:
            raise ValueError("single_factor_scope='selected' 时必须配置 single_factor_selected_factors。")
        missing = sorted(set(selected).difference(factor_columns))
        if missing:
            raise ValueError(f"single_factor_selected_factors 中存在未知因子: {missing}")
        return selected

    raise ValueError("single_factor_scope 只能是 'all'、'new' 或 'selected'。")


def make_factor_catalog_dummy_data(data: pd.DataFrame, rows: int = 4) -> pd.DataFrame:
    """构造极小的哑数据，用于快速生成因子名称目录而不是计算完整历史矩阵。"""
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index):
        start = data.index[0]
    else:
        start = pd.Timestamp("2025-01-01 09:00:00")
    index = pd.date_range(start=start, periods=rows, freq="30min")
    base = np.arange(rows, dtype="float64")
    return pd.DataFrame(
        {
            "open": 100.0 + base * 0.01,
            "high": 101.0 + base * 0.01,
            "low": 99.0 + base * 0.01,
            "close": 100.2 + base * 0.01,
            "volume": 1000.0 + base,
            "amt": 100000.0 + base * 100.0,
            "amount": 100000.0 + base * 100.0,
        },
        index=index,
    )


def make_dummy_related_data_map(config: Any, dummy_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """为因子名称目录构造跨品种哑数据，避免为了拿列名去读取真实 Wind/CSV。"""
    related_data_map: dict[str, pd.DataFrame] = {}
    main_symbol = str(getattr(config, "symbol", "")).upper()
    for symbol in getattr(config, "related_symbols", []) or []:
        symbol = str(symbol).strip()
        if not symbol or symbol.upper() == main_symbol:
            continue
        related_data_map[symbol] = dummy_data.copy()
    return related_data_map


def make_dummy_macro_data_map(config: Any, dummy_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """为因子名称目录构造宏观哑数据，避免为了拿列名触发 Wind。"""
    macro_data_map: dict[str, pd.DataFrame] = {}
    daily_index = pd.date_range(dummy_data.index.min().normalize(), periods=8, freq="D")
    values = 100.0 + np.arange(len(daily_index), dtype="float64") * 0.1
    for symbol in getattr(config, "macro_state_symbols", []) or []:
        symbol = str(symbol).strip()
        if symbol:
            macro_data_map[symbol] = pd.DataFrame({"close": values}, index=daily_index)
    return macro_data_map


def build_factor_name_catalog(data: pd.DataFrame, config: Any) -> list[str]:
    """用极小样本生成完整因子名称目录，供 single_factor_scope='new' 推导新增因子。"""
    dummy_data = make_factor_catalog_dummy_data(data)
    try:
        catalog_config = replace(config, factor_pruning_apply_to_build=False)
    except TypeError:
        catalog_config = config
    related_dummy_map = (
        make_dummy_related_data_map(catalog_config, dummy_data)
        if getattr(catalog_config, "enable_cross_asset_factors", False)
        else None
    )
    macro_dummy_map = (
        make_dummy_macro_data_map(catalog_config, dummy_data)
        if getattr(catalog_config, "enable_macro_state_factors", False)
        else None
    )
    catalog = build_factors(
        dummy_data,
        catalog_config,
        related_data_map=related_dummy_map,
        macro_data_map=macro_dummy_map,
        requested_factors=None,
    )
    return list(catalog.columns)


def build_calendar_factor_name_catalog(data: pd.DataFrame, config: Any) -> list[str]:
    """只生成 calendar_ 因子名称目录，用于新增日历因子的快速增量测试。"""
    dummy_data = make_factor_catalog_dummy_data(data)
    calendar_factors = add_calendar_seasonality_factors(dummy_data, config)
    return list(calendar_factors.columns)


def build_macro_factor_name_catalog(data: pd.DataFrame, config: Any) -> list[str]:
    """只生成 macro_ 因子名称目录，避免增量测试宏观因子时触发 Wind 或全量因子构建。"""
    dummy_data = make_factor_catalog_dummy_data(data)
    macro_dummy_map = make_dummy_macro_data_map(config, dummy_data)
    macro_factors = add_macro_state_factors(dummy_data, macro_dummy_map, config)
    return list(macro_factors.columns)


def resolve_single_factor_requested_factors(
    data: pd.DataFrame,
    config: Any,
) -> tuple[list[str] | None, dict[str, int] | None]:
    """根据单因子测试范围推导真实需要构建的因子名。"""
    scope = str(getattr(config, "single_factor_scope", "all")).lower()
    if scope == "all":
        return None, None
    if scope == "selected":
        selected = list(getattr(config, "single_factor_selected_factors", []) or [])
        return selected, None
    if scope != "new":
        return None, None

    start_factor_id = get_single_factor_new_factor_start_index(config)
    if start_factor_id >= MACRO_FACTOR_START_INDEX:
        macro_columns = build_macro_factor_name_catalog(data, config)
        offset = start_factor_id - MACRO_FACTOR_START_INDEX
        requested = macro_columns[offset:]
        pruned_names = load_pruned_factor_names(config)
        if pruned_names:
            requested = [factor_name for factor_name in requested if factor_name not in pruned_names]
        factor_id_map = {
            factor_name: MACRO_FACTOR_START_INDEX + factor_offset
            for factor_offset, factor_name in enumerate(macro_columns)
        }
        return requested, factor_id_map

    if start_factor_id >= CALENDAR_FACTOR_START_INDEX:
        calendar_columns = build_calendar_factor_name_catalog(data, config)
        offset = start_factor_id - CALENDAR_FACTOR_START_INDEX
        requested = calendar_columns[offset:]
        pruned_names = load_pruned_factor_names(config)
        if pruned_names:
            requested = [factor_name for factor_name in requested if factor_name not in pruned_names]
        factor_id_map = {
            factor_name: CALENDAR_FACTOR_START_INDEX + factor_offset
            for factor_offset, factor_name in enumerate(calendar_columns)
        }
        return requested, factor_id_map

    catalog_columns = build_factor_name_catalog(data, config)
    requested = catalog_columns[start_factor_id - 1 :]
    pruned_names = load_pruned_factor_names(config)
    if pruned_names:
        requested = [factor_name for factor_name in requested if factor_name not in pruned_names]
    factor_id_map = {
        factor_name: factor_id
        for factor_id, factor_name in enumerate(catalog_columns, start=1)
    }
    return requested, factor_id_map


def build_single_factor_matrix(data: pd.DataFrame, config: Any) -> pd.DataFrame:
    """为单因子回测构建因子矩阵，new/selected 模式下只计算本轮需要测试的因子。"""
    requested_factors, catalog_factor_id_map = resolve_single_factor_requested_factors(data, config)
    if requested_factors is None:
        factors = build_factors(data, config)
    else:
        print(f"单因子按需构建因子数量: {len(requested_factors)}")
        if not requested_factors:
            factors = pd.DataFrame(index=data.index)
        else:
            factors = build_factors(data, config, requested_factors=requested_factors)

    if catalog_factor_id_map:
        scoped_factor_id_map = {
            factor_name: catalog_factor_id_map[factor_name]
            for factor_name in factors.columns
            if factor_name in catalog_factor_id_map
        }
        factors.attrs["factor_id_map"] = scoped_factor_id_map
        factors.attrs["factor_label_map"] = {
            factor_name: format_factor_label(factor_name, factor_id)
            for factor_name, factor_id in scoped_factor_id_map.items()
        }
    return factors


def build_factors(
    data: pd.DataFrame,
    config: Any,
    related_data_map: dict[str, pd.DataFrame] | None = None,
    macro_data_map: dict[str, pd.DataFrame] | None = None,
    requested_factors: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """构建完整因子矩阵。

    输出包含：
    - 4 个基础因子：momentum、reversal、breakout、volume_confirm。
    - add_parametric_factors 生成的大量参数化因子。

    本函数也会补充 bar_return_cc / bar_return_oc 等内部计算字段，
    但最终只返回因子矩阵，不返回行情字段。
    """
    requested_set = set(requested_factors or [])
    pruned_factor_names = load_pruned_factor_names(config)
    if requested_set:
        requested_set = requested_set.difference(pruned_factor_names)
    if pruned_factor_names and not requested_factors:
        print(f"已应用因子淘汰清单，排除因子数量: {len(pruned_factor_names)}")

    def need_any_factor() -> bool:
        return not requested_set

    def filter_requested(frame: pd.DataFrame) -> pd.DataFrame:
        if pruned_factor_names:
            keep_not_pruned = [column for column in frame.columns if column not in pruned_factor_names]
            frame = frame[keep_not_pruned] if keep_not_pruned else pd.DataFrame(index=frame.index)
        if not requested_set:
            return frame
        keep_columns = [column for column in frame.columns if column in requested_set]
        return frame[keep_columns] if keep_columns else pd.DataFrame(index=frame.index)

    def has_requested_prefix(prefixes: tuple[str, ...]) -> bool:
        return any(factor_name.startswith(prefixes) for factor_name in requested_set)

    df = data.copy()

    df["bar_return_cc"] = df["close"].pct_change()
    df["bar_return_oc"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    df["volume"] = df.get("volume", pd.Series(0.0, index=df.index)).fillna(0.0)

    factor_parts = []
    base_factors = filter_requested(add_basic_factors(df, config))
    if not base_factors.empty:
        factor_parts.append(base_factors)

    parametric_prefixes = tuple(
        factor_name
        for factor_name in requested_set
        if factor_name not in {"momentum", "reversal", "breakout", "volume_confirm"}
        and not factor_name.startswith(
            (
                "calendar_",
                "macro_",
                "cross_",
                "crossmega_",
                "crossultra_",
                "crosshyper_",
                "crossomega_",
                "ultra_",
                "hyper_",
                "omega_",
            )
        )
    )
    if need_any_factor() or parametric_prefixes:
        parametric_factors = filter_requested(add_parametric_factors(df, config))
        if not parametric_factors.empty:
            factor_parts.append(parametric_factors)

    if getattr(config, "enable_cross_asset_factors", False):
        if related_data_map is None:
            related_data_map = fetch_related_intraday_data(config)
        cross_asset_factors = pd.DataFrame(index=df.index)
        if need_any_factor() or has_requested_prefix(("cross_", "crossmega_")):
            cross_asset_factors = filter_requested(add_cross_asset_factors(df, related_data_map, config))
        if not cross_asset_factors.empty:
            print(f"跨品种因子数量: {cross_asset_factors.shape[1]}")
            factor_parts.append(cross_asset_factors)

    complex_non_cross_factors = pd.DataFrame(index=df.index)
    if need_any_factor() or has_requested_prefix(("ultra_",)):
        complex_non_cross_factors = filter_requested(add_complex_non_cross_factors(df, config))
    if not complex_non_cross_factors.empty:
        factor_parts.append(complex_non_cross_factors)

    if getattr(config, "enable_cross_asset_factors", False):
        complex_cross_asset_factors = pd.DataFrame(index=df.index)
        if need_any_factor() or has_requested_prefix(("crossultra_",)):
            complex_cross_asset_factors = filter_requested(
                add_complex_cross_asset_factors(df, related_data_map or {}, config)
            )
        if not complex_cross_asset_factors.empty:
            print(f"复杂跨品种因子数量: {complex_cross_asset_factors.shape[1]}")
            factor_parts.append(complex_cross_asset_factors)

    hyper_non_cross_factors = pd.DataFrame(index=df.index)
    if need_any_factor() or has_requested_prefix(("hyper_",)):
        hyper_non_cross_factors = filter_requested(add_hyper_non_cross_factors(df, config))
    if not hyper_non_cross_factors.empty:
        factor_parts.append(hyper_non_cross_factors)

    if getattr(config, "enable_cross_asset_factors", False):
        hyper_cross_asset_factors = pd.DataFrame(index=df.index)
        if need_any_factor() or has_requested_prefix(("crosshyper_",)):
            hyper_cross_asset_factors = filter_requested(
                add_hyper_cross_asset_factors(df, related_data_map or {}, config)
            )
        if not hyper_cross_asset_factors.empty:
            print(f"高阶跨品种因子数量: {hyper_cross_asset_factors.shape[1]}")
            factor_parts.append(hyper_cross_asset_factors)

    omega_non_cross_factors = pd.DataFrame(index=df.index)
    if need_any_factor() or has_requested_prefix(("omega_",)):
        omega_non_cross_factors = filter_requested(add_omega_non_cross_factors(df, config))
    if not omega_non_cross_factors.empty:
        factor_parts.append(omega_non_cross_factors)

    if getattr(config, "enable_cross_asset_factors", False):
        omega_cross_asset_factors = pd.DataFrame(index=df.index)
        if need_any_factor() or has_requested_prefix(("crossomega_",)):
            omega_cross_asset_factors = filter_requested(
                add_omega_cross_asset_factors(df, related_data_map or {}, config)
            )
        if not omega_cross_asset_factors.empty:
            print(f"终极跨品种因子数量: {omega_cross_asset_factors.shape[1]}")
            factor_parts.append(omega_cross_asset_factors)

    calendar_factors = pd.DataFrame(index=df.index)
    if need_any_factor() or has_requested_prefix(("calendar_",)):
        calendar_factors = filter_requested(add_calendar_seasonality_factors(df, config))
    if not calendar_factors.empty:
        factor_parts.append(calendar_factors)

    if getattr(config, "enable_macro_state_factors", False):
        if macro_data_map is None:
            macro_data_map = fetch_macro_state_data(config)
        macro_factors = pd.DataFrame(index=df.index)
        if need_any_factor() or has_requested_prefix(("macro_",)):
            macro_factors = filter_requested(add_macro_state_factors(df, macro_data_map, config))
        if not macro_factors.empty:
            print(f"宏观状态因子数量: {macro_factors.shape[1]}")
            factor_parts.append(macro_factors)

    if not factor_parts:
        missing_preview = ", ".join(sorted(requested_set)[:10])
        raise ValueError(f"请求的因子当前都无法生成: {missing_preview}")
    return pd.concat(factor_parts, axis=1).copy()



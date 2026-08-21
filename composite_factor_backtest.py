from __future__ import annotations

"""综合因子 / XGBoost 滚动训练回测模块。

本文件把单因子信号合成为一个可交易的综合信号：
- 支持 selected / all / best 三种因子输入范围。
- 支持 signal / continuous / both 三种特征形式。
- 使用覆盖最小持仓期、成本与波动缓冲后的未来收益方向作为三分类目标。
- 采用滚动窗口训练和滚动预测，尽量模拟真实上线时只能使用历史数据的状态。
- 同时输出 XGBoost 策略和等权投票基准策略，方便判断模型是否真的优于简单合成。
"""

from pathlib import Path
from typing import Any
from dataclasses import replace
import datetime as dt
import hashlib
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import BacktestConfig, report_config_validation
from framework.experiment_utils import (
    copy_existing_files,
    get_experiment_run_dir,
    write_factor_count_snapshot,
    write_run_config,
)
from framework.runtime_utils import (
    build_config_hash,
    copy_file_atomic,
    run_tracked,
    write_json_atomic,
)
from framework.project_fingerprint import build_source_fingerprint_hash
from framework.output_layout import (
    apply_frequency_runtime_defaults,
    get_frequency_key,
    get_research_output_dir,
)
from framework.factor_library import get_factor_library_dir
from framework.factors import (
    build_factors,
    fetch_intraday_data,
    get_factor_columns,
    get_factor_id_map,
    get_factor_label_map,
    get_last_related_data_coverage,
    score_to_raw_signal,
    stop_wind,
)
from single_factor_backtest import (
    build_signal_from_score,
    calculate_metrics,
    calculate_wilson_interval,
    infer_annual_periods,
    plot_backtest_result,
    print_metrics,
    run_backtest,
    safe_run_backtest,
    choose_factor_direction,
    split_train_validation_test_index,
)


COMPOSITE_ARTIFACT_SCHEMA_VERSION = 1


def build_file_identity(path: Path) -> dict[str, Any]:
    """记录产物或输入文件的内容身份。"""
    path = path.resolve()
    identity: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists() or not path.is_file():
        return identity
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    identity.update(
        {
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        }
    )
    return identity


def get_composite_artifact_manifest_path(output_dir: Path | str) -> Path:
    """返回综合模型 latest 产物清单路径。"""
    return Path(output_dir) / "composite_artifact_manifest.json"


def write_composite_artifact_manifest(
    config: BacktestConfig,
    output_dir: Path,
    model_name: str,
    split_time: pd.Timestamp,
    validation_end_time: pd.Timestamp,
) -> Path:
    """写入综合回测产物身份，供组合层和实盘层校验。"""
    detail_path = output_dir / "composite_detail.csv"
    summary_path = output_dir / "composite_summary.csv"
    active_path = get_active_factor_library_path(config)
    detail = pd.read_csv(detail_path, index_col=0, parse_dates=True)
    manifest = {
        "schema_version": COMPOSITE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "composite_backtest",
        "status": "complete",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": str(getattr(config, "run_id", "") or ""),
        "symbol": str(config.symbol),
        "model_name": str(model_name),
        "config_sha256": build_config_hash(config),
        "source_sha256": build_source_fingerprint_hash(
            [
                Path("config.py"),
                Path("framework/data_loader.py"),
                Path("framework/factors.py"),
                Path("framework/factor_library.py"),
                Path("single_factor_backtest.py"),
                Path("composite_factor_backtest.py"),
                *sorted(Path("framework/factor_builders").glob("*.py")),
            ]
        ),
        "split_time": str(split_time),
        "validation_end_time": str(validation_end_time),
        "detail_rows": int(len(detail)),
        "detail_start": str(detail.index.min()) if not detail.empty else "",
        "detail_end": str(detail.index.max()) if not detail.empty else "",
        "inputs": {
            "active_factor_library": build_file_identity(active_path),
            "active_library_oos_audit": build_file_identity(
                output_dir / "active_library_oos_audit.json"
            ),
        },
        "outputs": {
            "composite_detail": build_file_identity(detail_path),
            "composite_summary": build_file_identity(summary_path),
        },
    }
    path = get_composite_artifact_manifest_path(output_dir)
    return write_json_atomic(path, manifest)


def load_composite_artifact_manifest(
    output_dir: Path | str,
    expected_symbol: str | None = None,
    verify_outputs: bool = True,
    verify_inputs: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """读取并验证综合产物清单，返回清单和失败原因。"""
    path = get_composite_artifact_manifest_path(output_dir)
    if not path.exists():
        return None, "缺少 composite_artifact_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"产物清单无法解析: {exc}"
    if int(manifest.get("schema_version", 0) or 0) != COMPOSITE_ARTIFACT_SCHEMA_VERSION:
        return None, "产物清单版本不兼容"
    if manifest.get("artifact_type") != "composite_backtest":
        return None, "产物类型不是 composite_backtest"
    if manifest.get("status") != "complete":
        return None, "综合产物状态不是 complete"
    if expected_symbol and str(manifest.get("symbol", "")).upper() != str(expected_symbol).upper():
        return None, "产物品种与组合品种不一致"
    if verify_inputs:
        saved_inputs = manifest.get("inputs", {})
        if not isinstance(saved_inputs, dict):
            return None, "产物清单 inputs 字段无效"
        saved_active = saved_inputs.get("active_factor_library")
        if not isinstance(saved_active, dict):
            return None, "产物清单缺少 active 因子库身份"
        active_path = Path(str(saved_active.get("path", "")))
        current_active = build_file_identity(active_path)
        if not current_active.get("exists"):
            return None, "综合回测使用的 active 因子库快照已缺失"
        if current_active.get("sha256") != saved_active.get("sha256"):
            return None, "综合回测使用的 active 因子库快照已变化"
        saved_audit = saved_inputs.get("active_library_oos_audit")
        if isinstance(saved_audit, dict) and saved_audit.get("exists"):
            audit_path = Path(str(saved_audit.get("path", "")))
            current_audit = build_file_identity(audit_path)
            if not current_audit.get("exists"):
                return None, "active 因子库样本外截止审计文件已缺失"
            if current_audit.get("sha256") != saved_audit.get("sha256"):
                return None, "active 因子库样本外截止审计文件已变化"
    if verify_outputs:
        expected_outputs = {
            "composite_detail": Path(output_dir) / "composite_detail.csv",
            "composite_summary": Path(output_dir) / "composite_summary.csv",
        }
        saved_outputs = manifest.get("outputs", {})
        if not isinstance(saved_outputs, dict):
            return None, "产物清单 outputs 字段无效"
        for name, output_path in expected_outputs.items():
            saved_identity = saved_outputs.get(name)
            if not isinstance(saved_identity, dict):
                return None, f"产物清单缺少必需文件身份: {name}"
            current_identity = build_file_identity(output_path)
            if not current_identity.get("exists"):
                return None, f"产物文件缺失: {name}"
            if current_identity.get("sha256") != saved_identity.get("sha256"):
                return None, f"产物内容已变化: {name}"
    return manifest, ""


TARGET_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}
CLASS_TO_TARGET = {class_id: target for target, class_id in TARGET_TO_CLASS.items()}


def get_active_factor_library_path(config: BacktestConfig) -> Path:
    """返回综合模型使用的 active 因子库路径。"""
    if getattr(config, "use_frozen_active_library", False):
        frozen_path = getattr(config, "frozen_active_library_path", None)
        if not frozen_path:
            raise ValueError("use_frozen_active_library=True 时必须配置 frozen_active_library_path。")
        active_path = Path(frozen_path)
        if not active_path.is_absolute():
            active_path = Path(config.output_dir) / active_path
        return active_path
    return get_factor_library_dir(config) / "active_factors.csv"


def freeze_active_factor_library_for_run(
    config: BacktestConfig,
    output_dir: Path,
    run_dir: Path | None,
) -> tuple[BacktestConfig, Path]:
    """在读取因子名之前冻结 active 库，并返回只引用该快照的运行配置。"""
    source_path = get_active_factor_library_path(config).resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"没有找到 active 因子库: {source_path}。请先运行 single_factor_backtest.py 更新因子库。"
        )
    if not bool(getattr(config, "composite_auto_freeze_active_library", True)):
        return config, source_path

    snapshot_dir = run_dir if run_dir is not None else output_dir
    snapshot_path = (snapshot_dir / "active_factors_snapshot.csv").resolve()
    copy_file_atomic(source_path, snapshot_path)
    active_library = pd.read_csv(snapshot_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {source_path}")

    def unique_values(column: str) -> list[str]:
        if column not in active_library.columns:
            return []
        return sorted(
            {
                str(value)
                for value in active_library[column].dropna()
                if str(value).strip()
            }
        )

    snapshot_manifest = {
        "schema_version": 1,
        "artifact_type": "active_factor_library_snapshot",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": str(getattr(config, "run_id", "") or ""),
        "symbol": str(config.symbol),
        "source": build_file_identity(source_path),
        "snapshot": build_file_identity(snapshot_path),
        "factor_count": int(active_library["因子"].notna().sum()),
        "training_cutoffs": (
            unique_values("因子筛选训练截止") or unique_values("训练截止")
        ),
        "validation_cutoffs": (
            unique_values("因子筛选验证截止") or unique_values("验证截止")
        ),
    }
    write_json_atomic(
        snapshot_dir / "active_factors_snapshot_manifest.json",
        snapshot_manifest,
    )
    runtime_config = replace(
        config,
        use_frozen_active_library=True,
        frozen_active_library_path=str(snapshot_path),
    )
    print(
        f"本轮 active 因子库已冻结: {snapshot_path}；"
        f"因子数量={snapshot_manifest['factor_count']}"
    )
    return runtime_config, snapshot_path


def audit_active_library_oos_cutoff(
    config: BacktestConfig,
    final_test_start: pd.Timestamp,
    output_dir: Path,
    run_dir: Path | None,
) -> dict[str, Any]:
    """确认 active 因子筛选使用的数据没有越过本次最终测试起点。"""
    policy = str(
        getattr(config, "composite_active_library_cutoff_policy", "auto")
    ).lower()
    active_path = get_active_factor_library_path(config).resolve()
    active_library = pd.read_csv(active_path)
    cutoff_column = ""
    for candidate in (
        "因子筛选验证截止",
        "验证截止",
        "因子筛选训练截止",
        "训练截止",
    ):
        if candidate in active_library.columns:
            cutoff_column = candidate
            break

    def comparable_timestamp(value: Any) -> pd.Timestamp | None:
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            return None
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp

    test_start = comparable_timestamp(final_test_start)
    parsed_cutoffs = (
        [
            timestamp
            for timestamp in (
                comparable_timestamp(value)
                for value in active_library[cutoff_column].dropna()
            )
            if timestamp is not None
        ]
        if cutoff_column
        else []
    )
    latest_cutoff = max(parsed_cutoffs) if parsed_cutoffs else None
    cutoff_valid = (
        test_start is not None
        and latest_cutoff is not None
        and latest_cutoff <= test_start
    )
    if policy == "off":
        status = "skipped"
        reason = "配置关闭 active 因子库时间截止审计"
    elif not cutoff_column:
        status = "failed"
        reason = "active 因子库缺少 验证截止/训练截止 元数据"
    elif latest_cutoff is None:
        status = "failed"
        reason = f"active 因子库的 {cutoff_column} 无法解析"
    elif not cutoff_valid:
        status = "failed"
        reason = (
            f"active 因子库最新{cutoff_column}={latest_cutoff} "
            f"晚于最终测试起点={test_start}"
        )
    else:
        status = "passed"
        reason = ""

    audit = {
        "schema_version": 1,
        "artifact_type": "active_library_oos_cutoff_audit",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "symbol": str(config.symbol),
        "policy": policy,
        "status": status,
        "reason": reason,
        "active_library": build_file_identity(active_path),
        "cutoff_column": cutoff_column,
        "latest_selection_cutoff": str(latest_cutoff) if latest_cutoff is not None else "",
        "final_test_start": str(test_start) if test_start is not None else "",
        "factor_count": int(active_library["因子"].notna().sum())
        if "因子" in active_library.columns
        else 0,
    }
    audit_path = write_json_atomic(
        output_dir / "active_library_oos_audit.json",
        audit,
    )
    if run_dir is not None:
        write_json_atomic(run_dir / audit_path.name, audit)

    if status == "failed":
        message = f"active 因子库样本外截止审计失败: {reason}"
        detected_future_selection = (
            latest_cutoff is not None
            and test_start is not None
            and latest_cutoff > test_start
        )
        if policy == "error" or (policy == "auto" and detected_future_selection):
            raise ValueError(message)
        print(f"警告: {message}")
    elif status == "passed":
        print(
            f"active 因子库样本外截止审计通过: "
            f"{cutoff_column}={latest_cutoff} <= 最终测试起点={test_start}"
        )
    return audit


def load_active_factor_names(config: BacktestConfig) -> list[str]:
    """先读取 active 因子名，用于综合回测按需构建因子矩阵。"""
    active_path = get_active_factor_library_path(config)
    if not active_path.exists():
        raise FileNotFoundError(
            f"没有找到 active 因子库: {active_path}。请先运行 single_factor_backtest.py 更新因子库。"
        )

    active_library = pd.read_csv(active_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {active_path}")
    active_names = [
        str(factor_name)
        for factor_name in active_library["因子"].dropna().tolist()
        if str(factor_name).strip()
    ]
    if not active_names:
        raise ValueError(
            f"active 因子库为空: {active_path}。综合回测已停止，"
            "不会回退计算全部因子；请先运行单因子流程补充 active 因子。"
        )
    return list(dict.fromkeys(active_names))


def build_factor_cache_meta(
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> dict[str, Any]:
    """生成 active 因子矩阵缓存的轻量校验信息。"""
    hash_columns = [
        column
        for column in ["open", "high", "low", "close", "volume", "amt", "amount"]
        if column in data.columns
    ]
    hash_frame = data[hash_columns] if hash_columns else data
    data_hash_values = pd.util.hash_pandas_object(hash_frame, index=True).to_numpy()
    data_content_hash = hashlib.sha256(data_hash_values.tobytes()).hexdigest()
    return {
        "symbol": config.symbol,
        "bar_frequency": get_frequency_key(config),
        "bar_size": int(config.bar_size),
        "start": str(data.index.min()) if len(data.index) else "",
        "end": str(data.index.max()) if len(data.index) else "",
        "rows": int(len(data.index)),
        "data_columns": hash_columns,
        "data_content_hash": data_content_hash,
        "requested_factors": list(requested_factors),
        "zscore_window": int(config.zscore_window),
        "signal_threshold": float(config.signal_threshold),
        "enable_cross_asset_factors": bool(getattr(config, "enable_cross_asset_factors", False)),
        "enable_macro_state_factors": bool(getattr(config, "enable_macro_state_factors", False)),
        "enable_external_daily_factors": bool(getattr(config, "enable_external_daily_factors", False)),
        "related_symbols": list(getattr(config, "related_symbols", []) or []),
        "cross_asset_factor_windows": list(getattr(config, "cross_asset_factor_windows", []) or []),
        "cross_asset_max_ffill_bars": int(getattr(config, "cross_asset_max_ffill_bars", 0) or 0),
        "cross_asset_max_factors": getattr(config, "cross_asset_max_factors", None),
        "macro_state_symbols": list(getattr(config, "macro_state_symbols", []) or []),
        "macro_state_field": str(getattr(config, "macro_state_field", "close") or "close"),
        "macro_state_windows": list(getattr(config, "macro_state_windows", []) or []),
        "macro_state_lag_daily_bars": int(getattr(config, "macro_state_lag_daily_bars", 0) or 0),
        "external_daily_sources": list(getattr(config, "external_daily_sources", []) or []),
        "external_daily_windows": list(getattr(config, "external_daily_windows", []) or []),
        "external_daily_lag_daily_bars": int(getattr(config, "external_daily_lag_daily_bars", 0) or 0),
        "factor_source_hash": build_source_fingerprint_hash(),
    }


def load_factor_matrix_cache(
    cache_path: Path,
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame | None:
    """读取并校验 active 因子矩阵缓存。"""
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(cached, pd.DataFrame):
        return None
    expected_meta = build_factor_cache_meta(data, requested_factors, config)
    if cached.attrs.get("cache_meta") != expected_meta:
        return None
    if not cached.index.equals(data.index):
        return None
    missing_columns = sorted(set(requested_factors).difference(cached.columns))
    if missing_columns:
        return None
    return cached[list(requested_factors)].copy()


def save_factor_matrix_cache(
    factors: pd.DataFrame,
    cache_path: Path,
    data: pd.DataFrame,
    requested_factors: list[str],
    config: BacktestConfig,
) -> None:
    """保存 active 因子矩阵缓存，供下一次综合回测复用。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = factors.copy()
    cached.attrs["cache_meta"] = build_factor_cache_meta(data, requested_factors, config)
    cached.to_pickle(cache_path)


def load_active_factor_pool(
    factors: pd.DataFrame,
    config: BacktestConfig,
) -> list[str]:
    """读取 active_factors.csv，并把它作为候选因子的硬边界。

    综合模型只考虑已经通过单因子入库流程的因子。
    xgboost_feature_scope 只在这个 active 池内部继续选择，而不是面对代码能生成的全部因子。
    """
    if getattr(config, "use_frozen_active_library", False):
        frozen_path = getattr(config, "frozen_active_library_path", None)
        if not frozen_path:
            raise ValueError("use_frozen_active_library=True 时必须配置 frozen_active_library_path。")
        active_path = Path(frozen_path)
        if not active_path.is_absolute():
            active_path = Path(config.output_dir) / active_path
    else:
        active_path = get_factor_library_dir(config) / "active_factors.csv"
    if not active_path.exists():
        raise FileNotFoundError(
            f"没有找到 active 因子库: {active_path}。请先运行 single_factor_backtest.py 更新因子库。"
        )

    active_library = pd.read_csv(active_path)
    if "因子" not in active_library.columns:
        raise KeyError(f"active 因子库缺少 '因子' 列: {active_path}")

    factor_columns = get_factor_columns(factors)
    factor_set = set(factor_columns)
    active_names = [
        str(factor_name)
        for factor_name in active_library["因子"].dropna().tolist()
        if str(factor_name) in factor_set
    ]
    if not active_names:
        raise ValueError(
            "active 因子库中没有任何因子能由 framework/factors.py 生成。"
            "请检查 active 因子库、related_symbols 或重新运行单因子回测。"
        )

    missing_count = int(active_library["因子"].notna().sum()) - len(active_names)
    if missing_count > 0:
        print(f"active 因子库中有 {missing_count} 个因子当前不可生成，已自动跳过。")

    return active_names


def select_best_factors_on_training(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
    split_time: pd.Timestamp,
) -> tuple[list[str], pd.DataFrame]:
    """在静态训练集上选择表现最好的因子。

    这是非滚动 best 模式使用的旧路径：
    先对每个因子做训练集单因子回测，再按训练夏普和累计收益排序，
    最后取前 xgboost_best_top_n 个作为 XGBoost 候选特征。

    如果启用 walk-forward feature selection，则不会走这个函数，
    因为每个滚动窗口都会重新选因子。
    """
    train_data = data.loc[data.index < split_time]
    rows = []

    for factor_name in get_factor_columns(factors):
        direction, direction_metrics = choose_factor_direction(
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
        train_signal = signal.loc[signal.index < split_time]
        _train_df, metrics, error = safe_run_backtest(train_data, train_signal, config)
        if error is not None:
            continue

        rows.append(
            {
                "因子": factor_name,
                "方向": direction,
                **metrics,
                **direction_metrics,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("训练段没有可用因子，无法选择 best 特征。")

    min_sharpe = float(getattr(config, "factor_library_min_sharpe", 0.0) or 0.0)
    min_total_return = float(getattr(config, "factor_library_min_total_return", 0.0) or 0.0)
    summary = summary.dropna(subset=["夏普比率", "累计收益"])
    summary = summary[
        (summary["夏普比率"] >= min_sharpe)
        & (summary["累计收益"] >= min_total_return)
    ]
    if summary.empty:
        raise ValueError(
            "训练段没有因子满足 factor_library_min_sharpe/"
            "factor_library_min_total_return。"
        )

    summary = summary.sort_values(["夏普比率", "累计收益"], ascending=False)
    selected = summary["因子"].head(config.xgboost_best_top_n).tolist()
    return selected, summary


def get_selected_factors(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig,
    split_time: pd.Timestamp,
) -> tuple[list[str], pd.DataFrame | None]:
    """根据 xgboost_feature_scope 决定 XGBoost 候选因子集合。

    active_factors.csv 是硬边界，所有模式都只能在 active 因子池内继续选择：
    - all：使用所有当前可生成的 active 因子。
    - selected：使用 config.selected_factors，但这些因子必须属于 active 因子池。
    - best：只在 active 因子池内做静态或滚动 best 选择。
    """
    active_factor_columns = load_active_factor_pool(factors, config)
    active_factors = factors[active_factor_columns]
    scope = config.xgboost_feature_scope.lower()
    if scope != "selected" and config.selected_factors:
        print("提示: selected_factors 仅在 xgboost_feature_scope='selected' 时生效，当前模式将忽略该列表。")

    if scope == "all":
        return active_factor_columns, None

    if scope == "best":
        if config.xgboost_walk_forward_feature_selection:
            return active_factor_columns, None
        return select_best_factors_on_training(data, active_factors, config, split_time)

    if scope != "selected":
        raise ValueError('xgboost_feature_scope 只能是 "selected", "all", 或 "best"。')

    if config.selected_factors:
        missing = sorted(set(config.selected_factors).difference(active_factor_columns))
        if missing:
            raise ValueError(
                "selected_factors 中存在不在 active 因子库内或当前不可生成的因子: "
                f"{missing}"
            )
        return list(config.selected_factors), None
    return active_factor_columns, None


def should_use_walk_forward_selection(config: BacktestConfig) -> bool:
    """判断是否启用滚动窗口内动态选因。"""
    return (
        config.xgboost_feature_scope.lower() == "best"
        and bool(config.xgboost_walk_forward_feature_selection)
    )


def build_factor_signal_features(
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """把多个因子的连续值转换成 -1/0/1 信号特征矩阵。

    这是当前框架的核心自变量之一：
    每个单因子的独立多空判断作为模型输入，让模型学习不同信号组合下的可交易未来方向。
    """
    missing = sorted(set(selected_factors).difference(factors.columns))
    if missing:
        raise ValueError(f"因子数据缺少以下字段: {missing}")

    if bool(factors.attrs.get("precomputed_factor_signals", False)):
        return (
            factors[selected_factors]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1.0, upper=1.0)
            .astype("float64")
        )

    feature_data = {
        factor_name: score_to_raw_signal(
            factors[factor_name].replace([np.inf, -np.inf], np.nan),
            config.signal_threshold,
        )
        for factor_name in selected_factors
    }
    return pd.DataFrame(feature_data, index=factors.index)


def calculate_signal_streak(signal: pd.Series) -> pd.Series:
    """统计同方向非零信号连续出现的 K 线数量。"""
    clean_signal = signal.fillna(0.0)
    group_id = clean_signal.ne(clean_signal.shift(1)).cumsum()
    streak = clean_signal.groupby(group_id).cumcount() + 1
    return streak.where(clean_signal != 0.0, 0.0).astype("float64")


def build_factor_state_features(
    continuous_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    selected_factors: list[str],
) -> pd.DataFrame:
    """为 XGBoost 构造动态因子状态特征。

    这些特征帮助模型区分刚刚翻转的信号和已经持续一段时间的信号，
    也帮助模型识别因子是在增强还是在衰减。
    """
    state_parts = []
    for factor_name in selected_factors:
        value = continuous_features[factor_name]
        signal = signal_features[factor_name]
        state_parts.extend(
            [
                value.diff().rename(f"{factor_name}_value_diff_1"),
                value.shift(1).rename(f"{factor_name}_value_lag_1"),
                signal.shift(1).rename(f"{factor_name}_signal_lag_1"),
                signal.diff().rename(f"{factor_name}_signal_change_1"),
                calculate_signal_streak(signal).rename(f"{factor_name}_signal_streak"),
            ]
        )
    return pd.concat(state_parts, axis=1)


def build_features_for_factors(
    factors: pd.DataFrame,
    signal_features: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """根据配置构造 XGBoost 最终特征矩阵。

    signal：只使用 -1/0/1 信号。
    continuous：只使用标准化后的连续因子值。
    both：每个因子同时放入 value 和 signal 两个特征。
    """
    mode = config.xgboost_feature_mode.lower()
    if mode not in {"signal", "continuous", "both"}:
        raise ValueError('xgboost_feature_mode 只能是 "signal", "continuous", 或 "both"。')

    continuous_features = factors[selected_factors].replace([np.inf, -np.inf], np.nan)
    feature_parts = []

    if mode == "signal":
        feature_parts.append(signal_features[selected_factors])
    elif mode == "continuous":
        feature_parts.append(continuous_features)
    else:
        for factor_name in selected_factors:
            feature_parts.append(continuous_features[factor_name].rename(f"{factor_name}_value"))
            feature_parts.append(signal_features[factor_name].rename(f"{factor_name}_signal"))

    if getattr(config, "xgboost_include_factor_state_features", False):
        feature_parts.append(
            build_factor_state_features(
                continuous_features,
                signal_features,
                selected_factors,
            )
        )
    return pd.concat(feature_parts, axis=1)


def build_xgboost_features(
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """构造 XGBoost 特征矩阵的便捷函数。"""
    signal_features = build_factor_signal_features(factors, selected_factors, config)
    return build_features_for_factors(factors, signal_features, selected_factors, config)


def get_feature_base_factor(feature_name: str, selected_factors: list[str]) -> str:
    """从模型特征名还原基础因子名。

    both 模式下会出现 xxx_value / xxx_signal，
    输出特征重要性时需要把它们映射回原始因子。
    """
    if feature_name in selected_factors:
        return feature_name
    for suffix in ("_value", "_signal"):
        if feature_name.endswith(suffix):
            base_name = feature_name[: -len(suffix)]
            if base_name in selected_factors:
                return base_name
    return feature_name


def get_xgboost_target_round_trip_cost(config: BacktestConfig) -> float:
    """返回标签需要覆盖的一次完整开平仓成本，单位为收益率。"""
    if bool(getattr(config, "xgboost_target_include_round_trip_cost", True)):
        return 2.0 * (
            max(0.0, float(config.commission_bps))
            + max(0.0, float(config.slippage_bps))
        ) / 10000.0
    return 0.0


def get_xgboost_target_neutral_threshold(config: BacktestConfig) -> float:
    """返回 XGBoost 分类目标中的静态中性收益阈值。

    静态阈值由一次完整开平仓的双边成本和额外中性缓冲组成。
    动态波动率缓冲在 calculate_xgboost_target_neutral_threshold 中另行叠加。
    """
    neutral_bps = config.xgboost_target_neutral_bps
    if neutral_bps is None:
        neutral_bps = 0.0
    return get_xgboost_target_round_trip_cost(config) + (
        max(0.0, float(neutral_bps)) / 10000.0
    )


def calculate_xgboost_target_volatility(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """估计信号时点可知的持有期收益波动率，不使用未来行情。"""
    window = max(
        20,
        int(getattr(config, "xgboost_target_dynamic_neutral_window", 240) or 240),
    )
    min_periods = max(20, window // 3)
    return_mode = str(
        getattr(config, "backtest_return_mode", "next_open_continuous")
        or "next_open_continuous"
    ).lower()
    if return_mode == "next_open_continuous":
        base_return = data["close"].replace(0, np.nan).pct_change()
    else:
        base_return = data["close"] / data["open"].replace(0, np.nan) - 1.0
    volatility = (
        base_return.rolling(window=window, min_periods=min_periods).std()
        * np.sqrt(get_xgboost_target_horizon(config))
    )
    return volatility.reindex(index).replace([np.inf, -np.inf], np.nan).astype("float64")


def calculate_xgboost_target_neutral_threshold(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """计算每个时点的标签中性阈值，只使用当前及历史行情。"""
    base_threshold = get_xgboost_target_neutral_threshold(config)
    threshold = pd.Series(base_threshold, index=index, dtype="float64")
    if not bool(getattr(config, "xgboost_target_use_dynamic_neutral_threshold", False)):
        return threshold

    multiplier = max(
        0.0,
        float(getattr(config, "xgboost_target_dynamic_neutral_multiplier", 0.0) or 0.0),
    )
    if multiplier <= 0:
        return threshold

    dynamic_threshold = calculate_xgboost_target_volatility(data, index, config) * multiplier
    # 波动缓冲叠加在双边成本之上，保证非中性标签代表成本后仍有足够统计边际。
    return threshold.add(dynamic_threshold.fillna(0.0), fill_value=0.0).astype("float64")


def get_xgboost_target_label_mode(config: BacktestConfig) -> str:
    """读取标签生成模式，并对未知值回退到旧版 threshold 模式。"""
    mode = str(getattr(config, "xgboost_target_label_mode", "threshold") or "threshold").lower()
    return mode if mode in {"threshold", "quantile"} else "threshold"


def calculate_quantile_target_bounds(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.DataFrame:
    """用已经落地的历史 horizon 收益滚动分位数生成标签上下边界。"""
    horizon = get_xgboost_target_horizon(config)
    history_return = calculate_future_horizon_return(data, data.index, config).shift(horizon)
    window = max(50, int(getattr(config, "xgboost_target_quantile_window", 1200) or 1200))
    min_periods = max(30, window // 4)
    lower_q = min(
        0.49,
        max(0.01, float(getattr(config, "xgboost_target_quantile_lower", 0.35) or 0.35)),
    )
    upper_q = max(
        0.51,
        min(0.99, float(getattr(config, "xgboost_target_quantile_upper", 0.65) or 0.65)),
    )
    if lower_q >= upper_q:
        lower_q, upper_q = 0.35, 0.65

    lower = history_return.rolling(window=window, min_periods=min_periods).quantile(lower_q)
    upper = history_return.rolling(window=window, min_periods=min_periods).quantile(upper_q)
    min_abs = max(
        0.0,
        float(getattr(config, "xgboost_target_quantile_min_abs_bps", 0.0) or 0.0) / 10000.0,
    )
    if min_abs > 0:
        lower = pd.concat([lower, pd.Series(-min_abs, index=lower.index)], axis=1).min(axis=1)
        upper = pd.concat([upper, pd.Series(min_abs, index=upper.index)], axis=1).max(axis=1)

    bounds = pd.DataFrame(
        {
            "target_lower_threshold": lower.reindex(index),
            "target_upper_threshold": upper.reindex(index),
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan)
    fallback_neutral = calculate_xgboost_target_neutral_threshold(data, index, config)
    bounds["target_lower_threshold"] = pd.concat(
        [bounds["target_lower_threshold"], -fallback_neutral],
        axis=1,
    ).min(axis=1)
    bounds["target_upper_threshold"] = pd.concat(
        [bounds["target_upper_threshold"], fallback_neutral],
        axis=1,
    ).max(axis=1)
    return bounds.astype("float64")


def calculate_xgboost_target_bounds(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.DataFrame:
    """计算 XGBoost 标签使用的上下边界。"""
    if get_xgboost_target_label_mode(config) == "quantile":
        return calculate_quantile_target_bounds(data, index, config)

    neutral_threshold = calculate_xgboost_target_neutral_threshold(data, index, config)
    return pd.DataFrame(
        {
            "target_lower_threshold": -neutral_threshold,
            "target_upper_threshold": neutral_threshold,
        },
        index=index,
    )


def select_factors_in_window(
    signal_features: pd.DataFrame,
    train_target: pd.Series,
    candidate_factors: list[str],
    config: BacktestConfig,
) -> tuple[list[str], pd.DataFrame]:
    """在单个滚动训练窗口内做特征选择。

    逻辑：
    1. 计算每个候选因子信号与训练目标的相关性。
    2. 先按绝对相关性筛出候选池。
    3. 再按因子之间的相关性做去重。
    4. 最多保留 xgboost_best_top_n 个因子。

    这样可以降低海量相似因子一起进入模型导致的过拟合风险。
    """
    frame = signal_features[candidate_factors].assign(__target__=train_target)
    frame = frame.dropna(subset=["__target__"])
    if frame.empty:
        return [], pd.DataFrame()

    target = frame["__target__"]
    factor_frame = frame[candidate_factors].replace([np.inf, -np.inf], np.nan)
    valid_factors = factor_frame.columns[factor_frame.std(ddof=0) > 0].tolist()
    if not valid_factors or target.std(ddof=0) == 0:
        return [], pd.DataFrame()

    factor_frame = factor_frame[valid_factors]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = factor_frame.corrwith(target).replace([np.inf, -np.inf], np.nan)
    summary = pd.DataFrame(
        {
            "因子": scores.index,
            "目标相关性": scores.values,
            "abs_score": scores.abs().values,
            "有效样本数": factor_frame.notna().sum().reindex(scores.index).values,
        }
    ).dropna(subset=["abs_score"])
    summary = summary[summary["有效样本数"] >= config.xgboost_min_train_samples]
    if summary.empty:
        return [], summary

    # 先放宽候选池，再做相关性去重；否则前几名高度相似时会挤掉其他信息源。
    candidate_limit = max(
        config.xgboost_best_top_n,
        config.xgboost_best_top_n * max(1, int(config.xgboost_candidate_multiplier)),
    )
    summary = summary.sort_values("abs_score", ascending=False).head(candidate_limit)

    selected = []
    selected_signals = factor_frame[summary["因子"].tolist()]
    max_corr = float(config.xgboost_max_feature_corr)
    for factor_name in summary["因子"]:
        if len(selected) >= config.xgboost_best_top_n:
            break
        if not selected:
            selected.append(factor_name)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pair_corr = selected_signals[selected].corrwith(selected_signals[factor_name]).abs()
        if pair_corr.dropna().lt(max_corr).all():
            selected.append(factor_name)

    summary["入选"] = summary["因子"].isin(selected)
    return selected, summary


def calculate_next_bar_direction(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """生成 XGBoost 的三分类预测目标。

    未来 horizon 根累计收益 > 中性阈值：1
    未来 horizon 根累计收益 < -中性阈值：-1
    落在中性区间：0
    """
    next_return = calculate_future_horizon_return(data, index, config)
    target_bounds = calculate_xgboost_target_bounds(data, index, config)
    direction = pd.Series(0.0, index=index, dtype="float64")
    direction[next_return > target_bounds["target_upper_threshold"]] = 1.0
    direction[next_return < target_bounds["target_lower_threshold"]] = -1.0
    direction[next_return.isna()] = np.nan
    return direction


def get_xgboost_target_horizon(config: BacktestConfig) -> int:
    """返回有效预测跨度，并可自动覆盖执行层最小持仓期。"""
    horizon = max(1, int(getattr(config, "xgboost_target_horizon", 1) or 1))
    if (
        bool(getattr(config, "xgboost_target_align_with_min_holding", True))
        and bool(getattr(config, "xgboost_use_position_rules", False))
    ):
        horizon = max(
            horizon,
            max(1, int(getattr(config, "xgboost_min_holding_bars", 1) or 1)),
        )
    return horizon


def calculate_future_horizon_return(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.Series:
    """计算从下一根 K 线开盘到目标跨度收盘的未来累计收益。

    horizon=1 时等价于旧版下一根 K 线 open 到 close 目标。
    horizon=3 时使用 open[t+1] 到 close[t+3] 的收益，并对齐到时点 t。
    """
    horizon = get_xgboost_target_horizon(config)
    entry_open = data["open"].replace(0, np.nan).shift(-1)
    exit_close = data["close"].shift(-horizon)
    return (exit_close / entry_open - 1.0).reindex(index)


def calculate_future_target_outcomes(
    data: pd.DataFrame,
    index: pd.Index,
    config: BacktestConfig,
) -> pd.DataFrame:
    """构造与真实持仓和交易成本一致的未来目标收益诊断。

    future_horizon_net_return 表示按未来价格方向交易、扣除完整双边成本后
    仍可获得的带方向机会收益；绝对波动不足以覆盖成本时记为 0。
    standardized 列再除以信号时点已知的事前持有期波动率。
    """
    gross_return = calculate_future_horizon_return(data, index, config).astype("float64")
    round_trip_cost = get_xgboost_target_round_trip_cost(config)
    net_magnitude = (gross_return.abs() - round_trip_cost).clip(lower=0.0)
    net_return = np.sign(gross_return) * net_magnitude
    volatility = calculate_xgboost_target_volatility(data, index, config)
    standardized_net_return = net_return / volatility.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "future_horizon_return": gross_return,
            "future_horizon_round_trip_cost": round_trip_cost,
            "future_horizon_net_return": net_return,
            "future_horizon_target_volatility": volatility,
            "future_horizon_standardized_net_return": standardized_net_return,
        },
        index=index,
    ).replace([np.inf, -np.inf], np.nan)


def calculate_next_bar_return(data: pd.DataFrame, index: pd.Index) -> pd.Series:
    """计算下一根K线 open 到 close 的收益，并对齐到当前时点。"""
    open_price = data["open"].replace(0, np.nan)
    return (data["close"] / open_price - 1.0).reindex(index).shift(-1)


def choose_signal_direction_on_training(
    raw_signal: pd.Series,
    target_return: pd.Series,
) -> float:
    """在训练窗口内判断一个合成信号是否需要反向。

    比较原始信号和反向信号在训练窗口的目标持有期净收益贡献。
    该逻辑用于 XGBoost 概率信号和等权投票基准的方向校准。
    """
    aligned = pd.DataFrame(
        {
            "signal": raw_signal,
            "target_return": target_return.reindex(raw_signal.index),
        }
    ).dropna()
    if aligned.empty or aligned["signal"].abs().sum() == 0:
        return 1.0

    forward_return = (aligned["signal"] * aligned["target_return"]).sum()
    inverse_return = (-aligned["signal"] * aligned["target_return"]).sum()
    return -1.0 if inverse_return > forward_return else 1.0


def probabilities_to_trade_signal(
    probabilities: pd.DataFrame,
    config: BacktestConfig,
    min_edge: float | None = None,
    min_probability: float | None = None,
) -> pd.Series:
    """把 XGBoost 三分类概率转换成 -1/0/1 交易信号。

    使用 prob_up - prob_down 作为方向优势。
    只有优势超过 xgboost_trade_min_edge，且方向概率超过 xgboost_trade_min_probability 时才开仓。
    """
    edge = probabilities["prob_up"] - probabilities["prob_down"]
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    flat_probability = probabilities["prob_flat"]
    min_edge = max(
        0.0,
        float(config.xgboost_trade_min_edge if min_edge is None else min_edge),
    )
    min_probability = max(
        0.0,
        float(config.xgboost_trade_min_probability if min_probability is None else min_probability),
    )
    max_flat_probability = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_trade_max_flat_probability", 1.0) or 1.0)),
    )
    min_directional_vs_flat_edge = max(
        0.0,
        float(getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0) or 0.0),
    )
    flat_allowed = (
        (flat_probability <= max_flat_probability)
        & ((directional_probability - flat_probability) >= min_directional_vs_flat_edge)
    )

    signal = pd.Series(0.0, index=probabilities.index, dtype="float64")
    long_mask = (edge >= min_edge) & (directional_probability >= min_probability) & flat_allowed
    short_mask = (edge <= -min_edge) & (directional_probability >= min_probability) & flat_allowed
    signal[long_mask] = 1.0
    signal[short_mask] = -1.0
    signal[probabilities[["prob_up", "prob_down", "prob_flat"]].isna().any(axis=1)] = np.nan
    return signal


def build_market_state_filter(data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """仅使用当前和历史 K 线构造简单的交易质量过滤器。"""
    window = max(20, int(getattr(config, "xgboost_trade_filter_window", 120) or 120))
    min_periods = max(20, window // 3)
    regime_window = max(20, int(getattr(config, "market_state_regime_window", window) or window))
    regime_min_periods = max(20, regime_window // 3)
    use_filters = bool(getattr(config, "xgboost_trade_use_market_filters", False))

    abs_intrabar_return = (data["close"] / data["open"].replace(0, np.nan) - 1.0).abs()
    vol_rank = abs_intrabar_return.rolling(window=window, min_periods=min_periods).rank(pct=True)
    close_return = data["close"].replace(0, np.nan).pct_change()
    rolling_return = data["close"].replace(0, np.nan) / data["close"].replace(0, np.nan).shift(regime_window) - 1.0
    rolling_abs_return_sum = close_return.abs().rolling(regime_window, min_periods=regime_min_periods).sum()
    trend_strength = (rolling_return.abs() / rolling_abs_return_sum.replace(0, np.nan)).clip(0.0, 1.0)
    trend_threshold = max(
        0.0,
        min(1.0, float(getattr(config, "market_state_trend_strength_threshold", 0.25) or 0.25)),
    )

    liquidity_series = None
    for column in ("amt", "amount", "volume"):
        if column in data.columns:
            liquidity_series = data[column].replace(0, np.nan).abs()
            if liquidity_series.notna().any():
                break
    if liquidity_series is None:
        liquidity_series = pd.Series(np.nan, index=data.index, dtype="float64")
    liquidity_rank = liquidity_series.rolling(window=window, min_periods=min_periods).rank(pct=True)

    trade_allowed = pd.Series(True, index=data.index, dtype="bool")
    if use_filters:
        min_vol_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_volatility_rank", 0.0) or 0.0)),
        )
        min_liq_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_liquidity_rank", 0.0) or 0.0)),
        )
        if min_vol_rank > 0:
            trade_allowed &= vol_rank.fillna(0.0) >= min_vol_rank
        if min_liq_rank > 0 and liquidity_rank.notna().any():
            trade_allowed &= liquidity_rank.fillna(0.0) >= min_liq_rank
    trade_allowed &= abs_intrabar_return.notna()

    volatility_regime = pd.Series("未知波动", index=data.index, dtype="object")
    volatility_regime[vol_rank <= 0.33] = "低波动"
    volatility_regime[(vol_rank > 0.33) & (vol_rank <= 0.66)] = "中波动"
    volatility_regime[vol_rank > 0.66] = "高波动"

    liquidity_regime = pd.Series("未知流动性", index=data.index, dtype="object")
    liquidity_regime[liquidity_rank <= 0.33] = "低流动性"
    liquidity_regime[(liquidity_rank > 0.33) & (liquidity_rank <= 0.66)] = "中流动性"
    liquidity_regime[liquidity_rank > 0.66] = "高流动性"

    trend_regime = pd.Series("未知趋势", index=data.index, dtype="object")
    trend_regime[(trend_strength >= trend_threshold) & (rolling_return > 0)] = "趋势上涨"
    trend_regime[(trend_strength >= trend_threshold) & (rolling_return < 0)] = "趋势下跌"
    trend_regime[(trend_strength < trend_threshold) & trend_strength.notna()] = "震荡"

    hour = pd.Series(data.index.hour, index=data.index)
    minute = pd.Series(data.index.minute, index=data.index)
    intraday_minutes = hour * 60 + minute
    session_regime = pd.Series("其他时段", index=data.index, dtype="object")
    session_regime[(intraday_minutes >= 9 * 60) & (intraday_minutes < 10 * 60 + 30)] = "早盘"
    session_regime[(intraday_minutes >= 10 * 60 + 30) & (intraday_minutes < 11 * 60 + 30)] = "上午后段"
    session_regime[(intraday_minutes >= 13 * 60 + 30) & (intraday_minutes < 15 * 60)] = "下午盘"
    session_regime[(intraday_minutes >= 21 * 60) | (intraday_minutes < 2 * 60 + 30)] = "夜盘"

    if bool(getattr(config, "xgboost_trade_use_regime_filter", False)):
        allowed_regimes = set(str(value) for value in (getattr(config, "allowed_market_state_regimes", []) or []))
        if allowed_regimes:
            regime_allowed = (
                trend_regime.isin(allowed_regimes)
                | volatility_regime.isin(allowed_regimes)
                | liquidity_regime.isin(allowed_regimes)
                | session_regime.isin(allowed_regimes)
            )
            trade_allowed &= regime_allowed.fillna(False)

    return pd.DataFrame(
        {
            "volatility_rank": vol_rank.astype("float64"),
            "liquidity_rank": liquidity_rank.astype("float64"),
            "trend_strength": trend_strength.astype("float64"),
            "rolling_regime_return": rolling_return.astype("float64"),
            "volatility_regime": volatility_regime,
            "liquidity_regime": liquidity_regime,
            "trend_regime": trend_regime,
            "session_regime": session_regime,
            "trade_allowed": trade_allowed.astype("float64"),
        },
        index=data.index,
    )


def build_confidence_position_size(
    probabilities: pd.DataFrame,
    config: BacktestConfig,
    min_edge: float | None = None,
    min_probability: float | None = None,
) -> pd.Series:
    """把模型置信度映射到 0 到 max_size 之间的目标仓位。"""
    edge = (probabilities["prob_up"] - probabilities["prob_down"]).abs()
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    flat_probability = probabilities["prob_flat"]
    min_edge = min(
        0.999,
        max(0.0, float(config.xgboost_trade_min_edge if min_edge is None else min_edge)),
    )
    min_probability = min(
        0.999,
        max(
            0.0,
            float(config.xgboost_trade_min_probability if min_probability is None else min_probability),
        ),
    )
    max_flat_probability = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_trade_max_flat_probability", 1.0) or 1.0)),
    )
    min_directional_vs_flat_edge = max(
        0.0,
        float(getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0) or 0.0),
    )
    flat_allowed = (
        (flat_probability <= max_flat_probability)
        & ((directional_probability - flat_probability) >= min_directional_vs_flat_edge)
    )
    max_size = max(0.0, float(getattr(config, "xgboost_position_size_max", 1.0) or 1.0))

    edge_strength = ((edge - min_edge) / max(1e-9, 1.0 - min_edge)).clip(lower=0.0, upper=1.0)
    probability_strength = (
        (directional_probability - min_probability) / max(1e-9, 1.0 - min_probability)
    ).clip(lower=0.0, upper=1.0)
    confidence = edge_strength.combine(probability_strength, max).clip(lower=0.0, upper=1.0)

    if bool(getattr(config, "xgboost_use_dynamic_position_sizing", False)):
        power = max(0.25, float(getattr(config, "xgboost_position_size_power", 1.0) or 1.0))
        min_size = min(
            max_size,
            max(0.0, float(getattr(config, "xgboost_position_size_min", 0.0) or 0.0)),
        )
        size = min_size + (max_size - min_size) * confidence.pow(power)
        size = size.where(confidence > 0, 0.0)
    else:
        size = (confidence > 0).astype("float64") * max_size

    size = size.where(flat_allowed.fillna(False), 0.0)
    size[probabilities[["prob_up", "prob_down", "prob_flat"]].isna().any(axis=1)] = np.nan
    return size.astype("float64")


def build_dynamic_confidence_position_size(
    probabilities: pd.DataFrame,
    min_edge: pd.Series,
    min_probability: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """使用逐行校准阈值，把置信度映射为仓位大小。"""
    edge = (probabilities["prob_up"] - probabilities["prob_down"]).abs()
    directional_probability = probabilities[["prob_up", "prob_down"]].max(axis=1)
    flat_probability = probabilities["prob_flat"]
    edge_threshold = min_edge.reindex(probabilities.index).fillna(float(config.xgboost_trade_min_edge))
    probability_threshold = min_probability.reindex(probabilities.index).fillna(
        float(config.xgboost_trade_min_probability)
    )
    max_flat_probability = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_trade_max_flat_probability", 1.0) or 1.0)),
    )
    min_directional_vs_flat_edge = max(
        0.0,
        float(getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0) or 0.0),
    )
    flat_allowed = (
        (flat_probability <= max_flat_probability)
        & ((directional_probability - flat_probability) >= min_directional_vs_flat_edge)
    )
    max_size = max(0.0, float(getattr(config, "xgboost_position_size_max", 1.0) or 1.0))

    edge_strength = (
        (edge - edge_threshold) / (1.0 - edge_threshold).clip(lower=1e-9)
    ).clip(lower=0.0, upper=1.0)
    probability_strength = (
        (directional_probability - probability_threshold)
        / (1.0 - probability_threshold).clip(lower=1e-9)
    ).clip(lower=0.0, upper=1.0)
    confidence = edge_strength.combine(probability_strength, max).clip(lower=0.0, upper=1.0)

    if bool(getattr(config, "xgboost_use_dynamic_position_sizing", False)):
        power = max(0.25, float(getattr(config, "xgboost_position_size_power", 1.0) or 1.0))
        min_size = min(
            max_size,
            max(0.0, float(getattr(config, "xgboost_position_size_min", 0.0) or 0.0)),
        )
        size = min_size + (max_size - min_size) * confidence.pow(power)
        size = size.where(confidence > 0, 0.0)
    else:
        size = (confidence > 0).astype("float64") * max_size

    size = size.where(flat_allowed.fillna(False), 0.0)
    size[probabilities[["prob_up", "prob_down", "prob_flat"]].isna().any(axis=1)] = np.nan
    return size.astype("float64")


def apply_position_rules(
    target_position: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """对模型目标仓位应用交易执行层规则。

    规则只使用当前和历史目标仓位，不读取未来收益：
    - 最小持仓期：降低刚开仓后立刻反向或清仓。
    - 反转冷却：刚退出后等待若干根 K 线再重新开仓。
    - 小变化忽略：同方向仓位微调不频繁交易。
    - 可选仓位平滑：让目标仓位逐步靠近模型输出。
    """
    clean_target = target_position.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    if not bool(getattr(config, "xgboost_use_position_rules", False)):
        return clean_target

    min_holding_bars = max(0, int(getattr(config, "xgboost_min_holding_bars", 0) or 0))
    cooldown_bars = max(0, int(getattr(config, "xgboost_reentry_cooldown_bars", 0) or 0))
    min_change = max(0.0, float(getattr(config, "xgboost_min_position_change", 0.0) or 0.0))
    alpha = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_position_smoothing_alpha", 1.0) or 1.0)),
    )

    filtered_values: list[float] = []
    current_position = 0.0
    holding_bars = 0
    cooldown_remaining = 0

    for desired_position in clean_target:
        desired_position = float(desired_position)
        current_sign = np.sign(current_position)
        desired_sign = np.sign(desired_position)
        next_position = desired_position

        if cooldown_remaining > 0 and current_sign == 0 and desired_sign != 0:
            next_position = 0.0
            desired_sign = 0.0
            cooldown_remaining -= 1
        elif cooldown_remaining > 0 and current_sign == 0:
            cooldown_remaining -= 1

        if current_sign != 0:
            is_exit = desired_sign == 0
            is_reverse = desired_sign != 0 and desired_sign != current_sign
            is_same_direction = desired_sign == current_sign

            if holding_bars < min_holding_bars and (is_exit or is_reverse):
                next_position = current_position
            elif is_reverse:
                next_position = 0.0 if cooldown_bars > 0 else desired_position
                cooldown_remaining = cooldown_bars
            elif is_exit:
                next_position = 0.0
                cooldown_remaining = cooldown_bars
            elif is_same_direction and abs(desired_position - current_position) < min_change:
                next_position = current_position

        if alpha < 1.0:
            next_position = alpha * next_position + (1.0 - alpha) * current_position

        if abs(next_position) < 1e-12:
            next_position = 0.0

        next_sign = np.sign(next_position)
        if next_sign == 0:
            holding_bars = 0
        elif next_sign == current_sign:
            holding_bars += 1
        else:
            holding_bars = 1

        current_position = float(next_position)
        filtered_values.append(current_position)

    return pd.Series(filtered_values, index=target_position.index, dtype="float64")


def build_confidence_rank_filter(
    confidence_score: pd.Series,
    config: BacktestConfig,
) -> pd.DataFrame:
    """基于近期置信度分位构造自适应过滤器。"""
    window = max(20, int(getattr(config, "xgboost_trade_confidence_rank_window", 240) or 240))
    min_periods = max(20, window // 3)
    rank = confidence_score.abs().rolling(window=window, min_periods=min_periods).rank(pct=True)

    allowed = pd.Series(True, index=confidence_score.index, dtype="bool")
    if bool(getattr(config, "xgboost_trade_use_confidence_rank_filter", False)):
        min_rank = min(
            1.0,
            max(0.0, float(getattr(config, "xgboost_trade_min_confidence_rank", 0.0) or 0.0)),
        )
        if min_rank > 0:
            allowed &= rank.fillna(0.0) >= min_rank
    allowed &= confidence_score.notna()

    return pd.DataFrame(
        {
            "confidence_rank": rank.astype("float64"),
            "confidence_trade_allowed": allowed.astype("float64"),
        },
        index=confidence_score.index,
    )


def calibrate_trade_thresholds_on_training(
    probabilities: pd.DataFrame,
    forward_return: pd.Series,
    trade_allowed: pd.Series,
    config: BacktestConfig,
) -> tuple[float, float]:
    """根据近期训练窗口选择交易阈值。"""
    base_edge = max(0.0, float(config.xgboost_trade_min_edge))
    base_probability = max(0.0, float(config.xgboost_trade_min_probability))
    if not bool(getattr(config, "xgboost_auto_calibrate_trade_thresholds", False)):
        return base_edge, base_probability

    edge_grid = sorted(
        {
            base_edge,
            *[
                max(base_edge, float(value))
                for value in (getattr(config, "xgboost_trade_edge_grid", []) or [])
            ],
        }
    )
    probability_grid = sorted(
        {
            base_probability,
            *[
                max(base_probability, float(value))
                for value in (getattr(config, "xgboost_trade_probability_grid", []) or [])
            ],
        }
    )
    min_trades = max(1, int(getattr(config, "xgboost_threshold_min_trades", 20) or 20))

    aligned_return = forward_return.reindex(probabilities.index)
    allowed = trade_allowed.reindex(probabilities.index).fillna(False).astype(bool)
    best_score = -np.inf
    best_pair = (base_edge, base_probability)

    for edge_threshold in edge_grid:
        for probability_threshold in probability_grid:
            raw_signal = probabilities_to_trade_signal(
                probabilities,
                config,
                min_edge=edge_threshold,
                min_probability=probability_threshold,
            ).where(allowed, 0.0)
            trade_mask = raw_signal.fillna(0.0) != 0
            if int(trade_mask.sum()) < min_trades:
                continue
            strategy_return = (raw_signal * aligned_return).replace([np.inf, -np.inf], np.nan).dropna()
            if strategy_return.empty:
                continue
            volatility = float(strategy_return.std(ddof=0))
            score = float(strategy_return.mean() / volatility) if volatility > 0 else float(strategy_return.mean())
            if score > best_score:
                best_score = score
                best_pair = (float(edge_threshold), float(probability_threshold))

    return best_pair


def vote_score_to_trade_signal(vote_score: pd.Series, config: BacktestConfig) -> pd.Series:
    """把等权投票分数转换成 -1/0/1 交易信号。"""
    min_abs_score = max(0.0, float(config.benchmark_vote_min_abs_score))
    signal = pd.Series(0.0, index=vote_score.index, dtype="float64")
    signal[vote_score > min_abs_score] = 1.0
    signal[vote_score < -min_abs_score] = -1.0
    signal[vote_score.isna()] = np.nan
    return signal


def build_backtest_signal_from_columns(
    signal: pd.DataFrame,
    score_column: str,
    raw_signal_column: str,
    position_column: str,
) -> pd.DataFrame:
    """把综合信号表中的指定列整理成 run_backtest 需要的标准格式。"""
    return pd.DataFrame(
        {
            "composite_score": signal[score_column],
            "raw_signal": signal[raw_signal_column],
            "position": signal[position_column],
        },
        index=signal.index,
    )


def safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    """安全计算相关系数，样本不足或常数序列时返回 NaN。"""
    aligned = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 3:
        return np.nan
    if aligned.iloc[:, 0].std(ddof=0) == 0 or aligned.iloc[:, 1].std(ddof=0) == 0:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method))


def calculate_multiclass_classification_metrics(
    target: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    """计算三分类 Balanced Accuracy、MCC 及准确率 Wilson 区间。"""
    aligned = pd.DataFrame({"target": target, "prediction": prediction}).dropna()
    if aligned.empty:
        return {
            "BalancedAccuracy": np.nan,
            "MCC": np.nan,
            "准确率Wilson下限": np.nan,
            "准确率Wilson上限": np.nan,
        }

    classes = [-1.0, 0.0, 1.0]
    confusion = pd.crosstab(aligned["target"], aligned["prediction"]).reindex(
        index=classes,
        columns=classes,
        fill_value=0,
    )
    matrix = confusion.to_numpy(dtype="float64")
    true_totals = matrix.sum(axis=1)
    predicted_totals = matrix.sum(axis=0)
    sample_count = float(matrix.sum())
    correct_count = float(np.trace(matrix))

    recalls = np.divide(
        np.diag(matrix),
        true_totals,
        out=np.full(len(classes), np.nan, dtype="float64"),
        where=true_totals > 0,
    )
    balanced_accuracy = float(np.nanmean(recalls)) if np.isfinite(recalls).any() else np.nan
    numerator = correct_count * sample_count - float(np.dot(true_totals, predicted_totals))
    denominator = np.sqrt(
        (sample_count**2 - float(np.dot(predicted_totals, predicted_totals)))
        * (sample_count**2 - float(np.dot(true_totals, true_totals)))
    )
    mcc = numerator / denominator if denominator > 0 else np.nan
    lower, upper = calculate_wilson_interval(int(correct_count), int(sample_count))
    return {
        "BalancedAccuracy": balanced_accuracy,
        "MCC": float(mcc) if np.isfinite(mcc) else np.nan,
        "准确率Wilson下限": lower,
        "准确率Wilson上限": upper,
    }


def calculate_multiclass_calibration_error(
    target: pd.Series,
    probabilities: pd.DataFrame,
    bins: int,
) -> dict[str, float]:
    """计算基于最大类别置信度的多分类 ECE 和 MCE。"""
    frame = probabilities[["prob_down", "prob_flat", "prob_up"]].copy()
    frame["target"] = target.map(TARGET_TO_CLASS)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {"概率校准误差ECE": np.nan, "概率校准最大误差MCE": np.nan}

    probability = frame[["prob_down", "prob_flat", "prob_up"]].clip(0.0, 1.0)
    probability = probability.div(probability.sum(axis=1).replace(0.0, np.nan), axis=0)
    valid = probability.notna().all(axis=1)
    probability = probability.loc[valid]
    labels = frame.loc[valid, "target"].astype(int).to_numpy()
    if probability.empty:
        return {"概率校准误差ECE": np.nan, "概率校准最大误差MCE": np.nan}

    values = probability.to_numpy(dtype="float64")
    confidence = values.max(axis=1)
    correct = (values.argmax(axis=1) == labels).astype("float64")
    bin_edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, len(bin_edges) - 2)
    weighted_error = 0.0
    max_error = 0.0
    for bin_id in range(len(bin_edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        weighted_error += float(mask.mean()) * error
        max_error = max(max_error, error)
    return {
        "概率校准误差ECE": weighted_error,
        "概率校准最大误差MCE": max_error,
    }


def calculate_block_bootstrap_statistics(
    strategy_returns: pd.Series,
    annual_periods: int,
    config: BacktestConfig,
) -> dict[str, float]:
    """使用循环移动区块自助法估计收益和标准 Sharpe 的不确定性。"""
    if not bool(getattr(config, "statistical_enable_block_bootstrap", True)):
        return {}
    values = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna().to_numpy("float64")
    sample_count = len(values)
    bootstrap_samples = max(1, int(getattr(config, "statistical_bootstrap_samples", 500) or 500))
    if sample_count < 8 or bootstrap_samples < 20:
        return {
            "BlockBootstrap有效样本数": float(sample_count),
            "BlockBootstrap重复次数": float(bootstrap_samples),
            "标准夏普置信下限": np.nan,
            "标准夏普置信上限": np.nan,
            "年化算术收益置信下限": np.nan,
            "年化算术收益置信上限": np.nan,
            "平均收益为零双侧P值": np.nan,
        }

    configured_block = int(getattr(config, "statistical_bootstrap_block_size", 0) or 0)
    block_size = configured_block if configured_block > 0 else int(round(np.sqrt(sample_count)))
    block_size = min(sample_count, max(1, block_size))
    block_count = int(np.ceil(sample_count / block_size))
    confidence_level = min(
        0.999,
        max(0.50, float(getattr(config, "statistical_confidence_level", 0.95) or 0.95)),
    )
    alpha = (1.0 - confidence_level) / 2.0
    rng = np.random.default_rng(int(getattr(config, "statistical_bootstrap_seed", 0) or 0))
    offsets = np.arange(block_size, dtype="int64")
    centered = values - values.mean()
    bootstrap_mean = np.empty(bootstrap_samples, dtype="float64")
    bootstrap_sharpe = np.full(bootstrap_samples, np.nan, dtype="float64")
    null_mean = np.empty(bootstrap_samples, dtype="float64")

    for sample_id in range(bootstrap_samples):
        starts = rng.integers(0, sample_count, size=block_count)
        indices = ((starts[:, None] + offsets) % sample_count).reshape(-1)[:sample_count]
        sample = values[indices]
        bootstrap_mean[sample_id] = sample.mean()
        sample_volatility = sample.std(ddof=1)
        if sample_volatility > 0:
            bootstrap_sharpe[sample_id] = (
                bootstrap_mean[sample_id] / sample_volatility * np.sqrt(annual_periods)
            )
        null_mean[sample_id] = centered[indices].mean()

    valid_sharpe = bootstrap_sharpe[np.isfinite(bootstrap_sharpe)]
    observed_mean = float(values.mean())
    p_value = (1.0 + float((np.abs(null_mean) >= abs(observed_mean)).sum())) / (
        bootstrap_samples + 1.0
    )
    return {
        "BlockBootstrap有效样本数": float(sample_count),
        "BlockBootstrap重复次数": float(bootstrap_samples),
        "BlockBootstrap区块长度": float(block_size),
        "标准夏普置信下限": float(np.quantile(valid_sharpe, alpha))
        if len(valid_sharpe)
        else np.nan,
        "标准夏普置信上限": float(np.quantile(valid_sharpe, 1.0 - alpha))
        if len(valid_sharpe)
        else np.nan,
        "年化算术收益置信下限": float(np.quantile(bootstrap_mean, alpha) * annual_periods),
        "年化算术收益置信上限": float(
            np.quantile(bootstrap_mean, 1.0 - alpha) * annual_periods
        ),
        "平均收益为零双侧P值": p_value,
        "平均收益为正Bootstrap置信度": float((bootstrap_mean > 0.0).mean()),
    }


def train_xgboost_classifier(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
    validation_groups: pd.Series | None = None,
) -> Any:
    """训练单个 XGBoost 三分类模型。

    输入标签为 -1/0/1，内部映射成 XGBoost 需要的 0/1/2 类别。
    模型参数全部来自 BacktestConfig，便于统一调参和复现实验。
    """
    import xgboost as xgb

    validation_matrix = None
    validation_size = 0
    if bool(getattr(config, "xgboost_use_validation_early_stopping", False)):
        validation_ratio = min(
            0.40,
            max(0.05, float(getattr(config, "xgboost_validation_ratio", 0.20) or 0.20)),
        )
        validation_size = int(len(train_features) * validation_ratio)
        validation_size = min(validation_size, max(0, len(train_features) - 50))

    if validation_size > 0:
        validation_start = len(train_features) - validation_size
        if validation_groups is not None:
            aligned_groups = validation_groups.reindex(train_features.index)
            boundary_group = aligned_groups.iloc[validation_start]
            group_positions = np.flatnonzero(aligned_groups.eq(boundary_group).to_numpy())
            if len(group_positions):
                validation_start = int(group_positions[0])
        fit_features = train_features.iloc[:validation_start]
        fit_target = train_target.reindex(fit_features.index)
        fit_weight = sample_weight.reindex(fit_features.index).fillna(1.0) if sample_weight is not None else None
        validation_features = train_features.iloc[validation_start:]
        validation_target = train_target.reindex(validation_features.index)
        validation_weight = (
            sample_weight.reindex(validation_features.index).fillna(1.0)
            if sample_weight is not None
            else None
        )
        if fit_target.nunique() < 2 or validation_target.dropna().empty:
            fit_features = train_features
            fit_target = train_target
            fit_weight = sample_weight
            validation_features = None
            validation_target = None
            validation_weight = None
    else:
        fit_features = train_features
        fit_target = train_target
        fit_weight = sample_weight
        validation_features = None
        validation_target = None
        validation_weight = None

    train_label = fit_target.map(TARGET_TO_CLASS)
    train_matrix = xgb.DMatrix(
        fit_features[feature_columns],
        label=train_label,
        weight=None if fit_weight is None else fit_weight.reindex(fit_features.index).fillna(1.0),
        feature_names=feature_columns,
    )
    if validation_features is not None and validation_target is not None:
        validation_matrix = xgb.DMatrix(
            validation_features[feature_columns],
            label=validation_target.map(TARGET_TO_CLASS),
            weight=(
                None
                if validation_weight is None
                else validation_weight.reindex(validation_features.index).fillna(1.0)
            ),
            feature_names=feature_columns,
        )

    evals = [(train_matrix, "train")]
    train_kwargs: dict[str, Any] = {"verbose_eval": False}
    early_stopping_rounds = int(getattr(config, "xgboost_early_stopping_rounds", 0) or 0)
    if validation_matrix is not None and early_stopping_rounds > 0:
        evals.append((validation_matrix, "validation"))
        train_kwargs["early_stopping_rounds"] = early_stopping_rounds

    return xgb.train(
        params={
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": str(getattr(config, "xgboost_tree_method", "hist") or "hist"),
            "max_depth": int(config.xgboost_max_depth),
            "eta": float(config.xgboost_learning_rate),
            "subsample": float(config.xgboost_subsample),
            "colsample_bytree": float(config.xgboost_colsample_bytree),
            "min_child_weight": float(config.xgboost_min_child_weight),
            "gamma": float(config.xgboost_gamma),
            "lambda": float(config.xgboost_reg_lambda),
            "alpha": float(config.xgboost_reg_alpha),
            "seed": int(config.xgboost_random_state),
            "nthread": int(getattr(config, "xgboost_nthread", -1) or -1),
            "verbosity": 0,
        },
        dtrain=train_matrix,
        evals=evals,
        num_boost_round=int(config.xgboost_n_estimators),
        **train_kwargs,
    )


def predict_xgboost_probability(model: Any, matrix: Any) -> np.ndarray:
    """在可用时使用早停得到的最佳轮次预测概率。"""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None and best_iteration >= 0:
        return model.predict(matrix, iteration_range=(0, int(best_iteration) + 1))
    return model.predict(matrix)


def get_enabled_composite_models(config: BacktestConfig) -> list[str]:
    """读取需要滚动对比的综合模型列表，并做轻量规范化。"""
    raw_models = getattr(config, "composite_model_names", ["xgboost"]) or ["xgboost"]
    alias = {
        "logistic": "logistic_regression",
        "lr": "logistic_regression",
        "elastic_net": "elastic_net_logistic",
        "elasticnet": "elastic_net_logistic",
        "enet": "elastic_net_logistic",
        "hist_gb": "hist_gradient_boosting",
        "histgb": "hist_gradient_boosting",
        "rf": "random_forest",
        "et": "extra_trees",
    }
    models: list[str] = []
    for model_name in raw_models:
        normalized = alias.get(str(model_name).strip().lower(), str(model_name).strip().lower())
        if normalized and normalized not in models:
            models.append(normalized)
    return models or ["xgboost"]


def train_sklearn_classifier(
    model_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
) -> Any:
    """训练 sklearn 三分类模型，用作 XGBoost 之外的模型对照。"""
    train_label = train_target.map(TARGET_TO_CLASS).astype(int)
    weight = None if sample_weight is None else sample_weight.reindex(train_features.index).fillna(1.0)

    if model_name == "logistic_regression":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(
                C=float(getattr(config, "composite_logistic_c", 1.0) or 1.0),
                max_iter=int(getattr(config, "composite_logistic_max_iter", 1000) or 1000),
                multi_class="auto",
                random_state=int(config.xgboost_random_state),
            ),
        )
        fit_kwargs = {"logisticregression__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "elastic_net_logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(
                C=float(getattr(config, "composite_elastic_net_c", 0.10) or 0.10),
                penalty="elasticnet",
                solver="saga",
                l1_ratio=float(
                    getattr(config, "composite_elastic_net_l1_ratio", 0.50) or 0.50
                ),
                max_iter=int(getattr(config, "composite_logistic_max_iter", 1000) or 1000),
                multi_class="auto",
                random_state=int(config.xgboost_random_state),
            ),
        )
        fit_kwargs = {"logisticregression__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            HistGradientBoostingClassifier(
                learning_rate=float(
                    getattr(config, "composite_hist_learning_rate", 0.05) or 0.05
                ),
                max_iter=int(getattr(config, "composite_hist_max_iter", 120) or 120),
                max_leaf_nodes=int(
                    getattr(config, "composite_hist_max_leaf_nodes", 15) or 15
                ),
                min_samples_leaf=int(
                    getattr(config, "composite_hist_min_samples_leaf", 20) or 20
                ),
                l2_regularization=float(
                    getattr(config, "composite_hist_l2_regularization", 2.0) or 2.0
                ),
                early_stopping=True,
                validation_fraction=0.20,
                n_iter_no_change=15,
                random_state=int(config.xgboost_random_state),
            ),
        )
        fit_kwargs = (
            {"histgradientboostingclassifier__sample_weight": weight}
            if weight is not None
            else {}
        )
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            RandomForestClassifier(
                n_estimators=int(getattr(config, "composite_sklearn_n_estimators", 120) or 120),
                max_depth=int(config.xgboost_max_depth) if int(config.xgboost_max_depth) > 0 else None,
                min_samples_leaf=max(1, int(getattr(config, "xgboost_min_child_weight", 1) or 1)),
                random_state=int(config.xgboost_random_state),
                n_jobs=int(getattr(config, "composite_sklearn_n_jobs", -1) or -1),
                class_weight=None,
            ),
        )
        fit_kwargs = {"randomforestclassifier__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            ExtraTreesClassifier(
                n_estimators=int(getattr(config, "composite_sklearn_n_estimators", 120) or 120),
                max_depth=int(config.xgboost_max_depth) if int(config.xgboost_max_depth) > 0 else None,
                min_samples_leaf=max(1, int(getattr(config, "xgboost_min_child_weight", 1) or 1)),
                random_state=int(config.xgboost_random_state),
                n_jobs=int(getattr(config, "composite_sklearn_n_jobs", -1) or -1),
                class_weight=None,
            ),
        )
        fit_kwargs = {"extratreesclassifier__sample_weight": weight} if weight is not None else {}
        model.fit(train_features[feature_columns], train_label, **fit_kwargs)
        return model

    raise ValueError(
        "未知综合模型: "
        f"{model_name}。可选: xgboost, logistic_regression, elastic_net_logistic, "
        "hist_gradient_boosting, random_forest, extra_trees。"
    )


def train_composite_classifier(
    model_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    feature_columns: list[str],
    config: BacktestConfig,
    sample_weight: pd.Series | None = None,
    validation_groups: pd.Series | None = None,
) -> Any:
    """按模型名称训练综合三分类模型。"""
    if model_name == "xgboost":
        return train_xgboost_classifier(
            train_features,
            train_target,
            feature_columns,
            config,
            sample_weight=sample_weight,
            validation_groups=validation_groups,
        )
    return train_sklearn_classifier(
        model_name,
        train_features,
        train_target,
        feature_columns,
        config,
        sample_weight=sample_weight,
    )


def predict_composite_probability(
    model_name: str,
    model: Any,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """统一输出三分类概率，列顺序固定为 down/flat/up。"""
    if model_name == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(features[feature_columns], feature_names=feature_columns)
        return predict_xgboost_probability(model, matrix)

    raw_probability = model.predict_proba(features[feature_columns])
    probability = np.zeros((len(features), 3), dtype="float64")
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)
    if classes is None:
        raise ValueError(f"{model_name} 没有 classes_，无法对齐三分类概率。")
    for source_column, class_id in enumerate(classes):
        if 0 <= int(class_id) <= 2:
            probability[:, int(class_id)] = raw_probability[:, source_column]
    row_sum = probability.sum(axis=1)
    empty_rows = row_sum <= 0
    if empty_rows.any():
        probability[empty_rows, 1] = 1.0
    return probability


def get_model_feature_importance(
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> pd.Series:
    """提取不同模型的特征重要性，无法提取时使用等权兜底。"""
    if model_name == "xgboost":
        return pd.Series(model.get_score(importance_type="gain"), dtype="float64")

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=feature_columns, dtype="float64")
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype="float64")
        importance = np.abs(coef).mean(axis=0)
        return pd.Series(importance, index=feature_columns, dtype="float64")
    return pd.Series(1.0, index=feature_columns, dtype="float64")


def build_training_sample_weights(
    train_target: pd.Series,
    config: BacktestConfig,
    timestamps: pd.Series | None = None,
) -> pd.Series:
    """构造逐样本权重，同时支持类别权重和时间衰减权重。"""
    neutral_weight = max(
        0.0,
        float(getattr(config, "xgboost_train_neutral_class_weight", 1.0) or 1.0),
    )
    nonzero_weight = max(
        0.0,
        float(getattr(config, "xgboost_train_nonzero_class_weight", 1.0) or 1.0),
    )
    weights = pd.Series(nonzero_weight, index=train_target.index, dtype="float64")
    weights[train_target == 0] = neutral_weight
    weights[train_target.isna()] = 0.0

    if bool(getattr(config, "xgboost_train_use_time_decay_weight", False)) and len(weights) > 1:
        half_life = float(getattr(config, "xgboost_train_time_decay_half_life", 0) or 0)
        if half_life > 0:
            min_time_weight = max(
                0.0,
                min(1.0, float(getattr(config, "xgboost_train_time_decay_min_weight", 0.0) or 0.0)),
            )
            if timestamps is not None:
                aligned_timestamps = pd.to_datetime(
                    timestamps.reindex(train_target.index),
                    errors="coerce",
                )
                ordered_times = pd.Index(aligned_timestamps.dropna().unique()).sort_values()
                time_rank = pd.Series(
                    np.arange(len(ordered_times), dtype="float64"),
                    index=ordered_times,
                )
                sample_rank = aligned_timestamps.map(time_rank)
                fallback_rank = pd.Series(
                    np.arange(len(weights), dtype="float64"),
                    index=weights.index,
                )
                sample_rank = sample_rank.fillna(fallback_rank)
                age_from_window_end = float(sample_rank.max()) - sample_rank.to_numpy(dtype="float64")
            else:
                age_from_window_end = len(weights) - 1 - np.arange(len(weights), dtype="float64")
            time_weight = np.power(0.5, age_from_window_end / half_life)
            time_weight = np.maximum(time_weight, min_time_weight)
            if bool(getattr(config, "xgboost_train_time_decay_normalize", True)):
                mean_weight = float(np.nanmean(time_weight))
                if np.isfinite(mean_weight) and mean_weight > 0:
                    time_weight = time_weight / mean_weight
            weights *= pd.Series(time_weight, index=weights.index, dtype="float64")

    return weights


def build_xgboost_rolling_signal(
    data: pd.DataFrame,
    factors: pd.DataFrame,
    selected_factors: list[str],
    config: BacktestConfig,
    predict_index: pd.Index,
    model_name: str = "xgboost",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame | None]:
    """执行指定模型的滚动训练、滚动预测并生成交易信号。

    每个预测时点只使用它之前的 train_window 根K线训练模型。
    每隔 xgboost_retrain_every 根K线重新训练一次，中间复用上一次模型。
    如果启用 walk-forward feature selection，则每次重新训练前都会在当前训练窗口内重新筛因子。

    返回：
    - signal：包含预测方向、概率、交易信号、持仓和基准投票信号。
    - feature_importance：滚动模型累计特征重要性。
    - features：最终构造的特征矩阵。
    - selection_summary：滚动选因明细，未启用时可能为 None。
    """
    model_name = str(model_name).strip().lower()
    if model_name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise ImportError("未安装 xgboost，请先安装: python -m pip install xgboost") from exc

    signal_features = build_factor_signal_features(factors, selected_factors, config)
    market_state = build_market_state_filter(data, config)
    use_walk_forward_selection = should_use_walk_forward_selection(config)
    if use_walk_forward_selection:
        features = pd.DataFrame(index=factors.index)
        feature_columns: list[str] = []
    else:
        features = build_features_for_factors(factors, signal_features, selected_factors, config)
        feature_columns = features.columns.tolist()
    target = calculate_next_bar_direction(data, factors.index, config)
    target_outcomes = calculate_future_target_outcomes(data, factors.index, config)
    target_economic_return = target_outcomes["future_horizon_net_return"]
    predict_index = pd.Index(predict_index).intersection(factors.index)
    predict_positions = factors.index.get_indexer(predict_index)
    predict_positions = predict_positions[predict_positions >= 0]

    predicted_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    vote_score = pd.Series(np.nan, index=factors.index, dtype="float64")
    xgboost_signal_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    vote_signal_direction = pd.Series(np.nan, index=factors.index, dtype="float64")
    calibrated_min_edge = pd.Series(np.nan, index=factors.index, dtype="float64")
    calibrated_min_probability = pd.Series(np.nan, index=factors.index, dtype="float64")
    probabilities = pd.DataFrame(
        np.nan,
        index=factors.index,
        columns=["prob_down", "prob_flat", "prob_up"],
        dtype="float64",
    )
    importance_sum = pd.Series(dtype="float64")
    model_count = 0
    model = None
    active_feature_columns: list[str] = []
    active_selected_factors = list(selected_factors)
    active_xgboost_signal_direction = 1.0
    active_vote_signal_direction = 1.0
    active_min_edge = float(config.xgboost_trade_min_edge)
    active_min_probability = float(config.xgboost_trade_min_probability)
    selection_rows = []

    train_window = int(config.xgboost_train_window)
    min_train_samples = int(config.xgboost_min_train_samples)
    retrain_every = max(1, int(config.xgboost_retrain_every))
    target_horizon = get_xgboost_target_horizon(config)
    progress_every = max(
        1,
        int(getattr(config, "xgboost_progress_every", 25) or 25),
    )

    # 对每个样本外时点滚动预测；position 永远是当前预测点。
    # 多周期目标需要额外剔除训练窗口尾部尚未完全落地的标签，避免未来函数。
    for step, position in enumerate(predict_positions):
        if step == 0 or step + 1 == len(predict_positions) or (step + 1) % progress_every == 0:
            print(
                f"{model_name}滚动预测进度: "
                f"{step + 1}/{len(predict_positions)} 个K线时点；"
                f"active候选因子={len(selected_factors)}；"
                f"本轮最多选因={config.xgboost_best_top_n}",
                end="\r",
            )
        train_start = max(0, position - train_window)
        train_end = position - target_horizon + 1
        if train_end <= train_start:
            continue
        train_target = target.iloc[train_start:train_end]

        # 首次预测或到达重训间隔时，使用当前历史窗口重新训练模型。
        if model is None or step % retrain_every == 0:
            if use_walk_forward_selection:
                active_selected_factors, window_selection = select_factors_in_window(
                    signal_features.iloc[train_start:train_end],
                    train_target,
                    selected_factors,
                    config,
                )
                if len(active_selected_factors) == 0:
                    continue
                features = build_features_for_factors(
                    factors,
                    signal_features,
                    active_selected_factors,
                    config,
                )
                feature_columns = features.columns.tolist()
                active_feature_columns = feature_columns

                if not window_selection.empty:
                    selected_labels = ",".join(active_selected_factors)
                    window_selection = window_selection.copy()
                    window_selection.insert(0, "预测时间", factors.index[position])
                    window_selection.insert(1, "训练开始", factors.index[train_start])
                    window_selection.insert(2, "训练结束", factors.index[train_end - 1])
                    window_selection["本轮入选因子"] = selected_labels
                    selection_rows.extend(window_selection.to_dict("records"))
            else:
                active_feature_columns = feature_columns

            # 合并逐列特征产生的内部碎片，避免 assign/fit 时反复复制内存。
            train_features = features.iloc[train_start:train_end].copy()
            train_frame = train_features.assign(__target__=train_target).dropna(subset=["__target__"])
            if bool(getattr(config, "xgboost_train_use_market_filters", False)):
                train_allowed_mask = (
                    market_state.loc[train_frame.index, "trade_allowed"].fillna(0.0) > 0
                )
                train_frame = train_frame.loc[train_allowed_mask]
            min_directional_samples = max(
                0,
                int(getattr(config, "xgboost_train_min_directional_samples", 0) or 0),
            )
            directional_sample_count = int((train_frame["__target__"] != 0).sum())
            if directional_sample_count < min_directional_samples:
                continue
            if len(train_frame) < min_train_samples or train_frame["__target__"].nunique() < 2:
                continue
            train_trade_allowed = market_state.loc[train_frame.index, "trade_allowed"].fillna(0.0) > 0
            train_sample_weight = build_training_sample_weights(train_frame["__target__"], config)

            model = train_composite_classifier(
                model_name,
                train_frame[active_feature_columns],
                train_frame["__target__"],
                active_feature_columns,
                config,
                sample_weight=train_sample_weight,
            )
            # 用训练窗口内的预测表现判断概率信号是否需要整体反向。
            if (
                config.xgboost_auto_calibrate_signal_direction
                or config.xgboost_auto_calibrate_trade_thresholds
            ):
                train_probability = predict_composite_probability(
                    model_name,
                    model,
                    train_frame[active_feature_columns],
                    active_feature_columns,
                )
                train_probability = pd.DataFrame(
                    train_probability,
                    index=train_frame.index,
                    columns=["prob_down", "prob_flat", "prob_up"],
                )
                active_min_edge, active_min_probability = calibrate_trade_thresholds_on_training(
                    train_probability,
                    target_economic_return,
                    train_trade_allowed,
                    config,
                )
                train_raw_signal = probabilities_to_trade_signal(
                    train_probability,
                    config,
                    min_edge=active_min_edge,
                    min_probability=active_min_probability,
                )
                train_raw_signal = train_raw_signal.where(train_trade_allowed, 0.0)
                if config.xgboost_auto_calibrate_signal_direction:
                    active_xgboost_signal_direction = choose_signal_direction_on_training(
                        train_raw_signal,
                        target_economic_return,
                    )
                else:
                    active_xgboost_signal_direction = 1.0
            else:
                active_min_edge = float(config.xgboost_trade_min_edge)
                active_min_probability = float(config.xgboost_trade_min_probability)
                active_xgboost_signal_direction = 1.0

            # 等权投票基准也可单独做方向校准，便于与 XGBoost 策略公平对照。
            if config.benchmark_vote_auto_calibrate_direction and active_selected_factors:
                train_vote_score = signal_features.loc[train_frame.index, active_selected_factors].mean(axis=1)
                train_vote_signal = vote_score_to_trade_signal(train_vote_score, config)
                train_vote_signal = train_vote_signal.where(train_trade_allowed, 0.0)
                active_vote_signal_direction = choose_signal_direction_on_training(
                    train_vote_signal,
                    target_economic_return,
                )
            else:
                active_vote_signal_direction = 1.0

            model_importance = get_model_feature_importance(model_name, model, active_feature_columns)
            importance_sum = importance_sum.add(pd.Series(model_importance), fill_value=0.0)
            model_count += 1
        elif use_walk_forward_selection and not active_feature_columns:
            continue

        # 当前时点只做一次预测，预测结果会在回测里 shift 成下一根K线实际持仓。
        current_features = features.iloc[[position]]
        class_probability = predict_composite_probability(
            model_name,
            model,
            current_features[active_feature_columns],
            active_feature_columns,
        )[0]
        probabilities.iloc[position] = class_probability
        predicted_class = int(np.argmax(class_probability))
        predicted_direction.iloc[position] = CLASS_TO_TARGET[predicted_class]
        xgboost_signal_direction.iloc[position] = active_xgboost_signal_direction
        calibrated_min_edge.iloc[position] = active_min_edge
        calibrated_min_probability.iloc[position] = active_min_probability

        if active_selected_factors:
            vote_score.iloc[position] = (
                active_vote_signal_direction
                * signal_features.iloc[position][active_selected_factors].mean()
            )
            vote_signal_direction.iloc[position] = active_vote_signal_direction

    if predicted_direction.reindex(predict_index).dropna().empty:
        raise ValueError(
            f"{model_name} 滚动窗口没有生成有效预测，请调小 xgboost_min_train_samples "
            "或 xgboost_train_window。"
        )

    if model_count > 0 and importance_sum.sum() > 0:
        feature_importance = importance_sum / importance_sum.sum()
    else:
        fallback_columns = active_feature_columns or feature_columns
        feature_importance = pd.Series(1.0 / len(fallback_columns), index=fallback_columns)

    signal = pd.DataFrame(index=factors.index)
    signal["xgboost_predicted_direction"] = predicted_direction
    signal["target_direction"] = target
    signal = signal.join(target_outcomes)
    target_bounds = calculate_xgboost_target_bounds(data, factors.index, config)
    signal = signal.join(target_bounds)
    signal["target_neutral_threshold"] = (
        target_bounds["target_upper_threshold"].abs()
        .combine(target_bounds["target_lower_threshold"].abs(), max)
        .astype("float64")
    )
    signal["target_label_mode"] = get_xgboost_target_label_mode(config)
    signal = signal.join(probabilities)
    signal["prob_edge"] = signal["prob_up"] - signal["prob_down"]
    signal["directional_probability"] = signal[["prob_up", "prob_down"]].max(axis=1)
    signal["calibrated_min_edge"] = calibrated_min_edge
    signal["calibrated_min_probability"] = calibrated_min_probability
    signal["xgboost_signal_direction"] = xgboost_signal_direction
    signal["calibrated_prob_edge"] = signal["prob_edge"] * signal["xgboost_signal_direction"]
    signal["composite_score"] = signal["calibrated_prob_edge"]
    signal = signal.join(market_state)
    confidence_filter = build_confidence_rank_filter(signal["calibrated_prob_edge"], config)
    signal = signal.join(confidence_filter)
    signal["trade_allowed"] = (
        signal["trade_allowed"].fillna(0.0) > 0
    ) & (signal["confidence_trade_allowed"].fillna(0.0) > 0)
    signal["trade_allowed"] = signal["trade_allowed"].astype("float64")
    signal["position_size"] = build_dynamic_confidence_position_size(
        probabilities,
        signal["calibrated_min_edge"],
        signal["calibrated_min_probability"],
        config,
    ).fillna(0.0)
    edge_threshold = signal["calibrated_min_edge"].fillna(float(config.xgboost_trade_min_edge))
    probability_threshold = signal["calibrated_min_probability"].fillna(
        float(config.xgboost_trade_min_probability)
    )
    max_flat_probability = min(
        1.0,
        max(0.0, float(getattr(config, "xgboost_trade_max_flat_probability", 1.0) or 1.0)),
    )
    min_directional_vs_flat_edge = max(
        0.0,
        float(getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0) or 0.0),
    )
    signal["directional_vs_flat_edge"] = signal["directional_probability"] - signal["prob_flat"]
    signal["flat_trade_allowed"] = (
        (signal["prob_flat"] <= max_flat_probability)
        & (signal["directional_vs_flat_edge"] >= min_directional_vs_flat_edge)
    ).astype("float64")
    raw_signal = pd.Series(0.0, index=signal.index, dtype="float64")
    raw_signal[
        (signal["prob_edge"] >= edge_threshold)
        & (signal["directional_probability"] >= probability_threshold)
        & (signal["flat_trade_allowed"].fillna(0.0) > 0)
    ] = 1.0
    raw_signal[
        (signal["prob_edge"] <= -edge_threshold)
        & (signal["directional_probability"] >= probability_threshold)
        & (signal["flat_trade_allowed"].fillna(0.0) > 0)
    ] = -1.0
    signal["raw_signal"] = (raw_signal * signal["xgboost_signal_direction"]).fillna(0.0)
    signal["raw_signal"] = signal["raw_signal"].where(signal["trade_allowed"].fillna(0.0) > 0, 0.0)
    signal["calibrated_predicted_direction"] = (
        signal["xgboost_predicted_direction"] * signal["xgboost_signal_direction"]
    )
    signal["raw_signal_before_position_rules"] = signal["raw_signal"]
    signal["target_position_before_rules"] = signal["raw_signal"] * signal["position_size"]
    signal["target_position"] = apply_position_rules(
        signal["target_position_before_rules"],
        config,
    )
    signal["raw_signal"] = np.sign(signal["target_position"]).astype("float64")
    signal["position"] = signal["target_position"].shift(1).fillna(0.0)
    signal["benchmark_vote_score"] = vote_score
    signal["benchmark_vote_signal_direction"] = vote_signal_direction
    signal["benchmark_vote_raw_signal"] = vote_score_to_trade_signal(vote_score, config).fillna(0.0)
    signal["benchmark_vote_raw_signal"] = signal["benchmark_vote_raw_signal"].where(
        signal["trade_allowed"].fillna(0.0) > 0,
        0.0,
    )
    signal["benchmark_vote_raw_signal_before_position_rules"] = signal["benchmark_vote_raw_signal"]
    signal["benchmark_vote_target_position_before_rules"] = (
        signal["benchmark_vote_raw_signal"] * signal["position_size"]
    )
    signal["benchmark_vote_target_position"] = apply_position_rules(
        signal["benchmark_vote_target_position_before_rules"],
        config,
    )
    signal["benchmark_vote_raw_signal"] = np.sign(signal["benchmark_vote_target_position"]).astype("float64")
    signal["benchmark_vote_position"] = signal["benchmark_vote_target_position"].shift(1).fillna(0.0)

    extra_columns = []
    for factor_name in selected_factors:
        if factor_name in active_selected_factors:
            extra_columns.append(factors[[factor_name]])
            extra_columns.append(signal_features[[factor_name]].rename(columns={factor_name: f"{factor_name}_signal"}))
    if not use_walk_forward_selection:
        feature_extra_columns = []
        for feature_name in feature_columns:
            if feature_name not in signal.columns:
                feature_extra_columns.append(feature_name)
        if feature_extra_columns:
            extra_columns.append(
                features[feature_extra_columns].rename(
                    columns={feature_name: f"feature_{feature_name}" for feature_name in feature_extra_columns}
                )
            )
    if extra_columns:
        signal = pd.concat([signal, *extra_columns], axis=1).copy()

    selection_summary = pd.DataFrame(selection_rows) if selection_rows else None
    return signal, feature_importance, features, selection_summary


def _normalize_capped_model_weights(values: pd.Series, max_weight: float) -> pd.Series:
    """把非负模型得分归一化，并用水位法落实单模型权重上限。"""
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    positive = values[values > 0]
    result = pd.Series(0.0, index=values.index, dtype="float64")
    if positive.empty:
        return result

    effective_cap = max(float(max_weight), 1.0 / len(positive))
    remaining = list(positive.index)
    remaining_mass = 1.0
    while remaining:
        scores = positive.loc[remaining]
        proposed = (
            pd.Series(remaining_mass / len(remaining), index=remaining)
            if float(scores.sum()) <= 0
            else scores / float(scores.sum()) * remaining_mass
        )
        capped = proposed[proposed > effective_cap + 1e-12]
        if capped.empty:
            result.loc[remaining] = proposed
            break
        result.loc[capped.index] = effective_cap
        remaining_mass -= effective_cap * len(capped)
        remaining = [name for name in remaining if name not in capped.index]
    return result


def build_historical_probability_ensemble(
    model_signals: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用严格滞后的滚动预测质量融合基础模型概率。

    时点 t 的权重只使用截至 t-horizon 已经成熟的标签，避免多周期目标把
    当前或未来收益泄露到模型权重。基础模型的训练窗口方向校准会先作用到
    down/up 概率，再进入融合。
    """
    requested = {
        str(name).strip().lower()
        for name in (getattr(config, "composite_ensemble_model_names", []) or [])
    }
    selected = {
        name: signal
        for name, signal in model_signals.items()
        if not requested or name.lower() in requested
    }
    min_models = max(2, int(getattr(config, "composite_ensemble_min_models", 2) or 2))
    if len(selected) < min_models:
        raise ValueError(f"概率融合至少需要 {min_models} 个成功基础模型，当前只有 {len(selected)} 个。")

    first_signal = next(iter(selected.values()))
    index = first_signal.index
    for signal in selected.values():
        index = index.intersection(signal.index, sort=False)
    if index.empty:
        raise ValueError("基础模型没有共同的滚动预测时点，无法构建概率融合。")

    target = first_signal["target_direction"].reindex(index).astype("float64")
    return_column = (
        "future_horizon_net_return"
        if "future_horizon_net_return" in first_signal.columns
        else "future_horizon_return"
    )
    future_return = first_signal[return_column].reindex(index).astype("float64")
    horizon = get_xgboost_target_horizon(config)
    window = max(2, int(getattr(config, "composite_ensemble_weight_window", 480) or 480))
    min_history = max(2, int(getattr(config, "composite_ensemble_min_history", 120) or 120))
    min_accuracy = float(
        getattr(config, "composite_ensemble_min_directional_accuracy", 0.48) or 0.48
    )
    min_corr = float(getattr(config, "composite_ensemble_min_edge_return_corr", -0.02))
    max_weight = float(getattr(config, "composite_ensemble_max_model_weight", 0.60) or 0.60)
    shrinkage = min(
        1.0,
        max(0.0, float(getattr(config, "composite_ensemble_equal_weight_shrinkage", 0.35) or 0.35)),
    )

    adjusted_probabilities: dict[str, pd.DataFrame] = {}
    scores = pd.DataFrame(0.0, index=index, columns=selected, dtype="float64")
    eligible = pd.DataFrame(False, index=index, columns=selected)
    accuracies = pd.DataFrame(np.nan, index=index, columns=selected, dtype="float64")
    correlations = pd.DataFrame(np.nan, index=index, columns=selected, dtype="float64")
    history_counts = pd.DataFrame(0.0, index=index, columns=selected, dtype="float64")

    for model_name, signal in selected.items():
        probability = signal[["prob_down", "prob_flat", "prob_up"]].reindex(index).astype("float64")
        direction = signal["xgboost_signal_direction"].reindex(index).fillna(1.0)
        reversed_mask = direction < 0
        adjusted = probability.copy()
        adjusted.loc[reversed_mask, "prob_down"] = probability.loc[reversed_mask, "prob_up"]
        adjusted.loc[reversed_mask, "prob_up"] = probability.loc[reversed_mask, "prob_down"]
        adjusted_probabilities[model_name] = adjusted

        edge = adjusted["prob_up"] - adjusted["prob_down"]
        directional = target.ne(0) & edge.notna()
        correct = pd.Series(np.nan, index=index, dtype="float64")
        correct.loc[directional] = (
            np.sign(edge.loc[directional]) == np.sign(target.loc[directional])
        ).astype("float64")
        count = directional.astype("float64").rolling(window, min_periods=1).sum().shift(horizon)
        accuracy = correct.rolling(window, min_periods=1).mean().shift(horizon)
        correlation = edge.rolling(
            window,
            min_periods=max(5, min_history // 2),
        ).corr(future_return).shift(horizon)
        is_eligible = (
            count.ge(min_history)
            & accuracy.ge(min_accuracy)
            & correlation.ge(min_corr)
            & adjusted.notna().all(axis=1)
        )
        performance_score = (
            (accuracy - min_accuracy).clip(lower=0.0)
            + (correlation - min_corr).clip(lower=0.0)
        ).where(is_eligible, 0.0)
        scores[model_name] = performance_score.fillna(0.0)
        eligible[model_name] = is_eligible.fillna(False)
        accuracies[model_name] = accuracy
        correlations[model_name] = correlation
        history_counts[model_name] = count.fillna(0.0)

    weights = pd.DataFrame(0.0, index=index, columns=selected, dtype="float64")
    for timestamp in index:
        eligible_names = eligible.columns[eligible.loc[timestamp]].tolist()
        if len(eligible_names) < min_models:
            continue
        dynamic = _normalize_capped_model_weights(
            scores.loc[timestamp, eligible_names],
            max_weight,
        )
        if float(dynamic.sum()) <= 0:
            dynamic[:] = 1.0 / len(dynamic)
        equal = pd.Series(1.0 / len(eligible_names), index=eligible_names)
        blended = (1.0 - shrinkage) * dynamic + shrinkage * equal
        weights.loc[timestamp, eligible_names] = _normalize_capped_model_weights(
            blended,
            max_weight,
        )

    ensemble_probability = pd.DataFrame(
        0.0,
        index=index,
        columns=["prob_down", "prob_flat", "prob_up"],
    )
    for model_name, probability in adjusted_probabilities.items():
        ensemble_probability = ensemble_probability.add(
            probability.mul(weights[model_name], axis=0),
            fill_value=0.0,
        )
    valid_weight = weights.sum(axis=1).gt(0)
    ensemble_probability.loc[~valid_weight] = np.nan
    probability_sum = ensemble_probability.sum(axis=1).replace(0.0, np.nan)
    ensemble_probability = ensemble_probability.div(probability_sum, axis=0)

    signal = first_signal.reindex(index).copy()
    signal[["prob_down", "prob_flat", "prob_up"]] = ensemble_probability
    signal["prob_edge"] = signal["prob_up"] - signal["prob_down"]
    signal["directional_probability"] = signal[["prob_up", "prob_down"]].max(axis=1)
    signal["xgboost_signal_direction"] = 1.0
    predicted_class = np.argmax(ensemble_probability.fillna(0.0).to_numpy(), axis=1)
    predicted_direction = pd.Series(
        [CLASS_TO_TARGET[int(value)] for value in predicted_class],
        index=index,
        dtype="float64",
    ).where(valid_weight)
    signal["xgboost_predicted_direction"] = predicted_direction
    signal["calibrated_predicted_direction"] = predicted_direction
    signal["calibrated_min_edge"] = float(config.xgboost_trade_min_edge)
    signal["calibrated_min_probability"] = float(config.xgboost_trade_min_probability)
    signal["calibrated_prob_edge"] = signal["prob_edge"]
    signal["composite_score"] = signal["calibrated_prob_edge"]

    market_state = build_market_state_filter(data, config).reindex(index)
    for column in market_state.columns:
        signal[column] = market_state[column]
    confidence_filter = build_confidence_rank_filter(signal["calibrated_prob_edge"], config)
    for column in confidence_filter.columns:
        signal[column] = confidence_filter[column]
    signal["trade_allowed"] = (
        (signal["trade_allowed"].fillna(0.0) > 0)
        & (signal["confidence_trade_allowed"].fillna(0.0) > 0)
        & valid_weight
    ).astype("float64")
    signal["position_size"] = build_dynamic_confidence_position_size(
        ensemble_probability,
        signal["calibrated_min_edge"],
        signal["calibrated_min_probability"],
        config,
    ).fillna(0.0)
    signal["directional_vs_flat_edge"] = signal["directional_probability"] - signal["prob_flat"]
    signal["flat_trade_allowed"] = (
        (signal["prob_flat"] <= float(getattr(config, "xgboost_trade_max_flat_probability", 1.0)))
        & (
            signal["directional_vs_flat_edge"]
            >= float(getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0))
        )
    ).astype("float64")
    raw_signal = probabilities_to_trade_signal(
        ensemble_probability,
        config,
        min_edge=float(config.xgboost_trade_min_edge),
        min_probability=float(config.xgboost_trade_min_probability),
    ).where(signal["flat_trade_allowed"].fillna(0.0) > 0, 0.0)
    signal["raw_signal"] = raw_signal.where(signal["trade_allowed"] > 0, 0.0).fillna(0.0)
    signal["raw_signal_before_position_rules"] = signal["raw_signal"]
    signal["target_position_before_rules"] = signal["raw_signal"] * signal["position_size"]
    signal["target_position"] = apply_position_rules(signal["target_position_before_rules"], config)
    signal["raw_signal"] = np.sign(signal["target_position"]).astype("float64")
    signal["position"] = signal["target_position"].shift(1).fillna(0.0)

    diagnostics = []
    for model_name in selected:
        model_diagnostic = pd.DataFrame(
            {
                "时间": index,
                "模型": model_name,
                "权重": weights[model_name].to_numpy(),
                "历史方向准确率": accuracies[model_name].to_numpy(),
                "历史概率收益相关性": correlations[model_name].to_numpy(),
                "成熟方向样本数": history_counts[model_name].to_numpy(),
                "是否合格": eligible[model_name].to_numpy(),
            }
        )
        diagnostics.append(model_diagnostic)
    return signal, pd.concat(diagnostics, ignore_index=True)


def save_prediction_diagnostics(backtest_df: pd.DataFrame, output_dir: Path) -> None:
    """保存独立于策略盈亏的模型预测质量诊断。"""
    required = {
        "target_direction",
        "future_horizon_return",
        "xgboost_predicted_direction",
        "calibrated_predicted_direction",
        "raw_signal",
        "calibrated_prob_edge",
    }
    if not required.issubset(backtest_df.columns):
        return

    optional_columns = [
        "trade_allowed",
        "confidence_trade_allowed",
        "confidence_rank",
        "calibrated_min_edge",
        "calibrated_min_probability",
    ]
    diagnostic_columns = list(required) + [col for col in optional_columns if col in backtest_df.columns]
    valid = backtest_df[diagnostic_columns].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_direction", "future_horizon_return", "xgboost_predicted_direction"]
    )
    if valid.empty:
        return

    raw_pred = valid["xgboost_predicted_direction"]
    calibrated_pred = valid["calibrated_predicted_direction"]
    target = valid["target_direction"]
    traded = valid["raw_signal"] != 0
    directional_target = target != 0

    diagnostics = pd.Series(
        {
            "样本数": int(len(valid)),
            "目标上涨占比": float((target > 0).mean()),
            "目标下跌占比": float((target < 0).mean()),
            "目标中性占比": float((target == 0).mean()),
            "原始三分类准确率": float((raw_pred == target).mean()),
            "校准后三分类准确率": float((calibrated_pred == target).mean()),
            "非中性目标方向准确率": float(
                (np.sign(calibrated_pred[directional_target]) == np.sign(target[directional_target])).mean()
            )
            if directional_target.any()
            else np.nan,
            "交易信号样本数": int(traded.sum()),
            "交易信号方向准确率": float(
                (np.sign(valid.loc[traded, "raw_signal"]) == np.sign(target[traded])).mean()
            )
            if traded.any()
            else np.nan,
            "可交易样本数": int((valid["trade_allowed"].fillna(0.0) > 0).sum())
            if "trade_allowed" in valid.columns
            else np.nan,
            "可交易样本方向准确率": float(
                (
                    np.sign(
                        valid.loc[valid["trade_allowed"].fillna(0.0) > 0, "calibrated_predicted_direction"]
                    )
                    == np.sign(target[valid["trade_allowed"].fillna(0.0) > 0])
                ).mean()
            )
            if "trade_allowed" in valid.columns and (valid["trade_allowed"].fillna(0.0) > 0).any()
            else np.nan,
            "高置信度样本数": int((valid["confidence_trade_allowed"].fillna(0.0) > 0).sum())
            if "confidence_trade_allowed" in valid.columns
            else np.nan,
            "概率差与未来收益相关性": float(
                valid["calibrated_prob_edge"].corr(valid["future_horizon_return"])
            ),
            "平均未来Horizon收益": float(valid["future_horizon_return"].mean()),
            "交易样本平均未来Horizon收益": float(valid.loc[traded, "future_horizon_return"].mean())
            if traded.any()
            else np.nan,
        },
        name="value",
    )
    diagnostics.to_csv(output_dir / "composite_prediction_diagnostics.csv", encoding="utf-8-sig")

    confusion = pd.crosstab(
        target,
        calibrated_pred,
        rownames=["真实方向"],
        colnames=["校准预测方向"],
        dropna=False,
    )
    confusion.to_csv(output_dir / "composite_prediction_confusion_matrix.csv", encoding="utf-8-sig")

    bins = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
    labels = [
        "<=-0.20",
        "-0.20~-0.10",
        "-0.10~-0.05",
        "-0.05~0",
        "0~0.05",
        "0.05~0.10",
        "0.10~0.20",
        ">0.20",
    ]
    valid = valid.copy()
    valid["概率差分箱"] = pd.cut(valid["calibrated_prob_edge"], bins=bins, labels=labels)
    edge_rows = []
    for bucket, bucket_df in valid.groupby("概率差分箱", observed=False):
        if bucket_df.empty:
            continue
        edge_rows.append(
            {
                "概率差分箱": bucket,
                "样本数": int(len(bucket_df)),
                "平均概率差": bucket_df["calibrated_prob_edge"].mean(),
                "平均未来Horizon收益": bucket_df["future_horizon_return"].mean(),
                "未来上涨占比": (bucket_df["future_horizon_return"] > 0).mean(),
                "未来下跌占比": (bucket_df["future_horizon_return"] < 0).mean(),
                "平均真实方向": bucket_df["target_direction"].mean(),
                "交易信号占比": (bucket_df["raw_signal"] != 0).mean(),
            }
        )
    pd.DataFrame(edge_rows).to_csv(
        output_dir / "composite_xgboost_edge_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def calculate_prediction_metrics_for_segment(
    segment_name: str,
    segment_df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, Any] | None:
    """计算不依赖交易规则和策略收益的纯预测质量指标。"""
    required = {
        "target_direction",
        "future_horizon_return",
        "xgboost_predicted_direction",
        "calibrated_predicted_direction",
        "calibrated_prob_edge",
    }
    if segment_df.empty or not required.issubset(segment_df.columns):
        return None

    optional_columns = [
        "prob_down",
        "prob_flat",
        "prob_up",
        "raw_signal",
        "xgboost_signal_direction",
        "future_horizon_net_return",
        "future_horizon_standardized_net_return",
    ]
    valid = segment_df[
        list(required) + [column for column in optional_columns if column in segment_df.columns]
    ]
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target_direction", "future_horizon_return", "xgboost_predicted_direction"]
    )
    if valid.empty:
        return None

    target = valid["target_direction"].astype("float64")
    raw_pred = valid["xgboost_predicted_direction"].astype("float64")
    calibrated_pred = valid["calibrated_predicted_direction"].astype("float64")
    future_return = valid["future_horizon_return"].astype("float64")
    prob_edge = valid["calibrated_prob_edge"].astype("float64")
    directional_target = target != 0
    predicted_directional = calibrated_pred != 0
    traded = valid.get("raw_signal", pd.Series(0.0, index=valid.index)).fillna(0.0) != 0
    raw_classification = calculate_multiclass_classification_metrics(target, raw_pred)
    calibrated_classification = calculate_multiclass_classification_metrics(
        target,
        calibrated_pred,
    )
    directional_correct = (
        np.sign(calibrated_pred[directional_target]) == np.sign(target[directional_target])
    )
    directional_lower, directional_upper = calculate_wilson_interval(
        int(directional_correct.sum()),
        int(directional_correct.count()),
    )
    future_net_return = valid.get("future_horizon_net_return", future_return).astype("float64")
    future_standardized_net_return = valid.get(
        "future_horizon_standardized_net_return",
        pd.Series(np.nan, index=valid.index),
    ).astype("float64")

    rows: dict[str, Any] = {
        "样本段": segment_name,
        "预测样本数": int(len(valid)),
        "目标上涨占比": float((target > 0).mean()),
        "目标下跌占比": float((target < 0).mean()),
        "目标中性占比": float((target == 0).mean()),
        "原始三分类准确率": float((raw_pred == target).mean()),
        "校准后三分类准确率": float((calibrated_pred == target).mean()),
        "原始三分类BalancedAccuracy": raw_classification["BalancedAccuracy"],
        "原始三分类MCC": raw_classification["MCC"],
        "校准后三分类BalancedAccuracy": calibrated_classification["BalancedAccuracy"],
        "校准后三分类MCC": calibrated_classification["MCC"],
        "校准准确率Wilson下限": calibrated_classification["准确率Wilson下限"],
        "校准准确率Wilson上限": calibrated_classification["准确率Wilson上限"],
        "方向目标样本数": int(directional_target.sum()),
        "方向目标准确率": (
            float((np.sign(calibrated_pred[directional_target]) == np.sign(target[directional_target])).mean())
            if directional_target.any()
            else np.nan
        ),
        "方向准确率Wilson下限": directional_lower,
        "方向准确率Wilson上限": directional_upper,
        "预测方向覆盖率": float(predicted_directional.mean()),
        "交易信号覆盖率": float(traded.mean()),
        "交易信号方向准确率": (
            float((np.sign(valid.loc[traded, "raw_signal"]) == np.sign(target[traded])).mean())
            if traded.any() and "raw_signal" in valid.columns
            else np.nan
        ),
        "概率差与未来收益Pearson": safe_corr(prob_edge, future_return, method="pearson"),
        "概率差与未来收益Spearman": safe_corr(prob_edge, future_return, method="spearman"),
        "概率差与未来净收益Pearson": safe_corr(
            prob_edge,
            future_net_return,
            method="pearson",
        ),
        "概率差与未来净收益Spearman": safe_corr(
            prob_edge,
            future_net_return,
            method="spearman",
        ),
        "概率差与标准化未来净收益Pearson": safe_corr(
            prob_edge,
            future_standardized_net_return,
            method="pearson",
        ),
        "概率差与标准化未来净收益Spearman": safe_corr(
            prob_edge,
            future_standardized_net_return,
            method="spearman",
        ),
        "平均绝对概率差": float(prob_edge.abs().mean()),
        "预测为多样本未来平均收益": float(future_return[calibrated_pred > 0].mean())
        if (calibrated_pred > 0).any()
        else np.nan,
        "预测为空样本未来平均收益": float(future_return[calibrated_pred < 0].mean())
        if (calibrated_pred < 0).any()
        else np.nan,
        "预测为中性样本未来平均绝对收益": float(future_return[calibrated_pred == 0].abs().mean())
        if (calibrated_pred == 0).any()
        else np.nan,
    }

    class_recalls = {}
    for class_value, class_name in [(-1.0, "下跌"), (0.0, "中性"), (1.0, "上涨")]:
        class_mask = target == class_value
        class_recalls[f"{class_name}召回率"] = (
            float((calibrated_pred[class_mask] == class_value).mean()) if class_mask.any() else np.nan
        )
    rows.update(class_recalls)

    if {"prob_down", "prob_flat", "prob_up"}.issubset(valid.columns):
        class_label = target.map(TARGET_TO_CLASS).astype("Int64")
        prob_matrix = valid[["prob_down", "prob_flat", "prob_up"]].astype("float64").copy()
        if "xgboost_signal_direction" in valid.columns:
            reverse_mask = valid["xgboost_signal_direction"].fillna(1.0) < 0
            original_down = prob_matrix.loc[reverse_mask, "prob_down"].copy()
            prob_matrix.loc[reverse_mask, "prob_down"] = prob_matrix.loc[reverse_mask, "prob_up"]
            prob_matrix.loc[reverse_mask, "prob_up"] = original_down
        calibration_metrics = calculate_multiclass_calibration_error(
            target,
            prob_matrix,
            int(getattr(config, "prediction_calibration_bins", 10) or 10)
            if config is not None
            else 10,
        )
        rows.update(calibration_metrics)
        prob_matrix = prob_matrix.clip(1e-12, 1.0)
        prob_matrix = prob_matrix.div(prob_matrix.sum(axis=1).replace(0, np.nan), axis=0)
        logloss_frame = prob_matrix.assign(__label__=class_label).dropna(
            subset=["prob_down", "prob_flat", "prob_up", "__label__"]
        )
        if not logloss_frame.empty:
            labels = logloss_frame["__label__"].astype(int).to_numpy()
            probs = logloss_frame[["prob_down", "prob_flat", "prob_up"]].to_numpy()
            rows["三分类LogLoss"] = float(-np.log(probs[np.arange(len(labels)), labels]).mean())
            one_hot = np.eye(3)[labels]
            rows["三分类BrierScore"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return rows


def calculate_trading_metrics_for_segment(
    segment_name: str,
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any] | None:
    """计算交易规则、仓位和成本之后的策略效果指标。"""
    required = {"strategy_net_return", "benchmark_return", "position"}
    if segment_df.empty or not required.issubset(segment_df.columns):
        return None
    cleaned = segment_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["strategy_net_return"])
    if cleaned.empty:
        return None
    annual_periods = infer_annual_periods(pd.DatetimeIndex(cleaned.index), config.annual_trading_days)
    metrics = calculate_metrics(
        cleaned["strategy_net_return"],
        cleaned["benchmark_return"],
        cleaned["position"],
        annual_periods,
    )
    turnover = cleaned.get("turnover", pd.Series(0.0, index=cleaned.index)).fillna(0.0)
    position = cleaned["position"].fillna(0.0)
    raw_signal = cleaned.get("raw_signal", pd.Series(0.0, index=cleaned.index)).fillna(0.0)
    target_position = cleaned.get("target_position", pd.Series(0.0, index=cleaned.index)).fillna(0.0)
    rows: dict[str, Any] = {
        "样本段": segment_name,
        "交易样本数": int(len(cleaned)),
        **metrics,
        "策略毛收益": float(cleaned.get("strategy_gross_return", cleaned["strategy_net_return"]).sum()),
        "策略净收益": float(cleaned["strategy_net_return"].sum()),
        "总换手": float(turnover.sum()),
        "平均换手": float(turnover.mean()),
        "实际持仓覆盖率": float((position != 0).mean()),
        "平均实际仓位绝对值": float(position.abs().mean()),
        "目标仓位覆盖率": float((target_position != 0).mean()),
        "平均目标仓位绝对值": float(target_position.abs().mean()),
        "交易信号覆盖率": float((raw_signal != 0).mean()),
    }
    rows.update(
        calculate_block_bootstrap_statistics(
            cleaned["strategy_net_return"],
            annual_periods,
            config,
        )
    )
    if "trade_allowed" in cleaned.columns:
        rows["交易过滤后可交易覆盖率"] = float(cleaned["trade_allowed"].fillna(0.0).mean())
    if "flat_trade_allowed" in cleaned.columns:
        rows["中性概率过滤后覆盖率"] = float(cleaned["flat_trade_allowed"].fillna(0.0).mean())
    if "confidence_trade_allowed" in cleaned.columns:
        rows["置信度过滤后覆盖率"] = float(cleaned["confidence_trade_allowed"].fillna(0.0).mean())
    return rows


def save_prediction_and_trading_reports(
    segment_backtests: dict[str, pd.DataFrame],
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """分别保存预测质量报告和交易结果报告。"""
    prediction_rows = []
    trading_rows = []
    for segment_name, segment_df in segment_backtests.items():
        prediction_row = calculate_prediction_metrics_for_segment(
            segment_name,
            segment_df,
            config,
        )
        if prediction_row is not None:
            prediction_rows.append(prediction_row)
        trading_row = calculate_trading_metrics_for_segment(segment_name, segment_df, config)
        if trading_row is not None:
            trading_rows.append(trading_row)

    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(
            output_dir / "composite_prediction_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if trading_rows:
        pd.DataFrame(trading_rows).to_csv(
            output_dir / "composite_trading_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def plot_train_test_backtest_result(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    title: str,
    score_label: str,
    file_name: str,
) -> Path:
    """并排绘制训练集、验证集和最终测试集回测面板。"""
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(30, 16),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.0]},
    )
    fig.suptitle(title, fontsize=16)

    panels = [
        ("训练集滚动预测", train_df),
        ("验证集滚动预测", validation_df),
        ("最终测试集滚动预测", test_df),
    ]
    for col, (panel_name, backtest_df) in enumerate(panels):
        axes[0, col].plot(backtest_df.index, backtest_df["nav"], label="策略净值", linewidth=1.6)
        axes[0, col].plot(
            backtest_df.index,
            backtest_df["benchmark_nav"],
            label="基准净值",
            linewidth=1.2,
            alpha=0.85,
        )
        axes[0, col].set_title(panel_name)
        axes[0, col].set_ylabel("净值")
        axes[0, col].legend(loc="upper left")
        axes[0, col].grid(alpha=0.3)

        axes[1, col].fill_between(
            backtest_df.index,
            backtest_df["drawdown"],
            0,
            color="#d62728",
            alpha=0.35,
        )
        axes[1, col].set_ylabel("回撤")
        axes[1, col].grid(alpha=0.3)

        axes[2, col].plot(
            backtest_df.index,
            backtest_df["composite_score"],
            label=score_label,
            linewidth=1.0,
        )
        axes[2, col].step(
            backtest_df.index,
            backtest_df["position"],
            label="实际仓位",
            color="#ff7f0e",
            linewidth=1.0,
            where="mid",
        )
        axes[2, col].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        axes[2, col].set_ylabel("分数 / 仓位")
        axes[2, col].legend(loc="upper left")
        axes[2, col].grid(alpha=0.3)

        if "future_horizon_return" in backtest_df.columns:
            axes[3, col].scatter(
                backtest_df["calibrated_prob_edge"],
                backtest_df["future_horizon_return"],
                s=8,
                alpha=0.35,
            )
            axes[3, col].axhline(0, color="black", linewidth=0.8, alpha=0.7)
            axes[3, col].axvline(0, color="black", linewidth=0.8, alpha=0.7)
            axes[3, col].set_xlabel("校准概率差")
            axes[3, col].set_ylabel("未来Horizon收益")
            axes[3, col].grid(alpha=0.3)
        else:
            axes[3, col].axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plot_path = output_dir / file_name
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def calculate_backtest_segment_metrics(
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, float]:
    """基于已有回测明细计算某个分段的绩效指标。"""
    annual_periods = infer_annual_periods(
        pd.DatetimeIndex(segment_df.index),
        config.annual_trading_days,
    )
    return calculate_metrics(
        segment_df["strategy_net_return"],
        segment_df["benchmark_return"],
        segment_df["position"],
        annual_periods,
    )


def append_segment_metric_row(
    rows: list[dict[str, Any]],
    segment_type: str,
    segment_name: str,
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """把一个分段的绩效指标追加到稳健性报告行列表。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if segment_df.empty or not required_columns.issubset(segment_df.columns):
        return
    try:
        metrics = calculate_backtest_segment_metrics(segment_df, config)
    except Exception:
        return
    rows.append(
        {
            "分段类型": segment_type,
            "分段": segment_name,
            "开始时间": segment_df.index.min(),
            "结束时间": segment_df.index.max(),
            **metrics,
        }
    )


def append_market_state_row(
    rows: list[dict[str, Any]],
    state_type: str,
    state_name: str,
    segment_df: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    """追加一行市场状态分层表现。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if segment_df.empty or not required_columns.issubset(segment_df.columns):
        return
    segment_df = segment_df.dropna(subset=["strategy_net_return"])
    if segment_df.empty:
        return
    try:
        metrics = calculate_backtest_segment_metrics(segment_df, config)
    except Exception:
        return

    traded = segment_df.get("raw_signal", pd.Series(0.0, index=segment_df.index)).fillna(0.0) != 0
    target_direction = segment_df.get("target_direction")
    raw_signal = segment_df.get("raw_signal")
    if target_direction is not None and raw_signal is not None and traded.any():
        direction_accuracy = float(
            (np.sign(raw_signal.loc[traded]) == np.sign(target_direction.loc[traded])).mean()
        )
    else:
        direction_accuracy = np.nan

    prob_edge = segment_df.get("calibrated_prob_edge")
    future_return = segment_df.get("future_horizon_return")
    if prob_edge is not None and future_return is not None and prob_edge.notna().sum() >= 3:
        edge_return_corr = float(prob_edge.corr(future_return))
    else:
        edge_return_corr = np.nan

    rows.append(
        {
            "状态类型": state_type,
            "状态": state_name,
            "样本数": int(len(segment_df)),
            "交易信号覆盖率": float(traded.mean()),
            "平均仓位绝对值": float(segment_df["position"].fillna(0.0).abs().mean()),
            "交易方向准确率": direction_accuracy,
            "概率差与未来收益相关性": edge_return_corr,
            "平均概率差": float(prob_edge.mean()) if prob_edge is not None else np.nan,
            "平均绝对概率差": float(prob_edge.abs().mean()) if prob_edge is not None else np.nan,
            **metrics,
        }
    )


def get_supported_pandas_frequency(candidates: list[str]) -> str:
    """在不同 pandas 版本之间选择可用的频率别名。"""
    for freq in candidates:
        try:
            pd.tseries.frequencies.to_offset(freq)
            return freq
        except ValueError:
            continue
    raise ValueError(f"当前 pandas 不支持这些频率别名: {candidates}")


def save_composite_robustness_report(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """保存综合策略分段稳健性报告，帮助识别收益是否过度集中。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if backtest_df.empty or not required_columns.issubset(backtest_df.columns):
        return

    rows: list[dict[str, Any]] = []
    cleaned = backtest_df.replace([np.inf, -np.inf], np.nan).copy()
    append_segment_metric_row(rows, "整体", "最终测试集", cleaned.dropna(subset=["strategy_net_return"]), config)

    month_freq = get_supported_pandas_frequency(["ME", "M"])
    quarter_freq = get_supported_pandas_frequency(["QE", "Q"])
    for period_name, freq in [("月度", month_freq), ("季度", quarter_freq)]:
        for period, segment_df in cleaned.groupby(pd.Grouper(freq=freq)):
            append_segment_metric_row(
                rows,
                period_name,
                str(period.date()) if pd.notna(period) else "未知",
                segment_df.dropna(subset=["strategy_net_return"]),
                config,
            )

    benchmark_abs_return = cleaned["benchmark_return"].abs().replace([np.inf, -np.inf], np.nan)
    if benchmark_abs_return.notna().sum() >= 20:
        vol_rank = benchmark_abs_return.rolling(120, min_periods=20).rank(pct=True)
        regime_map = {
            "低波动": vol_rank <= 0.33,
            "中波动": (vol_rank > 0.33) & (vol_rank <= 0.66),
            "高波动": vol_rank > 0.66,
        }
        for regime_name, mask in regime_map.items():
            append_segment_metric_row(
                rows,
                "波动状态",
                regime_name,
                cleaned.loc[mask.fillna(False)].dropna(subset=["strategy_net_return"]),
                config,
            )

    if "calibrated_prob_edge" in cleaned.columns:
        confidence = cleaned["calibrated_prob_edge"].abs().replace([np.inf, -np.inf], np.nan)
        if confidence.notna().sum() >= 20:
            rank = confidence.rolling(120, min_periods=20).rank(pct=True)
            regime_map = {
                "低置信度": rank <= 0.33,
                "中置信度": (rank > 0.33) & (rank <= 0.66),
                "高置信度": rank > 0.66,
            }
            for regime_name, mask in regime_map.items():
                append_segment_metric_row(
                    rows,
                    "模型置信度",
                    regime_name,
                    cleaned.loc[mask.fillna(False)].dropna(subset=["strategy_net_return"]),
                    config,
                )

    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "composite_robustness_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_market_state_report(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """保存按市场状态拆分的综合策略表现报告。"""
    required_columns = {"strategy_net_return", "benchmark_return", "position"}
    if backtest_df.empty or not required_columns.issubset(backtest_df.columns):
        return

    cleaned = backtest_df.replace([np.inf, -np.inf], np.nan).copy()
    cleaned = cleaned.dropna(subset=["strategy_net_return"])
    if cleaned.empty:
        return

    rows: list[dict[str, Any]] = []
    append_market_state_row(rows, "整体", "全部", cleaned, config)

    state_columns = [
        ("趋势状态", "trend_regime"),
        ("波动状态", "volatility_regime"),
        ("流动性状态", "liquidity_regime"),
        ("交易时段", "session_regime"),
    ]
    for state_type, column in state_columns:
        if column not in cleaned.columns:
            continue
        for state_name, segment_df in cleaned.groupby(column, dropna=False):
            label = "未知" if pd.isna(state_name) else str(state_name)
            append_market_state_row(rows, state_type, label, segment_df, config)

    combined_specs = [
        ("趋势×波动", ["trend_regime", "volatility_regime"]),
        ("趋势×流动性", ["trend_regime", "liquidity_regime"]),
        ("波动×流动性", ["volatility_regime", "liquidity_regime"]),
        ("趋势×时段", ["trend_regime", "session_regime"]),
    ]
    for state_type, columns in combined_specs:
        if not set(columns).issubset(cleaned.columns):
            continue
        combined = cleaned[columns].astype("string").fillna("未知").agg(" | ".join, axis=1)
        for state_name, segment_df in cleaned.groupby(combined, dropna=False):
            append_market_state_row(rows, state_type, str(state_name), segment_df, config)

    if rows:
        report = pd.DataFrame(rows)
        sort_columns = [column for column in ["状态类型", "状态"] if column in report.columns]
        if sort_columns:
            report = report.sort_values(sort_columns).reset_index(drop=True)
        report.to_csv(
            output_dir / "composite_market_state_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_cost_stress_report(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """基于同一组持仓重算不同交易成本下的策略表现。"""
    required_columns = {"strategy_gross_return", "turnover", "benchmark_return", "position"}
    if backtest_df.empty or not required_columns.issubset(backtest_df.columns):
        return

    annual_periods = infer_annual_periods(
        pd.DatetimeIndex(backtest_df.index),
        config.annual_trading_days,
    )
    rows = []
    for cost_bps in getattr(config, "cost_stress_bps_list", []) or []:
        cost_bps = float(cost_bps)
        net_return = backtest_df["strategy_gross_return"].fillna(0.0) - (
            backtest_df["turnover"].fillna(0.0) * cost_bps / 10000.0
        )
        metrics = calculate_metrics(
            net_return,
            backtest_df["benchmark_return"],
            backtest_df["position"],
            annual_periods,
        )
        rows.append(
            {
                "单边总成本bps": cost_bps,
                "平均单根成本": float((backtest_df["turnover"].fillna(0.0) * cost_bps / 10000.0).mean()),
                **metrics,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "composite_cost_stress_report.csv",
            index=False,
            encoding="utf-8-sig",
        )


def parse_comparison_model_segment(model_label: str) -> tuple[str, str]:
    """把模型比较表中的名称拆成模型名和样本段。"""
    label = str(model_label)
    suffix_map = {
        "_train": "训练集",
        "_validation": "验证集",
    }
    for suffix, segment in suffix_map.items():
        if label.endswith(suffix):
            return label[: -len(suffix)], segment
    return label, "最终测试集"


def safe_metric_retention(test_value: float, validation_value: float) -> float:
    """计算测试指标相对验证指标的保留比例。"""
    if not np.isfinite(test_value) or not np.isfinite(validation_value):
        return np.nan
    if validation_value <= 0:
        return np.nan
    return float(test_value / validation_value)


def save_validation_test_gap_report(
    comparison_rows: list[dict[str, Any]],
    output_dir: Path,
    config: BacktestConfig,
) -> None:
    """保存训练/验证/最终测试表现差异诊断，辅助识别过拟合和样本外衰减。"""
    if not bool(getattr(config, "composite_enable_validation_test_gap_report", True)):
        return
    if not comparison_rows:
        return

    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty or "模型" not in comparison_df.columns:
        return
    comparison_df = comparison_df[comparison_df.get("错误").isna()] if "错误" in comparison_df.columns else comparison_df
    if comparison_df.empty:
        return

    parsed = comparison_df["模型"].map(parse_comparison_model_segment)
    comparison_df = comparison_df.copy()
    comparison_df["基础模型"] = parsed.map(lambda value: value[0])
    comparison_df["样本段"] = parsed.map(lambda value: value[1])

    metric_columns = [
        "累计收益",
        "年化收益",
        "夏普比率",
        "最大回撤",
        "胜率",
        "交易次数",
        "样本K线数",
        "平均仓位",
    ]
    available_metrics = [column for column in metric_columns if column in comparison_df.columns]
    if not available_metrics:
        return

    sharpe_warn = max(0.0, float(getattr(config, "composite_gap_warn_sharpe_retention", 0.5) or 0.5))
    return_warn = max(0.0, float(getattr(config, "composite_gap_warn_return_retention", 0.5) or 0.5))
    rows: list[dict[str, Any]] = []
    for model_name, model_df in comparison_df.groupby("基础模型", dropna=False):
        by_segment = {
            str(row["样本段"]): row
            for _, row in model_df.drop_duplicates("样本段", keep="last").iterrows()
        }
        if "验证集" not in by_segment or "最终测试集" not in by_segment:
            continue

        validation_row = by_segment["验证集"]
        test_row = by_segment["最终测试集"]
        train_row = by_segment.get("训练集")
        row: dict[str, Any] = {"模型": model_name}
        for metric in available_metrics:
            train_value = float(train_row[metric]) if train_row is not None and pd.notna(train_row.get(metric)) else np.nan
            validation_value = float(validation_row[metric]) if pd.notna(validation_row.get(metric)) else np.nan
            test_value = float(test_row[metric]) if pd.notna(test_row.get(metric)) else np.nan
            row[f"训练_{metric}"] = train_value
            row[f"验证_{metric}"] = validation_value
            row[f"最终测试_{metric}"] = test_value
            row[f"测试减验证_{metric}"] = (
                test_value - validation_value
                if np.isfinite(test_value) and np.isfinite(validation_value)
                else np.nan
            )
            row[f"测试相对验证_{metric}"] = safe_metric_retention(test_value, validation_value)
            row[f"验证减训练_{metric}"] = (
                validation_value - train_value
                if np.isfinite(validation_value) and np.isfinite(train_value)
                else np.nan
            )

        sharpe_retention = row.get("测试相对验证_夏普比率", np.nan)
        return_retention = row.get("测试相对验证_累计收益", np.nan)
        sharpe_warning = np.isfinite(sharpe_retention) and sharpe_retention < sharpe_warn
        return_warning = np.isfinite(return_retention) and return_retention < return_warn
        row["样本外衰减预警"] = bool(sharpe_warning or return_warning)
        warning_reasons = []
        if sharpe_warning:
            warning_reasons.append("测试夏普相对验证衰减")
        if return_warning:
            warning_reasons.append("测试累计收益相对验证衰减")
        row["预警原因"] = "；".join(warning_reasons)
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "composite_validation_test_gap.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_composite_outputs(
    backtest_df: pd.DataFrame,
    metrics: dict[str, float],
    selected_factors: list[str],
    feature_importance: pd.Series,
    output_dir: Path,
    factor_id_map: dict[str, int],
    factor_label_map: dict[str, str],
    split_time: pd.Timestamp,
    validation_end_time: pd.Timestamp,
    config: BacktestConfig,
    primary_model_name: str = "xgboost",
    selection_summary: pd.DataFrame | None = None,
    segment_backtests: dict[str, pd.DataFrame] | None = None,
) -> None:
    """保存综合因子回测的全部输出文件。

    输出包括：
    - composite_detail.csv：逐K线信号、持仓、收益、净值明细。
    - composite_summary.csv：模型配置和绩效摘要。
    - composite_xgboost_feature_importance.csv：特征重要性。
    - composite_xgboost_feature_selection.csv：滚动选因明细。
    - composite_xgboost_edge_diagnostics.csv：概率优势分桶诊断。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(output_dir / "composite_detail.csv", encoding="utf-8-sig")

    summary = pd.Series(metrics, name="value")
    summary.loc["模型"] = f"{primary_model_name}_rolling_multiclass"
    summary.loc["绩效指标口径"] = "standard_sharpe_mean_over_vol_v2"
    summary.loc["训练验证切分时间"] = str(split_time)
    summary.loc["验证测试切分时间"] = str(validation_end_time)
    summary.loc["预测目标"] = "cost_and_volatility_adjusted_future_return(-1,0,1)"
    summary.loc["预测标签模式"] = get_xgboost_target_label_mode(config)
    summary.loc["预测目标基础跨度K线数"] = int(config.xgboost_target_horizon)
    summary.loc["预测目标跨度K线数"] = get_xgboost_target_horizon(config)
    summary.loc["目标跨度对齐最小持仓"] = bool(
        getattr(config, "xgboost_target_align_with_min_holding", True)
    )
    summary.loc["训练标签隔离K线数"] = max(0, get_xgboost_target_horizon(config) - 1)
    summary.loc["目标计入双边成本"] = bool(
        getattr(config, "xgboost_target_include_round_trip_cost", True)
    )
    summary.loc["目标双边成本bps"] = get_xgboost_target_round_trip_cost(config) * 10000.0
    summary.loc["目标额外中性缓冲bps"] = float(config.xgboost_target_neutral_bps or 0.0)
    summary.loc["目标静态中性区间bps"] = get_xgboost_target_neutral_threshold(config) * 10000.0
    summary.loc["启用动态目标中性区间"] = bool(
        getattr(config, "xgboost_target_use_dynamic_neutral_threshold", False)
    )
    summary.loc["动态目标中性区间窗口"] = int(
        getattr(config, "xgboost_target_dynamic_neutral_window", 0) or 0
    )
    summary.loc["动态目标中性区间波动率倍率"] = float(
        getattr(config, "xgboost_target_dynamic_neutral_multiplier", 0.0) or 0.0
    )
    summary.loc["目标分位窗口"] = int(getattr(config, "xgboost_target_quantile_window", 0) or 0)
    summary.loc["目标下分位"] = float(getattr(config, "xgboost_target_quantile_lower", 0.0) or 0.0)
    summary.loc["目标上分位"] = float(getattr(config, "xgboost_target_quantile_upper", 0.0) or 0.0)
    summary.loc["目标分位最小绝对阈值bps"] = float(
        getattr(config, "xgboost_target_quantile_min_abs_bps", 0.0) or 0.0
    )
    summary.loc["手续费bps"] = config.commission_bps
    summary.loc["滑点bps"] = config.slippage_bps
    summary.loc["候选池来源"] = (
        getattr(config, "frozen_active_library_path", None)
        if getattr(config, "use_frozen_active_library", False)
        else "active_factors.csv"
    )
    summary.loc["使用冻结因子库"] = bool(getattr(config, "use_frozen_active_library", False))
    summary.loc["自动冻结active因子库"] = bool(
        getattr(config, "composite_auto_freeze_active_library", True)
    )
    summary.loc["冻结因子库路径"] = str(get_active_factor_library_path(config))
    summary.loc["因子库截止审计策略"] = str(
        getattr(config, "composite_active_library_cutoff_policy", "auto")
    )
    summary.loc["特征范围"] = config.xgboost_feature_scope
    summary.loc["特征模式"] = config.xgboost_feature_mode
    summary.loc["使用因子状态特征"] = config.xgboost_include_factor_state_features
    summary.loc["滚动选因"] = config.xgboost_walk_forward_feature_selection
    summary.loc["候选因子数量"] = len(selected_factors)
    summary.loc["树最小子节点权重"] = config.xgboost_min_child_weight
    summary.loc["分裂最小损失下降"] = config.xgboost_gamma
    summary.loc["L2正则"] = config.xgboost_reg_lambda
    summary.loc["L1正则"] = config.xgboost_reg_alpha
    summary.loc["启用验证测试衰减诊断"] = bool(
        getattr(config, "composite_enable_validation_test_gap_report", True)
    )
    summary.loc["验证测试夏普保留率预警阈值"] = float(
        getattr(config, "composite_gap_warn_sharpe_retention", 0.5) or 0.5
    )
    summary.loc["验证测试收益保留率预警阈值"] = float(
        getattr(config, "composite_gap_warn_return_retention", 0.5) or 0.5
    )
    summary.loc["启用验证集EarlyStopping"] = config.xgboost_use_validation_early_stopping
    summary.loc["验证集比例"] = config.xgboost_validation_ratio
    summary.loc["EarlyStopping轮数"] = config.xgboost_early_stopping_rounds
    summary.loc["交易最小概率差"] = config.xgboost_trade_min_edge
    summary.loc["交易最小方向概率"] = config.xgboost_trade_min_probability
    summary.loc["交易最大中性概率"] = float(
        getattr(config, "xgboost_trade_max_flat_probability", 1.0) or 1.0
    )
    summary.loc["交易最小方向相对中性优势"] = float(
        getattr(config, "xgboost_trade_min_directional_vs_flat_edge", 0.0) or 0.0
    )
    summary.loc["自动校准交易阈值"] = config.xgboost_auto_calibrate_trade_thresholds
    summary.loc["阈值校准最少交易数"] = config.xgboost_threshold_min_trades
    summary.loc["启用市场状态过滤"] = config.xgboost_trade_use_market_filters
    summary.loc["过滤窗口"] = config.xgboost_trade_filter_window
    summary.loc["市场状态分层窗口"] = int(getattr(config, "market_state_regime_window", 120) or 120)
    summary.loc["趋势强度阈值"] = float(
        getattr(config, "market_state_trend_strength_threshold", 0.25) or 0.25
    )
    summary.loc["启用市场状态标签过滤"] = bool(
        getattr(config, "xgboost_trade_use_regime_filter", False)
    )
    summary.loc["允许交易市场状态"] = ",".join(
        str(value) for value in (getattr(config, "allowed_market_state_regimes", []) or [])
    )
    summary.loc["最小波动率分位"] = config.xgboost_trade_min_volatility_rank
    summary.loc["最小流动性分位"] = config.xgboost_trade_min_liquidity_rank
    summary.loc["动态仓位"] = config.xgboost_use_dynamic_position_sizing
    summary.loc["最小动态仓位"] = config.xgboost_position_size_min
    summary.loc["仓位幂次"] = config.xgboost_position_size_power
    summary.loc["最大仓位"] = config.xgboost_position_size_max
    summary.loc["启用持仓规则优化"] = bool(getattr(config, "xgboost_use_position_rules", False))
    summary.loc["最小持仓K线数"] = int(getattr(config, "xgboost_min_holding_bars", 0) or 0)
    summary.loc["反转冷却K线数"] = int(getattr(config, "xgboost_reentry_cooldown_bars", 0) or 0)
    summary.loc["最小仓位变化阈值"] = float(getattr(config, "xgboost_min_position_change", 0.0) or 0.0)
    summary.loc["仓位平滑系数"] = float(getattr(config, "xgboost_position_smoothing_alpha", 1.0) or 1.0)
    summary.loc["启用置信度分位过滤"] = config.xgboost_trade_use_confidence_rank_filter
    summary.loc["置信度分位窗口"] = config.xgboost_trade_confidence_rank_window
    summary.loc["最小置信度分位"] = config.xgboost_trade_min_confidence_rank
    summary.loc["训练集启用市场过滤"] = config.xgboost_train_use_market_filters
    summary.loc["训练最少方向样本数"] = config.xgboost_train_min_directional_samples
    summary.loc["训练非中性类别权重"] = config.xgboost_train_nonzero_class_weight
    summary.loc["训练中性类别权重"] = config.xgboost_train_neutral_class_weight
    summary.loc["训练样本时间衰减加权"] = bool(
        getattr(config, "xgboost_train_use_time_decay_weight", False)
    )
    summary.loc["训练样本时间衰减半衰期K线数"] = int(
        getattr(config, "xgboost_train_time_decay_half_life", 0) or 0
    )
    summary.loc["训练样本时间衰减最低权重"] = float(
        getattr(config, "xgboost_train_time_decay_min_weight", 0.0) or 0.0
    )
    summary.loc["训练样本时间衰减归一化"] = bool(
        getattr(config, "xgboost_train_time_decay_normalize", True)
    )
    summary.loc["XGBoost自动校准方向"] = config.xgboost_auto_calibrate_signal_direction
    summary.loc["等权投票自动校准方向"] = config.benchmark_vote_auto_calibrate_direction
    if "raw_signal" in backtest_df.columns:
        raw_signal = backtest_df["raw_signal"].fillna(0.0)
        summary.loc["信号覆盖率"] = float((raw_signal != 0).mean())
        summary.loc["做多信号数"] = int((raw_signal > 0).sum())
        summary.loc["做空信号数"] = int((raw_signal < 0).sum())
        summary.loc["空仓信号数"] = int((raw_signal == 0).sum())
    if "raw_signal_before_position_rules" in backtest_df.columns:
        raw_signal_before_rules = backtest_df["raw_signal_before_position_rules"].fillna(0.0)
        summary.loc["规则前信号覆盖率"] = float((raw_signal_before_rules != 0).mean())
    if "calibrated_prob_edge" in backtest_df.columns:
        summary.loc["平均校准概率差"] = float(backtest_df["calibrated_prob_edge"].mean())
        summary.loc["平均绝对校准概率差"] = float(backtest_df["calibrated_prob_edge"].abs().mean())
    if "trade_allowed" in backtest_df.columns:
        summary.loc["过滤后可交易覆盖率"] = float(backtest_df["trade_allowed"].fillna(0.0).mean())
    if "flat_trade_allowed" in backtest_df.columns:
        summary.loc["中性概率过滤后覆盖率"] = float(backtest_df["flat_trade_allowed"].fillna(0.0).mean())
    if "directional_vs_flat_edge" in backtest_df.columns:
        summary.loc["平均方向相对中性优势"] = float(backtest_df["directional_vs_flat_edge"].mean())
    if "position_size" in backtest_df.columns:
        summary.loc["平均目标仓位大小"] = float(backtest_df["position_size"].fillna(0.0).mean())
    if {"target_position", "target_position_before_rules"}.issubset(backtest_df.columns):
        target_position = backtest_df["target_position"].fillna(0.0)
        target_before_rules = backtest_df["target_position_before_rules"].fillna(0.0)
        summary.loc["规则后目标仓位覆盖率"] = float((target_position != 0).mean())
        summary.loc["规则前目标仓位覆盖率"] = float((target_before_rules != 0).mean())
        summary.loc["规则后目标仓位平均绝对值"] = float(target_position.abs().mean())
        summary.loc["规则前目标仓位平均绝对值"] = float(target_before_rules.abs().mean())
        summary.loc["规则后目标仓位平均变化"] = float(target_position.diff().abs().fillna(target_position.abs()).mean())
        summary.loc["规则前目标仓位平均变化"] = float(
            target_before_rules.diff().abs().fillna(target_before_rules.abs()).mean()
        )
    if "position" in backtest_df.columns:
        position = backtest_df["position"].fillna(0.0)
        summary.loc["实际持仓覆盖率"] = float((position != 0).mean())
        summary.loc["平均实际仓位绝对值"] = float(position.abs().mean())
        summary.loc["非零持仓平均绝对仓位"] = (
            float(position[position != 0].abs().mean()) if (position != 0).any() else 0.0
        )
    if "confidence_rank" in backtest_df.columns:
        summary.loc["平均置信度分位"] = float(backtest_df["confidence_rank"].fillna(0.0).mean())
    if "calibrated_min_edge" in backtest_df.columns:
        summary.loc["平均校准概率差阈值"] = float(backtest_df["calibrated_min_edge"].dropna().mean())
    if "calibrated_min_probability" in backtest_df.columns:
        summary.loc["平均校准方向概率阈值"] = float(
            backtest_df["calibrated_min_probability"].dropna().mean()
        )
    if not should_use_walk_forward_selection(config):
        summary.loc["选中因子编号"] = ",".join(str(factor_id_map[factor]) for factor in selected_factors)
        summary.loc["选中因子标签"] = ",".join(factor_label_map[factor] for factor in selected_factors)
        summary.loc["选中因子"] = ",".join(selected_factors)
    summary.to_csv(output_dir / "composite_summary.csv", encoding="utf-8-sig")

    importance_df = feature_importance.rename("importance").reset_index()
    importance_df.columns = ["特征", "importance"]
    importance_df["因子"] = importance_df["特征"].map(
        lambda feature_name: get_feature_base_factor(feature_name, selected_factors)
    )
    importance_df.insert(0, "因子编号", importance_df["因子"].map(factor_id_map))
    importance_df.insert(1, "因子标签", importance_df["因子"].map(factor_label_map))
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(
        output_dir / "composite_xgboost_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    factor_contribution = (
        importance_df.groupby(["因子编号", "因子标签", "因子"], dropna=False)["importance"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "重要性合计",
                "mean": "平均特征重要性",
                "count": "特征数量",
            }
        )
        .sort_values("重要性合计", ascending=False)
    )
    total_importance = factor_contribution["重要性合计"].sum()
    factor_contribution["重要性占比"] = (
        factor_contribution["重要性合计"] / total_importance
        if total_importance > 0
        else np.nan
    )
    factor_contribution.to_csv(
        output_dir / "composite_factor_contribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if selection_summary is not None:
        selection_summary.to_csv(
            output_dir / "composite_xgboost_feature_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_prediction_diagnostics(backtest_df, output_dir)
    if segment_backtests is not None:
        save_prediction_and_trading_reports(segment_backtests, output_dir, config)
    save_composite_robustness_report(backtest_df, output_dir, config)
    save_market_state_report(backtest_df, output_dir, config)
    save_cost_stress_report(backtest_df, output_dir, config)


def run_composite_backtest(config: BacktestConfig) -> dict[str, float]:
    """运行一次指定配置下的综合因子回测，并返回测试集绩效指标。"""
    config = apply_frequency_runtime_defaults(config)
    report_config_validation(config, "composite")
    output_dir = get_research_output_dir(config, "composite_factor")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = get_experiment_run_dir(config, "composite")
    config, _active_snapshot_path = freeze_active_factor_library_for_run(
        config,
        output_dir,
        run_dir,
    )
    write_run_config(config, output_dir)
    if run_dir is not None:
        write_run_config(config, run_dir, filename="effective_run_config.json")

    try:
        requested_factors = None
        if bool(getattr(config, "composite_build_active_only", True)):
            requested_factors = load_active_factor_names(config)
            print(f"综合回测按 active 因子按需构建: {len(requested_factors)} 个候选因子")

        print(f"读取 {config.symbol} 的 {get_frequency_key(config)} 数据...")
        data = fetch_intraday_data(config)

        print("构建因子...")
        cache_path = output_dir / "active_factor_matrix_cache.pkl"
        factors = None
        if requested_factors is not None and bool(getattr(config, "composite_use_factor_cache", True)):
            factors = load_factor_matrix_cache(cache_path, data, requested_factors, config)
            if factors is not None:
                print(f"已读取 active 因子矩阵缓存: {cache_path}")
        if factors is None:
            factors = build_factors(data, config, requested_factors=requested_factors)
            if requested_factors is not None and bool(getattr(config, "composite_use_factor_cache", True)):
                save_factor_matrix_cache(factors, cache_path, data, requested_factors, config)
                print(f"已保存 active 因子矩阵缓存: {cache_path}")
        if run_dir is not None:
            write_factor_count_snapshot(factors, run_dir)
            coverage = get_last_related_data_coverage()
            if not coverage.empty:
                coverage.to_csv(
                    output_dir / "related_data_coverage.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                coverage.to_csv(
                    run_dir / "related_data_coverage.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        split_time, validation_end_time = split_train_validation_test_index(
            data.index,
            config.auto_select_train_ratio,
            getattr(config, "auto_select_validation_ratio", 0.0),
        )
        audit_active_library_oos_cutoff(
            config,
            validation_end_time,
            output_dir,
            run_dir,
        )
        selected_factors, selection_summary = get_selected_factors(
            data,
            factors,
            config,
            split_time,
        )
        factor_id_map = get_factor_id_map(factors)
        factor_label_map = get_factor_label_map(factors)
        train_data = data.loc[data.index < split_time]
        train_factors = factors.loc[factors.index < split_time]
        validation_data = data.loc[(data.index >= split_time) & (data.index < validation_end_time)]
        validation_factors = factors.loc[
            (factors.index >= split_time) & (factors.index < validation_end_time)
        ]
        backtest_data = data.loc[data.index >= validation_end_time]
        backtest_factors = factors.loc[factors.index >= validation_end_time]

        if should_use_walk_forward_selection(config):
            print(
                "XGBoost候选因子池: active_factors.csv；可用active因子数量: "
                f"{len(selected_factors)}，每次重训滚动选取 Top {config.xgboost_best_top_n}"
            )
        else:
            print(
                "XGBoost特征因子(active池内): "
                + ", ".join(factor_label_map[factor] for factor in selected_factors)
            )
        print(
            "滚动训练参数: "
            f"window={config.xgboost_train_window}, "
            f"min_samples={config.xgboost_min_train_samples}, "
            f"retrain_every={config.xgboost_retrain_every}"
        )

        predict_start = min(
            max(1, int(config.xgboost_min_train_samples)),
            max(0, len(factors.index) - 1),
        )
        predict_index = factors.index[predict_start:]
        model_names = get_enabled_composite_models(config)
        print("综合模型对比: " + ", ".join(model_names))
        comparison_rows: list[dict[str, Any]] = []
        generated_paths: list[Path] = []
        primary_metrics: dict[str, float] | None = None
        benchmark_vote_metrics: dict[str, float] | None = None
        successful_model_signals: dict[str, pd.DataFrame] = {}

        for model_name in model_names:
            try:
                signal, feature_importance, _, rolling_selection_summary = build_xgboost_rolling_signal(
                    data,
                    factors,
                    selected_factors,
                    config,
                    predict_index,
                    model_name=model_name,
                )
            except ImportError as exc:
                print(f"\n跳过模型 {model_name}: {exc}")
                comparison_rows.append({"模型": model_name, "错误": str(exc)})
                continue
            successful_model_signals[model_name] = signal
            train_signal = signal.loc[train_factors.index]
            validation_signal = signal.loc[validation_factors.index]
            backtest_signal = signal.loc[backtest_factors.index]
            train_df, train_metrics = run_backtest(train_data, train_signal, config)
            validation_df, validation_metrics = run_backtest(
                validation_data,
                validation_signal,
                config,
            )
            backtest_df, metrics = run_backtest(backtest_data, backtest_signal, config)
            comparison_rows.extend(
                [
                    {"模型": f"{model_name}_train", **train_metrics},
                    {"模型": f"{model_name}_validation", **validation_metrics},
                    {"模型": model_name, **metrics},
                ]
            )

            report_filename = (
                "composite_report.png"
                if model_name == "xgboost"
                else f"composite_report_{model_name}.png"
            )
            plot_path = plot_train_test_backtest_result(
                train_df,
                validation_df,
                backtest_df,
                output_dir,
                f"{config.symbol} {model_name}多因子滚动窗口回测",
                "预测方向(-1/0/1)",
                report_filename,
            )
            generated_paths.append(plot_path)

            should_save_primary_outputs = model_name == "xgboost" or primary_metrics is None
            if should_save_primary_outputs:
                save_composite_outputs(
                    backtest_df,
                    metrics,
                    selected_factors,
                    feature_importance,
                    output_dir,
                    factor_id_map,
                    factor_label_map,
                    split_time,
                    validation_end_time,
                    config,
                    primary_model_name=model_name,
                    selection_summary=(
                        rolling_selection_summary
                        if rolling_selection_summary is not None
                        else selection_summary
                    ),
                    segment_backtests={
                        "训练集": train_df,
                        "验证集": validation_df,
                        "最终测试集": backtest_df,
                    },
                )
                write_composite_artifact_manifest(
                    config,
                    output_dir,
                    model_name,
                    split_time,
                    validation_end_time,
                )
                primary_metrics = metrics

            train_benchmark_vote_signal = build_backtest_signal_from_columns(
                train_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            validation_benchmark_vote_signal = build_backtest_signal_from_columns(
                validation_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            benchmark_vote_signal = build_backtest_signal_from_columns(
                backtest_signal,
                "benchmark_vote_score",
                "benchmark_vote_raw_signal",
                "benchmark_vote_position",
            )
            train_benchmark_vote_df, train_benchmark_vote_metrics = run_backtest(
                train_data,
                train_benchmark_vote_signal,
                config,
            )
            validation_benchmark_vote_df, validation_benchmark_vote_metrics = run_backtest(
                validation_data,
                validation_benchmark_vote_signal,
                config,
            )
            benchmark_vote_df, current_benchmark_vote_metrics = run_backtest(
                backtest_data,
                benchmark_vote_signal,
                config,
            )
            if benchmark_vote_metrics is None:
                benchmark_vote_metrics = current_benchmark_vote_metrics
                benchmark_plot_path = plot_train_test_backtest_result(
                    train_benchmark_vote_df,
                    validation_benchmark_vote_df,
                    benchmark_vote_df,
                    output_dir,
                    f"{config.symbol} 等权投票基准滚动窗口回测",
                    "等权投票分数",
                    "benchmark_vote_report.png",
                )
                generated_paths.append(benchmark_plot_path)
                comparison_rows.extend(
                    [
                        {"模型": "equal_vote_benchmark_train", **train_benchmark_vote_metrics},
                        {
                            "模型": "equal_vote_benchmark_validation",
                            **validation_benchmark_vote_metrics,
                        },
                        {"模型": "equal_vote_benchmark", **current_benchmark_vote_metrics},
                    ]
                )

        if bool(getattr(config, "composite_enable_probability_ensemble", True)):
            try:
                ensemble_signal, ensemble_diagnostics = build_historical_probability_ensemble(
                    successful_model_signals,
                    data,
                    config,
                )
                ensemble_diagnostics.to_csv(
                    output_dir / "composite_model_ensemble_weights.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                train_ensemble_signal = ensemble_signal.reindex(train_factors.index)
                validation_ensemble_signal = ensemble_signal.reindex(validation_factors.index)
                test_ensemble_signal = ensemble_signal.reindex(backtest_factors.index)
                train_ensemble_df, train_ensemble_metrics = run_backtest(
                    train_data,
                    train_ensemble_signal,
                    config,
                )
                validation_ensemble_df, validation_ensemble_metrics = run_backtest(
                    validation_data,
                    validation_ensemble_signal,
                    config,
                )
                test_ensemble_df, test_ensemble_metrics = run_backtest(
                    backtest_data,
                    test_ensemble_signal,
                    config,
                )
                comparison_rows.extend(
                    [
                        {"模型": "probability_ensemble_train", **train_ensemble_metrics},
                        {
                            "模型": "probability_ensemble_validation",
                            **validation_ensemble_metrics,
                        },
                        {"模型": "probability_ensemble", **test_ensemble_metrics},
                    ]
                )
                ensemble_plot_path = plot_train_test_backtest_result(
                    train_ensemble_df,
                    validation_ensemble_df,
                    test_ensemble_df,
                    output_dir,
                    f"{config.symbol} 历史绩效动态概率融合回测",
                    "融合预测方向(-1/0/1)",
                    "composite_report_probability_ensemble.png",
                )
                generated_paths.append(ensemble_plot_path)
            except ValueError as exc:
                print(f"\n跳过概率融合: {exc}")
                comparison_rows.append({"模型": "probability_ensemble", "错误": str(exc)})

        pd.DataFrame(comparison_rows).to_csv(
            output_dir / "composite_model_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_validation_test_gap_report(comparison_rows, output_dir, config)
        if run_dir is not None:
            copy_existing_files(
                [
                    output_dir / "composite_detail.csv",
                    output_dir / "composite_summary.csv",
                    output_dir / "composite_model_comparison.csv",
                    output_dir / "composite_validation_test_gap.csv",
                    output_dir / "composite_xgboost_feature_importance.csv",
                    output_dir / "composite_factor_contribution.csv",
                    output_dir / "composite_xgboost_feature_selection.csv",
                    output_dir / "composite_xgboost_edge_diagnostics.csv",
                    output_dir / "composite_robustness_report.csv",
                    output_dir / "composite_cost_stress_report.csv",
                    output_dir / "composite_prediction_diagnostics.csv",
                    output_dir / "composite_prediction_confusion_matrix.csv",
                    output_dir / "composite_model_ensemble_weights.csv",
                    output_dir / "composite_artifact_manifest.json",
                    output_dir / "related_data_coverage.csv",
                    *generated_paths,
                ],
                run_dir,
            )

        metrics = primary_metrics or {}
        print("\n========== 多模型综合因子滚动窗口回测 ==========")
        print_metrics(metrics)
        print("\n========== 等权投票基准回测 ==========")
        print_metrics(benchmark_vote_metrics or {})
        print("\n输出文件:")
        output_paths = [
            output_dir / "composite_detail.csv",
            output_dir / "composite_summary.csv",
            output_dir / "composite_model_comparison.csv",
            output_dir / "composite_validation_test_gap.csv",
            output_dir / "composite_xgboost_feature_importance.csv",
            output_dir / "composite_factor_contribution.csv",
            output_dir / "composite_robustness_report.csv",
            output_dir / "composite_cost_stress_report.csv",
            output_dir / "composite_model_ensemble_weights.csv",
            *generated_paths,
        ]
        for path in dict.fromkeys(output_paths):
            if path.exists():
                print(path)
        return metrics
    finally:
        # 资源释放由脚本入口或批量入口统一负责，这里只保持异常传播和语法结构完整。
        pass


def main() -> None:
    """综合因子回测脚本入口。"""
    config = BacktestConfig()

    def action() -> dict[str, float]:
        try:
            return run_composite_backtest(config)
        finally:
            stop_wind()

    run_tracked(config, "composite", action)


if __name__ == "__main__":
    main()

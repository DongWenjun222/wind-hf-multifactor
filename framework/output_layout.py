from __future__ import annotations

"""项目输出目录布局。

所有入口脚本通过这里获得多品种目录，避免各文件自行拼接路径后逐渐漂移。
读取函数保留旧版 ``by_symbol/<品种>`` 兼容，写入统一使用新版分层目录。
"""

from pathlib import Path
from copy import copy
from typing import Any


def get_frequency_key(config: Any) -> str:
    """返回规范化频率键，目前支持分钟线和日频。"""
    raw = str(getattr(config, "bar_frequency", "") or "").strip().lower()
    if raw in {"1d", "d", "day", "daily", "日频", "日线"}:
        return "1d"
    if raw.endswith("min") and raw[:-3].isdigit() and int(raw[:-3]) > 0:
        return f"{int(raw[:-3])}min"
    bar_size = max(1, int(getattr(config, "bar_size", 30) or 30))
    return f"{bar_size}min"


def is_daily_frequency(config: Any) -> bool:
    """判断当前配置是否使用日频行情。"""
    return get_frequency_key(config) == "1d"


def get_frequency_scoped_dir(base_dir: Path | str, config: Any) -> Path:
    """在基础目录下追加频率层；已带频率后缀时不重复追加。"""
    base = Path(base_dir)
    frequency = get_frequency_key(config)
    return base if base.name.lower() == frequency else base / frequency


def get_research_output_dir(config: Any, category: str) -> Path:
    """返回按品种、类别和频率隔离的研究目录。"""
    return get_frequency_scoped_dir(get_research_output_root(config) / category, config)


def get_trading_signal_output_dir(config: Any) -> Path:
    """返回按频率隔离的交易信号目录。"""
    return get_frequency_scoped_dir(Path(config.output_dir) / "trading_signals", config)


def apply_frequency_runtime_defaults(config: Any) -> Any:
    """返回频率适配后的浅拷贝，避免日频沿用分钟级训练窗口。"""
    if not is_daily_frequency(config):
        return config
    resolved = copy(config)
    mappings = {
        "zscore_window": "daily_zscore_window",
        "xgboost_train_window": "daily_xgboost_train_window",
        "xgboost_min_train_samples": "daily_xgboost_min_train_samples",
        "xgboost_retrain_every": "daily_xgboost_retrain_every",
        "qcut_window": "daily_qcut_window",
        "qcut_min_periods": "daily_qcut_min_periods",
        "composite_ensemble_weight_window": "daily_composite_ensemble_weight_window",
        "composite_ensemble_min_history": "daily_composite_ensemble_min_history",
        "composite_probability_calibration_window": "daily_composite_probability_calibration_window",
        "composite_probability_calibration_min_history": "daily_composite_probability_calibration_min_history",
        "composite_edge_calibration_window": "daily_composite_edge_calibration_window",
        "composite_edge_calibration_min_history": "daily_composite_edge_calibration_min_history",
        "composite_multi_window_train_windows": "daily_composite_multi_window_train_windows",
        "pooled_symbol_residual_window": "daily_pooled_symbol_residual_window",
        "pooled_symbol_residual_min_history": "daily_pooled_symbol_residual_min_history",
        "pooled_symbol_residual_retrain_every": "daily_pooled_symbol_residual_retrain_every",
    }
    for target, source in mappings.items():
        if hasattr(resolved, source):
            setattr(resolved, target, getattr(resolved, source))
    return resolved


def safe_symbol_dir_name(symbol: str) -> str:
    """把 Wind 品种代码转换成统一的大写目录名。"""
    return str(symbol).replace(".", "_").replace("/", "_").replace("-", "_").upper()


def is_symbol_scoped_output_dir(base_dir: Path | str, config: Any) -> bool:
    """判断 output_dir 是否已经是当前品种的 multi-symbol 结果根目录。"""
    base = Path(base_dir)
    symbol_dir = safe_symbol_dir_name(getattr(config, "symbol", ""))
    if not symbol_dir or base.name.upper() != symbol_dir:
        return False
    parent_name = base.parent.name.lower()
    return parent_name in {
        str(getattr(config, "multi_symbol_symbols_subdir", "symbols")).lower(),
        str(getattr(config, "multi_symbol_output_subdir", "by_symbol")).lower(),
    }


def get_research_output_root(config: Any) -> Path:
    """统一单品种与多品种流程使用的当前品种研究根目录。"""
    base = Path(config.output_dir)
    if not bool(getattr(config, "single_symbol_separate_output_dirs", True)):
        return base
    if is_symbol_scoped_output_dir(base, config):
        return base
    return (
        base
        / str(getattr(config, "multi_symbol_output_subdir", "by_symbol"))
        / str(getattr(config, "multi_symbol_symbols_subdir", "symbols"))
        / safe_symbol_dir_name(getattr(config, "symbol", ""))
    )


def get_multi_symbol_root(config: Any) -> Path:
    """返回多品种输出根目录。"""
    return Path(config.output_dir) / str(getattr(config, "multi_symbol_output_subdir", "by_symbol"))


def get_multi_symbol_symbols_dir(config: Any) -> Path:
    """返回各品种独立结果的父目录。"""
    return get_multi_symbol_root(config) / str(
        getattr(config, "multi_symbol_symbols_subdir", "symbols")
    )


def get_multi_symbol_summary_dir(config: Any) -> Path:
    """返回多品种运行状态、错误和清单目录。"""
    return get_frequency_scoped_dir(
        get_multi_symbol_root(config)
        / str(getattr(config, "multi_symbol_summary_subdir", "summary")),
        config,
    )


def get_multi_symbol_portfolio_dir(config: Any) -> Path:
    """返回多品种组合层结果目录。"""
    return get_frequency_scoped_dir(
        get_multi_symbol_root(config)
        / str(getattr(config, "multi_symbol_portfolio_subdir", "portfolio")),
        config,
    )


def get_multi_symbol_reports_dir(config: Any) -> Path:
    """返回多品种跨品种模型图表目录。"""
    return get_frequency_scoped_dir(
        get_multi_symbol_root(config)
        / str(getattr(config, "multi_symbol_reports_subdir", "reports")),
        config,
    )


def get_symbol_output_dir(config: Any, symbol: str) -> Path:
    """返回新版多品种单品种输出目录。"""
    return get_multi_symbol_symbols_dir(config) / safe_symbol_dir_name(symbol)


def get_legacy_symbol_output_dir(config: Any, symbol: str) -> Path:
    """返回旧版 ``by_symbol/<品种>`` 输出目录。"""
    return get_multi_symbol_root(config) / safe_symbol_dir_name(symbol)


def get_symbol_output_candidates(config: Any, symbol: str) -> list[Path]:
    """返回新版、旧版单品种目录候选，顺序代表读取优先级。"""
    paths = [get_symbol_output_dir(config, symbol), get_legacy_symbol_output_dir(config, symbol)]
    return list(dict.fromkeys(paths))


def resolve_existing_symbol_output_dir(config: Any, symbol: str) -> Path:
    """优先返回已经存在的品种目录，不存在时返回新版写入位置。"""
    for path in get_symbol_output_candidates(config, symbol):
        if path.exists():
            return path
    return get_symbol_output_dir(config, symbol)

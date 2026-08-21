from __future__ import annotations

"""项目统一命令行入口。

示例：
    python cli.py single --symbol C.DCE --scope new --start-index 126978
    python cli.py composite --symbol C.DCE --models xgboost,logistic_regression
    python cli.py multi --symbols C.DCE,M.DCE,Y.DCE --no-skip-existing
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from composite_factor_backtest import run_composite_backtest
from config import BacktestConfig, report_config_validation, resolve_symbol_universe
from framework.factors import build_single_factor_matrix, fetch_intraday_data, stop_wind
from multi_symbol_backtest import run_multi_symbol_backtest
from pooled_model_backtest import run_pooled_model_backtest
from framework.output_layout import apply_frequency_runtime_defaults, is_daily_frequency
from framework.runtime_utils import run_tracked
from single_factor_backtest import run_single_factor_backtests
from trading_signal import run_trading_signal_export


def parse_csv_values(value: str | None) -> list[str] | None:
    """解析逗号分隔的命令行列表。"""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_json_config(config: BacktestConfig, config_path: str | None) -> None:
    """从 JSON 文件覆盖配置中的已知字段。"""
    if not config_path:
        return
    with Path(config_path).open("r", encoding="utf-8") as file:
        overrides = json.load(file)
    if not isinstance(overrides, dict):
        raise ValueError("配置 JSON 顶层必须是对象。")
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"配置 JSON 包含未知参数: {name}")
        setattr(config, name, value)


def config_to_serializable_dict(config: BacktestConfig) -> dict[str, Any]:
    """把最终配置转换成可写入 JSON 的字典。"""
    return asdict(config)


def emit_final_config(config: BacktestConfig, args: argparse.Namespace) -> None:
    """按命令行要求打印或保存最终配置。"""
    config_dict = config_to_serializable_dict(config)
    if bool(getattr(args, "print_config", False)):
        print(json.dumps(config_dict, ensure_ascii=False, indent=2, sort_keys=True))

    save_path = getattr(args, "save_config", None)
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(config_dict, file, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"最终配置已保存: {path}")


def apply_common_overrides(config: BacktestConfig, args: argparse.Namespace) -> None:
    """应用各任务共享的常用参数覆盖。"""
    for arg_name, config_name in (
        ("symbol", "symbol"),
        ("start_time", "start_time"),
        ("end_time", "end_time"),
        ("frequency", "bar_frequency"),
        ("bar_size", "bar_size"),
        ("output_dir", "output_dir"),
        ("data_cache_dir", "data_cache_dir"),
        ("run_id", "run_id"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)
    frequency = getattr(args, "frequency", None)
    bar_size = getattr(args, "bar_size", None)
    if frequency and str(frequency).lower().endswith("min"):
        config.bar_size = int(str(frequency).lower().removesuffix("min"))
    elif bar_size is not None and not frequency:
        config.bar_frequency = f"{int(bar_size)}min"


def create_config(args: argparse.Namespace) -> BacktestConfig:
    """根据默认配置、JSON 配置和命令行覆盖构建最终运行配置。"""
    config = BacktestConfig()
    apply_json_config(config, getattr(args, "config_json", None))
    apply_common_overrides(config, args)

    if getattr(args, "scope", None) is not None:
        config.single_factor_scope = args.scope
    if getattr(args, "start_index", None) is not None:
        config.single_factor_new_factor_start_index = args.start_index
        config.single_factor_new_factor_start_index_by_symbol = {}
    range_start = getattr(args, "range_start", None)
    range_end = getattr(args, "range_end", None)
    if range_start is not None or range_end is not None:
        current_start, current_end = getattr(config, "single_factor_range", (1, 1))
        config.single_factor_range = (
            int(range_start if range_start is not None else current_start),
            int(range_end if range_end is not None else current_end),
        )
    selected_factors = parse_csv_values(getattr(args, "selected_factors", None))
    if selected_factors is not None:
        if getattr(args, "command", "") == "single":
            config.single_factor_selected_factors = selected_factors
        elif getattr(args, "command", "") == "composite":
            config.selected_factors = selected_factors
    if getattr(args, "feature_scope", None) is not None:
        config.xgboost_feature_scope = args.feature_scope
    if getattr(args, "models", None) is not None:
        config.composite_model_names = parse_csv_values(args.models) or []
    if getattr(args, "train_window", None) is not None:
        config.xgboost_train_window = args.train_window
    if getattr(args, "min_train_samples", None) is not None:
        config.xgboost_min_train_samples = args.min_train_samples
    if getattr(args, "retrain_every", None) is not None:
        config.xgboost_retrain_every = args.retrain_every
    if getattr(args, "symbols", None) is not None:
        config.symbols = resolve_symbol_universe(args.symbols)
        if getattr(args, "command", "") == "pooled":
            config.pooled_model_symbols = config.symbols
    if getattr(args, "pooled_scope", None) is not None:
        config.pooled_model_scope = args.pooled_scope
    if getattr(args, "pooled_feature_source", None) is not None:
        config.pooled_model_feature_source = args.pooled_feature_source
    if getattr(args, "pooled_max_features", None) is not None:
        config.pooled_model_max_features = args.pooled_max_features
    if getattr(args, "pooled_train_time_window", None) is not None:
        config.pooled_model_train_time_window = args.pooled_train_time_window
    if getattr(args, "pooled_max_train_rows", None) is not None:
        config.pooled_model_max_train_rows = args.pooled_max_train_rows
    if getattr(args, "pooled_model", None) is not None:
        config.pooled_model_name = args.pooled_model
    if getattr(args, "command", "") == "signal" and getattr(args, "mode", None) is not None:
        config.trading_signal_mode = args.mode
    if getattr(args, "skip_existing", None) is not None:
        config.multi_symbol_skip_existing = args.skip_existing
    if getattr(args, "run_single_factor", None) is not None:
        config.multi_symbol_run_single_factor = args.run_single_factor
    if getattr(args, "run_composite", None) is not None:
        config.multi_symbol_run_composite = args.run_composite
    if is_daily_frequency(config):
        # CLI 通用模型参数在日频下应覆盖 daily_* 默认值，而不是随后被默认值反向覆盖。
        if getattr(args, "train_window", None) is not None:
            config.daily_xgboost_train_window = int(args.train_window)
        if getattr(args, "min_train_samples", None) is not None:
            config.daily_xgboost_min_train_samples = int(args.min_train_samples)
        if getattr(args, "retrain_every", None) is not None:
            config.daily_xgboost_retrain_every = int(args.retrain_every)
    return apply_frequency_runtime_defaults(config)


def run_single(config: BacktestConfig) -> Any:
    """运行单因子流程。"""
    config = apply_frequency_runtime_defaults(config)
    try:
        print(f"读取 {config.symbol} 的 {config.bar_frequency} 数据...")
        data = fetch_intraday_data(config)
        print("按需构建单因子矩阵...")
        factors = build_single_factor_matrix(data, config)
        return run_single_factor_backtests(data, factors, config)
    finally:
        stop_wind()


def run_composite(config: BacktestConfig) -> Any:
    """运行综合因子流程。"""
    try:
        return run_composite_backtest(config)
    finally:
        stop_wind()


def run_multi(config: BacktestConfig) -> Any:
    """运行多品种流程。"""
    try:
        return run_multi_symbol_backtest(config)
    finally:
        stop_wind()


def run_pooled(config: BacktestConfig) -> Any:
    """运行多品种共享信息模型。"""
    try:
        return run_pooled_model_backtest(config)
    finally:
        stop_wind()


def run_signal(config: BacktestConfig, args: argparse.Namespace) -> Any:
    """导出最新交易信号。"""
    return run_trading_signal_export(
        config,
        symbols=getattr(args, "symbols", None) or getattr(args, "symbol", None),
        source=getattr(args, "source", "auto"),
        mode=getattr(args, "mode", None),
        output_path=getattr(args, "output", None),
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """添加全部流程可共享的运行参数。"""
    parser.add_argument("--config-json", help="JSON 配置覆盖文件路径。")
    parser.add_argument("--symbol", help="单品种运行时的 Wind 品种代码。")
    parser.add_argument("--start-time", help="回测开始时间。")
    parser.add_argument("--end-time", help="回测结束时间。")
    parser.add_argument(
        "--frequency",
        choices=["1d", "30min", "60min", "15min", "5min"],
        help="研究频率；1d 使用日频独立因子库和模型，其余使用分钟线。",
    )
    parser.add_argument("--bar-size", type=int, help="K 线周期，单位分钟。")
    parser.add_argument("--output-dir", help="结果输出目录。")
    parser.add_argument("--data-cache-dir", help="行情数据缓存目录。")
    parser.add_argument("--run-id", help="实验编号；不填则自动生成。")
    parser.add_argument("--print-config", action="store_true", help="运行前打印命令行覆盖后的最终配置。")
    parser.add_argument("--save-config", help="把命令行覆盖后的最终配置保存为 JSON。")
    parser.add_argument("--dry-run", action="store_true", help="只解析并输出最终配置，不执行回测或信号导出。")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(description="期货多因子研究框架统一运行入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="运行单因子回测与因子入库。")
    add_common_arguments(single)
    single.add_argument("--scope", choices=["all", "new", "range", "selected"], help="单因子构建范围。")
    single.add_argument("--start-index", type=int, help="new 模式下新增因子的起始编号。")
    single.add_argument("--range-start", type=int, help="range 模式下因子编号区间起点。")
    single.add_argument("--range-end", type=int, help="range 模式下因子编号区间终点。")
    single.add_argument("--selected-factors", help="selected 模式下逗号分隔的单因子名称列表。")

    composite = subparsers.add_parser("composite", help="运行综合因子滚动训练回测。")
    add_common_arguments(composite)
    composite.add_argument(
        "--feature-scope",
        choices=["all", "best", "selected"],
        help="综合模型在 active 池内的因子选择方式。",
    )
    composite.add_argument("--models", help="逗号分隔的模型列表。")
    composite.add_argument("--selected-factors", help="selected 模式下逗号分隔的 active 因子名称列表。")
    composite.add_argument("--train-window", type=int, help="滚动训练窗口长度。")
    composite.add_argument("--min-train-samples", type=int, help="最少训练样本数。")
    composite.add_argument("--retrain-every", type=int, help="每隔多少根 K 线重新训练。")

    multi = subparsers.add_parser("multi", help="运行多品种批量回测和组合汇总。")
    add_common_arguments(multi)
    multi.add_argument("--symbols", help="逗号分隔的品种列表。")
    multi.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否跳过已经存在结果的品种。",
    )
    multi.add_argument(
        "--run-single-factor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否为每个品种运行单因子流程并更新 active 因子库。",
    )
    multi.add_argument(
        "--run-composite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否为每个品种运行综合因子模型和单品种回测。",
    )

    pooled = subparsers.add_parser("pooled", help="运行多品种共享信息 pooled 模型。")
    add_common_arguments(pooled)
    pooled.add_argument("--symbols", help="逗号分隔的品种列表；不填则使用 config.symbols。")
    pooled.add_argument(
        "--pooled-scope",
        choices=["sector", "market"],
        help="共享模型范围：sector 按板块训练；market 全市场训练。",
    )
    pooled.add_argument(
        "--pooled-feature-source",
        choices=["active_union", "active_intersection"],
        help="共享特征来源：active_union 使用 active 并集；active_intersection 使用交集。",
    )
    pooled.add_argument("--pooled-max-features", type=int, help="共享模型最多使用的基础因子数。")
    pooled.add_argument("--pooled-train-time-window", type=int, help="共享模型每次滚动训练最多回看的历史时间点数量。")
    pooled.add_argument("--pooled-max-train-rows", type=int, help="共享模型每次滚动训练最多使用的 long-format 样本行数。")
    pooled.add_argument(
        "--pooled-model",
        choices=["xgboost", "logistic_regression", "random_forest", "extra_trees"],
        help="共享模型分类器。",
    )

    signal = subparsers.add_parser("signal", help="导出最新单品种或多品种交易信号。")
    add_common_arguments(signal)
    signal.add_argument("--symbols", help="逗号分隔的品种列表；不填则使用 config.symbols。")
    signal.add_argument(
        "--mode",
        choices=["compute", "model", "vote", "detail"],
        default=None,
        help=(
            "信号生成模式：compute/model 复用综合回测滚动模型；"
            "vote 使用 active 因子加权投票；detail 读取已有 composite_detail.csv。"
        ),
    )
    signal.add_argument(
        "--source",
        choices=["auto", "single", "multi"],
        default="auto",
        help=(
            "信号来源：single 读根目录 composite_factor；"
            "multi 读 by_symbol/symbols；auto 优先多品种目录。"
        ),
    )
    signal.add_argument("--output", help="输出 CSV 路径。")
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    config = create_config(args)
    report_config_validation(config, args.command)
    emit_final_config(config, args)
    if bool(getattr(args, "dry_run", False)):
        print("dry-run 模式：已跳过实际运行。")
        return
    runners = {
        "single": run_single,
        "composite": run_composite,
        "multi": run_multi,
        "pooled": run_pooled,
        "signal": lambda current_config: run_signal(current_config, args),
    }
    run_tracked(config, args.command, lambda: runners[args.command](config))


if __name__ == "__main__":
    main()

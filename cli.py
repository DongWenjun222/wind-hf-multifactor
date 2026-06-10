from __future__ import annotations

"""项目统一命令行入口。

示例：
    python cli.py single --symbol C.DCE --scope new --start-index 126978
    python cli.py composite --symbol C.DCE --models xgboost,logistic_regression
    python cli.py multi --symbols C.DCE,M.DCE,Y.DCE --no-skip-existing
"""

import argparse
import json
from pathlib import Path
from typing import Any

from composite_factor_backtest import run_composite_backtest
from config import BacktestConfig, resolve_symbol_universe
from factors import build_single_factor_matrix, fetch_intraday_data, stop_wind
from multi_symbol_backtest import run_multi_symbol_backtest
from runtime_utils import run_tracked
from single_factor_backtest import run_single_factor_backtests


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


def apply_common_overrides(config: BacktestConfig, args: argparse.Namespace) -> None:
    """应用各任务共享的常用参数覆盖。"""
    for arg_name, config_name in (
        ("symbol", "symbol"),
        ("start_time", "start_time"),
        ("end_time", "end_time"),
        ("bar_size", "bar_size"),
        ("output_dir", "output_dir"),
        ("data_cache_dir", "data_cache_dir"),
        ("run_id", "run_id"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)


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
    if getattr(args, "skip_existing", None) is not None:
        config.multi_symbol_skip_existing = args.skip_existing
    return config


def run_single(config: BacktestConfig) -> Any:
    """运行单因子流程。"""
    try:
        print(f"读取 {config.symbol} 的 {config.bar_size} 分钟数据...")
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


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """添加全部流程可共享的运行参数。"""
    parser.add_argument("--config-json", help="JSON 配置覆盖文件路径。")
    parser.add_argument("--symbol", help="单品种运行时的 Wind 品种代码。")
    parser.add_argument("--start-time", help="回测开始时间。")
    parser.add_argument("--end-time", help="回测结束时间。")
    parser.add_argument("--bar-size", type=int, help="K 线周期，单位分钟。")
    parser.add_argument("--output-dir", help="结果输出目录。")
    parser.add_argument("--data-cache-dir", help="行情数据缓存目录。")
    parser.add_argument("--run-id", help="实验编号；不填则自动生成。")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(description="期货多因子研究框架统一运行入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="运行单因子回测与因子入库。")
    add_common_arguments(single)
    single.add_argument("--scope", choices=["all", "new", "selected"], help="单因子构建范围。")
    single.add_argument("--start-index", type=int, help="new 模式下新增因子的起始编号。")

    composite = subparsers.add_parser("composite", help="运行综合因子滚动训练回测。")
    add_common_arguments(composite)
    composite.add_argument(
        "--feature-scope",
        choices=["all", "best", "selected"],
        help="综合模型在 active 池内的因子选择方式。",
    )
    composite.add_argument("--models", help="逗号分隔的模型列表。")
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
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()
    config = create_config(args)
    runners = {
        "single": run_single,
        "composite": run_composite,
        "multi": run_multi,
    }
    run_tracked(config, args.command, lambda: runners[args.command](config))


if __name__ == "__main__":
    main()

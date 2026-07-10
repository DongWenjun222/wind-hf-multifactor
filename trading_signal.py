from __future__ import annotations

"""最新交易信号导出工具。

本脚本不重新训练模型，只读取已经生成的 composite_detail.csv，
提取最新一根 K 线对应的下一根目标仓位，生成单品种或多品种交易指令表。
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig, resolve_symbol_universe
from factors import safe_symbol_name


SIGNAL_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "prob_down",
    "prob_flat",
    "prob_up",
    "prob_edge",
    "directional_probability",
    "calibrated_prob_edge",
    "calibrated_min_edge",
    "calibrated_min_probability",
    "trade_allowed",
    "confidence_rank",
    "confidence_trade_allowed",
    "position_size",
    "raw_signal",
    "raw_signal_before_position_rules",
    "target_position_before_rules",
    "target_position",
    "position",
    "benchmark_vote_score",
    "benchmark_vote_raw_signal",
    "benchmark_vote_target_position",
    "benchmark_vote_position",
    "volatility_rank",
    "liquidity_rank",
    "trend_strength",
    "volatility_regime",
    "liquidity_regime",
    "trend_regime",
    "session_regime",
]


def normalize_signal_symbols(symbols: str | list[str] | None, config: BacktestConfig) -> list[str]:
    """解析交易信号需要导出的品种列表。"""
    if symbols:
        return resolve_symbol_universe(symbols)
    return [config.symbol]


def get_symbol_composite_detail_path(config: BacktestConfig, symbol: str, source: str) -> Path:
    """根据来源模式返回某个品种的综合回测明细路径。"""
    source = str(source).lower()
    if source == "single":
        return Path(config.output_dir) / "composite_factor" / "composite_detail.csv"

    symbol_dir = safe_symbol_name(symbol).upper()
    multi_path = (
        Path(config.output_dir)
        / getattr(config, "multi_symbol_output_subdir", "by_symbol")
        / symbol_dir
        / "composite_factor"
        / "composite_detail.csv"
    )
    if source == "multi":
        return multi_path

    single_path = Path(config.output_dir) / "composite_factor" / "composite_detail.csv"
    return multi_path if multi_path.exists() else single_path


def read_latest_signal_row(detail_path: Path) -> pd.Series:
    """读取 composite_detail.csv 中最新一根有效信号。"""
    if not detail_path.exists():
        raise FileNotFoundError(f"没有找到综合回测明细: {detail_path}")
    detail = pd.read_csv(detail_path, index_col=0, parse_dates=True)
    if detail.empty:
        raise ValueError(f"综合回测明细为空: {detail_path}")

    signal_like_columns = [
        column
        for column in ["target_position", "raw_signal", "calibrated_prob_edge", "position"]
        if column in detail.columns
    ]
    if signal_like_columns:
        valid = detail.dropna(subset=signal_like_columns, how="all")
        if not valid.empty:
            detail = valid
    return detail.iloc[-1]


def safe_float(value: Any, default: float = np.nan) -> float:
    """安全转换为 float。"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_optional_value(row: pd.Series, column: str, default: Any = np.nan) -> Any:
    """读取可选列。"""
    return row[column] if column in row.index else default


def infer_next_target_position(row: pd.Series) -> float:
    """推导下一根 K 线目标仓位。"""
    if "target_position" in row.index and pd.notna(row["target_position"]):
        return safe_float(row["target_position"], 0.0)
    raw_signal = safe_float(get_optional_value(row, "raw_signal", 0.0), 0.0)
    position_size = safe_float(get_optional_value(row, "position_size", 1.0), 1.0)
    return raw_signal * position_size


def describe_direction(value: float, eps: float = 1e-9) -> str:
    """把仓位或信号数值转成中文方向。"""
    if value > eps:
        return "做多"
    if value < -eps:
        return "做空"
    return "空仓"


def describe_adjustment(delta: float, eps: float = 1e-9) -> str:
    """把调仓量转成可读动作。"""
    if delta > eps:
        return "加多/减空"
    if delta < -eps:
        return "减多/加空"
    return "不调仓"


def build_symbol_signal(config: BacktestConfig, symbol: str, source: str = "auto") -> dict[str, Any]:
    """生成单个品种的最新交易信号行。"""
    detail_path = get_symbol_composite_detail_path(config, symbol, source)
    row = read_latest_signal_row(detail_path)
    signal_time = row.name
    current_position = safe_float(get_optional_value(row, "position", 0.0), 0.0)
    next_target_position = infer_next_target_position(row)
    adjustment = next_target_position - current_position

    result: dict[str, Any] = {
        "品种": symbol,
        "信号时间": signal_time,
        "信号来源文件": str(detail_path),
        "最新收盘价": safe_float(get_optional_value(row, "close", np.nan)),
        "当前实际仓位": current_position,
        "下一根目标仓位": next_target_position,
        "调仓量": adjustment,
        "交易方向": describe_direction(next_target_position),
        "调仓动作": describe_adjustment(adjustment),
    }
    for column in SIGNAL_COLUMNS:
        if column in row.index and column not in result:
            result[column] = row[column]
    return result


def build_trading_signals(
    config: BacktestConfig,
    symbols: list[str],
    source: str = "auto",
) -> pd.DataFrame:
    """批量生成最新交易信号表。"""
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            rows.append(build_symbol_signal(config, symbol, source=source))
        except Exception as exc:
            rows.append(
                {
                    "品种": symbol,
                    "信号时间": "",
                    "信号来源文件": "",
                    "最新收盘价": np.nan,
                    "当前实际仓位": np.nan,
                    "下一根目标仓位": np.nan,
                    "调仓量": np.nan,
                    "交易方向": "",
                    "调仓动作": "",
                    "错误": str(exc),
                }
            )
    return pd.DataFrame(rows)


def save_trading_signals(
    signals: pd.DataFrame,
    config: BacktestConfig,
    output_path: str | None = None,
) -> Path:
    """保存最新交易信号表。"""
    if output_path:
        path = Path(output_path)
    else:
        path = Path(config.output_dir) / "trading_signals" / "trading_signals_latest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_trading_signal_export(
    config: BacktestConfig,
    symbols: str | list[str] | None = None,
    source: str = "auto",
    output_path: str | None = None,
) -> pd.DataFrame:
    """生成并保存最新交易信号。"""
    selected_symbols = normalize_signal_symbols(symbols, config)
    signals = build_trading_signals(config, selected_symbols, source=source)
    path = save_trading_signals(signals, config, output_path)
    print(f"最新交易信号已保存: {path}")
    if not signals.empty:
        display_columns = [
            column
            for column in ["品种", "信号时间", "交易方向", "当前实际仓位", "下一根目标仓位", "调仓量", "调仓动作", "错误"]
            if column in signals.columns
        ]
        print(signals[display_columns].to_string(index=False))
    return signals


def build_parser() -> argparse.ArgumentParser:
    """创建独立脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="导出最新单品种或多品种交易信号")
    parser.add_argument("--symbols", help="逗号分隔品种列表；不填则使用 config.symbols。")
    parser.add_argument(
        "--source",
        choices=["auto", "single", "multi"],
        default="auto",
        help="信号来源：single 读根目录 composite_factor；multi 读 by_symbol；auto 优先 by_symbol。",
    )
    parser.add_argument("--output", help="输出 CSV 路径；不填则写入 output_dir/trading_signals/trading_signals_latest.csv。")
    parser.add_argument("--output-dir", help="覆盖 config.output_dir。")
    return parser


def main() -> None:
    """独立脚本入口。"""
    args = build_parser().parse_args()
    config = BacktestConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    run_trading_signal_export(
        config,
        symbols=args.symbols,
        source=args.source,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

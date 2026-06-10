from __future__ import annotations

"""数据质量报告工具。

运行后会检查主品种、相关品种和宏观日频数据的基础质量，并输出 CSV：
- 行数、起止时间、重复时间戳。
- 缺失率、零成交量比例、OHLC 关系异常。
- K 线间隔是否接近配置周期。
- 极端收益占比。
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig
from data_loader import (
    fetch_intraday_data,
    fetch_macro_state_data,
    fetch_related_intraday_data,
    stop_wind,
)


def infer_interval_stats(index: pd.DatetimeIndex, expected_minutes: int | None = None) -> dict[str, Any]:
    """推断时间索引间隔质量。"""
    if len(index) < 2:
        return {
            "推断间隔分钟": np.nan,
            "间隔异常数量": 0,
            "间隔异常占比": np.nan,
            "最大间隔分钟": np.nan,
        }
    intervals = pd.Series(index).sort_values().diff().dropna().dt.total_seconds().div(60.0)
    intervals = intervals[intervals > 0]
    if intervals.empty:
        return {
            "推断间隔分钟": np.nan,
            "间隔异常数量": 0,
            "间隔异常占比": np.nan,
            "最大间隔分钟": np.nan,
        }
    inferred = float(intervals.round().mode().iloc[0])
    expected = float(expected_minutes or inferred)
    abnormal = (intervals.round() != round(expected))
    return {
        "推断间隔分钟": inferred,
        "间隔异常数量": int(abnormal.sum()),
        "间隔异常占比": float(abnormal.mean()),
        "最大间隔分钟": float(intervals.max()),
    }


def summarize_intraday_data(
    data: pd.DataFrame,
    symbol: str,
    role: str,
    expected_bar_size: int,
    extreme_return_threshold: float,
) -> dict[str, Any]:
    """汇总分钟行情质量。"""
    index = pd.DatetimeIndex(data.index)
    close = data.get("close", pd.Series(dtype="float64")).replace(0, np.nan)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    ohlc_violation = pd.Series(False, index=data.index)
    if {"open", "high", "low", "close"}.issubset(data.columns):
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        close_ = data["close"]
        ohlc_violation = (
            (high < low)
            | (open_ > high)
            | (open_ < low)
            | (close_ > high)
            | (close_ < low)
        )
    volume = data.get("volume", pd.Series(np.nan, index=data.index))
    row = {
        "数据类型": role,
        "代码": symbol,
        "行数": int(len(data)),
        "开始时间": str(index.min()) if len(index) else "",
        "结束时间": str(index.max()) if len(index) else "",
        "重复时间戳数量": int(pd.Index(index).duplicated().sum()),
        "价格缺失率": float(data[["open", "high", "low", "close"]].isna().mean().mean())
        if {"open", "high", "low", "close"}.issubset(data.columns)
        else np.nan,
        "成交量缺失率": float(volume.isna().mean()) if len(volume) else np.nan,
        "零成交量占比": float((volume.fillna(0.0) == 0).mean()) if len(volume) else np.nan,
        "OHLC异常数量": int(ohlc_violation.fillna(False).sum()),
        "OHLC异常占比": float(ohlc_violation.fillna(False).mean()) if len(ohlc_violation) else np.nan,
        "极端收益数量": int((returns.abs() > extreme_return_threshold).sum()),
        "极端收益占比": float((returns.abs() > extreme_return_threshold).mean()) if len(returns) else np.nan,
        "最大单根收益": float(returns.max()) if returns.notna().any() else np.nan,
        "最小单根收益": float(returns.min()) if returns.notna().any() else np.nan,
    }
    row.update(infer_interval_stats(index, expected_bar_size))
    return row


def summarize_macro_data(data: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """汇总宏观日频数据质量。"""
    index = pd.DatetimeIndex(data.index)
    close = data.get("close", pd.Series(dtype="float64")).replace(0, np.nan)
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return {
        "数据类型": "macro_daily",
        "代码": symbol,
        "行数": int(len(data)),
        "开始时间": str(index.min()) if len(index) else "",
        "结束时间": str(index.max()) if len(index) else "",
        "重复时间戳数量": int(pd.Index(index).duplicated().sum()),
        "价格缺失率": float(close.isna().mean()) if len(close) else np.nan,
        "成交量缺失率": np.nan,
        "零成交量占比": np.nan,
        "OHLC异常数量": np.nan,
        "OHLC异常占比": np.nan,
        "极端收益数量": int((returns.abs() > 0.05).sum()),
        "极端收益占比": float((returns.abs() > 0.05).mean()) if len(returns) else np.nan,
        "最大单根收益": float(returns.max()) if returns.notna().any() else np.nan,
        "最小单根收益": float(returns.min()) if returns.notna().any() else np.nan,
        **infer_interval_stats(index, None),
    }


def build_data_quality_report(config: BacktestConfig, extreme_return_threshold: float = 0.08) -> pd.DataFrame:
    """构建完整数据质量报告。"""
    rows: list[dict[str, Any]] = []
    main_data = fetch_intraday_data(config)
    rows.append(
        summarize_intraday_data(
            main_data,
            config.symbol,
            "main_intraday",
            config.bar_size,
            extreme_return_threshold,
        )
    )

    related_data_map = fetch_related_intraday_data(config)
    for symbol, data in related_data_map.items():
        rows.append(
            summarize_intraday_data(
                data,
                symbol,
                "related_intraday",
                config.bar_size,
                extreme_return_threshold,
            )
        )

    macro_data_map = fetch_macro_state_data(config)
    for symbol, data in macro_data_map.items():
        rows.append(summarize_macro_data(data, symbol))
    return pd.DataFrame(rows)


def save_data_quality_report(
    config: BacktestConfig,
    output_path: Path | None = None,
    extreme_return_threshold: float = 0.08,
) -> Path:
    """保存数据质量报告。"""
    if output_path is None:
        output_path = Path(config.output_dir) / "data_quality_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_data_quality_report(config, extreme_return_threshold)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成主品种、相关品种和宏观数据质量报告。")
    parser.add_argument("--symbol", help="覆盖 config.symbol。")
    parser.add_argument("--output", help="输出 CSV 路径，默认 output_dir/data_quality_report.csv。")
    parser.add_argument("--extreme-return-threshold", type=float, default=0.08, help="分钟线极端收益阈值。")
    args = parser.parse_args()

    config = BacktestConfig()
    if args.symbol:
        config.symbol = args.symbol
    try:
        output_path = save_data_quality_report(
            config,
            Path(args.output) if args.output else None,
            args.extreme_return_threshold,
        )
        print(f"数据质量报告已保存: {output_path}")
    finally:
        stop_wind()


if __name__ == "__main__":
    main()

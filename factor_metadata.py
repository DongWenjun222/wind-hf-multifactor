from __future__ import annotations

"""因子元数据生成工具。

用途：
- 为当前可生成的因子目录生成 factor_metadata.csv。
- 记录因子编号、类别、来源模块、是否跨品种、是否宏观/日历、复杂度估计等。
- 便于后续做因子解释、家族聚类、相关性去重、AI 因子治理和硬删除。
"""

import argparse
from pathlib import Path

import pandas as pd

from config import BacktestConfig
from factor_taxonomy import classify_factor
from factors import build_factor_name_catalog


def build_factor_metadata(config: BacktestConfig) -> pd.DataFrame:
    """构建完整因子目录元数据表。"""
    dummy_index = pd.date_range(config.start_time, periods=120, freq=f"{config.bar_size}min")
    dummy_data = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.2,
            "volume": 1000.0,
            "amt": 100000.0,
            "amount": 100000.0,
        },
        index=dummy_index,
    )
    factor_names = build_factor_name_catalog(dummy_data, config)
    rows = []
    for factor_id, factor_name in enumerate(factor_names, start=1):
        row = classify_factor(factor_name)
        row["因子编号"] = factor_id
        row["因子标签"] = f"{factor_id}_{factor_name}"
        rows.append(row)
    columns = [
        "因子编号",
        "因子标签",
        "因子",
        "因子家族",
        "来源文件",
        "是否跨品种",
        "是否宏观",
        "是否日历",
        "是否基础因子",
        "估计窗口",
        "名称片段数",
        "复杂度估计",
    ]
    return pd.DataFrame(rows)[columns]


def save_factor_metadata(config: BacktestConfig, output_path: Path | None = None) -> Path:
    """生成并保存因子元数据。"""
    if output_path is None:
        output_path = Path(config.output_dir) / "factor_metadata.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_factor_metadata(config)
    metadata.to_csv(output_path, index=False, encoding="utf-8-sig")
    family_summary = (
        metadata.groupby("因子家族")
        .agg(因子数量=("因子", "count"), 最小编号=("因子编号", "min"), 最大编号=("因子编号", "max"))
        .reset_index()
    )
    family_summary.to_csv(
        output_path.with_name(output_path.stem + "_family_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成因子元数据表。")
    parser.add_argument("--output", help="输出 CSV 路径，默认写入 output_dir/factor_metadata.csv。")
    args = parser.parse_args()
    config = BacktestConfig()
    output_path = save_factor_metadata(config, Path(args.output) if args.output else None)
    print(f"因子元数据已保存: {output_path}")
    print(f"因子家族汇总已保存: {output_path.with_name(output_path.stem + '_family_summary.csv')}")


if __name__ == "__main__":
    main()

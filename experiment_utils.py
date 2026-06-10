from __future__ import annotations

"""实验追踪辅助工具。

项目会保留现有的“latest”输出文件以便快速查看，同时也会把关键结果复制到带时间戳的
实验目录中。这样即使后续运行覆盖了最新文件，历史研究结果仍然可以追踪。
"""

from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import shutil
from typing import Iterable

import pandas as pd

from config import BacktestConfig
from factor_library import get_factor_library_dir


def get_run_id(config: BacktestConfig, run_type: str) -> str:
    """返回当前脚本执行对应的稳定实验编号。"""
    if getattr(config, "run_id", None):
        return f"{config.run_id}_{run_type}"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{run_type}"


def get_experiment_run_dir(config: BacktestConfig, run_type: str) -> Path | None:
    """创建并返回实验目录；如果关闭实验目录功能则返回 None。"""
    if not getattr(config, "enable_experiment_run_dirs", True):
        return None

    run_dir = Path(config.output_dir) / "runs" / get_run_id(config, run_type)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(config: BacktestConfig, output_dir: Path) -> None:
    """保存本次运行实际使用的完整配置。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(asdict(config), config_file, ensure_ascii=False, indent=2, default=str)


def copy_existing_files(paths: Iterable[Path], target_dir: Path) -> None:
    """把已经存在的结果文件复制到实验目录。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists() and path.is_file():
            shutil.copy2(path, target_dir / path.name)


def snapshot_active_factor_library(config: BacktestConfig, target_dir: Path) -> Path | None:
    """把当前 active 因子库复制到实验目录。"""
    active_path = get_factor_library_dir(config) / "active_factors.csv"
    if not active_path.exists():
        return None

    snapshot_path = target_dir / "active_factors_snapshot.csv"
    shutil.copy2(active_path, snapshot_path)
    return snapshot_path


def write_factor_count_snapshot(factors: pd.DataFrame, target_dir: Path) -> Path:
    """写入简洁的因子数量快照，便于快速比较实验。"""
    cross_count = int(sum(str(column).startswith("cross_") for column in factors.columns))
    snapshot = {
        "factor_count": int(factors.shape[1]),
        "cross_asset_factor_count": cross_count,
        "non_cross_factor_count": int(factors.shape[1] - cross_count),
    }
    path = target_dir / "factor_count.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)
    return path

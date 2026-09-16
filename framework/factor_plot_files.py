from __future__ import annotations

"""单因子 latest 图表文件的安全清理工具。"""

from pathlib import Path
from typing import Any

import pandas as pd

from .output_layout import get_research_output_dir


def delete_single_factor_plot_files(
    config: Any,
    factor_names: list[str],
    *catalog_frames: pd.DataFrame,
) -> list[Path]:
    """删除指定因子的 latest 回测图，不触碰历史 runs 快照。"""
    normalized_names = {str(name).strip() for name in factor_names if str(name).strip()}
    if not normalized_names:
        return []

    single_factor_dir = get_research_output_dir(config, "single_factor").resolve()
    if not single_factor_dir.exists():
        return []

    frames = [
        frame
        for frame in catalog_frames
        if frame is not None and not frame.empty and "因子" in frame.columns
    ]
    catalog = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if catalog.empty:
        return []
    selected = catalog[catalog["因子"].astype(str).isin(normalized_names)].copy()
    expected_names: set[str] = set()
    if "因子标签" in selected.columns:
        expected_names.update(
            f"{label}_report.png"
            for label in selected["因子标签"].dropna().astype(str)
            if label.strip()
        )
    if "因子编号" in selected.columns:
        for _, row in selected.iterrows():
            factor_id = pd.to_numeric(row.get("因子编号"), errors="coerce")
            factor_name = str(row.get("因子", "")).strip()
            if pd.notna(factor_id) and factor_name:
                expected_names.add(f"{int(factor_id)}_{factor_name}_report.png")

    candidates: set[Path] = set()
    if "图片文件" in selected.columns:
        for raw_path in selected["图片文件"].dropna().astype(str):
            if not raw_path.strip():
                continue
            path = Path(raw_path)
            resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            candidates.add(resolved)
    for path in single_factor_dir.rglob("*.png"):
        if path.name in expected_names:
            candidates.add(path.resolve())

    deleted: list[Path] = []
    for path in sorted(candidates, key=str):
        try:
            path.relative_to(single_factor_dir)
        except ValueError:
            print(f"跳过不在当前单因子目录内的图片路径: {path}")
            continue
        if path.is_file() and path.suffix.lower() == ".png":
            try:
                path.unlink()
                deleted.append(path)
            except OSError as exc:
                print(f"单因子图片删除失败: {path}, {exc}")
    return deleted

from __future__ import annotations

"""active 因子逻辑审查档案与增量识别工具。"""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .factor_library import get_factor_library_dir
from .factor_taxonomy import classify_factor


LOGIC_REVIEW_VERSION = "1"
LOGIC_REVIEW_COLUMNS = [
    "因子",
    "逻辑审查状态",
    "逻辑类型",
    "逻辑说明",
    "主要风险",
    "逻辑审查置信度",
    "逻辑审查时间",
    "逻辑审查模型",
    "逻辑审查版本",
    "逻辑审查指纹",
    "因子家族",
    "来源文件",
]
ACTIVE_LOGIC_COLUMNS = [
    "逻辑审查状态",
    "逻辑类型",
    "逻辑说明",
    "主要风险",
    "逻辑审查置信度",
    "逻辑审查时间",
    "逻辑审查模型",
    "逻辑审查版本",
    "逻辑审查指纹",
]
VALID_LOGIC_STATUSES = {"approved", "rejected", "uncertain"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_factor_logic_reviews_path(config: Any) -> Path:
    """返回当前输出目录和频率对应的永久逻辑审查档案。"""
    return get_factor_library_dir(config) / "factor_logic_reviews.csv"


def load_factor_logic_reviews(config: Any) -> pd.DataFrame:
    """读取审查档案；损坏或不存在时返回结构完整的空表。"""
    path = get_factor_logic_reviews_path(config)
    if not path.exists():
        return pd.DataFrame(columns=LOGIC_REVIEW_COLUMNS)
    try:
        reviews = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"因子逻辑审查档案读取失败，将保留原文件且不复用记录: {path}, 原因: {exc}")
        return pd.DataFrame(columns=LOGIC_REVIEW_COLUMNS)
    for column in LOGIC_REVIEW_COLUMNS:
        if column not in reviews.columns:
            reviews[column] = ""
    reviews["因子"] = reviews["因子"].fillna("").astype(str).str.strip()
    reviews = reviews[reviews["因子"].ne("")]
    return reviews[LOGIC_REVIEW_COLUMNS].drop_duplicates("因子", keep="last").reset_index(drop=True)


def save_factor_logic_reviews(reviews: pd.DataFrame, config: Any) -> Path:
    """原子保存审查档案，避免批量审查中断后留下半个 CSV。"""
    path = get_factor_logic_reviews_path(config)
    output = reviews.copy()
    for column in LOGIC_REVIEW_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output = output[LOGIC_REVIEW_COLUMNS].drop_duplicates("因子", keep="last")
    temp_path = path.with_name(f".{path.name}.tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)
    return path


def _source_digest(source_file: str, project_root: Path) -> str:
    source_path = project_root / str(source_file)
    try:
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        return "source-unavailable"


def build_factor_logic_context(
    factors: pd.DataFrame,
    *,
    project_root: Path | None = None,
    review_version: str = LOGIC_REVIEW_VERSION,
) -> pd.DataFrame:
    """构建因子审查上下文和公式来源指纹。"""
    root = Path(project_root or PROJECT_ROOT).resolve()
    if factors.empty or "因子" not in factors.columns:
        return pd.DataFrame(
            columns=["因子", "因子家族", "来源文件", "复杂度估计", "逻辑审查指纹"]
        )
    rows: list[dict[str, Any]] = []
    source_digests: dict[str, str] = {}
    for factor_name in factors["因子"].dropna().astype(str).str.strip().drop_duplicates():
        if not factor_name:
            continue
        metadata = classify_factor(factor_name)
        source_file = str(metadata["来源文件"])
        source_digest = source_digests.setdefault(
            source_file,
            _source_digest(source_file, root),
        )
        fingerprint_payload = "|".join(
            [factor_name, str(metadata["因子家族"]), source_file, source_digest, review_version]
        )
        rows.append(
            {
                "因子": factor_name,
                "因子家族": metadata["因子家族"],
                "来源文件": source_file,
                "复杂度估计": metadata["复杂度估计"],
                "是否跨品种": bool(metadata["是否跨品种"]),
                "是否宏观": bool(metadata["是否宏观"]),
                "是否日历": bool(metadata["是否日历"]),
                "逻辑审查指纹": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def get_pending_logic_reviews(
    active: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    project_root: Path | None = None,
    review_version: str = LOGIC_REVIEW_VERSION,
) -> pd.DataFrame:
    """只返回从未审查或因构造器/规则版本变化而失效的 active 因子。"""
    context = build_factor_logic_context(
        active,
        project_root=project_root,
        review_version=review_version,
    )
    if context.empty:
        return context
    if reviews.empty:
        return context.reset_index(drop=True)
    latest = reviews.drop_duplicates("因子", keep="last").set_index("因子")
    completed_fingerprints = latest["逻辑审查指纹"].fillna("").astype(str).to_dict()
    completed_statuses = latest["逻辑审查状态"].fillna("").astype(str).str.lower().to_dict()
    pending_mask = context.apply(
        lambda row: (
            completed_fingerprints.get(str(row["因子"])) != str(row["逻辑审查指纹"])
            or completed_statuses.get(str(row["因子"])) not in VALID_LOGIC_STATUSES
        ),
        axis=1,
    )
    return context.loc[pending_mask].reset_index(drop=True)


def merge_factor_logic_reviews(
    frame: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    project_root: Path | None = None,
    review_version: str = LOGIC_REVIEW_VERSION,
) -> pd.DataFrame:
    """把仍匹配当前公式指纹的最新审查结论附加到因子表。"""
    if frame.empty or "因子" not in frame.columns:
        return frame.copy()
    output = frame.drop(columns=[c for c in ACTIVE_LOGIC_COLUMNS if c in frame.columns]).copy()
    context = build_factor_logic_context(
        output,
        project_root=project_root,
        review_version=review_version,
    )[["因子", "逻辑审查指纹"]]
    latest = reviews.drop_duplicates("因子", keep="last").copy()
    if latest.empty:
        output["逻辑审查状态"] = "pending"
        for column in ACTIVE_LOGIC_COLUMNS:
            if column != "逻辑审查状态":
                output[column] = ""
        return output
    for column in LOGIC_REVIEW_COLUMNS:
        if column not in latest.columns:
            latest[column] = ""
    valid = context.merge(latest, on=["因子", "逻辑审查指纹"], how="left")
    review_columns = ["因子", *ACTIVE_LOGIC_COLUMNS]
    output = output.merge(valid[review_columns], on="因子", how="left")
    output["逻辑审查状态"] = output["逻辑审查状态"].fillna("pending")
    return output

from __future__ import annotations

"""项目源码指纹工具。"""

import hashlib
from pathlib import Path


DEFAULT_FINGERPRINT_FILES = [
    Path("config.py"),
    Path("data_loader.py"),
    Path("factors.py"),
    Path("factor_builders/__init__.py"),
    Path("factor_builders/basic.py"),
    Path("factor_builders/parametric.py"),
    Path("factor_builders/calendar.py"),
    Path("factor_builders/cross_asset.py"),
    Path("factor_builders/macro_state.py"),
    Path("factor_builders/non_cross.py"),
    Path("factor_builders/common.py"),
]


def hash_file(path: Path) -> str:
    """计算单个文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_fingerprint(paths: list[Path] | None = None) -> dict[str, str]:
    """返回影响因子构建的源码文件哈希。"""
    paths = paths or DEFAULT_FINGERPRINT_FILES
    fingerprint: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            fingerprint[str(path).replace("\\", "/")] = hash_file(path)
    return fingerprint


def build_source_fingerprint_hash(paths: list[Path] | None = None) -> str:
    """把源码文件哈希折叠成一个总哈希。"""
    fingerprint = build_source_fingerprint(paths)
    digest = hashlib.sha256()
    for path, file_hash in sorted(fingerprint.items()):
        digest.update(path.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return digest.hexdigest()

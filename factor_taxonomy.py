from __future__ import annotations

"""因子分类与治理工具。

本模块只根据因子名称做轻量分类，不读取行情、不构建完整因子矩阵。
它被因子元数据导出和因子库入库共同使用，避免两个地方维护两套家族规则。
"""

import re
from typing import Any


BASIC_FACTORS = {"momentum", "reversal", "breakout", "volume_confirm"}


def estimate_factor_complexity(factor_name: str, family: str, token_count: int) -> str:
    """给因子一个粗粒度复杂度标签，方便后续治理。"""
    if family == "basic":
        return "low"
    if family in {"calendar", "macro_state"} and token_count <= 5:
        return "medium"
    if any(prefix in factor_name for prefix in ("omega_", "crossomega_", "hyper_", "crosshyper_")):
        return "high"
    if token_count >= 7:
        return "high"
    if token_count >= 5:
        return "medium"
    return "low"


def classify_factor(factor_name: str) -> dict[str, Any]:
    """根据因子名推断因子家族和工程属性。"""
    factor_name = str(factor_name)
    if factor_name in BASIC_FACTORS:
        family = "basic"
        source_file = "factor_builders/basic.py"
    elif factor_name.startswith("calendar_"):
        family = "calendar"
        source_file = "factor_builders/calendar.py"
    elif factor_name.startswith("macro_"):
        family = "macro_state"
        source_file = "factor_builders/macro_state.py"
    elif factor_name.startswith(("cross_", "crossmega_", "crossultra_", "crosshyper_", "crossomega_")):
        family = "cross_asset"
        source_file = "factor_builders/cross_asset.py"
    elif factor_name.startswith(("ultra_", "hyper_", "omega_")):
        family = "non_cross_complex"
        source_file = "factor_builders/non_cross.py"
    else:
        family = "parametric"
        source_file = "factor_builders/parametric.py"

    tokens = factor_name.split("_")
    window_match = re.search(r"_(\d+)$", factor_name)
    window = int(window_match.group(1)) if window_match else None
    return {
        "因子": factor_name,
        "因子家族": family,
        "来源文件": source_file,
        "是否跨品种": family == "cross_asset",
        "是否宏观": family == "macro_state",
        "是否日历": family == "calendar",
        "是否基础因子": family == "basic",
        "估计窗口": window,
        "名称片段数": len(tokens),
        "复杂度估计": estimate_factor_complexity(factor_name, family, len(tokens)),
    }


def get_factor_family(factor_name: str) -> str:
    """返回因子家族名称。"""
    return str(classify_factor(factor_name)["因子家族"])

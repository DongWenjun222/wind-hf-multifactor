from __future__ import annotations

"""未来函数与数据泄露静态审计工具。

这个脚本不会证明代码一定无泄露，但可以把高风险写法集中列出来：
- shift(-n)、bfill、全样本统计、未来标签构造等。
- 对已知合理位置做 allowlist 标注，方便人工复核。
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


DEFAULT_SOURCE_PATTERNS = ["*.py", "factor_builders/*.py"]
DEFAULT_EXCLUDE_PARTS = {"__pycache__", ".git", "wind_hf_multifactor_output"}


@dataclass(frozen=True)
class AuditRule:
    name: str
    pattern: re.Pattern[str]
    severity: str
    description: str


AUDIT_RULES = [
    AuditRule(
        "negative_shift",
        re.compile(r"\.shift\s*\(\s*-\s*\d+"),
        "high",
        "出现负向 shift，可能引用未来数据；如果用于构造预测标签，需要确认训练窗口已隔离。",
    ),
    AuditRule(
        "bfill",
        re.compile(r"\.bfill\s*\(|fillna\s*\([^)]*method\s*=\s*['\"]bfill['\"]"),
        "high",
        "出现向后填充，可能把未来数据填到过去。",
    ),
    AuditRule(
        "expanding",
        re.compile(r"\.expanding\s*\("),
        "medium",
        "出现 expanding 统计，需确认只在历史方向上使用，且没有全样本拟合。",
    ),
    AuditRule(
        "global_mean_std",
        re.compile(r"\.(mean|std|quantile|rank)\s*\([^)]*\)"),
        "low",
        "出现非 rolling 的整体统计，需确认不是用全样本信息构造交易时点特征。",
    ),
    AuditRule(
        "fit_transform",
        re.compile(r"\.fit_transform\s*\("),
        "medium",
        "出现 fit_transform，需确认只在训练窗口内拟合，未使用测试集信息。",
    ),
]


ALLOWLIST_HINTS = [
    {
        "file": "composite_factor_backtest.py",
        "patterns": ["build_forward_return_target", "build_target_return_series", "shift(-"],
        "reason": "综合模型预测目标需要未来收益标签；应配合滚动训练标签隔离复核。",
    },
    {
        "file": "leakage_audit.py",
        "patterns": ["shift(-"],
        "reason": "审计规则文本本身。",
    },
]


def iter_source_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    """收集待审计源码文件。"""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    unique_files = []
    seen = set()
    for path in sorted(files):
        if not path.is_file():
            continue
        if any(part in DEFAULT_EXCLUDE_PARTS for part in path.parts):
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)
    return unique_files


def get_allowlist_reason(path: Path, line: str) -> str:
    """返回命中 allowlist 的说明，未命中则为空。"""
    normalized = str(path).replace("\\", "/")
    for item in ALLOWLIST_HINTS:
        if item["file"] not in normalized:
            continue
        if any(pattern in line for pattern in item["patterns"]):
            return str(item["reason"])
    return ""


def audit_file(path: Path, root: Path) -> list[dict[str, object]]:
    """审计单个文件。"""
    rows: list[dict[str, object]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    relative_path = path.relative_to(root)
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for rule in AUDIT_RULES:
            if not rule.pattern.search(line):
                continue
            allow_reason = get_allowlist_reason(relative_path, line)
            rows.append(
                {
                    "文件": str(relative_path),
                    "行号": line_no,
                    "规则": rule.name,
                    "严重级别": rule.severity,
                    "是否白名单解释": bool(allow_reason),
                    "白名单说明": allow_reason,
                    "说明": rule.description,
                    "代码": stripped[:300],
                }
            )
    return rows


def run_leakage_audit(root: Path, output_path: Path) -> Path:
    """运行静态审计并保存 CSV。"""
    files = iter_source_files(root, DEFAULT_SOURCE_PATTERNS)
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(audit_file(path, root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["文件", "行号", "规则", "严重级别", "是否白名单解释", "白名单说明", "说明", "代码"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行未来函数与数据泄露静态审计。")
    parser.add_argument("--root", default=".", help="项目根目录。")
    parser.add_argument(
        "--output",
        default="wind_hf_multifactor_output/leakage_audit_report.csv",
        help="审计报告输出路径。",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    result_path = run_leakage_audit(root, output_path)
    print(f"泄露审计报告已保存: {result_path}")


if __name__ == "__main__":
    main()

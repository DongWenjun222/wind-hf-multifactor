from __future__ import annotations

"""因子硬删除工具。

这个脚本用于把多品种回测形成的淘汰池真正落到因子构造源码层面：
1. 读取 factor_prune_list.csv，把候选淘汰因子合并进 factor_hard_delete_pool.csv。
2. 扫描 factors.py 和 factor_builders/*.py 中形如 (f"factor_name_{window}", formula), 的一行公式。
3. 若硬删池中的因子能匹配到某条公式模板，则删除该公式行。
4. 执行 --apply 前会自动备份实际修改的源码文件；不带 --apply 时只生成报告，不改代码。

注意：
- 对于 omega_{transform}_{input}_{window} 这类由多层循环动态拼接出来的因子，本脚本不会盲目删除
  transform 或 input，因为那会一次性误伤大量相关结构。它们会进入报告的 unsupported_dynamic_template。
- “硬删”是结构级删除：例如 ret_2 匹配到 f"ret_{window}" 后，会删除 ret_{window} 这条公式，
  因而 ret_3、ret_5 等同一结构也会一起不再生成。
"""

import argparse
import csv
import datetime as dt
import re
import shutil
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("wind_hf_multifactor_output")
DEFAULT_FACTORS_FILE = Path("factors.py")
DEFAULT_FACTOR_BUILDERS_DIR = Path("factor_builders")
DEFAULT_PRUNE_LIST = DEFAULT_OUTPUT_DIR / "factor_prune_list.csv"
DEFAULT_HARD_DELETE_POOL = DEFAULT_OUTPUT_DIR / "factor_hard_delete_pool.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "factor_hard_delete_report.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV，兼容 utf-8-sig。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """写出 CSV，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def get_factor_name(row: dict[str, str]) -> str:
    """兼容中文/英文因子列名。"""
    return str(row.get("因子") or row.get("factor") or "").strip()


def merge_prune_list_into_pool(prune_list_path: Path, pool_path: Path) -> list[dict[str, object]]:
    """把软淘汰名单合并进硬删除池。"""
    now = dt.datetime.now().isoformat(timespec="seconds")
    existing_rows = read_csv_rows(pool_path)
    pool_by_factor = {
        get_factor_name(row): dict(row)
        for row in existing_rows
        if get_factor_name(row)
    }

    for row in read_csv_rows(prune_list_path):
        factor_name = get_factor_name(row)
        if not factor_name:
            continue
        merged = dict(row)
        merged["因子"] = factor_name
        merged["入池时间"] = merged.get("入池时间") or now
        merged["来源"] = merged.get("来源") or str(prune_list_path)
        merged["硬删状态"] = merged.get("硬删状态") or "pending"
        pool_by_factor[factor_name] = {**pool_by_factor.get(factor_name, {}), **merged}

    rows = sorted(pool_by_factor.values(), key=lambda item: str(item.get("因子", "")))
    fieldnames = sorted({field for row in rows for field in row.keys()})
    preferred = ["因子", "硬删状态", "入池时间", "来源", "测试品种数", "最佳初筛夏普", "最佳初筛累计收益", "淘汰原因"]
    ordered_fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    write_csv_rows(pool_path, rows, ordered_fieldnames)
    return rows


def fstring_template_to_regex(template: str) -> re.Pattern[str]:
    """把 f-string 因子模板转成可匹配真实因子名的正则。"""
    pattern = re.escape(template)
    pattern = re.sub(r"\\\{[^{}]+\\\}", r"[^_]+(?:_[^_]+)*", pattern)
    return re.compile(f"^{pattern}$")


def get_default_source_files() -> list[Path]:
    """返回当前模块化结构下需要扫描的因子公式源码文件。"""
    source_files = [DEFAULT_FACTORS_FILE]
    if DEFAULT_FACTOR_BUILDERS_DIR.exists():
        source_files.extend(sorted(DEFAULT_FACTOR_BUILDERS_DIR.glob("*.py")))
    return [path for path in source_files if path.exists()]


def collect_removable_formula_templates(source_files: list[Path]) -> list[dict[str, object]]:
    """扫描可安全按行删除的 f-string 公式模板。"""
    templates = []
    for source_file in source_files:
        lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped.startswith("(f\""):
                continue
            if not stripped.endswith(","):
                continue
            match = re.match(r'\(f"([^"]+)"\s*,', stripped)
            if not match:
                continue
            template = match.group(1)
            templates.append(
                {
                    "source_file": source_file,
                    "line_no": line_no,
                    "line": line,
                    "template": template,
                    "regex": fstring_template_to_regex(template),
                }
            )
    return templates


def match_factors_to_templates(
    pool_rows: list[dict[str, object]],
    templates: list[dict[str, object]],
) -> tuple[dict[Path, set[int]], list[dict[str, object]]]:
    """把硬删池中的因子匹配到源码公式模板。"""
    lines_to_delete: dict[Path, set[int]] = {}
    report_rows = []
    for row in pool_rows:
        factor_name = str(row.get("因子", "")).strip()
        if not factor_name:
            continue

        matched_template = None
        for template_info in templates:
            regex = template_info["regex"]
            if regex.match(factor_name):
                matched_template = template_info
                break

        if matched_template is None:
            status = "unsupported_dynamic_template"
            if factor_name in {"momentum", "reversal", "breakout", "volume_confirm"}:
                status = "unsupported_base_factor"
            report_rows.append(
                {
                    "因子": factor_name,
                    "硬删状态": status,
                    "源码文件": "",
                    "源码行号": "",
                    "匹配模板": "",
                }
            )
            continue

        line_no = int(matched_template["line_no"])
        source_file = Path(matched_template["source_file"])
        lines_to_delete.setdefault(source_file, set()).add(line_no)
        report_rows.append(
            {
                "因子": factor_name,
                "硬删状态": "matched_formula_line",
                "源码文件": str(source_file),
                "源码行号": line_no,
                "匹配模板": matched_template["template"],
            }
        )

    return lines_to_delete, report_rows


def apply_delete_lines(factors_file: Path, lines_to_delete: set[int]) -> Path:
    """备份并删除指定源码行。"""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = factors_file.with_suffix(factors_file.suffix + f".bak_{timestamp}")
    shutil.copy2(factors_file, backup_path)

    lines = factors_file.read_text(encoding="utf-8").splitlines(keepends=True)
    kept_lines = [
        line
        for line_no, line in enumerate(lines, start=1)
        if line_no not in lines_to_delete
    ]
    factors_file.write_text("".join(kept_lines), encoding="utf-8")
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="把因子淘汰池硬删除到因子构造源码。")
    parser.add_argument(
        "--factors-file",
        default=None,
        help="兼容旧用法：只扫描指定的单个源码文件；默认扫描 factors.py 和 factor_builders/*.py。",
    )
    parser.add_argument("--prune-list", default=str(DEFAULT_PRUNE_LIST), help="软淘汰清单 CSV。")
    parser.add_argument("--pool", default=str(DEFAULT_HARD_DELETE_POOL), help="硬删除池 CSV。")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="硬删除报告 CSV。")
    parser.add_argument("--apply", action="store_true", help="真正修改匹配到的源码文件；不加则只预演。")
    args = parser.parse_args()

    source_files = [Path(args.factors_file)] if args.factors_file else get_default_source_files()
    prune_list_path = Path(args.prune_list)
    pool_path = Path(args.pool)
    report_path = Path(args.report)

    if not source_files:
        raise FileNotFoundError("没有找到可扫描的因子构造源码文件。")
    missing_files = [path for path in source_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"找不到因子构造源码文件: {missing_files}")

    pool_rows = merge_prune_list_into_pool(prune_list_path, pool_path)
    templates = collect_removable_formula_templates(source_files)
    lines_to_delete, report_rows = match_factors_to_templates(pool_rows, templates)

    unique_report_rows = []
    seen = set()
    for row in report_rows:
        key = (row["因子"], row["源码文件"], row["源码行号"], row["匹配模板"], row["硬删状态"])
        if key in seen:
            continue
        seen.add(key)
        unique_report_rows.append(row)

    write_csv_rows(
        report_path,
        unique_report_rows,
        ["因子", "硬删状态", "源码文件", "源码行号", "匹配模板"],
    )

    if args.apply and lines_to_delete:
        backup_paths = []
        for source_file, file_lines in lines_to_delete.items():
            backup_paths.append(apply_delete_lines(source_file, file_lines))
        print(f"已硬删公式行数: {sum(len(file_lines) for file_lines in lines_to_delete.values())}")
        for backup_path in backup_paths:
            print(f"源码备份: {backup_path}")
    elif args.apply:
        print("没有找到可安全硬删的公式行，源码未修改。")
    else:
        print("预演完成，未修改源码。加 --apply 才会真正硬删。")

    print(f"硬删除池: {pool_path}")
    print(f"硬删除报告: {report_path}")
    print(f"扫描源码文件数: {len(source_files)}")
    print(f"可硬删源码行数: {sum(len(file_lines) for file_lines in lines_to_delete.values())}")


if __name__ == "__main__":
    main()

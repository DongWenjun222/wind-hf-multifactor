from __future__ import annotations

"""清理或归档历史实验输出目录。

默认只预演，不会真正删除或压缩任何文件。确认报告无误后，再加 --apply 执行。
"""

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CleanupAction:
    """单个待清理目录的动作记录。"""

    runs_dir: Path
    run_dir: Path
    action: str
    archive_path: Path | None
    size_bytes: int
    reason: str


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="清理/归档 wind_hf_multifactor_output 下过多的 runs 历史实验目录。"
    )
    parser.add_argument(
        "--output-dir",
        default="wind_hf_multifactor_output",
        help="回测输出根目录，默认 wind_hf_multifactor_output。",
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=5,
        help="每个 runs/ 目录下保留最近多少个实验目录，默认 5。",
    )
    parser.add_argument(
        "--mode",
        choices=["archive", "delete"],
        default="archive",
        help="archive 表示压缩旧 run 后删除原目录；delete 表示直接删除旧 run。默认 archive。",
    )
    parser.add_argument(
        "--archive-dir",
        default="archived_runs",
        help="归档 zip 保存目录。相对路径会放在 output-dir 下，默认 archived_runs。",
    )
    parser.add_argument(
        "--report-file",
        default="output_cleanup_report.csv",
        help="清理报告文件名。相对路径会放在 output-dir 下。",
    )
    parser.add_argument(
        "--clean-temp-dirs",
        action="store_true",
        help="同时清理 output-dir 下名称以 _tmp 开头的临时目录。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行清理。默认不加该参数时只生成预演报告。",
    )
    return parser.parse_args()


def get_dir_size(path: Path) -> int:
    """递归计算目录大小。"""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def resolve_under_output(output_dir: Path, path_value: str) -> Path:
    """把相对路径解析到 output_dir 下。"""
    path = Path(path_value)
    return path if path.is_absolute() else output_dir / path


def is_inside(child: Path, parent: Path) -> bool:
    """判断 child 是否位于 parent 内部。"""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_runs_dirs(output_dir: Path, archive_dir: Path) -> list[Path]:
    """查找所有 runs 目录，排除归档目录内部。"""
    runs_dirs = []
    for path in output_dir.rglob("runs"):
        if not path.is_dir():
            continue
        if is_inside(path, archive_dir):
            continue
        runs_dirs.append(path)
    return sorted(set(runs_dirs))


def list_run_children(runs_dir: Path) -> list[Path]:
    """列出某个 runs/ 下的实验子目录，按时间从新到旧排序。"""
    children = [path for path in runs_dir.iterdir() if path.is_dir()]
    return sorted(
        children,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def safe_archive_name(output_dir: Path, run_dir: Path) -> str:
    """把 run 目录相对路径转成安全 zip 文件名。"""
    relative = run_dir.relative_to(output_dir)
    return "__".join(relative.parts) + ".zip"


def archive_directory(source_dir: Path, archive_path: Path) -> None:
    """把目录压缩成 zip。"""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for item in source_dir.rglob("*"):
            if item.is_file():
                zip_file.write(item, item.relative_to(source_dir.parent))


def collect_cleanup_actions(
    output_dir: Path,
    keep_runs: int,
    mode: str,
    archive_dir: Path,
    clean_temp_dirs: bool,
) -> list[CleanupAction]:
    """收集所有待执行动作。"""
    keep_runs = max(0, int(keep_runs))
    actions: list[CleanupAction] = []

    for runs_dir in find_runs_dirs(output_dir, archive_dir):
        run_dirs = list_run_children(runs_dir)
        for run_dir in run_dirs[keep_runs:]:
            archive_path = (
                archive_dir / safe_archive_name(output_dir, run_dir)
                if mode == "archive"
                else None
            )
            actions.append(
                CleanupAction(
                    runs_dir=runs_dir,
                    run_dir=run_dir,
                    action=mode,
                    archive_path=archive_path,
                    size_bytes=get_dir_size(run_dir),
                    reason=f"keep_latest_{keep_runs}_runs_per_runs_dir",
                )
            )

    if clean_temp_dirs:
        for temp_dir in output_dir.iterdir():
            if temp_dir.is_dir() and temp_dir.name.startswith("_tmp"):
                actions.append(
                    CleanupAction(
                        runs_dir=output_dir,
                        run_dir=temp_dir,
                        action="delete",
                        archive_path=None,
                        size_bytes=get_dir_size(temp_dir),
                        reason="temporary_output_dir",
                    )
                )

    return actions


def execute_actions(actions: list[CleanupAction], apply: bool) -> None:
    """执行或预演清理动作。"""
    if not apply:
        return

    for action in actions:
        if not action.run_dir.exists():
            continue
        if action.action == "archive":
            if action.archive_path is None:
                raise ValueError("archive 模式必须提供 archive_path。")
            archive_directory(action.run_dir, action.archive_path)
            shutil.rmtree(action.run_dir)
        elif action.action == "delete":
            shutil.rmtree(action.run_dir)
        else:
            raise ValueError(f"未知清理动作: {action.action}")


def save_report(
    output_dir: Path,
    report_file: Path,
    actions: list[CleanupAction],
    apply: bool,
) -> None:
    """保存清理报告。"""
    rows = []
    for action in actions:
        rows.append(
            {
                "是否执行": bool(apply),
                "动作": action.action,
                "原因": action.reason,
                "runs目录": str(action.runs_dir),
                "待处理目录": str(action.run_dir),
                "归档文件": str(action.archive_path) if action.archive_path else "",
                "大小MB": round(action.size_bytes / 1024 / 1024, 3),
            }
        )
    report = pd.DataFrame(rows)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_file, index=False, encoding="utf-8-sig")

    total_mb = sum(action.size_bytes for action in actions) / 1024 / 1024
    status = "已执行" if apply else "预演"
    print(f"{status}清理动作数量: {len(actions)}")
    print(f"预计/释放空间: {total_mb:.2f} MB")
    print(f"清理报告: {report_file}")
    if not apply:
        print("当前只是预演，没有删除或压缩任何文件；确认报告后可加 --apply 真正执行。")


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    archive_dir = resolve_under_output(output_dir, args.archive_dir)
    report_file = resolve_under_output(output_dir, args.report_file)
    actions = collect_cleanup_actions(
        output_dir=output_dir,
        keep_runs=args.keep_runs,
        mode=args.mode,
        archive_dir=archive_dir,
        clean_temp_dirs=bool(args.clean_temp_dirs),
    )
    execute_actions(actions, apply=bool(args.apply))
    save_report(output_dir, report_file, actions, apply=bool(args.apply))


if __name__ == "__main__":
    main()

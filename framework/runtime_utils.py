from __future__ import annotations

"""统一运行日志与执行清单工具。

该模块不改变研究逻辑，只负责把脚本执行过程留痕：
- 控制台输出同步写入日志文件。
- 记录运行状态、耗时、配置摘要和异常堆栈。
- 在实验目录中写出 execution_manifest.json。
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from typing import Any, Callable, TextIO, TypeVar
import warnings

from config import BacktestConfig
from .experiment_utils import get_experiment_run_dir, write_run_config


T = TypeVar("T")


def write_json_atomic(path: Path | str, payload: Any) -> Path:
    """在同一目录内原子写入 JSON，避免中断后留下不完整清单。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target


def copy_file_atomic(source: Path | str, target: Path | str) -> Path:
    """原子复制文件，确保消费者只会看到旧文件或完整的新文件。"""
    source_path = Path(source)
    target_path = Path(target)
    if source_path.resolve() == target_path.resolve():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target_path


def configure_warning_output(config: BacktestConfig) -> None:
    """根据配置控制 warning 输出。"""
    if bool(getattr(config, "suppress_warnings", True)):
        warnings.filterwarnings("ignore")
    else:
        warnings.resetwarnings()


class TeeStream:
    """将 stdout/stderr 同时写到终端和日志文件。"""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return False


def make_stable_run_id(config: BacktestConfig) -> str:
    """确保本次 CLI 运行的不同阶段共享同一个可追踪编号。"""
    run_id = getattr(config, "run_id", None)
    if run_id:
        return str(run_id)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_id = run_id
    return run_id


def get_log_path(config: BacktestConfig, run_type: str) -> Path:
    """生成运行日志路径。"""
    run_id = make_stable_run_id(config)
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{run_id}_{run_type}.log"


def serialize_result_summary(result: Any) -> Any:
    """把返回值压缩为适合写入执行清单的摘要。"""
    if result is None:
        return None
    if hasattr(result, "shape"):
        return {"type": type(result).__name__, "shape": list(result.shape)}
    if isinstance(result, dict):
        return {
            str(key): value
            for key, value in result.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return {"type": type(result).__name__}


def collect_output_files(output_dir: Path, started_at: dt.datetime) -> list[dict[str, Any]]:
    """收集本次运行期间更新的输出文件，便于排错和复盘。"""
    if not output_dir.exists():
        return []
    output_files: list[dict[str, Any]] = []
    threshold = started_at.timestamp() - 1.0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.stat().st_mtime < threshold:
            continue
        output_files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": int(path.stat().st_size),
                "modified_time": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return output_files


def build_config_hash(config: BacktestConfig) -> str:
    """计算配置摘要哈希，便于判断两次实验配置是否相同。"""
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_execution_manifest(
    config: BacktestConfig,
    run_type: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    log_path: Path,
    status: str,
    result: Any = None,
    error: BaseException | None = None,
    error_traceback: str | None = None,
) -> Path:
    """写出统一执行清单。"""
    run_dir = get_experiment_run_dir(config, run_type)
    manifest_dir = run_dir or (Path(config.output_dir) / "runs" / f"{config.run_id}_{run_type}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(config, manifest_dir)
    manifest = {
        "run_id": str(config.run_id),
        "run_type": run_type,
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
        "config_sha256": build_config_hash(config),
        "log_path": str(log_path),
        "result_summary": serialize_result_summary(result),
        "error": str(error) if error else None,
        "traceback": error_traceback,
        "output_files_updated": collect_output_files(Path(config.output_dir), started_at),
    }
    manifest_path = manifest_dir / "execution_manifest.json"
    return write_json_atomic(manifest_path, manifest)


def run_tracked(
    config: BacktestConfig,
    run_type: str,
    action: Callable[[], T],
) -> T:
    """带日志和执行清单运行一个研究任务。"""
    configure_warning_output(config)
    make_stable_run_id(config)
    log_path = get_log_path(config, run_type)
    started_at = dt.datetime.now()
    result: T | None = None
    error: BaseException | None = None
    error_traceback: str | None = None
    status = "success"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = TeeStream(sys.stdout, log_file)
        tee_stderr = TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"运行编号: {config.run_id}; 任务类型: {run_type}")
            print(f"日志文件: {log_path}")
            try:
                result = action()
            except BaseException as exc:
                status = "failed"
                error = exc
                error_traceback = traceback.format_exc()
                print(f"任务执行失败: {exc}")
                print(error_traceback, end="")
            finally:
                finished_at = dt.datetime.now()
                manifest_path = write_execution_manifest(
                    config=config,
                    run_type=run_type,
                    started_at=started_at,
                    finished_at=finished_at,
                    log_path=log_path,
                    status=status,
                    result=result,
                    error=error,
                    error_traceback=error_traceback,
                )
                print(f"执行清单: {manifest_path}")
                print(f"运行耗时(秒): {(finished_at - started_at).total_seconds():.3f}")

    if error is not None:
        raise error
    return result  # type: ignore[return-value]

from __future__ import annotations

"""项目一致性检查工具。

用于快速发现配置、CLI 和文档之间的常见漂移。这个脚本不运行回测，
只做轻量静态检查，适合每次调整默认参数或入口说明后执行。
"""

import argparse
from pathlib import Path

from cli import build_parser
from config import BacktestConfig


def read_text(path: Path) -> str:
    """读取文本文件。"""
    return path.read_text(encoding="utf-8")


def collect_checks() -> list[tuple[bool, str]]:
    """收集一致性检查结果。"""
    config = BacktestConfig()
    parser = build_parser()
    subparsers = parser._subparsers._group_actions[0].choices
    single_parser = subparsers["single"]
    composite_parser = subparsers["composite"]
    multi_parser = subparsers["multi"]
    pooled_parser = subparsers["pooled"]
    scope_action = next(action for action in single_parser._actions if "--scope" in action.option_strings)
    cli_scopes = set(scope_action.choices or [])
    option_strings_by_parser = {
        "single": {option for action in single_parser._actions for option in action.option_strings},
        "composite": {option for action in composite_parser._actions for option in action.option_strings},
        "multi": {option for action in multi_parser._actions for option in action.option_strings},
        "pooled": {option for action in pooled_parser._actions for option in action.option_strings},
    }

    docs = {
        "PROJECT_DOCUMENTATION.md": read_text(Path("PROJECT_DOCUMENTATION.md")),
        "README.md": read_text(Path("README.md")),
    }
    checks: list[tuple[bool, str]] = []
    checks.append(
        (
            str(config.single_factor_scope) in cli_scopes,
            f"CLI single --scope 支持当前默认 single_factor_scope={config.single_factor_scope!r}",
        )
    )
    checks.append(
        (
            {"all", "new", "range", "selected"}.issubset(cli_scopes),
            "CLI single --scope 支持 all/new/range/selected 四种模式",
        )
    )
    checks.append(
        (
            {"--selected-factors", "--range-start", "--range-end"}.issubset(option_strings_by_parser["single"]),
            "CLI single 支持 selected 因子列表和 range 编号区间覆盖",
        )
    )
    checks.append(
        (
            {"--selected-factors", "--feature-scope"}.issubset(option_strings_by_parser["composite"]),
            "CLI composite 支持 selected 因子列表和 feature scope 覆盖",
        )
    )
    checks.append(
        (
            {"--run-single-factor", "--no-run-single-factor", "--run-composite", "--no-run-composite"}.issubset(
                option_strings_by_parser["multi"]
            ),
            "CLI multi 支持显式控制单因子流程和综合因子流程是否运行",
        )
    )
    checks.append(
        (
            {"--pooled-train-time-window", "--pooled-max-train-rows"}.issubset(
                option_strings_by_parser["pooled"]
            ),
            "CLI pooled 支持训练时间窗口和 long-format 最大样本行数覆盖",
        )
    )
    common_config_options = {"--print-config", "--save-config", "--dry-run"}
    for command_name, option_strings in option_strings_by_parser.items():
        checks.append(
            (
                common_config_options.issubset(option_strings),
                f"CLI {command_name} 支持最终配置打印、保存和 dry-run",
            )
        )
    for filename, text in docs.items():
        checks.append(
            (
                f'| `single_factor_scope` | `"{config.single_factor_scope}"` |' in text,
                f"{filename} 中 single_factor_scope 默认值与 config.py 一致",
            )
        )
        checks.append(
            (
                "默认 compute 模式读取 active 因子" in text
                and "detail 模式" in text,
                f"{filename} 中 trading_signal.py 描述区分 compute/detail 模式",
            )
        )
        checks.append(
            (
                "pooled_model_train_time_window" in text
                and "long-format 样本" in text,
                f"{filename} 中 pooled 训练窗口和 long-format 行数说明完整",
            )
        )
    return checks


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="检查配置、CLI 和文档之间是否存在常见不一致。")
    parser.add_argument("--strict", action="store_true", help="存在失败检查时返回非零退出码。")
    args = parser.parse_args()

    checks = collect_checks()
    failed = [(ok, message) for ok, message in checks if not ok]
    for ok, message in checks:
        status = "通过" if ok else "失败"
        print(f"[{status}] {message}")
    print(f"一致性检查: {len(checks) - len(failed)}/{len(checks)} 通过")
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

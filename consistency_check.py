from __future__ import annotations

"""项目一致性检查工具。

用于快速发现配置、CLI 和文档之间的常见漂移。这个脚本不运行回测，
只做轻量静态检查，适合每次调整默认参数或入口说明后执行。
"""

import argparse
from pathlib import Path

from cli import build_parser, create_config
from config import BacktestConfig
from framework.project_fingerprint import get_default_fingerprint_files


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
    signal_parser = subparsers["signal"]
    scope_action = next(action for action in single_parser._actions if "--scope" in action.option_strings)
    cli_scopes = set(scope_action.choices or [])
    signal_mode_action = next(
        action for action in signal_parser._actions if "--mode" in action.option_strings
    )
    signal_modes = set(signal_mode_action.choices or [])
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
            {"compute", "model", "vote", "detail"}.issubset(signal_modes),
            "CLI signal 明确区分模型、投票和历史明细模式",
        )
    )
    range_args = parser.parse_args(
        ["single", "--scope", "range", "--range-start", "7", "--range-end", "11", "--dry-run"]
    )
    range_config = create_config(range_args)
    checks.append(
        (
            tuple(range_config.single_factor_range) == (7, 11),
            "CLI single 的显式 range 编号区间优先于 config.py 默认值",
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
                "compute/model 模式复用综合回测" in text
                and "vote" in text
                and "detail" in text,
                f"{filename} 中 trading_signal.py 描述区分模型、投票和明细模式",
            )
        )
        checks.append(
            (
                "pooled_model_train_time_window" in text
                and "long-format 样本" in text,
                f"{filename} 中 pooled 训练窗口和 long-format 行数说明完整",
            )
        )
        checks.append(
            (
                "backtest_return_mode" in text
                and "previous_position * gap_return" in text,
                f"{filename} 中默认连续持仓收益口径说明与代码一致",
            )
        )
        checks.append(
            (
                "label_available_time" in text
                and "完整时间戳" in text,
                f"{filename} 中 pooled 标签可用时间和横截面截断规则说明完整",
            )
        )
        checks.append(
            (
                ".single_pipeline_state.json" in text
                and ".composite_pipeline_state.json" in text,
                f"{filename} 中多品种安全断点状态文件说明完整",
            )
        )
        checks.append(
            (
                "composite_artifact_manifest.json" in text
                and "multi_symbol_portfolio_inputs.csv" in text
                and "multi_symbol_require_composite_artifact_manifest" in text
                and "multi_symbol_clear_stale_portfolio_outputs" in text,
                f"{filename} 中综合产物校验和组合输入审计说明完整",
            )
        )
        checks.append(
            (
                "current_root_and_symbol_key_artifacts" in text
                and "原子替换" in text
                and "不会递归收录全部历史 runs" in text,
                f"{filename} 中有界运行清单和原子 JSON 写入说明完整",
            )
        )
        checks.append(
            (
                "composite_auto_freeze_active_library" in text
                and "composite_active_library_cutoff_policy" in text
                and "active_library_oos_audit.json" in text
                and "因子筛选验证截止" in text
                and "effective_run_config.json" in text,
                f"{filename} 中 active 因子库冻结和样本外截止审计说明完整",
            )
        )
        checks.append(
            (
                "最终测试集不参与" in text
                and "不再兼容使用旧测试列" in text
                and "factor_library_min_selection_win_rate" in text
                and "迁移别名" in text
                and "missing_traceable_train_metrics" in text
                and "最终测试集表现不会用于决定是否淘汰因子" in text
                and "最终测试指标不会参与实时权重" in text,
                f"{filename} 中因子入库最终测试集隔离规则说明完整",
            )
        )
    fingerprint_names = {path.as_posix() for path in get_default_fingerprint_files()}
    checks.append(
        (
            "framework/factor_builders/external_daily.py" in fingerprint_names,
            "源码指纹自动覆盖 external_daily.py 等全部因子构建模块",
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

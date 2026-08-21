from __future__ import annotations

"""使用 Codex 对 active 因子做增量逻辑审查，并持久化剔除明确无逻辑的因子。"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from config import BacktestConfig
from factor_library_manager import (
    exclude_factors,
    load_active_library,
    load_pre_active_library,
    persist_library_state,
)
from framework.factor_library import (
    get_factor_library_dir,
    load_existing_factor_library,
    load_manual_factor_exclusions,
    load_manual_factor_protections,
)
from framework.factor_logic_review import (
    LOGIC_REVIEW_COLUMNS,
    LOGIC_REVIEW_VERSION,
    get_pending_logic_reviews,
    load_factor_logic_reviews,
    merge_factor_logic_reviews,
    save_factor_logic_reviews,
)
from framework.output_layout import (
    apply_frequency_runtime_defaults,
    resolve_existing_symbol_output_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="增量审查 active 因子的经济逻辑；已审查且公式未变化的因子自动跳过。"
    )
    parser.add_argument("--frequency", choices=["30min", "1d"], default=None)
    parser.add_argument("--symbol", default=None, help="例如 C.DCE、CU.SHF。")
    parser.add_argument("--output-dir", default=None, help="显式指定品种输出目录。")
    parser.add_argument("--batch-size", type=int, default=None, help="每次交给 AI 的因子数量。")
    parser.add_argument("--model", default=None, help="可选的 Codex 模型名称；默认沿用本机配置。")
    parser.add_argument(
        "--codex-path",
        default=None,
        help="Codex CLI 可执行文件路径；通常自动识别，IDE 环境 PATH 不完整时可显式指定。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出待审查因子，不调用 AI、不修改文件。",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="保存审查结论并同步列，但不自动剔除 rejected 因子。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有审查指纹，重新审查全部 active 因子。",
    )
    return parser


def build_audit_config(args: argparse.Namespace) -> BacktestConfig:
    base = BacktestConfig()
    symbol = str(args.symbol or base.symbol)
    config = replace(base, symbol=symbol)
    if args.frequency:
        config.bar_frequency = args.frequency
    if args.output_dir:
        config.output_dir = str(Path(args.output_dir))
    elif args.symbol and symbol != str(base.symbol):
        config.output_dir = str(resolve_existing_symbol_output_dir(base, symbol))
    return apply_frequency_runtime_defaults(config)


def build_source_evidence(batch: pd.DataFrame, max_lines_per_file: int = 160) -> str:
    """从构造器中提取与本批因子名称片段相关的带行号代码。"""
    excerpts: list[str] = []
    for source_file, group in batch.groupby("来源文件", sort=False):
        source_path = PROJECT_ROOT / str(source_file)
        try:
            lines = source_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            excerpts.append(f"### {source_file}\n来源读取失败: {exc}")
            continue
        tokens: set[str] = set()
        for factor_name in group["因子"].astype(str):
            tokens.update(
                token.lower()
                for token in factor_name.replace("-", "_").split("_")
                if len(token) >= 3 and not token.isdigit()
            )
        matched: set[int] = set(range(min(35, len(lines))))
        scored_matches: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            lowered = line.lower()
            score = sum(len(token) for token in tokens if token in lowered)
            if score:
                scored_matches.append((score, index))
        for _, index in sorted(scored_matches, key=lambda item: (-item[0], item[1]))[:50]:
            matched.update(range(max(0, index - 2), min(len(lines), index + 3)))
        selected = sorted(matched)[:max_lines_per_file]
        rendered = "\n".join(f"{index + 1:04d}: {lines[index]}" for index in selected)
        excerpts.append(f"### {source_file}\n{rendered}")
    return "\n\n".join(excerpts)


def build_review_prompt(batch: pd.DataFrame) -> str:
    records = batch[
        ["因子", "因子家族", "来源文件", "复杂度估计", "是否跨品种", "是否宏观", "是否日历"]
    ].to_dict(orient="records")
    return f"""你是中国商品期货量化研究的因子逻辑审查员。请审查下面 active 因子。

下面已经附上从对应构造器中按名称片段提取的带行号代码证据。你仍可只读打开来源文件补充核验，禁止修改文件。
目标是判断因子是否至少存在合理、可陈述、无明显未来数据的市场机制，不评价本次回测收益高低。

判定标准：
1. approved：存在清晰的趋势、反转、波动、流动性、量价、季节性、宏观或跨品种传导逻辑，公式也没有明显退化。
2. rejected：明确没有可成立的传导机制，属于无意义随机拼接、数学恒等/近恒等退化，或明确依赖未来数据。
3. uncertain：仅凭代码和名称无法可靠确认。复杂、解释较弱或证据不足不能直接 rejected。
4. 不得因为参数看起来奇怪、公式复杂、测试集表现差就判 rejected。
5. 必须基于代码确认输入、变换、窗口、滞后和除零处理；没有检查代码不能完成本批审查。
6. 每个输入因子必须且只能返回一条记录；置信度范围为 0 到 1，code_inspected 必须如实填写。

待审查因子：
{json.dumps(records, ensure_ascii=False, indent=2)}

构造器代码证据：
{build_source_evidence(batch)}
"""


def _review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string"},
                        "status": {"type": "string", "enum": ["approved", "rejected", "uncertain"]},
                        "logic_type": {"type": "string"},
                        "logic_explanation": {"type": "string"},
                        "main_risk": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "code_inspected": {"type": "boolean"},
                    },
                    "required": [
                        "factor",
                        "status",
                        "logic_type",
                        "logic_explanation",
                        "main_risk",
                        "confidence",
                        "code_inspected",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def resolve_codex_executable(explicit_path: str | None = None) -> Path:
    """定位 Codex CLI，兼容 IDE 启动的 Python 没有继承完整 PATH。"""
    configured = str(explicit_path or os.environ.get("CODEX_CLI_PATH", "")).strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return configured_path.resolve()
        raise FileNotFoundError(f"配置的 Codex CLI 不存在: {configured_path}")

    for executable_name in ("codex", "codex.exe", "codex.cmd"):
        resolved = shutil.which(executable_name)
        if resolved and Path(resolved).is_file():
            return Path(resolved).resolve()

    user_home = Path.home()
    candidates: list[Path] = []
    for extension_root in (
        user_home / ".vscode" / "extensions",
        user_home / ".vscode-insiders" / "extensions",
    ):
        candidates.extend(
            extension_root.glob("openai.chatgpt-*/bin/windows-x86_64/codex.exe")
        )
    candidates = [path for path in candidates if path.is_file()]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "找不到 Codex CLI。请确认 VS Code Codex 扩展已安装，或在 config.py 设置 "
        "factor_logic_review_codex_path，也可以设置环境变量 CODEX_CLI_PATH。"
    )


def run_codex_review(
    batch: pd.DataFrame,
    *,
    model: str | None = None,
    codex_path: str | None = None,
) -> list[dict[str, Any]]:
    """在只读沙箱中调用本机 Codex，并读取结构化审查结果。"""
    with tempfile.TemporaryDirectory(prefix="factor_logic_audit_") as temp_dir:
        temp = Path(temp_dir)
        schema_path = temp / "schema.json"
        result_path = temp / "result.json"
        schema_path.write_text(json.dumps(_review_schema(), ensure_ascii=False), encoding="utf-8")
        executable = resolve_codex_executable(codex_path)
        command = [
            str(executable),
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
            "-",
        ]
        if model:
            command[2:2] = ["--model", str(model)]
        if executable.suffix.lower() in {".cmd", ".bat"}:
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", *command]
        try:
            completed = subprocess.run(
                command,
                input=build_review_prompt(batch),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=PROJECT_ROOT,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Codex CLI 启动失败: {executable}。可通过 --codex-path 或 "
                "config.factor_logic_review_codex_path 显式指定。"
            ) from exc
        if completed.returncode != 0 or not result_path.exists():
            detail = (completed.stderr or completed.stdout or "未知错误").strip()[-2000:]
            raise RuntimeError(f"Codex 因子逻辑审查失败: {detail}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return list(payload.get("reviews", []))


def normalize_review_results(
    batch: pd.DataFrame,
    raw_reviews: list[dict[str, Any]],
    *,
    model_name: str,
    min_reject_confidence: float,
) -> pd.DataFrame:
    """校验 AI 输出，并把低置信度拒绝降级为 uncertain。"""
    expected = set(batch["因子"].astype(str))
    received = [str(item.get("factor", "")).strip() for item in raw_reviews]
    if len(received) != len(set(received)) or set(received) != expected:
        missing = sorted(expected - set(received))
        unexpected = sorted(set(received) - expected)
        raise ValueError(f"AI 审查结果因子集合不一致，缺少={missing}，多余={unexpected}")
    context = batch.set_index("因子")
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for item in raw_reviews:
        factor_name = str(item["factor"]).strip()
        if not bool(item.get("code_inspected", False)):
            raise ValueError(f"AI 未完成来源代码检查，拒绝把本批标记为已审查: {factor_name}")
        status = str(item["status"]).strip().lower()
        confidence = max(0.0, min(1.0, float(item["confidence"])))
        risk = str(item["main_risk"]).strip()
        if status == "rejected" and confidence < min_reject_confidence:
            status = "uncertain"
            risk = f"原AI拒绝置信度不足，已降级待复核；{risk}"
        rows.append(
            {
                "因子": factor_name,
                "逻辑审查状态": status,
                "逻辑类型": str(item["logic_type"]).strip(),
                "逻辑说明": str(item["logic_explanation"]).strip(),
                "主要风险": risk,
                "逻辑审查置信度": confidence,
                "逻辑审查时间": now,
                "逻辑审查模型": model_name,
                "逻辑审查版本": LOGIC_REVIEW_VERSION,
                "逻辑审查指纹": context.loc[factor_name, "逻辑审查指纹"],
                "因子家族": context.loc[factor_name, "因子家族"],
                "来源文件": context.loc[factor_name, "来源文件"],
            }
        )
    return pd.DataFrame(rows, columns=LOGIC_REVIEW_COLUMNS)


def main() -> None:
    args = build_parser().parse_args()
    config = build_audit_config(args)
    library_dir = get_factor_library_dir(config)
    master = load_existing_factor_library(config)
    master_is_complete = not master.empty and not bool(master.attrs.get("factor_library_partial", False))
    active = load_active_library(config, master)
    if active.empty or "因子" not in active.columns:
        raise ValueError(f"active 因子库为空，无法审查: {library_dir / 'active_factors.csv'}")
    reviews = load_factor_logic_reviews(config)
    if args.force:
        pending = get_pending_logic_reviews(active, reviews.iloc[0:0], project_root=PROJECT_ROOT)
    else:
        pending = get_pending_logic_reviews(active, reviews, project_root=PROJECT_ROOT)
    print(
        f"逻辑审查目录: {library_dir}；active={len(active)}；"
        f"已有有效审查={len(active) - len(pending)}；待审查={len(pending)}"
    )
    if args.dry_run:
        columns = [column for column in ["因子编号", "因子", "因子家族", "来源文件"] if column in pending.columns]
        print(pending[columns].to_string(index=False) if not pending.empty else "没有待审查因子。")
        return

    batch_size = max(1, int(args.batch_size or config.factor_logic_review_batch_size))
    min_confidence = float(config.factor_logic_review_min_reject_confidence)
    model_name = str(args.model or config.factor_logic_review_model or "codex-default")
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size].copy()
        raw = run_codex_review(
            batch,
            model=args.model or config.factor_logic_review_model,
            codex_path=args.codex_path or config.factor_logic_review_codex_path,
        )
        additions = normalize_review_results(
            batch,
            raw,
            model_name=model_name,
            min_reject_confidence=min_confidence,
        )
        reviews = pd.concat([reviews, additions], ignore_index=True).drop_duplicates("因子", keep="last")
        save_factor_logic_reviews(reviews, config)
        print(f"逻辑审查进度: {min(start + len(batch), len(pending))}/{len(pending)}")

    active = merge_factor_logic_reviews(active, reviews, project_root=PROJECT_ROOT)
    if not master.empty:
        master = merge_factor_logic_reviews(master, reviews, project_root=PROJECT_ROOT)
    persist_library_state(config, active, master, master_is_complete=master_is_complete)

    rejected = reviews[
        reviews["因子"].astype(str).isin(active["因子"].astype(str))
        & reviews["逻辑审查状态"].eq("rejected")
        & (pd.to_numeric(reviews["逻辑审查置信度"], errors="coerce") >= min_confidence)
    ]["因子"].astype(str).tolist()
    protected_names = set(load_manual_factor_protections(config)["因子"].astype(str))
    protected_rejections = [factor_name for factor_name in rejected if factor_name in protected_names]
    rejected = [factor_name for factor_name in rejected if factor_name not in protected_names]
    if protected_rejections:
        print(
            f"AI判为rejected但受手工保护、未自动剔除: {len(protected_rejections)} 个；"
            "如需删除请使用 factor_library_manager.py exclude。"
        )
    if rejected and not args.no_exclude:
        pre_active = load_pre_active_library(config, master)
        exclusion_master = master
        if not master_is_complete:
            active_rows = active.copy()
            pre_active_rows = pre_active.copy()
            active_rows["因子库状态"] = "active"
            pre_active_rows["因子库状态"] = "pre_active"
            exclusion_master = pd.concat([active_rows, pre_active_rows], ignore_index=True, sort=False)
        exclude_factors(
            config,
            rejected,
            "AI逻辑审查不通过，详细原因见 factor_logic_reviews.csv",
            active,
            exclusion_master,
            load_manual_factor_exclusions(config),
            master_is_complete=master_is_complete,
        )
    counts = reviews[reviews["因子"].isin(active["因子"])]["逻辑审查状态"].value_counts()
    print(f"审查档案已保存: {library_dir / 'factor_logic_reviews.csv'}")
    print("本轮active审查状态: " + "；".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()

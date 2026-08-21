from __future__ import annotations

"""人工审核并维护 pre_active / active 因子库。

示例：
python factor_library_manager.py list --frequency 1d
python factor_library_manager.py approve 305_body_range_5 --reason "图形与逻辑复核通过" --frequency 1d
python factor_library_manager.py revoke 305_body_range_5 --reason "暂停使用" --frequency 1d
python factor_library_manager.py protect 305_body_range_5 --reason "人工确认长期保留" --frequency 1d
python factor_library_manager.py unprotect 305_body_range_5 --frequency 1d
python factor_library_manager.py exclude 305_body_range_5 --reason "图形不稳定" --frequency 1d
python factor_library_manager.py restore 305_body_range_5 --frequency 1d
"""

import argparse
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import BacktestConfig
from framework.factor_library import (
    get_factor_library_dir,
    load_existing_factor_library,
    load_manual_factor_approvals,
    load_manual_factor_exclusions,
    load_manual_factor_protections,
    save_factor_library,
    save_manual_factor_approvals,
    save_manual_factor_exclusions,
    save_manual_factor_protections,
)
from framework.output_layout import (
    apply_frequency_runtime_defaults,
    get_research_output_dir,
    resolve_existing_symbol_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按名称、标签或编号审批、保护、剔除或恢复因子。")
    parser.add_argument(
        "action",
        choices=["list", "approve", "revoke", "protect", "unprotect", "exclude", "restore"],
    )
    parser.add_argument("factors", nargs="*", help="因子名称、因子标签或因子编号。")
    parser.add_argument("--reason", default="人工复核", help="本次审批、保护、撤销或剔除的原因。")
    parser.add_argument("--frequency", choices=["30min", "1d"], default=None)
    parser.add_argument("--symbol", default=None, help="例如 C.DCE、CU.SHF。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="显式指定该品种输出根目录；通常无需设置。",
    )
    return parser


def build_maintenance_config(args: argparse.Namespace) -> BacktestConfig:
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


def load_active_library(config: BacktestConfig, master: pd.DataFrame) -> pd.DataFrame:
    active_path = get_factor_library_dir(config) / "active_factors.csv"
    if active_path.exists():
        return pd.read_csv(active_path, encoding="utf-8-sig")
    if "因子库状态" in master.columns:
        return master[master["因子库状态"].eq("active")].copy()
    return master.iloc[0:0].copy()


def load_pre_active_library(config: BacktestConfig, master: pd.DataFrame) -> pd.DataFrame:
    """读取等待人工审核的规则候选池。"""
    path = get_factor_library_dir(config) / "pre_active_factors.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    if "因子库状态" in master.columns:
        return master[master["因子库状态"].eq("pre_active")].copy()
    return master.iloc[0:0].copy()


def resolve_factor_names(
    tokens: list[str],
    active: pd.DataFrame,
    master: pd.DataFrame,
    exclusions: pd.DataFrame,
    *additional_catalogs: pd.DataFrame,
) -> list[str]:
    frames = [
        frame
        for frame in (active, master, exclusions, *additional_catalogs)
        if frame is not None and not frame.empty
    ]
    catalog = (
        pd.concat(frames, ignore_index=True, sort=False).copy()
        if frames
        else pd.DataFrame()
    )
    if "因子" not in catalog.columns:
        raise ValueError("当前因子库没有可识别的因子记录。")
    catalog = catalog.dropna(subset=["因子"])
    names = set(catalog["因子"].astype(str))
    labels = (
        catalog.dropna(subset=["因子标签"]).set_index("因子标签")["因子"].astype(str).to_dict()
        if "因子标签" in catalog.columns
        else {}
    )
    if "因子编号" in catalog.columns:
        id_catalog = pd.DataFrame(
            {
                "_id": pd.to_numeric(catalog["因子编号"], errors="coerce").to_numpy(),
                "因子": catalog["因子"].astype(str).to_numpy(),
            }
        ).dropna(subset=["_id"])
        ids = id_catalog.set_index("_id")["因子"].to_dict()
    else:
        ids = {}

    resolved: list[str] = []
    missing: list[str] = []
    for raw_token in tokens:
        token = str(raw_token).strip()
        if token in names:
            resolved.append(token)
        elif token in labels:
            resolved.append(labels[token])
        elif token.isdigit() and float(int(token)) in ids:
            resolved.append(ids[float(int(token))])
        elif re.fullmatch(r"\d+_.+", token) and token.split("_", 1)[1] in names:
            # 兼容旧治理清单没有持久化“因子标签”列的情况。
            resolved.append(token.split("_", 1)[1])
        else:
            missing.append(token)
    if missing:
        raise ValueError(f"找不到这些因子: {missing}")
    return list(dict.fromkeys(resolved))


def build_factor_identity_records(
    factor_names: list[str],
    *catalogs: pd.DataFrame,
) -> pd.DataFrame:
    """从 active/master 等表提取稳定编号和标签，写入长期治理清单。"""
    frames = [frame for frame in catalogs if frame is not None and not frame.empty and "因子" in frame.columns]
    catalog = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    rows = []
    for factor_name in factor_names:
        factor_rows = (
            catalog[catalog["因子"].astype(str).eq(str(factor_name))]
            if not catalog.empty
            else pd.DataFrame()
        )
        factor_id = ""
        factor_label = ""
        if not factor_rows.empty:
            if "因子编号" in factor_rows.columns:
                valid_ids = pd.to_numeric(factor_rows["因子编号"], errors="coerce").dropna()
                if not valid_ids.empty:
                    factor_id = int(valid_ids.iloc[-1])
            if "因子标签" in factor_rows.columns:
                valid_labels = factor_rows["因子标签"].dropna().astype(str).str.strip()
                valid_labels = valid_labels[valid_labels.ne("")]
                if not valid_labels.empty:
                    factor_label = valid_labels.iloc[-1]
        if not factor_label and factor_id != "":
            factor_label = f"{factor_id}_{factor_name}"
        rows.append({"因子编号": factor_id, "因子标签": factor_label, "因子": factor_name})
    return pd.DataFrame(rows, columns=["因子编号", "因子标签", "因子"])


def enrich_factor_identity_records(
    records: pd.DataFrame,
    *catalogs: pd.DataFrame,
) -> pd.DataFrame:
    """为旧版治理清单补齐编号和标签，同时保留原有治理字段。"""
    if records.empty or "因子" not in records.columns:
        return records.copy()
    output = records.copy()
    identities = build_factor_identity_records(
        output["因子"].dropna().astype(str).tolist(),
        output,
        *catalogs,
    ).set_index("因子")
    for column in ("因子编号", "因子标签"):
        if column not in output.columns:
            output[column] = ""
        # pandas 3 的 StringDtype 不允许直接写入整数编号；治理清单需要同时
        # 兼容旧版空字符串和新版数值编号，因此迁移期间统一使用 object。
        output[column] = output[column].astype("object")
        existing = output[column]
        missing = existing.isna() | existing.astype(str).str.strip().isin({"", "nan"})
        replacements = output.loc[missing, "因子"].map(identities[column]).astype("object")
        output.loc[missing, column] = replacements.to_numpy()
    return output


def persist_library_state(
    config: BacktestConfig,
    active: pd.DataFrame,
    master: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    library_dir = get_factor_library_dir(config)
    if not master_is_complete:
        pre_active = (
            master[master["因子库状态"].eq("pre_active")].copy()
            if not master.empty and "因子库状态" in master.columns
            else master.iloc[0:0].copy()
        )
        for frame, filename in (
            (active, "active_factors.csv"),
            (pre_active, "pre_active_factors.csv"),
        ):
            target = library_dir / filename
            temp = target.with_name(f".{target.name}.tmp")
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
            temp.replace(target)
        print("完整主库当前不可读；已安全更新 active/pre_active，原主库保持不变。")
        return
    if master.empty:
        active.to_csv(
            library_dir / "active_factors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return
    rejected = (
        master[~master["因子库状态"].isin(["active", "pre_active"])].copy()
        if "因子库状态" in master.columns
        else master.iloc[0:0].copy()
    )
    save_factor_library(active, master, rejected, config)


def delete_factor_plot_files(
    config: BacktestConfig,
    factor_names: list[str],
    active: pd.DataFrame,
    master: pd.DataFrame,
) -> list[Path]:
    """删除当前 latest 单因子目录中的对应回测图，不触碰历史 runs 快照。"""
    single_factor_dir = get_research_output_dir(config, "single_factor").resolve()
    if not single_factor_dir.exists():
        return []

    frames = [frame for frame in (active, master) if not frame.empty and "因子" in frame.columns]
    catalog = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if catalog.empty:
        return []
    selected = catalog[catalog["因子"].astype(str).isin(factor_names)].copy()
    expected_names: set[str] = set()
    if "因子标签" in selected.columns:
        expected_names.update(
            f"{label}_report.png"
            for label in selected["因子标签"].dropna().astype(str)
            if label.strip()
        )
    if "因子编号" in selected.columns:
        for _, row in selected.iterrows():
            factor_id = pd.to_numeric(row.get("因子编号"), errors="coerce")
            factor_name = str(row.get("因子", "")).strip()
            if pd.notna(factor_id) and factor_name:
                expected_names.add(f"{int(factor_id)}_{factor_name}_report.png")

    candidates: set[Path] = set()
    if "图片文件" in selected.columns:
        for raw_path in selected["图片文件"].dropna().astype(str):
            if not raw_path.strip():
                continue
            path = Path(raw_path)
            resolved = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
            candidates.add(resolved)
    for path in single_factor_dir.rglob("*.png"):
        if path.name in expected_names:
            candidates.add(path.resolve())

    deleted: list[Path] = []
    for path in sorted(candidates, key=str):
        try:
            path.relative_to(single_factor_dir)
        except ValueError:
            print(f"跳过不在当前单因子目录内的图片路径: {path}")
            continue
        if path.is_file() and path.suffix.lower() == ".png":
            try:
                path.unlink()
                deleted.append(path)
            except OSError as exc:
                print(f"单因子图片删除失败: {path}, {exc}")
    return deleted


def exclude_factors(
    config: BacktestConfig,
    factor_names: list[str],
    reason: str,
    active: pd.DataFrame,
    master: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    additions = build_factor_identity_records(factor_names, active, master)
    additions["人工排除原因"] = str(reason)
    additions["人工排除时间"] = now
    exclusions = pd.concat([exclusions, additions], ignore_index=True)
    exclusions = exclusions.drop_duplicates("因子", keep="last")
    save_manual_factor_exclusions(exclusions, config)
    protections = load_manual_factor_protections(config)
    protections = protections[~protections["因子"].astype(str).isin(factor_names)].copy()
    save_manual_factor_protections(protections, config)
    approvals = load_manual_factor_approvals(config)
    approvals = approvals[~approvals["因子"].astype(str).isin(factor_names)].copy()
    save_manual_factor_approvals(approvals, config)

    deleted_plots = delete_factor_plot_files(
        config,
        factor_names,
        active,
        master,
    )
    previous_count = len(active)
    if "因子" in active.columns:
        active = active[~active["因子"].astype(str).isin(factor_names)].copy()
    if not master.empty and "因子" in master.columns:
        mask = master["因子"].astype(str).isin(factor_names)
        master.loc[mask, "因子库状态"] = "retired"
        master.loc[mask, "拒绝原因"] = "manual_exclusion"
        master.loc[mask, "人工排除原因"] = str(reason)
        master.loc[mask, "人工排除时间"] = now
        if "图片文件" in master.columns:
            master.loc[mask, "图片文件"] = ""
    persist_library_state(
        config,
        active,
        master,
        master_is_complete=master_is_complete,
    )
    print(
        f"已人工剔除 {len(factor_names)} 个因子；"
        f"active数量: {previous_count} -> {len(active)}；删除回测图: {len(deleted_plots)} 张"
    )


def protect_factors(
    config: BacktestConfig,
    factor_names: list[str],
    reason: str,
    active: pd.DataFrame,
    master: pd.DataFrame,
    protections: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    """保护当前 active，使其不被任何自动筛选或 AI 审查剔除。"""
    active_names = set(active.get("因子", pd.Series(dtype="object")).dropna().astype(str))
    invalid = [factor_name for factor_name in factor_names if factor_name not in active_names]
    if invalid:
        raise ValueError(f"只有当前 active 因子可以保护，这些因子不符合条件: {invalid}")
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    additions = build_factor_identity_records(factor_names, active, master)
    additions["手工保护原因"] = str(reason)
    additions["手工保护时间"] = now
    protections = pd.concat([protections, additions], ignore_index=True).drop_duplicates(
        "因子", keep="last"
    )
    save_manual_factor_protections(protections, config)
    # active/master 列较多时先整理内存块，避免连续增加治理列触发碎片化警告。
    active = active.copy()
    active_mask = active["因子"].astype(str).isin(factor_names)
    active.loc[active_mask, "是否手工保护"] = True
    active.loc[active_mask, "手工保护原因"] = str(reason)
    active.loc[active_mask, "手工保护时间"] = now
    if not master.empty and "因子" in master.columns:
        master = master.copy()
        master_mask = master["因子"].astype(str).isin(factor_names)
        master.loc[master_mask, "是否手工保护"] = True
        master.loc[master_mask, "手工保护原因"] = str(reason)
        master.loc[master_mask, "手工保护时间"] = now
    persist_library_state(config, active, master, master_is_complete=master_is_complete)
    print(f"已手工保护 {len(factor_names)} 个 active 因子；自动流程不会将其剔除。")


def unprotect_factors(
    config: BacktestConfig,
    factor_names: list[str],
    active: pd.DataFrame,
    master: pd.DataFrame,
    protections: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    """解除保护但暂不剔除；下次重筛时恢复普通 active 规则。"""
    protections = protections[~protections["因子"].astype(str).isin(factor_names)].copy()
    save_manual_factor_protections(protections, config)
    if "因子" in active.columns:
        active_mask = active["因子"].astype(str).isin(factor_names)
        active.loc[active_mask, "是否手工保护"] = False
        active.loc[active_mask, "手工保护原因"] = ""
        active.loc[active_mask, "手工保护时间"] = ""
    if not master.empty and "因子" in master.columns:
        master_mask = master["因子"].astype(str).isin(factor_names)
        master.loc[master_mask, "是否手工保护"] = False
        master.loc[master_mask, "手工保护原因"] = ""
        master.loc[master_mask, "手工保护时间"] = ""
    persist_library_state(config, active, master, master_is_complete=master_is_complete)
    print(f"已解除 {len(factor_names)} 个因子的手工保护；下次重筛恢复普通规则。")


def approve_factors(
    config: BacktestConfig,
    factor_names: list[str],
    reason: str,
    active: pd.DataFrame,
    master: pd.DataFrame,
    approvals: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    """把当前 pre_active 因子人工批准为真实 active。"""
    if master.empty or "因子库状态" not in master.columns:
        raise ValueError("完整因子主库为空，无法审批。请先运行单因子回测。")
    state_by_factor = master.drop_duplicates("因子", keep="last").set_index("因子")["因子库状态"]
    invalid = [name for name in factor_names if state_by_factor.get(name) != "pre_active"]
    if invalid:
        raise ValueError(f"只有当前 pre_active 因子可以审批，这些因子不符合条件: {invalid}")

    now = pd.Timestamp.now().isoformat(timespec="seconds")
    additions = build_factor_identity_records(factor_names, active, master)
    additions["人工审批原因"] = str(reason)
    additions["人工审批时间"] = now
    additions["审批来源"] = "manual_review"
    approvals = pd.concat([approvals, additions], ignore_index=True).drop_duplicates(
        "因子", keep="last"
    )
    save_manual_factor_approvals(approvals, config)
    mask = master["因子"].astype(str).isin(factor_names)
    master.loc[mask, "因子库状态"] = "active"
    master.loc[mask, "人工审批原因"] = str(reason)
    master.loc[mask, "人工审批时间"] = now
    active = pd.concat([active, master.loc[mask]], ignore_index=True, sort=False)
    active = active.drop_duplicates("因子", keep="last")
    persist_library_state(
        config,
        active,
        master,
        master_is_complete=master_is_complete,
    )
    print(f"已人工批准 {len(factor_names)} 个因子进入真实 active 因子库。")


def revoke_factors(
    config: BacktestConfig,
    factor_names: list[str],
    active: pd.DataFrame,
    master: pd.DataFrame,
    approvals: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    """撤销 active 资格，但保留规则候选资格和回测图。"""
    approvals = approvals[~approvals["因子"].astype(str).isin(factor_names)].copy()
    save_manual_factor_approvals(approvals, config)
    if "因子" in active.columns:
        active = active[~active["因子"].astype(str).isin(factor_names)].copy()
    if not master.empty and "因子" in master.columns:
        mask = master["因子"].astype(str).isin(factor_names)
        active_mask = mask & master["因子库状态"].eq("active")
        master.loc[active_mask, "因子库状态"] = "pre_active"
        master.loc[mask, "人工审批原因"] = ""
        master.loc[mask, "人工审批时间"] = ""
    persist_library_state(
        config,
        active,
        master,
        master_is_complete=master_is_complete,
    )
    print(f"已撤销 {len(factor_names)} 个因子的 active 资格，合格因子退回 pre_active。")


def restore_factors(
    config: BacktestConfig,
    factor_names: list[str],
    active: pd.DataFrame,
    master: pd.DataFrame,
    exclusions: pd.DataFrame,
    *,
    master_is_complete: bool = True,
) -> None:
    exclusions = exclusions[~exclusions["因子"].astype(str).isin(factor_names)].copy()
    save_manual_factor_exclusions(exclusions, config)
    if not master.empty and "因子" in master.columns:
        mask = master["因子"].astype(str).isin(factor_names)
        manual_mask = mask & master.get("拒绝原因", pd.Series("", index=master.index)).eq(
            "manual_exclusion"
        )
        master.loc[manual_mask, "因子库状态"] = "rejected"
        master.loc[manual_mask, "拒绝原因"] = "manual_restore_pending_retest"
        master.loc[mask, "人工排除原因"] = ""
        master.loc[mask, "人工排除时间"] = ""
    persist_library_state(
        config,
        active,
        master,
        master_is_complete=master_is_complete,
    )
    print(f"已恢复 {len(factor_names)} 个因子的参选资格；下次单因子重筛后才可能重新进入active。")


def main() -> None:
    args = build_parser().parse_args()
    config = build_maintenance_config(args)
    master = load_existing_factor_library(config)
    master_is_complete = not master.empty and not bool(
        master.attrs.get("factor_library_partial", False)
    )
    active = load_active_library(config, master)
    pre_active = load_pre_active_library(config, master)
    if not master_is_complete:
        active_rows = active.copy()
        pre_active_rows = pre_active.copy()
        active_rows["因子库状态"] = "active"
        pre_active_rows["因子库状态"] = "pre_active"
        master = pd.concat(
            [active_rows, pre_active_rows],
            ignore_index=True,
            sort=False,
        ).copy()
    approvals = load_manual_factor_approvals(config)
    exclusions = load_manual_factor_exclusions(config)
    protections = load_manual_factor_protections(config)
    identity_catalogs = (active, pre_active, master)
    if not approvals.empty:
        approvals = enrich_factor_identity_records(approvals, *identity_catalogs)
        save_manual_factor_approvals(approvals, config)
    if not exclusions.empty:
        exclusions = enrich_factor_identity_records(exclusions, *identity_catalogs)
        save_manual_factor_exclusions(exclusions, config)
    if not protections.empty:
        protections = enrich_factor_identity_records(protections, *identity_catalogs)
        save_manual_factor_protections(protections, config)
    library_dir = get_factor_library_dir(config)
    print(f"维护目录: {library_dir}")

    if args.action == "list":
        print(
            f"pre_active因子数量: {len(pre_active)}；active因子数量: {len(active)}；"
            f"人工审批数量: {len(approvals)}；手工保护数量: {len(protections)}；"
            f"人工排除数量: {len(exclusions)}"
        )
        if not pre_active.empty:
            print("\n待审核pre_active：")
            columns = [column for column in ["因子编号", "因子标签", "因子"] if column in pre_active.columns]
            print(pre_active[columns].to_string(index=False))
        if not exclusions.empty:
            print(exclusions.to_string(index=False))
        if not protections.empty:
            print("\n手工保护因子：")
            print(protections.to_string(index=False))
        return
    if not args.factors:
        raise ValueError("该操作必须至少提供一个因子名称、标签或编号。")
    factor_names = resolve_factor_names(
        args.factors,
        pd.concat([active, pre_active], ignore_index=True, sort=False),
        master,
        exclusions,
        protections,
        approvals,
    )
    if args.action == "approve":
        approve_factors(
            config,
            factor_names,
            args.reason,
            active,
            master,
            approvals,
            master_is_complete=master_is_complete,
        )
    elif args.action == "revoke":
        revoke_factors(
            config,
            factor_names,
            active,
            master,
            approvals,
            master_is_complete=master_is_complete,
        )
    elif args.action == "exclude":
        exclude_factors(
            config,
            factor_names,
            args.reason,
            active,
            master,
            exclusions,
            master_is_complete=master_is_complete,
        )
    elif args.action == "protect":
        protect_factors(
            config,
            factor_names,
            args.reason,
            active,
            master,
            protections,
            master_is_complete=master_is_complete,
        )
    elif args.action == "unprotect":
        unprotect_factors(
            config,
            factor_names,
            active,
            master,
            protections,
            master_is_complete=master_is_complete,
        )
    else:
        restore_factors(
            config,
            factor_names,
            active,
            master,
            exclusions,
            master_is_complete=master_is_complete,
        )


if __name__ == "__main__":
    main()

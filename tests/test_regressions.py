from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from composite_factor_backtest import (
    audit_active_library_oos_cutoff,
    build_factor_signal_features,
    build_historical_probability_ensemble,
    build_training_sample_weights,
    calculate_block_bootstrap_statistics,
    calculate_future_target_outcomes,
    calculate_next_bar_direction,
    calculate_prediction_metrics_for_segment,
    freeze_active_factor_library_for_run,
    get_enabled_composite_models,
    get_xgboost_target_horizon,
    load_active_factor_names,
    load_composite_artifact_manifest,
    predict_composite_probability,
    run_composite_backtest,
    train_composite_classifier,
    write_composite_artifact_manifest,
)
from config import BacktestConfig, validate_backtest_config
from framework.factor_library import (
    build_factor_library,
    build_compact_rejected_library,
    build_selection_metric,
    conservative_pair,
    get_existing_factor_library_storage_path,
    get_factor_library_dir,
    get_selection_config_value,
    load_manual_factor_approvals,
    load_manual_factor_exclusions,
    load_existing_factor_library,
    rank_single_factor_summary,
    save_factor_library,
)
from factor_library_manager import (
    approve_factors,
    exclude_factors,
    resolve_factor_names,
    revoke_factors,
    restore_factors,
)
from framework.data_loader import (
    filter_completed_daily_bars,
    get_data_cache_path,
    get_local_data_candidates,
)
from framework.factors import (
    EXPANDED_FACTOR_END_INDEX,
    EXPANDED_FACTOR_START_INDEX,
    FAMILY_EXPANSION_END_INDEX,
    FAMILY_EXPANSION_START_INDEX,
    SECOND_FAMILY_EXPANSION_END_INDEX,
    SECOND_FAMILY_EXPANSION_START_INDEX,
    THIRD_FAMILY_EXPANSION_END_INDEX,
    THIRD_FAMILY_EXPANSION_START_INDEX,
    TOTAL_FACTOR_END_INDEX,
    build_factors,
    build_single_factor_matrix,
    resolve_single_factor_requested_factors,
)
from framework.factor_taxonomy import classify_factor
from multi_symbol_backtest import (
    collect_multi_symbol_manifest_outputs,
    pipeline_state_matches,
    read_single_factor_summary_for_pruning,
    save_multi_symbol_portfolio,
    save_multi_symbol_model_report,
    save_pipeline_state,
    should_skip_composite_pipeline,
    summarize_active_library,
)
from framework.output_layout import (
    apply_frequency_runtime_defaults,
    get_multi_symbol_portfolio_dir,
    get_multi_symbol_reports_dir,
    get_multi_symbol_summary_dir,
    get_research_output_dir,
    get_symbol_output_dir as get_organized_symbol_output_dir,
    resolve_existing_symbol_output_dir,
)
from framework.project_fingerprint import get_default_fingerprint_files
from framework.runtime_utils import write_json_atomic
from single_factor_backtest import (
    calculate_metrics,
    get_active_factor_plot_names,
    get_expensive_diagnostic_skip_reason,
    get_reusable_single_factor_plot,
    prefilter_correlated_single_factors,
    run_backtest,
)
from framework.factor_builders.parametric import add_parametric_factors
from framework.factor_builders.expanded import (
    EXPANDED_FACTOR_COUNT,
    LEGACY_EXPANDED_FACTOR_COUNT,
    SECOND_EXPANDED_FACTOR_COUNT,
    add_expanded_factors,
    get_expanded_factor_names,
)
from framework.factor_builders.family_expansion import (
    FAMILY_EXPANSION_COUNT,
    FAMILY_EXPANSION_TOTAL_COUNT,
    FAMILY_PREFIXES,
    get_family_expansion_names,
)
from framework.factor_builders.family_expansion2 import (
    SECOND_FAMILY_COUNT,
    SECOND_FAMILY_PREFIXES,
    SECOND_FAMILY_TOTAL_COUNT,
    get_second_family_expansion_names,
)
from framework.factor_builders.family_expansion3 import (
    THIRD_FAMILY_COUNT,
    THIRD_FAMILY_PREFIXES,
    THIRD_FAMILY_TOTAL_COUNT,
    get_third_family_expansion_names,
)
from trading_signal import (
    get_factor_weight_series,
    get_live_model_predict_index,
    load_cached_factor_inputs,
    load_live_factor_inputs,
)


class FrameworkRegressionTests(unittest.TestCase):

    def test_factor_library_uses_compressed_master_and_compact_rejected_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            config.factor_library_storage_format = "pickle"
            library_dir = get_factor_library_dir(config)
            full = pd.DataFrame(
                {
                    "因子": ["factor_a", "factor_b"],
                    "因子编号": [1, 2],
                    "因子库状态": ["active", "rejected"],
                    "拒绝原因": ["", "low_selection_sharpe"],
                    "初筛夏普": [1.5, -0.5],
                    "测试月度收益集中度": [0.2, 0.9],
                }
            )
            active = full.iloc[[0]].copy()
            rejected = full.iloc[[1]].copy()
            full.to_csv(library_dir / "factor_library_all.csv", index=False)

            save_factor_library(active, full, rejected, config)

            master_path = get_existing_factor_library_storage_path(config)
            self.assertEqual(master_path, library_dir / "factor_library_all.pkl.gz")
            self.assertFalse((library_dir / "factor_library_all.csv").exists())
            loaded = load_existing_factor_library(config)
            pd.testing.assert_frame_equal(loaded, full)
            compact = pd.read_csv(library_dir / "rejected_factors.csv")
            self.assertIn("拒绝原因", compact.columns)
            self.assertNotIn("测试月度收益集中度", compact.columns)
            self.assertEqual(build_compact_rejected_library(rejected).shape[0], 1)
    def test_organized_multi_symbol_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            expected_root = Path(temp_dir) / "by_symbol"
            self.assertEqual(
                get_organized_symbol_output_dir(config, "CU.SHF"),
                expected_root / "symbols" / "CU_SHF",
            )
            self.assertEqual(get_multi_symbol_summary_dir(config), expected_root / "summary" / "30min")
            self.assertEqual(get_multi_symbol_portfolio_dir(config), expected_root / "portfolio" / "30min")
            self.assertEqual(get_multi_symbol_reports_dir(config), expected_root / "reports" / "30min")

            legacy = expected_root / "CU_SHF"
            legacy.mkdir(parents=True)
            self.assertEqual(resolve_existing_symbol_output_dir(config, "CU.SHF"), legacy)

    def test_multi_symbol_model_report_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            symbol_dir = Path(temp_dir) / "symbol_result"
            composite_dir = symbol_dir / "composite_factor" / "30min"
            composite_dir.mkdir(parents=True)
            index = pd.date_range("2025-01-01", periods=80, freq="30min")
            pd.DataFrame(
                {
                    "strategy_net_return": np.tile([0.001, -0.0004], 40),
                    "benchmark_return": np.tile([0.0003, -0.0002], 40),
                    "position": np.ones(80),
                },
                index=index,
            ).to_csv(composite_dir / "composite_detail.csv")
            summary = pd.DataFrame(
                [
                    {
                        "品种": "CU.SHF",
                        "输出目录": str(symbol_dir),
                        "综合因子状态": "完成",
                    }
                ]
            )
            report_dir = Path(temp_dir) / "reports"

            result = save_multi_symbol_model_report(summary, config, report_dir)

            self.assertEqual(result, report_dir / "multi_symbol_model_backtest_report.png")
            self.assertTrue(result.is_file())
            report_summary = pd.read_csv(report_dir / "multi_symbol_model_backtest_summary.csv")
            self.assertEqual(report_summary.loc[0, "品种"], "CU.SHF")

    def test_factor_ranking_is_invariant_to_final_test_metrics(self) -> None:
        config = BacktestConfig()
        summary = pd.DataFrame(
            {
                "因子": ["factor_a", "factor_b"],
                "训练夏普比率": [1.5, 0.8],
                "验证夏普比率": [1.0, 0.8],
                "训练累计收益": [0.10, 0.05],
                "验证累计收益": [0.08, 0.05],
                "测试夏普比率": [-100.0, 100.0],
                "测试累计收益": [-0.99, 9.0],
            }
        )
        _, ranked_before = rank_single_factor_summary(summary, config)
        changed_test = summary.copy()
        changed_test["测试夏普比率"] *= -1
        changed_test["测试累计收益"] *= -1
        _, ranked_after = rank_single_factor_summary(changed_test, config)

        self.assertEqual(
            ranked_before["因子"].tolist(),
            ranked_after["因子"].tolist(),
        )
        self.assertEqual(ranked_before.iloc[0]["因子"], "factor_a")

    def test_factor_selection_metric_never_falls_back_to_test(self) -> None:
        frame = pd.DataFrame(
            {
                "训练胜率": [0.55, 0.60, np.nan],
                "验证胜率": [0.65, np.nan, np.nan],
                "测试胜率": [0.0, 1.0, 1.0],
            }
        )
        selected, source = build_selection_metric(
            frame,
            "验证胜率",
            "训练胜率",
        )
        self.assertEqual(selected.tolist()[:2], [0.65, 0.60])
        self.assertTrue(pd.isna(selected.iloc[2]))
        self.assertEqual(source.tolist(), ["验证胜率", "训练胜率", "不可用"])

    def test_conservative_factor_metric_requires_traceable_training_value(self) -> None:
        frame = pd.DataFrame(
            {
                "训练夏普比率": [1.2, np.nan, 0.8],
                "验证夏普比率": [0.9, 9.0, np.nan],
                "初筛夏普": [100.0, 100.0, 100.0],
                "测试夏普比率": [100.0, 100.0, 100.0],
            }
        )
        metric = conservative_pair(frame, "训练夏普比率", "验证夏普比率")
        self.assertAlmostEqual(metric.iloc[0], 0.9)
        self.assertTrue(pd.isna(metric.iloc[1]))
        self.assertAlmostEqual(metric.iloc[2], 0.8)

    def test_multi_symbol_pruning_summary_ignores_final_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_dir = Path(temp_dir) / "single_factor"
            summary_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "因子": ["factor_a", "legacy_test_only"],
                    "训练夏普比率": [1.2, np.nan],
                    "验证夏普比率": [0.8, np.nan],
                    "训练累计收益": [0.10, np.nan],
                    "验证累计收益": [0.05, np.nan],
                    "测试夏普比率": [-99.0, 999.0],
                    "测试累计收益": [-9.0, 99.0],
                }
            ).to_csv(summary_dir / "single_factor_all_summary.csv", index=False)

            result = read_single_factor_summary_for_pruning("C.DCE", temp_dir)
            self.assertEqual(result["因子"].tolist(), ["factor_a"])
            self.assertAlmostEqual(result.iloc[0]["初筛夏普"], 0.8)
            self.assertAlmostEqual(result.iloc[0]["初筛累计收益"], 0.05)

    def test_multi_symbol_active_summary_never_falls_back_to_test(self) -> None:
        active = pd.DataFrame(
            {
                "因子": ["legacy_factor"],
                "初筛夏普": [np.nan],
                "测试夏普比率": [99.0],
            }
        )
        summary = summarize_active_library(active)
        self.assertTrue(pd.isna(summary["active平均入库夏普"]))
        self.assertTrue(pd.isna(summary["active最高入库夏普"]))

    def test_live_factor_weights_are_invariant_to_final_test_metrics(self) -> None:
        active = pd.DataFrame(
            {
                "因子": ["factor_a", "factor_b"],
                "训练夏普比率": [1.0, 2.0],
                "验证夏普比率": [0.8, 1.5],
                "测试夏普比率": [100.0, -100.0],
                "测试累计收益": [99.0, -99.0],
            }
        )
        before = get_factor_weight_series(active, ["factor_a", "factor_b"])
        changed = active.copy()
        changed["测试夏普比率"] *= -1
        changed["测试累计收益"] *= -1
        after = get_factor_weight_series(changed, ["factor_a", "factor_b"])
        pd.testing.assert_series_equal(before, after)
        self.assertAlmostEqual(before.loc["factor_a"], 0.8 / 2.3)
        self.assertAlmostEqual(before.loc["factor_b"], 1.5 / 2.3)

    def test_legacy_factor_selection_config_explicitly_overrides_canonical(self) -> None:
        config = BacktestConfig()
        self.assertIsNone(
            get_selection_config_value(
                config,
                "factor_library_min_selection_win_rate",
                "factor_library_min_test_win_rate",
            )
        )
        config.factor_library_min_test_win_rate = 0.6
        self.assertEqual(
            get_selection_config_value(
                config,
                "factor_library_min_selection_win_rate",
                "factor_library_min_test_win_rate",
            ),
            0.6,
        )

    def test_factor_library_requires_samples_but_not_majority_win_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            self.assertFalse(config.factor_library_require_manual_approval)
            summary = pd.DataFrame(
                {
                    "因子": ["enough_trades", "too_few_trades"],
                    "初筛有效": [True, True],
                    "训练夏普比率": [1.5, 1.5],
                    "验证夏普比率": [1.2, 1.2],
                    "训练累计收益": [0.10, 0.10],
                    "验证累计收益": [0.04, 0.04],
                    "训练胜率": [0.40, 0.90],
                    "验证胜率": [0.40, 0.90],
                    "训练交易次数": [30, 5],
                    "验证交易次数": [10, 2],
                }
            )
            factors = pd.DataFrame(
                {
                    "enough_trades": np.arange(40, dtype="float64"),
                    "too_few_trades": np.arange(40, dtype="float64") * 2,
                }
            )

            active, library_all, _ = build_factor_library(summary, factors, config)

            self.assertEqual(active["因子"].tolist(), ["enough_trades"])
            rejected = library_all.set_index("因子").loc["too_few_trades"]
            self.assertEqual(rejected["拒绝原因"], "low_selection_trade_count")

    def test_factor_library_requires_manual_approval_before_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_storage_format = "pickle"
            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            config.factor_library_require_manual_approval = True
            summary = pd.DataFrame(
                {
                    "因子": ["factor_a"],
                    "初筛有效": [True],
                    "训练夏普比率": [1.5],
                    "验证夏普比率": [1.2],
                    "训练累计收益": [0.10],
                    "验证累计收益": [0.04],
                    "训练交易次数": [30],
                    "验证交易次数": [10],
                }
            )
            factors = pd.DataFrame({"factor_a": np.arange(40, dtype="float64")})

            active, master, rejected = build_factor_library(summary, factors, config)
            self.assertTrue(active.empty)
            self.assertEqual(master.loc[0, "因子库状态"], "pre_active")
            self.assertTrue(rejected.empty)
            save_factor_library(active, master, rejected, config)

            approvals = load_manual_factor_approvals(config)
            approve_factors(
                config,
                ["factor_a"],
                "人工复核通过",
                active,
                master,
                approvals,
            )
            approved_active = pd.read_csv(
                get_factor_library_dir(config) / "active_factors.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(approved_active["因子"].tolist(), ["factor_a"])
            self.assertTrue(
                pd.read_csv(
                    get_factor_library_dir(config) / "pre_active_factors.csv",
                    encoding="utf-8-sig",
                ).empty
            )

            approved_master = load_existing_factor_library(config)
            revoke_factors(
                config,
                ["factor_a"],
                approved_active,
                approved_master,
                load_manual_factor_approvals(config),
            )
            self.assertTrue(
                pd.read_csv(
                    get_factor_library_dir(config) / "active_factors.csv",
                    encoding="utf-8-sig",
                ).empty
            )
            self.assertEqual(
                load_existing_factor_library(config).loc[0, "因子库状态"],
                "pre_active",
            )

    def test_manual_approval_works_when_full_master_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            library_dir = get_factor_library_dir(config)
            parquet_path = library_dir / "factor_library_all.parquet"
            parquet_path.write_bytes(b"unreadable master sentinel")
            partial_master = pd.DataFrame(
                {
                    "因子编号": [1],
                    "因子": ["factor_a"],
                    "因子库状态": ["pre_active"],
                }
            )
            active = partial_master.iloc[0:0].copy()

            approve_factors(
                config,
                ["factor_a"],
                "人工复核通过",
                active,
                partial_master,
                load_manual_factor_approvals(config),
                master_is_complete=False,
            )

            self.assertEqual(parquet_path.read_bytes(), b"unreadable master sentinel")
            saved_active = pd.read_csv(
                library_dir / "active_factors.csv",
                encoding="utf-8-sig",
            )
            saved_pre_active = pd.read_csv(
                library_dir / "pre_active_factors.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(saved_active["因子"].tolist(), ["factor_a"])
            self.assertTrue(saved_pre_active.empty)

    def test_factor_library_recovers_partial_csv_view_when_master_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            library_dir = get_factor_library_dir(config)
            (library_dir / "factor_library_all.parquet").write_bytes(b"invalid parquet")
            pd.DataFrame({"因子": ["factor_active"]}).to_csv(
                library_dir / "active_factors.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame({"因子": ["factor_pending"]}).to_csv(
                library_dir / "pre_active_factors.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(
                {"因子": ["factor_rejected"], "因子库状态": ["rejected"]}
            ).to_csv(
                library_dir / "rejected_factors.csv", index=False, encoding="utf-8-sig"
            )

            restored = load_existing_factor_library(config)

            self.assertEqual(len(restored), 3)
            self.assertTrue(restored.attrs.get("factor_library_partial"))
            statuses = restored.set_index("因子")["因子库状态"].to_dict()
            self.assertEqual(statuses["factor_active"], "active")
            self.assertEqual(statuses["factor_pending"], "pre_active")
            self.assertEqual(statuses["factor_rejected"], "rejected")

    def test_existing_active_is_migrated_to_manual_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            config.factor_library_require_manual_approval = True
            library_dir = get_factor_library_dir(config)
            pd.DataFrame({"因子": ["factor_a"]}).to_csv(
                library_dir / "active_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary = pd.DataFrame(
                {
                    "因子": ["factor_a"],
                    "初筛有效": [True],
                    "训练夏普比率": [1.5],
                    "验证夏普比率": [1.2],
                    "训练累计收益": [0.10],
                    "验证累计收益": [0.04],
                    "训练交易次数": [30],
                    "验证交易次数": [10],
                }
            )

            active, master, _ = build_factor_library(
                summary,
                pd.DataFrame({"factor_a": np.arange(40, dtype="float64")}),
                config,
            )

            self.assertEqual(active["因子"].tolist(), ["factor_a"])
            self.assertEqual(master.loc[0, "因子库状态"], "active")
            approvals = load_manual_factor_approvals(config)
            self.assertEqual(approvals["因子"].tolist(), ["factor_a"])
            self.assertEqual(approvals.loc[0, "审批来源"], "legacy_active_migration")

    def test_manual_exclusion_immediately_updates_and_persists_factor_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_storage_format = "pickle"
            master = pd.DataFrame(
                {
                    "因子编号": [1],
                    "因子": ["factor_a"],
                    "因子标签": ["1_factor_a"],
                    "因子库状态": ["active"],
                    "拒绝原因": [""],
                }
            )
            active = master.copy()
            plot_dir = get_research_output_dir(config, "single_factor")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "1_factor_a_report.png"
            plot_path.write_bytes(b"test png placeholder")
            master["图片文件"] = str(plot_path)
            active["图片文件"] = str(plot_path)
            exclusions = load_manual_factor_exclusions(config)

            exclude_factors(
                config,
                ["factor_a"],
                "人工图形复核不通过",
                active,
                master,
                exclusions,
            )

            saved_active = pd.read_csv(
                get_factor_library_dir(config) / "active_factors.csv",
                encoding="utf-8-sig",
            )
            saved_master = load_existing_factor_library(config)
            saved_exclusions = load_manual_factor_exclusions(config)
            self.assertTrue(saved_active.empty)
            self.assertFalse(plot_path.exists())
            self.assertEqual(saved_master.loc[0, "拒绝原因"], "manual_exclusion")
            self.assertEqual(saved_master.loc[0, "图片文件"], "")
            self.assertEqual(saved_exclusions["因子"].tolist(), ["factor_a"])
            self.assertEqual(
                resolve_factor_names(
                    ["1_factor_a"],
                    saved_active,
                    saved_master,
                    saved_exclusions,
                ),
                ["factor_a"],
            )

            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            fresh_summary = pd.DataFrame(
                {
                    "因子": ["factor_a"],
                    "初筛有效": [True],
                    "训练夏普比率": [2.0],
                    "验证夏普比率": [1.8],
                    "训练累计收益": [0.20],
                    "验证累计收益": [0.10],
                    "训练胜率": [0.60],
                    "验证胜率": [0.60],
                    "训练交易次数": [50],
                    "验证交易次数": [20],
                }
            )
            rebuilt_active, rebuilt_master, _ = build_factor_library(
                fresh_summary,
                pd.DataFrame({"factor_a": np.arange(60, dtype="float64")}),
                config,
            )
            self.assertTrue(rebuilt_active.empty)
            self.assertEqual(rebuilt_master.loc[0, "拒绝原因"], "manual_exclusion")

            restore_factors(
                config,
                ["factor_a"],
                rebuilt_active,
                rebuilt_master,
                saved_exclusions,
            )
            restored_master = load_existing_factor_library(config)
            self.assertTrue(load_manual_factor_exclusions(config).empty)
            self.assertEqual(
                restored_master.loc[0, "拒绝原因"],
                "manual_restore_pending_retest",
            )

    def test_expensive_factor_diagnostics_only_skip_guaranteed_rejections(self) -> None:
        config = BacktestConfig()
        train_metrics = {
            "夏普比率": 1.5,
            "累计收益": 0.10,
            "胜率": 0.40,
            "交易次数": 30.0,
            "最大回撤": -0.08,
        }
        validation_metrics = {
            "夏普比率": 1.2,
            "累计收益": 0.04,
            "胜率": 0.40,
            "交易次数": 10.0,
            "最大回撤": -0.05,
        }
        self.assertIsNone(
            get_expensive_diagnostic_skip_reason(
                train_metrics,
                validation_metrics,
                0.50,
                0.50,
                config,
            )
        )

        validation_metrics["交易次数"] = 2.0
        self.assertEqual(
            get_expensive_diagnostic_skip_reason(
                train_metrics,
                validation_metrics,
                0.50,
                0.50,
                config,
            ),
            "low_selection_trade_count",
        )
        config.single_factor_defer_expensive_diagnostics = False
        self.assertIsNone(
            get_expensive_diagnostic_skip_reason(
                train_metrics,
                validation_metrics,
                0.50,
                0.50,
                config,
            )
        )

    def test_single_factor_plots_cover_all_active_factors_by_default(self) -> None:
        config = BacktestConfig()
        active = pd.DataFrame({"因子": ["factor_a", "factor_b", "factor_c"]})

        self.assertEqual(
            get_active_factor_plot_names(active, config),
            ["factor_a", "factor_b", "factor_c"],
        )

        config.single_factor_plot_all_active = False
        config.single_factor_plot_top_n = 2
        self.assertEqual(
            get_active_factor_plot_names(active, config),
            ["factor_a", "factor_b"],
        )

    def test_existing_single_factor_plot_is_reused_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            output_dir = Path(temp_dir)
            plot_path = output_dir / "1_factor_a_report.png"
            plot_path.write_bytes(b"existing plot")

            self.assertEqual(
                get_reusable_single_factor_plot(output_dir, "1_factor_a", config),
                plot_path,
            )
            config.single_factor_reuse_existing_plots = False
            self.assertIsNone(
                get_reusable_single_factor_plot(output_dir, "1_factor_a", config)
            )
            config.single_factor_reuse_existing_plots = True
            plot_path.write_bytes(b"")
            self.assertIsNone(
                get_reusable_single_factor_plot(output_dir, "1_factor_a", config)
            )

    def test_factor_selection_thresholds_are_validated_after_legacy_resolution(self) -> None:
        config = BacktestConfig()
        config.factor_library_min_selection_win_rate = 1.1
        with self.assertRaisesRegex(ValueError, "min_selection_win_rate"):
            validate_backtest_config(config, "single")

        config.factor_library_min_selection_win_rate = 0.5
        config.factor_library_min_test_win_rate = -0.1
        with self.assertRaisesRegex(ValueError, "min_selection_win_rate"):
            validate_backtest_config(config, "single")

    def test_factor_ranking_rejects_test_only_legacy_summary(self) -> None:
        config = BacktestConfig()
        summary = pd.DataFrame(
            {
                "因子": ["factor_a"],
                "测试夏普比率": [9.0],
                "测试累计收益": [1.0],
            }
        )
        with self.assertRaisesRegex(KeyError, "不能使用最终测试集"):
            rank_single_factor_summary(summary, config)

    def test_factor_ranking_marks_validation_only_row_untraceable(self) -> None:
        config = BacktestConfig()
        summary = pd.DataFrame(
            {
                "因子": ["traceable", "validation_only"],
                "训练夏普比率": [1.2, np.nan],
                "验证夏普比率": [1.0, 9.0],
                "训练累计收益": [0.10, np.nan],
                "验证累计收益": [0.08, 9.0],
            }
        )
        _, ranked = rank_single_factor_summary(summary, config)
        validation_only = ranked.loc[ranked["因子"] == "validation_only"].iloc[0]
        self.assertFalse(bool(validation_only["初筛有效"]))
        self.assertEqual(validation_only["初筛样本"], "不可追溯")

    def test_composite_run_freezes_active_library_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            library_dir = get_factor_library_dir(config)
            run_dir = Path(temp_dir) / "runs" / "test_composite"
            output_dir = get_research_output_dir(config, "composite_factor")
            library_dir.mkdir(parents=True, exist_ok=True)
            source_path = library_dir / "active_factors.csv"
            pd.DataFrame(
                {
                    "因子": ["factor_a"],
                    "训练截止": ["2025-01-01"],
                    "验证截止": ["2025-02-01"],
                }
            ).to_csv(source_path, index=False)

            runtime_config, snapshot_path = freeze_active_factor_library_for_run(
                config,
                output_dir,
                run_dir,
            )
            pd.DataFrame({"因子": ["factor_b"]}).to_csv(source_path, index=False)

            self.assertTrue(runtime_config.use_frozen_active_library)
            self.assertEqual(Path(runtime_config.frozen_active_library_path), snapshot_path)
            self.assertEqual(load_active_factor_names(runtime_config), ["factor_a"])
            self.assertTrue((run_dir / "active_factors_snapshot_manifest.json").exists())
            self.assertFalse(config.use_frozen_active_library)

            audit = audit_active_library_oos_cutoff(
                runtime_config,
                pd.Timestamp("2025-03-01"),
                output_dir,
                run_dir,
            )
            self.assertEqual(audit["status"], "passed")

    def test_active_library_cutoff_audit_rejects_future_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.use_frozen_active_library = True
            active_path = Path(temp_dir) / "future_active.csv"
            config.frozen_active_library_path = str(active_path)
            pd.DataFrame(
                {
                    "因子": ["factor_a"],
                    "验证截止": ["2025-04-01"],
                }
            ).to_csv(active_path, index=False)
            output_dir = Path(temp_dir) / "composite_factor"

            with self.assertRaisesRegex(ValueError, "晚于最终测试起点"):
                audit_active_library_oos_cutoff(
                    config,
                    pd.Timestamp("2025-03-01"),
                    output_dir,
                    None,
                )
            audit = json.loads(
                (output_dir / "active_library_oos_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["status"], "failed")

    def test_active_library_cutoff_auto_allows_legacy_metadata_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.use_frozen_active_library = True
            active_path = Path(temp_dir) / "legacy_active.csv"
            config.frozen_active_library_path = str(active_path)
            pd.DataFrame({"因子": ["factor_a"]}).to_csv(active_path, index=False)

            audit = audit_active_library_oos_cutoff(
                config,
                pd.Timestamp("2025-03-01"),
                Path(temp_dir) / "composite_factor",
                None,
            )
            self.assertEqual(audit["status"], "failed")
            self.assertIn("缺少", audit["reason"])

    def test_atomic_json_write_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            write_json_atomic(path, {"version": 1, "items": [1, 2]})
            write_json_atomic(path, {"version": 2})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_multi_manifest_output_index_ignores_unrelated_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_dir = Path(temp_dir) / "by_symbol"
            symbol_dir = summary_dir / "C_DCE"
            (symbol_dir / "composite_factor").mkdir(parents=True)
            (symbol_dir / "runs" / "old_run").mkdir(parents=True)
            (summary_dir / "multi_symbol_summary.csv").write_text("current", encoding="utf-8")
            (symbol_dir / ".composite_pipeline_state.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (symbol_dir / "composite_factor" / "composite_detail.csv").write_text(
                "current",
                encoding="utf-8",
            )
            unrelated_path = symbol_dir / "runs" / "old_run" / "large_old_output.csv"
            unrelated_path.write_text("old", encoding="utf-8")
            summary = pd.DataFrame(
                [{"品种": "C.DCE", "输出目录": str(symbol_dir)}]
            )

            entries = collect_multi_symbol_manifest_outputs(summary, summary_dir)
            paths = {str(entry["path"]).replace("\\", "/") for entry in entries}

            self.assertIn("multi_symbol_summary.csv", paths)
            self.assertIn("C_DCE/.composite_pipeline_state.json", paths)
            self.assertIn("C_DCE/composite_factor/composite_detail.csv", paths)
            self.assertNotIn("C_DCE/runs/old_run/large_old_output.csv", paths)

    def test_composite_artifact_manifest_detects_modified_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.run_id = "test_run"
            output_dir = get_research_output_dir(config, "composite_factor")
            library_dir = get_factor_library_dir(config)
            output_dir.mkdir(parents=True)
            library_dir.mkdir(parents=True, exist_ok=True)
            active_path = library_dir / "active_factors.csv"
            pd.DataFrame({"因子": ["factor_a"]}).to_csv(active_path, index=False)
            index = pd.date_range("2025-01-01", periods=3, freq="30min")
            detail_path = output_dir / "composite_detail.csv"
            pd.DataFrame({"strategy_net_return": [0.0, 0.01, -0.01]}, index=index).to_csv(
                detail_path
            )
            pd.Series({"累计收益": 0.0}).to_csv(output_dir / "composite_summary.csv")

            write_composite_artifact_manifest(
                config,
                output_dir,
                "xgboost",
                index[1],
                index[2],
            )
            manifest, error = load_composite_artifact_manifest(
                output_dir,
                expected_symbol=config.symbol,
            )
            self.assertIsNotNone(manifest)
            self.assertEqual(error, "")

            with active_path.open("a", encoding="utf-8") as file:
                file.write("\n")
            manifest, error = load_composite_artifact_manifest(output_dir)
            self.assertIsNone(manifest)
            self.assertIn("active", error)

            pd.DataFrame({"因子": ["factor_a"]}).to_csv(active_path, index=False)
            write_composite_artifact_manifest(
                config,
                output_dir,
                "xgboost",
                index[1],
                index[2],
            )
            with detail_path.open("a", encoding="utf-8") as file:
                file.write("\n")
            manifest, error = load_composite_artifact_manifest(
                output_dir,
                expected_symbol=config.symbol,
            )
            self.assertIsNone(manifest)
            self.assertIn("composite_detail", error)

            write_composite_artifact_manifest(
                config,
                output_dir,
                "xgboost",
                index[1],
                index[2],
            )
            manifest_path = output_dir / "composite_artifact_manifest.json"
            invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invalid_manifest["outputs"] = {}
            manifest_path.write_text(
                json.dumps(invalid_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest, error = load_composite_artifact_manifest(output_dir)
            self.assertIsNone(manifest)
            self.assertIn("composite_detail", error)

    def test_empty_current_multi_run_clears_stale_portfolio_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            summary_dir = Path(temp_dir) / "by_symbol"
            summary_dir.mkdir(parents=True)
            stale_csv = summary_dir / "multi_symbol_portfolio_summary.csv"
            stale_png = summary_dir / "multi_symbol_portfolio_report.png"
            stale_csv.write_text("old", encoding="utf-8")
            stale_png.write_bytes(b"old")
            summary = pd.DataFrame(
                [
                    {
                        "品种": "C.DCE",
                        "输出目录": str(Path(temp_dir) / "C_DCE"),
                        "综合因子状态": "未运行",
                    }
                ]
            )

            save_multi_symbol_portfolio(summary, config, summary_dir)

            self.assertFalse(stale_csv.exists())
            self.assertFalse(stale_png.exists())
            inputs = pd.read_csv(summary_dir / "multi_symbol_portfolio_inputs.csv")
            self.assertFalse(bool(inputs.loc[0, "进入组合"]))
            self.assertIn("未完成", inputs.loc[0, "拒绝原因"])

    def test_live_factor_cache_keeps_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            output_dir = get_research_output_dir(config, "composite_factor")
            output_dir.mkdir(parents=True)
            index = pd.date_range("2025-01-01", periods=10, freq="30min")
            factors = pd.DataFrame({"factor_a": np.arange(10.0)}, index=index)
            factors.to_pickle(output_dir / "active_factor_matrix_cache.pkl")
            price_data = pd.DataFrame(
                {
                    "open": np.arange(10.0) + 100.0,
                    "close": np.arange(10.0) + 100.5,
                },
                index=index,
            )
            price_data.tail(3).to_csv(output_dir / "composite_detail.csv")

            cached = load_cached_factor_inputs(
                config,
                ["factor_a"],
                price_data=price_data,
            )

            self.assertIsNotNone(cached)
            cached_data, cached_factors, _ = cached
            self.assertEqual(len(cached_data), 10)
            self.assertEqual(len(cached_factors), 10)

    def test_live_signal_rebuilds_only_active_factors_when_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            self.assertTrue(config.trading_signal_rebuild_missing_factors)
            symbol_dir = Path(temp_dir) / "by_symbol" / "CU_SHF"
            library_dir = symbol_dir / "factor_library" / "30min"
            library_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"因子": ["factor_a"]}).to_csv(
                library_dir / "active_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )
            index = pd.date_range("2025-01-01", periods=10, freq="30min")
            price_data = pd.DataFrame(
                {
                    "open": np.arange(10.0) + 100.0,
                    "high": np.arange(10.0) + 101.0,
                    "low": np.arange(10.0) + 99.0,
                    "close": np.arange(10.0) + 100.5,
                    "volume": np.arange(10.0) + 1.0,
                },
                index=index,
            )
            built_factors = pd.DataFrame({"factor_a": np.arange(10.0)}, index=index)

            with patch("trading_signal.fetch_intraday_data", return_value=price_data), patch(
                "trading_signal.build_factors",
                return_value=built_factors,
            ) as build_mock:
                _, _, factors, active_factors, source = load_live_factor_inputs(
                    config,
                    "CU.SHF",
                    "multi",
                )

            self.assertEqual(active_factors, ["factor_a"])
            self.assertEqual(list(factors.columns), ["factor_a"])
            self.assertIn("实时构建", source)
            self.assertEqual(build_mock.call_args.kwargs["requested_factors"], ["factor_a"])

    def test_live_model_predict_index_aligns_to_retrain_boundary(self) -> None:
        config = BacktestConfig()
        config.xgboost_min_train_samples = 100
        config.xgboost_retrain_every = 125
        config.xgboost_trade_confidence_rank_window = 240
        config.trading_signal_model_history_bars = 240
        index = pd.date_range("2025-01-01", periods=1000, freq="30min")

        predict_index = get_live_model_predict_index(index, config)

        self.assertEqual(predict_index[0], index[725])
        self.assertEqual(predict_index[-1], index[-1])
        self.assertGreaterEqual(len(predict_index), 240)

    def test_precomputed_factor_signals_are_not_thresholded_twice(self) -> None:
        config = BacktestConfig()
        index = pd.date_range("2025-01-01", periods=3, freq="30min")
        factors = pd.DataFrame({"factor_a": [-1.0, 0.0, 1.0]}, index=index)
        factors.attrs["precomputed_factor_signals"] = True

        signals = build_factor_signal_features(factors, ["factor_a"], config)

        pd.testing.assert_series_equal(
            signals["factor_a"],
            factors["factor_a"],
            check_names=True,
        )

    def test_parametric_builder_only_returns_requested_window_and_factor(self) -> None:
        config = BacktestConfig()
        config.zscore_window = 20
        index = pd.date_range("2025-01-01 09:00:00", periods=40, freq="30min")
        close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.linspace(1000.0, 1200.0, len(index)),
                "bar_return_cc": close.pct_change(),
                "bar_return_oc": close / (close - 0.2) - 1.0,
            },
            index=index,
        )

        factors = add_parametric_factors(
            frame,
            config,
            requested_factors={"ret_2"},
        )

        self.assertEqual(list(factors.columns), ["ret_2"])

    def test_expanded_factor_universe_has_two_stable_hundred_thousand_name_blocks(self) -> None:
        names = get_expanded_factor_names()
        self.assertEqual(len(names), EXPANDED_FACTOR_COUNT)
        self.assertEqual(len(set(names)), EXPANDED_FACTOR_COUNT)
        self.assertEqual(LEGACY_EXPANDED_FACTOR_COUNT, 100_000)
        self.assertEqual(SECOND_EXPANDED_FACTOR_COUNT, 100_000)
        self.assertEqual(names[0], "expanded_retcc_meangap_w2_l0_s1")
        self.assertEqual(
            names[LEGACY_EXPANDED_FACTOR_COUNT - 1],
            "expanded_rangevol_efficiency_w233_l5_s6",
        )
        self.assertTrue(names[LEGACY_EXPANDED_FACTOR_COUNT].startswith("expanded2_"))
        self.assertTrue(names[-1].startswith("expanded2_"))
        self.assertEqual(
            EXPANDED_FACTOR_END_INDEX - EXPANDED_FACTOR_START_INDEX + 1,
            EXPANDED_FACTOR_COUNT,
        )

    def test_expanded_factor_range_and_builder_are_strictly_requested_only(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "range"
        config.single_factor_range = (EXPANDED_FACTOR_START_INDEX, EXPANDED_FACTOR_START_INDEX + 9)
        config.zscore_window = 20
        index = pd.date_range("2025-01-01", periods=100, freq="D")
        close = pd.Series(np.linspace(100.0, 115.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.003,
                "low": close * 0.997,
                "close": close,
                "volume": np.linspace(1000.0, 1500.0, len(index)),
                "amt": np.linspace(100000.0, 180000.0, len(index)),
            },
            index=index,
        )
        requested, factor_id_map = resolve_single_factor_requested_factors(data, config)
        self.assertEqual(len(requested or []), 10)
        self.assertEqual(factor_id_map[requested[0]], EXPANDED_FACTOR_START_INDEX)
        self.assertEqual(factor_id_map[requested[-1]], EXPANDED_FACTOR_START_INDEX + 9)

        selected = {requested[0], requested[-1]}
        direct = add_expanded_factors(data.assign(bar_return_cc=close.pct_change()), config, selected)
        self.assertEqual(set(direct.columns), selected)
        factors = build_factors(data, config, requested_factors=list(selected))
        self.assertEqual(set(factors.columns), selected)

    def test_second_expanded_block_keeps_old_ids_and_builds_on_demand(self) -> None:
        config = BacktestConfig()
        second_start = EXPANDED_FACTOR_START_INDEX + LEGACY_EXPANDED_FACTOR_COUNT
        config.single_factor_scope = "range"
        config.single_factor_range = (second_start, second_start + 2)
        config.zscore_window = 20
        index = pd.date_range("2025-01-01", periods=120, freq="D")
        close = pd.Series(np.linspace(100.0, 118.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.003,
                "low": close * 0.997,
                "close": close,
                "volume": np.linspace(1000.0, 1600.0, len(index)),
                "amt": np.linspace(100000.0, 190000.0, len(index)),
            },
            index=index,
        )

        requested, factor_id_map = resolve_single_factor_requested_factors(data, config)

        self.assertEqual(len(requested or []), 3)
        self.assertTrue(all(name.startswith("expanded2_") for name in requested or []))
        self.assertEqual(factor_id_map[requested[0]], 230001)
        self.assertEqual(factor_id_map[requested[-1]], 230003)
        built = build_factors(data, config, requested_factors=requested)
        self.assertEqual(set(built.columns), set(requested))

    def test_four_family_expansion_blocks_have_exact_counts_and_stable_ids(self) -> None:
        names = get_family_expansion_names()
        self.assertEqual(FAMILY_EXPANSION_COUNT, 30_000)
        self.assertEqual(FAMILY_EXPANSION_TOTAL_COUNT, 120_000)
        self.assertEqual(len(names), FAMILY_EXPANSION_TOTAL_COUNT)
        self.assertEqual(len(set(names)), FAMILY_EXPANSION_TOTAL_COUNT)
        for block_index, prefix in enumerate(FAMILY_PREFIXES):
            self.assertTrue(names[block_index * FAMILY_EXPANSION_COUNT].startswith(f"{prefix}_"))
            self.assertTrue(
                names[(block_index + 1) * FAMILY_EXPANSION_COUNT - 1].startswith(f"{prefix}_")
            )
        self.assertEqual(FAMILY_EXPANSION_START_INDEX, EXPANDED_FACTOR_END_INDEX + 1)
        self.assertEqual(
            FAMILY_EXPANSION_END_INDEX - FAMILY_EXPANSION_START_INDEX + 1,
            FAMILY_EXPANSION_TOTAL_COUNT,
        )
        self.assertEqual(FAMILY_EXPANSION_END_INDEX, 450_000)

    def test_four_family_expansions_resolve_and_build_strictly_on_demand(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "range"
        config.zscore_window = 20
        index = pd.date_range("2024-01-01", periods=180, freq="D")
        close = pd.Series(np.linspace(100.0, 125.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": np.linspace(1000.0, 1800.0, len(index)),
                "amt": np.linspace(100000.0, 220000.0, len(index)),
            },
            index=index,
        )
        starts = [FAMILY_EXPANSION_START_INDEX + i * FAMILY_EXPANSION_COUNT for i in range(4)]
        requested: list[str] = []
        for expected_prefix, start in zip(FAMILY_PREFIXES, starts):
            config.single_factor_range = (start, start)
            names, factor_id_map = resolve_single_factor_requested_factors(data, config)
            self.assertEqual(len(names or []), 1)
            self.assertTrue(names[0].startswith(f"{expected_prefix}_"))
            self.assertEqual(factor_id_map[names[0]], start)
            requested.extend(names)

        related = data.copy()
        related["close"] = close * 1.01 + np.sin(np.arange(len(index)))
        built = build_factors(
            data,
            config,
            related_data_map={"M.DCE": related},
            requested_factors=requested,
        )
        self.assertEqual(set(built.columns), set(requested))
        expected_families = ["parametric", "calendar", "non_cross_complex", "cross_asset"]
        self.assertEqual(
            [classify_factor(name)["因子家族"] for name in requested],
            expected_families,
        )

    def test_second_five_family_blocks_have_exact_counts_and_stable_ids(self) -> None:
        names = get_second_family_expansion_names()
        old_names = get_family_expansion_names()
        self.assertEqual(SECOND_FAMILY_COUNT, 20_000)
        self.assertEqual(SECOND_FAMILY_TOTAL_COUNT, 100_000)
        self.assertEqual(len(names), SECOND_FAMILY_TOTAL_COUNT)
        self.assertEqual(len(set(names)), SECOND_FAMILY_TOTAL_COUNT)
        self.assertFalse(set(names).intersection(old_names))
        for block_index, prefix in enumerate(SECOND_FAMILY_PREFIXES):
            self.assertTrue(names[block_index * SECOND_FAMILY_COUNT].startswith(f"{prefix}_"))
            self.assertTrue(
                names[(block_index + 1) * SECOND_FAMILY_COUNT - 1].startswith(f"{prefix}_")
            )
        self.assertEqual(SECOND_FAMILY_EXPANSION_START_INDEX, FAMILY_EXPANSION_END_INDEX + 1)
        self.assertEqual(
            SECOND_FAMILY_EXPANSION_END_INDEX - SECOND_FAMILY_EXPANSION_START_INDEX + 1,
            SECOND_FAMILY_TOTAL_COUNT,
        )
        self.assertEqual(SECOND_FAMILY_EXPANSION_END_INDEX, 550_000)

    def test_second_five_family_expansions_build_strictly_on_demand(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "range"
        config.zscore_window = 20
        index = pd.date_range("2023-01-01", periods=300, freq="D")
        close = pd.Series(100.0 + np.linspace(0.0, 20.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": 1000.0 + np.arange(len(index), dtype="float64"),
                "amt": 100000.0 + np.arange(len(index), dtype="float64") * 100.0,
            },
            index=index,
        )
        starts = [
            SECOND_FAMILY_EXPANSION_START_INDEX + block * SECOND_FAMILY_COUNT
            for block in range(5)
        ]
        requested: list[str] = []
        for expected_prefix, start in zip(SECOND_FAMILY_PREFIXES, starts):
            config.single_factor_range = (start, start)
            names, factor_id_map = resolve_single_factor_requested_factors(data, config)
            self.assertEqual(len(names or []), 1)
            self.assertTrue(names[0].startswith(f"{expected_prefix}_"))
            self.assertEqual(factor_id_map[names[0]], start)
            requested.extend(names)

        related = data.copy()
        related["close"] = close * 1.01 + np.sin(np.arange(len(index)))
        built = build_factors(
            data,
            config,
            related_data_map={"M.DCE": related},
            requested_factors=requested,
        )
        self.assertEqual(set(built.columns), set(requested))
        self.assertEqual(
            [classify_factor(name)["因子家族"] for name in requested],
            ["cross_asset", "non_cross_complex", "expanded", "parametric", "calendar"],
        )

    def test_third_five_family_blocks_have_exact_counts_and_stable_ids(self) -> None:
        names = get_third_family_expansion_names()
        old_names = get_second_family_expansion_names()
        self.assertEqual(THIRD_FAMILY_COUNT, 10_000)
        self.assertEqual(THIRD_FAMILY_TOTAL_COUNT, 50_000)
        self.assertEqual(len(names), THIRD_FAMILY_TOTAL_COUNT)
        self.assertEqual(len(set(names)), THIRD_FAMILY_TOTAL_COUNT)
        self.assertFalse(set(names).intersection(old_names))
        for block_index, prefix in enumerate(THIRD_FAMILY_PREFIXES):
            self.assertTrue(names[block_index * THIRD_FAMILY_COUNT].startswith(f"{prefix}_"))
            self.assertTrue(
                names[(block_index + 1) * THIRD_FAMILY_COUNT - 1].startswith(f"{prefix}_")
            )
        self.assertEqual(
            THIRD_FAMILY_EXPANSION_START_INDEX,
            SECOND_FAMILY_EXPANSION_END_INDEX + 1,
        )
        self.assertEqual(
            THIRD_FAMILY_EXPANSION_END_INDEX - THIRD_FAMILY_EXPANSION_START_INDEX + 1,
            THIRD_FAMILY_TOTAL_COUNT,
        )
        self.assertEqual(TOTAL_FACTOR_END_INDEX, 600_000)

    def test_third_five_family_expansions_build_strictly_on_demand(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "range"
        config.zscore_window = 20
        index = pd.date_range("2023-01-01", periods=500, freq="D")
        close = pd.Series(100.0 + np.linspace(0.0, 20.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": 1000.0 + np.arange(len(index), dtype="float64"),
                "amt": 100000.0 + np.arange(len(index), dtype="float64") * 100.0,
            },
            index=index,
        )
        starts = [
            THIRD_FAMILY_EXPANSION_START_INDEX + block * THIRD_FAMILY_COUNT
            for block in range(5)
        ]
        requested: list[str] = []
        for expected_prefix, start in zip(THIRD_FAMILY_PREFIXES, starts):
            config.single_factor_range = (start, start)
            names, factor_id_map = resolve_single_factor_requested_factors(data, config)
            self.assertEqual(len(names or []), 1)
            self.assertTrue(names[0].startswith(f"{expected_prefix}_"))
            self.assertEqual(factor_id_map[names[0]], start)
            requested.extend(names)

        related = data.copy()
        related["close"] = close * 1.01 + np.sin(np.arange(len(index)))
        built = build_factors(
            data,
            config,
            related_data_map={"M.DCE": related},
            requested_factors=requested,
        )
        self.assertEqual(set(built.columns), set(requested))
        self.assertEqual(
            [classify_factor(name)["因子家族"] for name in requested],
            ["cross_asset", "non_cross_complex", "expanded", "parametric", "calendar"],
        )

    def test_expanded_new_scope_respects_batch_size(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "new"
        config.single_factor_auto_update_start_index = False
        config.single_factor_new_factor_start_index = EXPANDED_FACTOR_START_INDEX + 25
        config.single_factor_new_factor_start_index_by_symbol = {}
        config.single_factor_new_factor_batch_size = 7
        index = pd.date_range("2025-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            index=index,
        )
        requested, factor_id_map = resolve_single_factor_requested_factors(data, config)
        self.assertEqual(len(requested or []), 7)
        self.assertEqual(factor_id_map[requested[0]], EXPANDED_FACTOR_START_INDEX + 25)
        self.assertEqual(factor_id_map[requested[-1]], EXPANDED_FACTOR_START_INDEX + 31)

    def test_empty_requested_factor_list_never_means_build_all(self) -> None:
        config = BacktestConfig()
        index = pd.date_range("2025-01-01", periods=40, freq="30min")
        data = pd.DataFrame(
            {
                "open": np.linspace(100.0, 101.0, len(index)),
                "high": np.linspace(100.5, 101.5, len(index)),
                "low": np.linspace(99.5, 100.5, len(index)),
                "close": np.linspace(100.1, 101.1, len(index)),
                "volume": np.linspace(1000.0, 1100.0, len(index)),
            },
            index=index,
        )
        with self.assertRaisesRegex(ValueError, "拒绝静默回退"):
            build_factors(data, config, requested_factors=[])

    def test_large_factor_families_build_requested_columns_only(self) -> None:
        config = BacktestConfig()
        config.zscore_window = 30
        config.enable_cross_asset_factors = False
        config.enable_macro_state_factors = False
        config.enable_external_daily_factors = False
        index = pd.date_range("2025-01-01", periods=80, freq="30min")
        close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": np.linspace(1000.0, 1300.0, len(index)),
            },
            index=index,
        )
        requested = [
            "ultra_mean_bar_return_2",
            "hyper_mean_ret_x_vol_2",
            "omega_mean_ret_pressure_2",
            "calendar_time_sin",
        ]
        factors = build_factors(data, config, requested_factors=requested)
        self.assertEqual(list(factors.columns), requested)

    def test_empty_active_library_stops_before_factor_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            active_path = Path(temp_dir) / "active_factors.csv"
            pd.DataFrame(columns=["因子"]).to_csv(active_path, index=False)
            config = BacktestConfig()
            config.use_frozen_active_library = True
            config.frozen_active_library_path = str(active_path)
            with self.assertRaisesRegex(ValueError, "不会回退计算全部因子"):
                load_active_factor_names(config)

    def test_composite_empty_active_stops_before_loading_market_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            library_dir = get_factor_library_dir(config)
            library_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["因子"]).to_csv(
                library_dir / "active_factors.csv",
                index=False,
            )
            config.enable_experiment_run_dirs = False
            with patch("composite_factor_backtest.fetch_intraday_data") as fetch_data:
                with self.assertRaisesRegex(ValueError, "不会回退计算全部因子"):
                    run_composite_backtest(config)
            fetch_data.assert_not_called()

    def test_cross_asset_large_families_build_requested_columns_only(self) -> None:
        config = BacktestConfig()
        config.zscore_window = 30
        config.enable_cross_asset_factors = True
        config.cross_asset_factor_windows = [2]
        config.enable_macro_state_factors = False
        config.enable_external_daily_factors = False
        index = pd.date_range("2025-01-01", periods=80, freq="30min")
        close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
        data = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": np.linspace(1000.0, 1300.0, len(index)),
            },
            index=index,
        )
        related = data.copy()
        related[["open", "high", "low", "close"]] *= 1.1
        requested = [
            "crossultra_mean_m_dce_relative_return_2",
            "crossomega_mean_m_dce_rel_ret_2",
        ]
        factors = build_factors(
            data,
            config,
            related_data_map={"M.DCE": related},
            requested_factors=requested,
        )
        self.assertEqual(list(factors.columns), requested)

    def test_partial_scope_builds_old_active_as_reference_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.single_factor_scope = "range"
            library_dir = get_factor_library_dir(config)
            library_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"因子": ["old_active"], "因子编号": [1]}).to_csv(
                library_dir / "active_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )
            index = pd.date_range("2025-01-01 09:00:00", periods=4, freq="30min")
            data = pd.DataFrame(
                {
                    "open": [100.0] * 4,
                    "high": [101.0] * 4,
                    "low": [99.0] * 4,
                    "close": [100.0] * 4,
                },
                index=index,
            )

            def fake_build(
                _data: pd.DataFrame,
                _config: BacktestConfig,
                requested_factors: list[str] | None = None,
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        factor: np.arange(len(_data), dtype="float64")
                        for factor in requested_factors or []
                    },
                    index=_data.index,
                )

            with (
                patch(
                    "framework.factors.resolve_single_factor_requested_factors",
                    return_value=(["new_factor"], {"new_factor": 2}),
                ),
                patch("framework.factors.build_factors", side_effect=fake_build),
            ):
                factors = build_single_factor_matrix(data, config)

            self.assertEqual(factors.attrs["backtest_factor_columns"], ["new_factor"])
            self.assertEqual(factors.attrs["existing_active_factor_columns"], ["old_active"])
            self.assertEqual(list(factors.columns), ["new_factor", "old_active"])
            self.assertEqual(factors.attrs["factor_id_map"]["old_active"], 1)

    def test_pooled_time_decay_is_equal_within_timestamp(self) -> None:
        config = BacktestConfig()
        config.xgboost_train_use_time_decay_weight = True
        config.xgboost_train_time_decay_half_life = 1
        config.xgboost_train_time_decay_min_weight = 0.0
        config.xgboost_train_time_decay_normalize = False
        config.xgboost_train_neutral_class_weight = 1.0
        config.xgboost_train_nonzero_class_weight = 1.0
        target = pd.Series([1.0, -1.0, 1.0, -1.0])
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-01 09:00:00",
                    "2025-01-01 09:00:00",
                    "2025-01-01 09:30:00",
                    "2025-01-01 09:30:00",
                ]
            )
        )
        weights = build_training_sample_weights(target, config, timestamps=timestamps)
        self.assertAlmostEqual(weights.iloc[0], weights.iloc[1])
        self.assertAlmostEqual(weights.iloc[2], weights.iloc[3])
        self.assertAlmostEqual(weights.iloc[0], 0.5)
        self.assertAlmostEqual(weights.iloc[2], 1.0)

    def test_continuous_return_accounts_for_gap_with_previous_position(self) -> None:
        config = BacktestConfig()
        config.backtest_return_mode = "next_open_continuous"
        config.commission_bps = 0.0
        config.slippage_bps = 0.0
        index = pd.date_range("2025-01-01 09:00:00", periods=3, freq="30min")
        data = pd.DataFrame(
            {
                "open": [100.0, 110.0, 100.0],
                "high": [100.0, 120.0, 100.0],
                "low": [100.0, 110.0, 100.0],
                "close": [100.0, 120.0, 100.0],
                "bar_return_oc": [0.0, 120.0 / 110.0 - 1.0, 0.0],
            },
            index=index,
        )
        signal = pd.DataFrame(
            {
                "composite_score": [0.0, 1.0, 1.0],
                "raw_signal": [0.0, 1.0, 1.0],
                "position": [0.0, 1.0, 1.0],
            },
            index=index,
        )
        detail, _ = run_backtest(data, signal, config)
        self.assertAlmostEqual(detail.loc[index[1], "strategy_gross_return"], 0.10)
        self.assertAlmostEqual(detail.loc[index[2], "strategy_gross_return"], -1.0 / 6.0)

    def test_backtest_preserves_intermittent_missing_score_bars(self) -> None:
        config = BacktestConfig()
        config.commission_bps = 0.0
        config.slippage_bps = 0.0
        index = pd.date_range("2025-01-01 09:00:00", periods=4, freq="30min")
        data = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
            },
            index=index,
        )
        signal = pd.DataFrame(
            {
                "composite_score": [1.0, np.nan, -1.0, 1.0],
                "raw_signal": [1.0, 0.0, -1.0, 1.0],
                "position": [0.0, 1.0, 0.0, -1.0],
            },
            index=index,
        )
        detail, _ = run_backtest(data, signal, config)

        self.assertEqual(list(detail.index), list(index))
        self.assertTrue(pd.isna(detail.loc[index[1], "composite_score"]))

    def test_factor_fingerprint_includes_external_daily_builder(self) -> None:
        paths = {str(path).replace("\\", "/") for path in get_default_fingerprint_files()}
        self.assertIn("framework/factor_builders/external_daily.py", paths)

    def test_pipeline_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            save_pipeline_state(config, "single")
            self.assertTrue(pipeline_state_matches(config, "single"))
            config.signal_threshold += 0.01
            self.assertFalse(pipeline_state_matches(config, "single"))

    def test_composite_pipeline_state_tracks_active_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            library_dir = get_factor_library_dir(config)
            library_dir.mkdir(parents=True, exist_ok=True)
            active_path = library_dir / "active_factors.csv"
            pd.DataFrame({"因子": ["factor_a"]}).to_csv(active_path, index=False)

            save_pipeline_state(config, "composite")
            self.assertTrue(pipeline_state_matches(config, "composite"))
            pd.DataFrame({"因子": ["factor_b"]}).to_csv(active_path, index=False)
            self.assertFalse(pipeline_state_matches(config, "composite"))

    def test_composite_skip_requires_valid_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.multi_symbol_skip_existing = True
            detail_dir = get_research_output_dir(config, "composite_factor")
            detail_dir.mkdir(parents=True)
            pd.DataFrame({"strategy_net_return": [0.0]}).to_csv(
                detail_dir / "composite_detail.csv"
            )
            save_pipeline_state(config, "composite")

            self.assertFalse(should_skip_composite_pipeline(config))

    def test_daily_frequency_uses_independent_data_and_research_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            minute = BacktestConfig()
            minute.output_dir = temp_dir
            minute.data_cache_dir = str(Path(temp_dir) / "data")
            minute.bar_frequency = "30min"

            daily = BacktestConfig()
            daily.output_dir = temp_dir
            daily.data_cache_dir = str(Path(temp_dir) / "data")
            daily.bar_frequency = "1d"

            self.assertEqual(
                get_factor_library_dir(minute),
                Path(temp_dir) / "factor_library" / "30min",
            )
            self.assertEqual(
                get_factor_library_dir(daily),
                Path(temp_dir) / "factor_library" / "1d",
            )
            self.assertEqual(
                get_data_cache_path(daily),
                Path(temp_dir) / "data" / "market" / "1d" / "C_DCE_1d_data.csv",
            )
            self.assertFalse(
                any("30min" in str(path) for path in get_local_data_candidates(daily))
            )

    def test_daily_frequency_applies_daily_training_windows(self) -> None:
        config = BacktestConfig()
        config.bar_frequency = "1d"
        config.xgboost_train_window = 9999
        config.xgboost_min_train_samples = 9999
        config.xgboost_retrain_every = 9999

        resolved = apply_frequency_runtime_defaults(config)

        self.assertEqual(resolved.xgboost_train_window, config.daily_xgboost_train_window)
        self.assertEqual(
            resolved.xgboost_min_train_samples,
            config.daily_xgboost_min_train_samples,
        )
        self.assertEqual(resolved.xgboost_retrain_every, config.daily_xgboost_retrain_every)
        self.assertEqual(
            resolved.composite_ensemble_weight_window,
            config.daily_composite_ensemble_weight_window,
        )
        self.assertEqual(
            resolved.composite_ensemble_min_history,
            config.daily_composite_ensemble_min_history,
        )
        self.assertEqual(config.xgboost_train_window, 9999)

    def test_new_composite_models_produce_aligned_three_class_probabilities(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("当前 Python 环境未安装 scikit-learn。")
        config = BacktestConfig()
        index = pd.date_range("2025-01-01", periods=90, freq="D")
        features = pd.DataFrame(
            {
                "factor_a": np.sin(np.arange(90) / 4.0),
                "factor_b": np.cos(np.arange(90) / 7.0),
                "factor_c": np.arange(90) % 5,
            },
            index=index,
        )
        features.loc[index[3], "factor_b"] = np.nan
        target = pd.Series(np.resize([-1.0, 0.0, 1.0], 90), index=index)

        for model_name in ("elastic_net_logistic", "hist_gradient_boosting"):
            model = train_composite_classifier(
                model_name,
                features,
                target,
                list(features.columns),
                config,
            )
            probability = predict_composite_probability(
                model_name,
                model,
                features.iloc[-5:],
                list(features.columns),
            )
            self.assertEqual(probability.shape, (5, 3))
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-8)

        config.composite_model_names = ["enet", "histgb", "lr"]
        self.assertEqual(
            get_enabled_composite_models(config),
            ["elastic_net_logistic", "hist_gradient_boosting", "logistic_regression"],
        )

    def test_probability_ensemble_weights_only_use_matured_labels(self) -> None:
        config = BacktestConfig()
        config.xgboost_target_horizon = 3
        config.composite_ensemble_weight_window = 20
        config.composite_ensemble_min_history = 8
        config.composite_ensemble_min_directional_accuracy = 0.0
        config.composite_ensemble_min_edge_return_corr = -1.0
        config.composite_ensemble_max_model_weight = 0.60
        config.xgboost_trade_min_edge = 0.0
        config.xgboost_trade_min_probability = 0.0
        index = pd.date_range("2025-01-01", periods=60, freq="D")
        target = pd.Series(np.where(np.arange(60) % 2 == 0, 1.0, -1.0), index=index)
        future_return = target * 0.01
        data = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + np.arange(60) * 0.1,
                "volume": 1000.0,
            },
            index=index,
        )

        def make_signal(correct: bool) -> pd.DataFrame:
            predicted = target.copy()
            if not correct:
                predicted.iloc[::3] *= -1
            up = pd.Series(np.where(predicted > 0, 0.70, 0.10), index=index)
            down = pd.Series(np.where(predicted < 0, 0.70, 0.10), index=index)
            return pd.DataFrame(
                {
                    "target_direction": target,
                    "future_horizon_return": future_return,
                    "prob_down": down,
                    "prob_flat": 0.20,
                    "prob_up": up,
                    "xgboost_signal_direction": 1.0,
                },
                index=index,
            )

        signals = {"model_a": make_signal(True), "model_b": make_signal(False)}
        ensemble, diagnostics = build_historical_probability_ensemble(signals, data, config)
        valid = ensemble[["prob_down", "prob_flat", "prob_up"]].dropna()
        self.assertFalse(valid.empty)
        np.testing.assert_allclose(valid.sum(axis=1), 1.0, atol=1e-10)
        self.assertLessEqual(float(diagnostics["权重"].max()), 0.60 + 1e-12)

        timestamp = index[40]
        changed = {name: frame.copy() for name, frame in signals.items()}
        change_start = index.get_loc(timestamp) - config.xgboost_target_horizon + 1
        for frame in changed.values():
            frame.iloc[change_start:, frame.columns.get_loc("target_direction")] *= -1
            frame.iloc[change_start:, frame.columns.get_loc("future_horizon_return")] *= -1
        _, changed_diagnostics = build_historical_probability_ensemble(changed, data, config)
        original_weight = diagnostics.loc[diagnostics["时间"] == timestamp, "权重"].to_numpy()
        changed_weight = changed_diagnostics.loc[
            changed_diagnostics["时间"] == timestamp,
            "权重",
        ].to_numpy()
        np.testing.assert_allclose(original_weight, changed_weight, atol=1e-12)

    def test_daily_frequency_drops_unfinished_current_bar(self) -> None:
        config = BacktestConfig()
        config.bar_frequency = "1d"
        config.daily_bar_ready_time = "15:30"
        index = pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12"])
        data = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=index)

        before_close = filter_completed_daily_bars(
            data,
            config,
            now=pd.Timestamp("2026-08-11 10:00:00"),
        )
        after_close = filter_completed_daily_bars(
            data,
            config,
            now=pd.Timestamp("2026-08-11 16:00:00"),
        )

        self.assertEqual(list(before_close.index), [pd.Timestamp("2026-08-10")])
        self.assertEqual(
            list(after_close.index),
            [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
        )

    def test_single_factor_corr_prefilter_excludes_final_test_covariates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.factor_selection_covariate_scope = "train_validation"
            config.auto_select_train_ratio = 0.60
            config.auto_select_validation_ratio = 0.20
            config.single_factor_enable_corr_prefilter = True
            config.single_factor_corr_prefilter_threshold = 0.95
            config.single_factor_corr_prefilter_sample_rows = 20
            index = pd.date_range("2025-01-01", periods=100, freq="D")
            factor_a = np.arange(100, dtype="float64")
            factor_b = np.sin(np.arange(100, dtype="float64"))
            # 最终测试段故意完全复制 factor_a；严格筛选不应看到这段分布。
            factor_b[80:] = factor_a[80:]
            factors = pd.DataFrame({"factor_a": factor_a, "factor_b": factor_b}, index=index)

            kept = prefilter_correlated_single_factors(
                factors,
                ["factor_a", "factor_b"],
                config,
                Path(temp_dir),
            )

            self.assertEqual(kept, ["factor_a", "factor_b"])
            summary = pd.read_csv(
                Path(temp_dir) / "single_factor_corr_prefilter_summary.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(summary.loc[0, "相关性样本范围"], "train_validation")
            self.assertIn("2025-03-21", summary.loc[0, "抽样截止"])

    def test_factor_library_corr_excludes_final_test_covariates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_selection_covariate_scope = "train_validation"
            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = True
            config.factor_library_use_signal_corr = False
            config.factor_library_max_corr = 0.80
            rng = np.random.default_rng(123)
            factor_a = rng.normal(size=100)
            factor_b = rng.normal(size=100)
            dominant_test_pattern = np.arange(1, 16, dtype="float64") * 1000.0
            factor_a[85:] = dominant_test_pattern
            factor_b[85:] = dominant_test_pattern
            factors = pd.DataFrame({"factor_a": factor_a, "factor_b": factor_b})
            summary = pd.DataFrame(
                {
                    "因子": ["factor_a", "factor_b"],
                    "初筛有效": [True, True],
                    "训练夏普比率": [1.8, 1.7],
                    "验证夏普比率": [1.4, 1.3],
                    "训练累计收益": [0.10, 0.09],
                    "验证累计收益": [0.05, 0.04],
                    "训练交易次数": [40, 40],
                    "验证交易次数": [20, 20],
                }
            )

            active, master, _ = build_factor_library(summary, factors, config)

            self.assertEqual(active["因子"].tolist(), ["factor_a", "factor_b"])
            self.assertTrue((master["相关性样本数"] == 85).all())
            self.assertTrue((master["相关性样本范围"] == "train_validation").all())

    def test_target_horizon_cost_and_holding_period_are_aligned(self) -> None:
        config = BacktestConfig()
        config.xgboost_target_horizon = 1
        config.xgboost_target_align_with_min_holding = True
        config.xgboost_use_position_rules = True
        config.xgboost_min_holding_bars = 2
        config.xgboost_target_use_dynamic_neutral_threshold = False
        config.xgboost_target_neutral_bps = 0.0
        config.commission_bps = 0.5
        config.slippage_bps = 1.0
        index = pd.date_range("2025-01-01", periods=6, freq="D")
        data = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": [100.0, 100.0, 100.02, 100.10, 99.90, 100.0],
            },
            index=index,
        )

        target = calculate_next_bar_direction(data, index, config)
        outcomes = calculate_future_target_outcomes(data, index, config)

        self.assertEqual(get_xgboost_target_horizon(config), 2)
        self.assertEqual(target.iloc[0], 0.0)
        self.assertEqual(target.iloc[1], 1.0)
        self.assertAlmostEqual(outcomes.iloc[0]["future_horizon_round_trip_cost"], 0.0003)
        self.assertAlmostEqual(outcomes.iloc[0]["future_horizon_net_return"], 0.0)
        self.assertAlmostEqual(outcomes.iloc[1]["future_horizon_net_return"], 0.0007)

    def test_metrics_report_standard_sharpe_separately_from_cagr_vol(self) -> None:
        index = pd.date_range("2025-01-01", periods=6, freq="D")
        returns = pd.Series([0.01, -0.004, 0.006, -0.002, 0.008, 0.001], index=index)
        metrics = calculate_metrics(
            returns,
            pd.Series(0.0, index=index),
            pd.Series(1.0, index=index),
            annual_periods=252,
        )
        expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

        self.assertAlmostEqual(metrics["夏普比率"], expected_sharpe)
        self.assertAlmostEqual(metrics["标准夏普比率"], expected_sharpe)
        self.assertNotAlmostEqual(metrics["CAGR波动比"], expected_sharpe)

    def test_prediction_report_includes_robust_classification_and_net_edge_metrics(self) -> None:
        config = BacktestConfig()
        index = pd.date_range("2025-01-01", periods=12, freq="D")
        target = pd.Series([-1.0, 0.0, 1.0] * 4, index=index)
        prediction = pd.Series([-1.0, 0.0, 1.0, -1.0, 1.0, 1.0] * 2, index=index)
        probabilities = pd.DataFrame(
            {
                "prob_down": np.where(prediction < 0, 0.70, 0.15),
                "prob_flat": np.where(prediction == 0, 0.70, 0.15),
                "prob_up": np.where(prediction > 0, 0.70, 0.15),
            },
            index=index,
        )
        future_return = target * 0.01
        frame = probabilities.assign(
            target_direction=target,
            future_horizon_return=future_return,
            future_horizon_net_return=target * 0.009,
            future_horizon_standardized_net_return=target * 0.5,
            xgboost_predicted_direction=prediction,
            calibrated_predicted_direction=prediction,
            calibrated_prob_edge=probabilities["prob_up"] - probabilities["prob_down"],
            xgboost_signal_direction=1.0,
            raw_signal=prediction,
        )

        metrics = calculate_prediction_metrics_for_segment("测试集", frame, config)

        self.assertIsNotNone(metrics)
        self.assertIn("校准后三分类MCC", metrics)
        self.assertIn("校准后三分类BalancedAccuracy", metrics)
        self.assertIn("概率校准误差ECE", metrics)
        self.assertIn("概率差与未来净收益Spearman", metrics)
        self.assertIn("方向准确率Wilson下限", metrics)

    def test_block_bootstrap_statistics_are_reproducible(self) -> None:
        config = BacktestConfig()
        config.statistical_bootstrap_samples = 50
        config.statistical_bootstrap_block_size = 5
        returns = pd.Series(np.sin(np.arange(100)) * 0.001 + 0.0001)

        first = calculate_block_bootstrap_statistics(returns, 252, config)
        second = calculate_block_bootstrap_statistics(returns, 252, config)

        self.assertEqual(first, second)
        self.assertIn("标准夏普置信下限", first)
        self.assertIn("平均收益为零双侧P值", first)

    def test_daily_cli_explicit_training_window_has_priority(self) -> None:
        from cli import build_parser, create_config

        args = build_parser().parse_args(
            [
                "composite",
                "--frequency",
                "1d",
                "--train-window",
                "360",
                "--min-train-samples",
                "90",
                "--retrain-every",
                "10",
            ]
        )
        config = create_config(args)

        self.assertEqual(config.xgboost_train_window, 360)
        self.assertEqual(config.xgboost_min_train_samples, 90)
        self.assertEqual(config.xgboost_retrain_every, 10)

    def test_config_validation_isolated_by_pipeline(self) -> None:
        config = BacktestConfig()
        config.xgboost_train_window = 0

        validate_backtest_config(config, "single")
        config.trading_signal_mode = "vote"
        validate_backtest_config(config, "signal")
        config.trading_signal_mode = "model"
        with self.assertRaisesRegex(ValueError, "xgboost_train_window"):
            validate_backtest_config(config, "signal")
        with self.assertRaisesRegex(ValueError, "xgboost_train_window"):
            validate_backtest_config(config, "composite")


if __name__ == "__main__":
    unittest.main()

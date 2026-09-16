from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from composite_factor_backtest import (
    audit_active_library_oos_cutoff,
    build_factor_signal_features,
    build_historical_edge_ensemble,
    build_historical_probability_ensemble,
    build_two_stage_targets,
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
    predict_fast_two_stage_net_return,
    resolve_multi_window_train_windows,
    run_composite_backtest,
    train_composite_classifier,
    train_fast_two_stage_net_return_model,
    two_stage_predictions_to_probabilities,
    uses_fast_two_stage_model,
    write_composite_artifact_manifest,
)
from config import (
    BacktestConfig,
    LIQUID_STOCK_INDEX_FUTURES,
    resolve_symbol_universe,
    validate_backtest_config,
)
from framework.factor_library import (
    build_factor_library,
    build_compact_rejected_library,
    build_selection_metric,
    conservative_pair,
    get_final_test_performance_reject_reason,
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
    normalize_intraday_data,
    resolve_related_symbols,
)
from framework.factors import (
    EXPANDED_FACTOR_END_INDEX,
    EXPANDED_FACTOR_START_INDEX,
    FAMILY_EXPANSION_END_INDEX,
    FAMILY_EXPANSION_START_INDEX,
    FIFTH_FAMILY_EXPANSION_END_INDEX,
    FIFTH_FAMILY_EXPANSION_START_INDEX,
    FOURTH_FAMILY_EXPANSION_END_INDEX,
    FOURTH_FAMILY_EXPANSION_START_INDEX,
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
    build_symbol_config,
    collect_multi_symbol_manifest_outputs,
    pipeline_state_matches,
    read_single_factor_summary_for_pruning,
    save_multi_symbol_portfolio,
    save_multi_symbol_model_report,
    save_pipeline_state,
    should_skip_composite_pipeline,
    summarize_active_library,
)
from pooled_model_backtest import (
    apply_pooled_symbol_residual_correction,
    build_grouped_symbol_map,
    build_symbol_signal_from_predictions,
)
from framework.model_calibration import (
    rolling_temperature_calibrate_binary,
    rolling_temperature_calibrate_multiclass,
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
    _evaluate_single_factor_without_plot,
    build_single_factor_walk_forward_folds,
    calculate_metrics,
    cleanup_removed_active_factor_plots,
    evaluate_single_factor_walk_forward,
    get_active_factor_plot_names,
    get_expensive_diagnostic_skip_reason,
    get_reusable_single_factor_plot,
    prefilter_correlated_single_factors,
    prebuild_and_prefilter_factor_names,
    run_backtest,
    run_single_factor_backtests,
    split_train_validation_test_index,
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
from framework.factor_builders.family_expansion4 import (
    FOURTH_FAMILY_COUNT,
    FOURTH_FAMILY_PREFIXES,
    FOURTH_FAMILY_TOTAL_COUNT,
    get_fourth_family_expansion_names,
)
from framework.factor_builders.family_expansion5 import (
    FIFTH_FAMILY_COUNT,
    FIFTH_FAMILY_PREFIXES,
    FIFTH_FAMILY_TOTAL_COUNT,
    get_fifth_family_expansion_names,
)
from trading_signal import (
    get_factor_weight_series,
    get_live_model_predict_index,
    load_cached_factor_inputs,
    load_live_factor_inputs,
)


class FrameworkRegressionTests(unittest.TestCase):

    def test_stock_index_futures_are_available_as_independent_and_default_universes(self) -> None:
        config = BacktestConfig()

        self.assertEqual(
            resolve_symbol_universe("liquid_stock_index"),
            LIQUID_STOCK_INDEX_FUTURES,
        )
        self.assertTrue(set(LIQUID_STOCK_INDEX_FUTURES).issubset(config.symbols))
        self.assertTrue(
            set(LIQUID_STOCK_INDEX_FUTURES).issubset(
                resolve_symbol_universe("liquid_futures")
            )
        )
        self.assertTrue(
            all(config.multi_symbol_group_map[symbol] == "股指期货" for symbol in LIQUID_STOCK_INDEX_FUTURES)
        )

    def test_stock_index_futures_use_stock_index_related_symbols(self) -> None:
        config = BacktestConfig()
        config.symbol = "IF.CFE"

        self.assertEqual(
            resolve_related_symbols(config),
            ["000300.SH", "IH.CFE", "IC.CFE", "IM.CFE"],
        )

        config.symbol = "C.DCE"
        self.assertEqual(resolve_related_symbols(config), config.related_symbols)

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

    def test_factor_library_final_test_gate_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_min_predictive_score = None
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            summary = pd.DataFrame(
                {
                    "因子": ["test_pass", "test_fail"],
                    "训练夏普比率": [1.5, 1.5],
                    "验证夏普比率": [1.2, 1.2],
                    "训练累计收益": [0.10, 0.10],
                    "验证累计收益": [0.04, 0.04],
                    "训练交易次数": [40, 40],
                    "验证交易次数": [15, 15],
                    "测试夏普比率": [1.1, -0.5],
                    "测试累计收益": [0.03, -0.02],
                    "测试交易次数": [12, 12],
                }
            )
            factors = pd.DataFrame(
                {
                    "test_pass": np.arange(40, dtype="float64"),
                    "test_fail": np.arange(40, dtype="float64") * 2,
                }
            )

            active_without_gate, _, _ = build_factor_library(summary, factors, config)
            self.assertEqual(set(active_without_gate["因子"]), {"test_pass", "test_fail"})

            config.factor_library_require_test_performance = True
            active_with_gate, library_all, _ = build_factor_library(summary, factors, config)
            indexed = library_all.set_index("因子")

            self.assertEqual(active_with_gate["因子"].tolist(), ["test_pass"])
            self.assertTrue(bool(indexed.loc["test_pass", "最终测试入库表现通过"]))
            self.assertFalse(bool(indexed.loc["test_fail", "最终测试入库表现通过"]))
            self.assertEqual(indexed.loc["test_fail", "拒绝原因"], "low_test_sharpe")

    def test_final_test_gate_reuses_enabled_selection_thresholds(self) -> None:
        config = BacktestConfig()
        config.factor_library_require_test_performance = True
        config.factor_library_min_selection_win_rate = 0.50
        config.factor_library_min_selection_trades = 10
        config.factor_library_min_selection_signal_coverage = 0.20
        config.factor_library_min_selection_rank_ic = 0.01
        config.factor_library_min_selection_monotonicity = 0.50
        config.factor_library_max_selection_drawdown = -0.20
        passing = pd.Series(
            {
                "测试夏普比率": 1.2,
                "测试累计收益": 0.05,
                "测试胜率": 0.55,
                "测试交易次数": 12,
                "测试信号覆盖率": 0.30,
                "测试RankIC": 0.03,
                "测试分组单调性": 0.70,
                "测试最大回撤": -0.10,
            }
        )

        self.assertEqual(
            get_final_test_performance_reject_reason(passing, config),
            "",
        )
        failing_cases = {
            "测试胜率": (0.50, "low_test_win_rate"),
            "测试交易次数": (9, "low_test_trade_count"),
            "测试信号覆盖率": (0.10, "low_test_signal_coverage"),
            "测试RankIC": (0.0, "low_test_rank_ic"),
            "测试分组单调性": (0.40, "low_test_monotonicity"),
            "测试最大回撤": (-0.30, "high_test_drawdown"),
        }
        for column, (value, expected_reason) in failing_cases.items():
            with self.subTest(column=column):
                candidate = passing.copy()
                candidate[column] = value
                self.assertEqual(
                    get_final_test_performance_reject_reason(candidate, config),
                    expected_reason,
                )

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

    def test_automatic_active_rotation_deletes_only_removed_factor_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            plot_dir = get_research_output_dir(config, "single_factor")
            plot_dir.mkdir(parents=True, exist_ok=True)
            removed_plot = plot_dir / "1_factor_a_report.png"
            retained_plot = plot_dir / "2_factor_b_report.png"
            removed_plot.write_bytes(b"removed")
            retained_plot.write_bytes(b"retained")
            history_dir = Path(temp_dir) / "runs" / "old_single"
            history_dir.mkdir(parents=True, exist_ok=True)
            history_plot = history_dir / removed_plot.name
            history_plot.write_bytes(b"history")

            previous_active = pd.DataFrame(
                {
                    "因子编号": [1, 2],
                    "因子": ["factor_a", "factor_b"],
                    "因子标签": ["1_factor_a", "2_factor_b"],
                    "图片文件": [str(removed_plot), str(retained_plot)],
                }
            )
            current_active = previous_active.iloc[[1]].copy()
            library_all = previous_active.copy()

            removed_names, deleted_paths = cleanup_removed_active_factor_plots(
                previous_active,
                current_active,
                library_all,
                config,
            )

            self.assertEqual(removed_names, ["factor_a"])
            self.assertEqual(deleted_paths, [removed_plot.resolve()])
            self.assertFalse(removed_plot.exists())
            self.assertTrue(retained_plot.exists())
            self.assertTrue(history_plot.exists())
            self.assertEqual(
                library_all.loc[library_all["因子"].eq("factor_a"), "图片文件"].iloc[0],
                "",
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
        config.single_factor_scope = "range"
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
            config.single_factor_scope = "range"
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

    def test_active_scope_forces_all_factor_plots_to_be_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.single_factor_scope = "active"
            config.single_factor_plot_all_active = False
            config.single_factor_plot_top_n = 1
            config.single_factor_reuse_existing_plots = True
            active = pd.DataFrame(
                {"因子": ["factor_a", "factor_b", "factor_c"]}
            )
            output_dir = Path(temp_dir)
            plot_path = output_dir / "1_factor_a_report.png"
            plot_path.write_bytes(b"existing plot")

            self.assertEqual(
                get_active_factor_plot_names(active, config),
                ["factor_a", "factor_b", "factor_c"],
            )
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
            config.composite_factor_pool_scope = "active"
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

    def test_composite_protected_pool_freezes_only_protected_active_factors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.bar_frequency = "30min"
            config.composite_factor_pool_scope = "protected"
            library_dir = get_factor_library_dir(config)
            run_dir = Path(temp_dir) / "runs" / "protected_composite"
            output_dir = get_research_output_dir(config, "composite_factor")
            library_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "因子": ["factor_a", "factor_b", "factor_c"],
                    "是否手工保护": [False, True, "是"],
                    "训练截止": ["2025-01-01"] * 3,
                    "验证截止": ["2025-02-01"] * 3,
                }
            ).to_csv(library_dir / "active_factors.csv", index=False)

            runtime_config, snapshot_path = freeze_active_factor_library_for_run(
                config,
                output_dir,
                run_dir,
            )

            self.assertEqual(load_active_factor_names(runtime_config), ["factor_b", "factor_c"])
            snapshot = pd.read_csv(snapshot_path)
            self.assertEqual(snapshot["因子"].tolist(), ["factor_b", "factor_c"])
            manifest = json.loads(
                (run_dir / "active_factors_snapshot_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["factor_pool_scope"], "protected")
            self.assertEqual(manifest["factor_count"], 2)

    def test_composite_protected_pool_stops_when_no_factor_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.composite_factor_pool_scope = "protected"
            library_dir = get_factor_library_dir(config)
            library_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"因子": ["factor_a"], "是否手工保护": [False]}
            ).to_csv(library_dir / "active_factors.csv", index=False)

            with self.assertRaisesRegex(ValueError, "没有手工保护因子"):
                load_active_factor_names(config)

    def test_composite_factor_pool_scope_is_validated(self) -> None:
        config = BacktestConfig()
        config.composite_factor_pool_scope = "unknown"
        with self.assertRaisesRegex(ValueError, "composite_factor_pool_scope"):
            validate_backtest_config(config, "composite")

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
        self.assertEqual(THIRD_FAMILY_EXPANSION_END_INDEX, 600_000)

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

    def test_fourth_five_family_blocks_have_exact_counts_and_stable_ids(self) -> None:
        names = get_fourth_family_expansion_names()
        old_names = get_third_family_expansion_names()
        self.assertEqual(FOURTH_FAMILY_COUNT, 20_000)
        self.assertEqual(FOURTH_FAMILY_TOTAL_COUNT, 100_000)
        self.assertEqual(len(names), FOURTH_FAMILY_TOTAL_COUNT)
        self.assertEqual(len(set(names)), FOURTH_FAMILY_TOTAL_COUNT)
        self.assertFalse(set(names).intersection(old_names))
        for block_index, prefix in enumerate(FOURTH_FAMILY_PREFIXES):
            self.assertTrue(names[block_index * FOURTH_FAMILY_COUNT].startswith(f"{prefix}_"))
            self.assertTrue(
                names[(block_index + 1) * FOURTH_FAMILY_COUNT - 1].startswith(f"{prefix}_")
            )
        self.assertEqual(
            FOURTH_FAMILY_EXPANSION_START_INDEX,
            THIRD_FAMILY_EXPANSION_END_INDEX + 1,
        )
        self.assertEqual(
            FOURTH_FAMILY_EXPANSION_END_INDEX - FOURTH_FAMILY_EXPANSION_START_INDEX + 1,
            FOURTH_FAMILY_TOTAL_COUNT,
        )
        self.assertEqual(FOURTH_FAMILY_EXPANSION_END_INDEX, 700_000)

    def test_fifth_five_family_blocks_have_exact_counts_and_stable_ids(self) -> None:
        names = get_fifth_family_expansion_names()
        old_names = get_fourth_family_expansion_names()
        self.assertEqual(FIFTH_FAMILY_COUNT, 20_000)
        self.assertEqual(FIFTH_FAMILY_TOTAL_COUNT, 100_000)
        self.assertEqual(len(names), FIFTH_FAMILY_TOTAL_COUNT)
        self.assertEqual(len(set(names)), FIFTH_FAMILY_TOTAL_COUNT)
        self.assertFalse(set(names).intersection(old_names))
        for block_index, prefix in enumerate(FIFTH_FAMILY_PREFIXES):
            self.assertTrue(names[block_index * FIFTH_FAMILY_COUNT].startswith(f"{prefix}_"))
            self.assertTrue(
                names[(block_index + 1) * FIFTH_FAMILY_COUNT - 1].startswith(f"{prefix}_")
            )
        self.assertEqual(
            FIFTH_FAMILY_EXPANSION_START_INDEX,
            FOURTH_FAMILY_EXPANSION_END_INDEX + 1,
        )
        self.assertEqual(
            FIFTH_FAMILY_EXPANSION_END_INDEX - FIFTH_FAMILY_EXPANSION_START_INDEX + 1,
            FIFTH_FAMILY_TOTAL_COUNT,
        )
        self.assertEqual(TOTAL_FACTOR_END_INDEX, 800_000)
        expected_families = [
            "cross_asset", "non_cross_complex", "expanded", "parametric", "calendar"
        ]
        for block, expected_family in enumerate(expected_families):
            self.assertEqual(
                classify_factor(names[block * FIFTH_FAMILY_COUNT])["因子家族"],
                expected_family,
            )

    def test_fourth_five_family_expansions_build_strictly_on_demand(self) -> None:
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
            FOURTH_FAMILY_EXPANSION_START_INDEX + block * FOURTH_FAMILY_COUNT
            for block in range(5)
        ]
        requested: list[str] = []
        for expected_prefix, start in zip(FOURTH_FAMILY_PREFIXES, starts):
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

    def test_fifth_five_family_expansions_build_strictly_on_demand(self) -> None:
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
            FIFTH_FAMILY_EXPANSION_START_INDEX + block * FIFTH_FAMILY_COUNT
            for block in range(5)
        ]
        requested: list[str] = []
        for expected_prefix, start in zip(FIFTH_FAMILY_PREFIXES, starts):
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
            config.composite_factor_pool_scope = "active"
            config.use_frozen_active_library = True
            config.frozen_active_library_path = str(active_path)
            with self.assertRaisesRegex(ValueError, "不会回退计算全部因子"):
                load_active_factor_names(config)

    def test_composite_empty_active_stops_before_loading_market_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.composite_factor_pool_scope = "active"
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

    def test_active_scope_resolves_current_symbol_frequency_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.symbol = "IF.CFE"
            config.bar_frequency = "1d"
            config.single_factor_scope = "active"
            library_dir = get_factor_library_dir(config)
            pd.DataFrame(
                {
                    "因子": ["momentum", "reversal"],
                    "因子编号": [11, 29],
                }
            ).to_csv(
                library_dir / "active_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )

            requested, factor_id_map = resolve_single_factor_requested_factors(
                pd.DataFrame(),
                config,
            )

            self.assertEqual(requested, ["momentum", "reversal"])
            self.assertEqual(factor_id_map, {"momentum": 11, "reversal": 29})
            self.assertIn("IF_CFE", str(library_dir))
            self.assertEqual(library_dir.name, "1d")

    def test_active_scope_missing_library_fails_without_full_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.symbol = "IF.CFE"
            config.bar_frequency = "1d"
            config.single_factor_scope = "active"

            with self.assertRaisesRegex(ValueError, "active_factors.csv"):
                resolve_single_factor_requested_factors(pd.DataFrame(), config)

    def test_active_scope_builds_every_active_factor_without_corr_prefilter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.single_factor_scope = "active"
            config.single_factor_enable_corr_prefilter = True
            config.single_factor_prebuild_corr_prefilter = True
            library_dir = get_factor_library_dir(config)
            pd.DataFrame(
                {
                    "因子": ["momentum", "reversal"],
                    "因子编号": [1, 2],
                }
            ).to_csv(
                library_dir / "active_factors.csv",
                index=False,
                encoding="utf-8-sig",
            )
            index = pd.date_range("2025-01-01", periods=80, freq="D")
            close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
            data = pd.DataFrame(
                {
                    "open": close.shift(1).fillna(close.iloc[0]),
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                },
                index=index,
            )
            requested, factor_id_map = resolve_single_factor_requested_factors(data, config)

            kept = prebuild_and_prefilter_factor_names(
                data,
                config,
                requested or [],
                factor_id_map,
                Path(temp_dir),
            )
            factors = build_single_factor_matrix(
                data,
                config,
                requested_factors_override=kept,
                catalog_factor_id_map_override=factor_id_map,
                progress_factor_columns=requested,
            )

            self.assertEqual(kept, ["momentum", "reversal"])
            self.assertEqual(factors.attrs["backtest_factor_columns"], kept)
            self.assertEqual(factors.attrs["factor_id_map"], factor_id_map)
            self.assertEqual(set(factors.columns), set(kept))

    def test_active_scope_is_accepted_by_config_and_cli(self) -> None:
        from cli import build_parser, create_config

        args = build_parser().parse_args(
            ["single", "--symbol", "IF.CFE", "--frequency", "1d", "--scope", "active"]
        )
        config = create_config(args)

        self.assertEqual(config.single_factor_scope, "active")
        validate_backtest_config(config, "single")

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
            minute.symbol = "C.DCE"
            minute.bar_frequency = "30min"

            daily = BacktestConfig()
            daily.output_dir = temp_dir
            daily.data_cache_dir = str(Path(temp_dir) / "data")
            daily.symbol = "C.DCE"
            daily.bar_frequency = "1d"

            self.assertEqual(
                get_factor_library_dir(minute),
                Path(temp_dir)
                / "by_symbol"
                / "symbols"
                / "C_DCE"
                / "factor_library"
                / "30min",
            )
            self.assertEqual(
                get_factor_library_dir(daily),
                Path(temp_dir)
                / "by_symbol"
                / "symbols"
                / "C_DCE"
                / "factor_library"
                / "1d",
            )
            self.assertEqual(
                get_data_cache_path(daily),
                Path(temp_dir) / "data" / "market" / "1d" / "C_DCE_1d_data.csv",
            )
            self.assertFalse(
                any("30min" in str(path) for path in get_local_data_candidates(daily))
            )

    def test_single_symbol_outputs_are_isolated_and_match_multi_symbol_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            c_config = BacktestConfig()
            c_config.output_dir = temp_dir
            c_config.symbol = "C.DCE"
            m_config = BacktestConfig()
            m_config.output_dir = temp_dir
            m_config.symbol = "M.DCE"

            c_library = get_factor_library_dir(c_config)
            m_library = get_factor_library_dir(m_config)
            self.assertNotEqual(c_library, m_library)
            self.assertIn("C_DCE", c_library.parts)
            self.assertIn("M_DCE", m_library.parts)

            multi_m_config = build_symbol_config(c_config, "M.DCE")
            self.assertEqual(get_factor_library_dir(multi_m_config), m_library)
            self.assertEqual(
                get_research_output_dir(multi_m_config, "single_factor"),
                get_research_output_dir(m_config, "single_factor"),
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

    def test_probability_calibration_only_uses_matured_labels(self) -> None:
        config = BacktestConfig()
        config.composite_probability_calibration_enabled = True
        config.composite_probability_calibration_window = 24
        config.composite_probability_calibration_min_history = 8
        config.composite_probability_calibration_retrain_every = 1
        config.composite_probability_temperature_grid = [0.5, 1.0, 2.0]
        index = pd.date_range("2025-01-01", periods=60, freq="D")
        target = pd.Series(np.where(np.arange(60) % 2 == 0, 1.0, -1.0), index=index)
        probabilities = pd.DataFrame(
            {
                "prob_down": np.where(target < 0, 0.70, 0.10),
                "prob_flat": 0.20,
                "prob_up": np.where(target > 0, 0.70, 0.10),
            },
            index=index,
        )
        trade_probability = pd.Series(0.75, index=index)
        horizon = 3
        timestamp = index[40]

        calibrated, temperature, _ = rolling_temperature_calibrate_multiclass(
            probabilities,
            target,
            config,
            horizon=horizon,
        )
        calibrated_trade, trade_temperature, _ = rolling_temperature_calibrate_binary(
            trade_probability,
            target.abs(),
            config,
            horizon=horizon,
        )
        changed_target = target.copy()
        changed_target.iloc[index.get_loc(timestamp) - horizon + 1 :] *= -1
        changed, changed_temperature, _ = rolling_temperature_calibrate_multiclass(
            probabilities,
            changed_target,
            config,
            horizon=horizon,
        )
        changed_trade, changed_trade_temperature, _ = rolling_temperature_calibrate_binary(
            trade_probability,
            changed_target.abs(),
            config,
            horizon=horizon,
        )

        np.testing.assert_allclose(
            calibrated.loc[timestamp],
            changed.loc[timestamp],
            atol=1e-12,
        )
        self.assertEqual(temperature.loc[timestamp], changed_temperature.loc[timestamp])
        self.assertEqual(calibrated_trade.loc[timestamp], changed_trade.loc[timestamp])
        self.assertEqual(
            trade_temperature.loc[timestamp],
            changed_trade_temperature.loc[timestamp],
        )

    def test_edge_ensemble_uses_unified_model_edge_score(self) -> None:
        config = BacktestConfig()
        config.xgboost_target_horizon = 2
        config.composite_ensemble_weight_window = 20
        config.composite_ensemble_min_history = 5
        config.composite_ensemble_min_directional_accuracy = 0.0
        config.composite_ensemble_min_edge_return_corr = -1.0
        config.composite_ensemble_min_abs_edge = 0.0
        config.xgboost_two_stage_min_trade_probability = 0.0
        index = pd.date_range("2025-01-01", periods=40, freq="D")
        target = pd.Series(np.where(np.arange(40) % 2 == 0, 1.0, -1.0), index=index)
        future_return = target * 0.01
        data = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            index=index,
        )

        def make_signal(scale: float) -> pd.DataFrame:
            # 概率投影故意给出相反方向，验证融合读取的是 model_edge_score。
            return pd.DataFrame(
                {
                    "target_direction": target,
                    "future_horizon_return": future_return,
                    "future_horizon_standardized_net_return": target * 0.4,
                    "model_edge_score": target * scale,
                    "calibrated_trade_probability": 0.8,
                    "prob_down": np.where(target > 0, 0.7, 0.1),
                    "prob_flat": 0.2,
                    "prob_up": np.where(target < 0, 0.7, 0.1),
                    "xgboost_signal_direction": 1.0,
                },
                index=index,
            )

        ensemble, diagnostics = build_historical_edge_ensemble(
            {"model_a": make_signal(0.3), "model_b": make_signal(0.2)},
            data,
            config,
        )

        valid = ensemble["model_edge_score"].dropna()
        self.assertFalse(valid.empty)
        self.assertTrue((np.sign(valid) == np.sign(target.loc[valid.index])).all())
        self.assertEqual(ensemble["probability_semantics"].dropna().iloc[-1], "diagnostic_projection")
        self.assertIn("历史边际收益相关性", diagnostics.columns)
        np.testing.assert_allclose(
            ensemble.loc[valid.index, ["prob_down", "prob_flat", "prob_up"]].sum(axis=1),
            1.0,
            atol=1e-10,
        )

    def test_daily_multi_window_defaults_include_primary_window(self) -> None:
        config = BacktestConfig()
        config.bar_frequency = "1d"
        resolved = apply_frequency_runtime_defaults(config)

        self.assertEqual(
            resolve_multi_window_train_windows(resolved),
            [504, 252, 1008],
        )

    def test_pooled_symbol_residual_only_uses_matured_symbol_history(self) -> None:
        config = BacktestConfig()
        config.pooled_model_hierarchy_mode = "global_symbol_residual"
        config.pooled_symbol_residual_enabled = True
        config.pooled_symbol_residual_window = 20
        config.pooled_symbol_residual_min_history = 5
        config.pooled_symbol_residual_prior_count = 0.0
        timestamps = pd.date_range("2025-01-01", periods=50, freq="D")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": "A.DCE",
                "label_available_time": timestamps + pd.Timedelta(days=2),
                "future_horizon_standardized_net_return": 0.4,
                "global_model_edge_score": 0.0,
            }
        )
        current_time = timestamps[30]

        original = apply_pooled_symbol_residual_correction(frame, config)
        changed_frame = frame.copy()
        changed_frame.loc[
            changed_frame["label_available_time"] > current_time,
            "future_horizon_standardized_net_return",
        ] = -10.0
        changed = apply_pooled_symbol_residual_correction(changed_frame, config)
        original_value = original.loc[original["timestamp"] == current_time].iloc[0]
        changed_value = changed.loc[changed["timestamp"] == current_time].iloc[0]

        self.assertGreater(original_value["symbol_residual_adjustment"], 0.0)
        self.assertAlmostEqual(
            original_value["symbol_residual_adjustment"],
            changed_value["symbol_residual_adjustment"],
        )
        self.assertEqual(
            build_grouped_symbol_map(["A.DCE", "CU.SHF"], config),
            {"全市场全局模型": ["A.DCE", "CU.SHF"]},
        )

    def test_pooled_residual_handles_symbol_first_seen_between_refreshes(self) -> None:
        config = BacktestConfig()
        config.pooled_model_hierarchy_mode = "global_symbol_residual"
        config.pooled_symbol_residual_enabled = True
        config.pooled_symbol_residual_window = 20
        config.pooled_symbol_residual_min_history = 2
        config.pooled_symbol_residual_retrain_every = 5
        timestamps = pd.date_range("2025-01-01", periods=8, freq="D")
        frame = pd.DataFrame(
            {
                "timestamp": list(timestamps) + list(timestamps[3:]),
                "symbol": ["A.DCE"] * len(timestamps) + ["B.DCE"] * 5,
                "label_available_time": list(timestamps + pd.Timedelta(days=1))
                + list(timestamps[3:] + pd.Timedelta(days=1)),
                "future_horizon_standardized_net_return": 0.2,
                "global_model_edge_score": 0.0,
            }
        )

        corrected = apply_pooled_symbol_residual_correction(frame, config)

        first_b = corrected[corrected["symbol"] == "B.DCE"].iloc[0]
        self.assertEqual(first_b["symbol_residual_history_count"], 0)
        self.assertEqual(first_b["symbol_residual_adjustment"], 0.0)

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

    def test_prebuild_corr_prefilter_uses_short_research_sample_before_full_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.single_factor_scope = "range"
            config.single_factor_enable_corr_prefilter = True
            config.single_factor_prebuild_corr_prefilter = True
            config.single_factor_corr_prefilter_threshold = 0.95
            config.single_factor_corr_prefilter_sample_rows = 20
            config.single_factor_prebuild_warmup_rows = 20
            config.single_factor_prebuild_batch_size = 1
            config.auto_select_train_ratio = 0.60
            config.auto_select_validation_ratio = 0.20
            index = pd.date_range("2025-01-01", periods=100, freq="D")
            data = pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": np.linspace(100.0, 110.0, len(index)),
                    "volume": 1000.0,
                },
                index=index,
            )
            observed: dict[str, object] = {}

            def fake_build(preview_data, _config, **kwargs):
                observed["rows"] = len(preview_data)
                observed["end"] = preview_data.index[-1]
                observed.setdefault("requested_batches", []).append(
                    kwargs["requested_factors_override"]
                )
                values = np.arange(len(preview_data), dtype="float64")
                return pd.DataFrame(
                    {"factor_a": values, "factor_b": values},
                    index=preview_data.index,
                )

            with patch(
                "single_factor_backtest.build_single_factor_matrix",
                side_effect=fake_build,
            ):
                kept = prebuild_and_prefilter_factor_names(
                    data,
                    config,
                    ["factor_a", "factor_b"],
                    {"factor_a": 1, "factor_b": 2},
                    Path(temp_dir),
                )

            self.assertEqual(kept, ["factor_a"])
            self.assertEqual(observed["rows"], 70)
            self.assertEqual(
                observed["requested_batches"],
                [["factor_a"], ["factor_b"]],
            )
            self.assertLess(observed["end"], index[80])

    def test_parallel_single_factor_worker_matches_serial_result(self) -> None:
        config = BacktestConfig()
        config.single_factor_defer_expensive_diagnostics = False
        config.single_factor_walk_forward_enabled = False
        config.statistical_enable_block_bootstrap = False
        config.qcut_window = 30
        config.qcut_min_periods = 15
        index = pd.date_range("2024-01-01", periods=320, freq="D")
        close = pd.Series(100.0 + np.cumsum(np.sin(np.arange(320) / 7.0) + 0.1), index=index)
        data = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0 + np.arange(320),
            },
            index=index,
        )
        split_time, validation_end = split_train_validation_test_index(
            data.index,
            config.auto_select_train_ratio,
            config.auto_select_validation_ratio,
        )
        train = data.loc[data.index < split_time]
        validation = data.loc[(data.index >= split_time) & (data.index < validation_end)]
        test = data.loc[data.index >= validation_end]
        score_a = pd.Series(np.sin(np.arange(320) / 5.0), index=index)
        score_b = pd.Series(np.cos(np.arange(320) / 9.0), index=index)

        def evaluate(name: str, factor_id: int, score: pd.Series):
            return _evaluate_single_factor_without_plot(
                name,
                f"{factor_id}_{name}",
                factor_id,
                score,
                data,
                train,
                validation,
                test,
                config,
                split_time,
                validation_end,
            )

        serial_row, serial_qcut, serial_skipped, serial_walk_forward = evaluate(
            "factor_a",
            1,
            score_a,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(evaluate, "factor_a", 1, score_a),
                executor.submit(evaluate, "factor_b", 2, score_b),
            ]
            (
                parallel_row,
                parallel_qcut,
                parallel_skipped,
                parallel_walk_forward,
            ) = futures[0].result()
            futures[1].result()

        self.assertEqual(serial_skipped, parallel_skipped)
        self.assertEqual(serial_qcut, parallel_qcut)
        self.assertEqual(serial_walk_forward, parallel_walk_forward)
        self.assertEqual(serial_row.keys(), parallel_row.keys())
        for key in serial_row:
            left, right = serial_row[key], parallel_row[key]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertEqual(left, right, key)

    def test_walk_forward_folds_do_not_touch_final_test(self) -> None:
        config = BacktestConfig()
        config.single_factor_walk_forward_folds = 4
        config.single_factor_walk_forward_initial_train_ratio = 0.50
        config.single_factor_walk_forward_embargo_bars = 1
        config.single_factor_walk_forward_min_validation_bars = 5
        index = pd.date_range("2024-01-01", periods=100, freq="D")
        final_test_start = index[85]

        folds = build_single_factor_walk_forward_folds(
            index,
            final_test_start,
            config,
        )

        self.assertEqual(len(folds), 4)
        validation_timestamps: list[pd.Timestamp] = []
        for fold in folds:
            train_index = fold["train_index"]
            validation_index = fold["validation_index"]
            self.assertLess(train_index[-1], validation_index[0])
            self.assertLess(validation_index[-1], final_test_start)
            validation_timestamps.extend(validation_index.tolist())
        self.assertEqual(len(validation_timestamps), len(set(validation_timestamps)))

    def test_walk_forward_evaluation_reports_non_overlapping_oos_metrics(self) -> None:
        config = BacktestConfig()
        config.single_factor_walk_forward_folds = 4
        config.single_factor_walk_forward_min_validation_bars = 10
        config.factor_library_min_walk_forward_folds = 3
        index = pd.date_range("2024-01-01", periods=220, freq="D")
        close = pd.Series(
            100.0 + np.cumsum(np.sin(np.arange(len(index)) / 6.0) + 0.05),
            index=index,
        )
        data = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
            },
            index=index,
        )
        score = pd.Series(np.sin(np.arange(len(index)) / 5.0), index=index)
        final_test_start = index[187]

        metrics, records = evaluate_single_factor_walk_forward(
            data,
            score,
            "factor_a",
            "1_factor_a",
            1,
            config,
            final_test_start,
        )

        self.assertEqual(metrics["WalkForward状态"], "已计算")
        self.assertEqual(metrics["WalkForward有效折数"], 4)
        self.assertEqual(len(records), 4)
        self.assertGreater(metrics["WalkForward样本K线数"], 0)
        self.assertTrue(
            all(pd.Timestamp(record["验证结束"]) < final_test_start for record in records)
        )

    def test_factor_ranking_prefers_walk_forward_over_fixed_validation(self) -> None:
        config = BacktestConfig()
        config.single_factor_keep_top_n = 2
        summary = pd.DataFrame(
            {
                "因子": ["fixed_period_winner", "walk_forward_winner"],
                "训练夏普比率": [3.0, 1.2],
                "验证夏普比率": [2.8, 1.1],
                "训练累计收益": [0.30, 0.12],
                "验证累计收益": [0.20, 0.08],
                "WalkForward状态": ["已计算", "已计算"],
                "WalkForward有效折数": [4, 4],
                "WalkForward夏普比率": [-0.5, 1.5],
                "WalkForward累计收益": [-0.03, 0.10],
                "WalkForward夏普中位数": [-0.4, 1.2],
                "WalkForward夏普最差值": [-1.0, 0.2],
                "WalkForward盈利折占比": [0.25, 0.75],
                "WalkForward方向一致率": [0.50, 0.75],
                "WalkForwardRankIC中位数": [-0.04, 0.06],
                "WalkForwardRankIC正向折占比": [0.25, 0.75],
                "WalkForward方向命中率": [0.45, 0.56],
            }
        )

        _, ranked = rank_single_factor_summary(summary, config)
        ranked = ranked.set_index("因子")

        self.assertEqual(ranked.sort_values("初筛科研综合评分").index[-1], "walk_forward_winner")
        self.assertEqual(ranked.loc["walk_forward_winner", "初筛样本"], "WalkForward样本外")
        self.assertAlmostEqual(ranked.loc["walk_forward_winner", "初筛夏普"], 1.5)
        self.assertAlmostEqual(ranked.loc["fixed_period_winner", "初筛夏普"], -0.5)

    def test_factor_library_rejects_unstable_walk_forward_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_storage_format = "pickle"
            config.factor_library_min_sharpe = 0.0
            config.factor_library_min_train_sharpe = 0.0
            config.factor_library_min_predictive_score = None
            config.factor_library_min_selection_trades = 0
            config.factor_library_min_train_trades = 0
            config.factor_library_min_walk_forward_trades = 0
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            summary = pd.DataFrame(
                {
                    "因子": ["unstable_factor"],
                    "训练夏普比率": [2.0],
                    "验证夏普比率": [1.5],
                    "训练累计收益": [0.20],
                    "验证累计收益": [0.10],
                    "训练交易次数": [40],
                    "验证交易次数": [20],
                    "WalkForward状态": ["已计算"],
                    "WalkForward有效折数": [4],
                    "WalkForward夏普比率": [1.2],
                    "WalkForward累计收益": [0.08],
                    "WalkForward夏普中位数": [0.5],
                    "WalkForward夏普最差值": [-0.7],
                    "WalkForward盈利折占比": [0.25],
                    "WalkForward方向一致率": [0.75],
                    "WalkForwardRankIC中位数": [0.04],
                    "WalkForwardRankIC正向折占比": [0.75],
                    "WalkForward方向命中率": [0.55],
                    "WalkForward胜率": [0.52],
                    "WalkForward交易次数": [40],
                    "WalkForward信号覆盖率": [0.40],
                    "WalkForward最大回撤": [-0.08],
                }
            )
            factors = pd.DataFrame(
                {"unstable_factor": np.arange(100, dtype="float64")}
            )

            active, master, _ = build_factor_library(summary, factors, config)

            self.assertTrue(active.empty)
            self.assertEqual(
                master.iloc[0]["拒绝原因"],
                "low_walk_forward_positive_fold_ratio",
            )

    def test_factor_library_retest_realigns_walk_forward_masks_after_history_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.factor_library_storage_format = "pickle"
            config.factor_library_min_sharpe = 0.0
            config.factor_library_min_train_sharpe = 0.0
            config.factor_library_min_predictive_score = None
            config.factor_library_min_selection_trades = 0
            config.factor_library_min_train_trades = 0
            config.factor_library_min_walk_forward_trades = 0
            config.factor_library_enable_family_quota = False
            config.factor_library_use_value_corr = False
            config.factor_library_use_signal_corr = False
            retest_summary = pd.DataFrame(
                {
                    "因子编号": [1],
                    "因子": ["factor_a"],
                    "训练夏普比率": [2.0],
                    "验证夏普比率": [1.5],
                    "训练累计收益": [0.20],
                    "验证累计收益": [0.10],
                    "训练交易次数": [40],
                    "验证交易次数": [20],
                    "WalkForward状态": ["已计算"],
                    "WalkForward有效折数": [4],
                    "WalkForward夏普比率": [1.2],
                    "WalkForward累计收益": [0.08],
                    "WalkForward夏普中位数": [0.8],
                    "WalkForward夏普最差值": [0.1],
                    "WalkForward盈利折占比": [0.75],
                    "WalkForward方向一致率": [0.75],
                    "WalkForwardRankIC中位数": [0.04],
                    "WalkForwardRankIC正向折占比": [0.75],
                    "WalkForward方向命中率": [0.55],
                    "WalkForward胜率": [0.52],
                    "WalkForward交易次数": [40],
                    "WalkForward信号覆盖率": [0.40],
                    "WalkForward最大回撤": [-0.08],
                }
            )
            historical = retest_summary.copy()
            historical["因子库状态"] = "active"
            historical["拒绝原因"] = ""
            save_factor_library(
                historical,
                historical,
                historical.iloc[0:0].copy(),
                config,
            )
            factors = pd.DataFrame(
                {"factor_a": np.arange(100, dtype="float64")}
            )

            active, master, _ = build_factor_library(
                retest_summary,
                factors,
                config,
            )

            self.assertEqual(active["因子"].tolist(), ["factor_a"])
            self.assertEqual(master.loc[0, "因子库状态"], "active")
            self.assertTrue(bool(master.loc[0, "WalkForward有效"]))

    def test_empty_market_data_fails_before_factor_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "行情数据为空"):
            normalize_intraday_data(pd.DataFrame())

    def test_invalid_stock_index_future_exchange_is_rejected(self) -> None:
        config = BacktestConfig()
        config.symbol = "IF.DCE"

        with self.assertRaisesRegex(ValueError, "IF.CFE"):
            validate_backtest_config(config, "single")

    def test_parallel_single_factor_pipeline_preserves_factor_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BacktestConfig()
            config.output_dir = temp_dir
            config.enable_experiment_run_dirs = False
            config.factor_library_storage_format = "pickle"
            config.single_factor_parallel_workers = 2
            config.single_factor_enable_corr_prefilter = False
            config.single_factor_plot_all = False
            config.single_factor_plot_all_active = False
            config.single_factor_plot_top_n = 0
            config.single_factor_defer_expensive_diagnostics = True
            config.factor_library_min_sharpe = 999.0
            config.factor_library_min_train_sharpe = 999.0
            config.factor_library_min_predictive_score = None
            config.statistical_enable_block_bootstrap = False
            index = pd.date_range("2024-01-01", periods=180, freq="D")
            close = pd.Series(100.0 + np.cumsum(np.sin(np.arange(180) / 8.0)), index=index)
            data = pd.DataFrame(
                {
                    "open": close.shift(1).fillna(close.iloc[0]),
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + np.arange(180),
                },
                index=index,
            )
            factors = pd.DataFrame(
                {
                    "factor_a": np.sin(np.arange(180) / 5.0),
                    "factor_b": np.cos(np.arange(180) / 7.0),
                },
                index=index,
            )
            factors.attrs["factor_id_map"] = {"factor_a": 1, "factor_b": 2}
            factors.attrs["factor_label_map"] = {
                "factor_a": "1_factor_a",
                "factor_b": "2_factor_b",
            }
            factors.attrs["backtest_factor_columns"] = ["factor_a", "factor_b"]
            factors.attrs["progress_factor_columns"] = ["factor_a", "factor_b"]

            run_single_factor_backtests(data, factors, config)

            summary = pd.read_csv(
                get_research_output_dir(config, "single_factor")
                / "single_factor_all_summary.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(set(summary["因子"]), {"factor_a", "factor_b"})
            self.assertEqual(
                summary.set_index("因子")["因子编号"].astype(int).to_dict(),
                {"factor_a": 1, "factor_b": 2},
            )
            self.assertTrue(summary["错误"].fillna("").eq("").all())

    def test_prefiltered_factor_ids_remain_available_for_progress_updates(self) -> None:
        config = BacktestConfig()
        config.single_factor_scope = "range"
        index = pd.date_range("2025-01-01", periods=30, freq="D")
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

        def fake_build(_data, _config, requested_factors=None):
            return pd.DataFrame(
                {name: np.arange(len(_data), dtype="float64") for name in requested_factors},
                index=_data.index,
            )

        with patch("framework.factors.build_factors", side_effect=fake_build):
            factors = build_single_factor_matrix(
                data,
                config,
                requested_factors_override=["factor_b"],
                catalog_factor_id_map_override={"factor_a": 10, "factor_b": 11},
                progress_factor_columns=["factor_a", "factor_b"],
                include_active_references=False,
            )

        self.assertEqual(factors.attrs["factor_id_map"], {"factor_b": 11})
        self.assertEqual(
            factors.attrs["progress_factor_id_map"],
            {"factor_a": 10, "factor_b": 11},
        )

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

    def test_fast_two_stage_is_limited_to_xgboost(self) -> None:
        config = BacktestConfig()
        config.xgboost_decision_mode = "two_stage_net_return"

        self.assertTrue(uses_fast_two_stage_model("xgboost", config))
        self.assertFalse(uses_fast_two_stage_model("random_forest", config))
        self.assertFalse(uses_fast_two_stage_model("logistic_regression", config))
        config.xgboost_decision_mode = "direction_classification"
        self.assertFalse(uses_fast_two_stage_model("xgboost", config))
        self.assertFalse(uses_fast_two_stage_model("extra_trees", config))

    def test_fast_two_stage_uses_reduced_regression_rounds(self) -> None:
        try:
            import sklearn  # noqa: F401
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("当前测试环境未安装 scikit-learn 或 xgboost")
        config = BacktestConfig()
        config.xgboost_n_estimators = 20
        config.xgboost_two_stage_regression_round_ratio = 0.50
        rng = np.random.default_rng(42)
        index = pd.RangeIndex(100)
        features = pd.DataFrame(
            {
                "feature_a": rng.normal(size=len(index)),
                "feature_b": rng.normal(size=len(index)),
            },
            index=index,
        )
        latent_return = features["feature_a"] * 0.6 - features["feature_b"] * 0.2
        direction = np.sign(latent_return).where(latent_return.abs() > 0.20, 0.0)
        targets = build_two_stage_targets(direction, latent_return, config)

        model = train_fast_two_stage_net_return_model(
            "xgboost",
            features,
            targets["trade_target"],
            targets["return_target"],
            features.columns.tolist(),
            config,
            sample_weight=pd.Series(1.0, index=index),
        )
        prediction = predict_fast_two_stage_net_return(
            "xgboost",
            model,
            features.iloc[-10:],
            features.columns.tolist(),
        )
        probability = two_stage_predictions_to_probabilities(prediction)

        self.assertEqual(model.return_model.num_boosted_rounds(), 10)
        self.assertTrue(prediction["trade_probability"].between(0.0, 1.0).all())
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)

    def test_fast_two_stage_supports_all_auxiliary_model_families(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("当前测试环境未安装 scikit-learn")
        config = BacktestConfig()
        config.composite_sklearn_n_estimators = 20
        config.composite_hist_max_iter = 20
        config.xgboost_two_stage_regression_round_ratio = 0.50
        rng = np.random.default_rng(17)
        index = pd.RangeIndex(120)
        features = pd.DataFrame(
            {
                "feature_a": rng.normal(size=len(index)),
                "feature_b": rng.normal(size=len(index)),
                "feature_c": rng.normal(size=len(index)),
            },
            index=index,
        )
        latent_return = features["feature_a"] * 0.5 - features["feature_b"] * 0.2
        direction = np.sign(latent_return).where(latent_return.abs() > 0.15, 0.0)
        targets = build_two_stage_targets(direction, latent_return, config)
        weights = pd.Series(1.0, index=index)

        for model_name in (
            "logistic_regression",
            "elastic_net_logistic",
            "hist_gradient_boosting",
            "random_forest",
            "extra_trees",
        ):
            with self.subTest(model=model_name):
                model = train_fast_two_stage_net_return_model(
                    model_name,
                    features,
                    targets["trade_target"],
                    targets["return_target"],
                    features.columns.tolist(),
                    config,
                    sample_weight=weights,
                )
                prediction = predict_fast_two_stage_net_return(
                    model_name,
                    model,
                    features.iloc[-5:],
                    features.columns.tolist(),
                )
                self.assertTrue(prediction.notna().all().all())

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
            decision_mode="direction_classification",
            probability_semantics="calibrated_class_probability",
            model_edge_semantics="rolling_calibrated_standardized_net_return_edge",
        )
        for column in ("prob_down", "prob_flat", "prob_up"):
            frame[f"uncalibrated_{column}"] = frame[column]

        metrics = calculate_prediction_metrics_for_segment("测试集", frame, config)

        self.assertIsNotNone(metrics)
        self.assertIn("校准后三分类MCC", metrics)
        self.assertIn("校准后三分类BalancedAccuracy", metrics)
        self.assertIn("概率校准误差ECE", metrics)
        self.assertIn("概率差与未来净收益Spearman", metrics)
        self.assertIn("方向准确率Wilson下限", metrics)
        self.assertIn("校准前三分类LogLoss", metrics)
        self.assertIn("校准后三分类LogLoss", metrics)
        self.assertIn("三分类LogLoss改善", metrics)
        self.assertEqual(
            metrics["模型边际口径"],
            "rolling_calibrated_standardized_net_return_edge",
        )

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

    def test_cli_exposes_calibration_multi_window_and_pooled_hierarchy(self) -> None:
        from cli import build_parser, create_config

        composite_args = build_parser().parse_args(
            [
                "composite",
                "--no-probability-calibration",
                "--edge-calibration",
                "--multi-window",
                "--multi-window-train-windows",
                "300,600,1200",
            ]
        )
        composite_config = create_config(composite_args)
        self.assertFalse(composite_config.composite_probability_calibration_enabled)
        self.assertTrue(composite_config.composite_edge_calibration_enabled)
        self.assertEqual(
            composite_config.composite_multi_window_train_windows,
            [300, 600, 1200],
        )

        pooled_args = build_parser().parse_args(
            [
                "pooled",
                "--pooled-hierarchy-mode",
                "group_only",
                "--no-pooled-symbol-residual",
            ]
        )
        pooled_config = create_config(pooled_args)
        self.assertEqual(pooled_config.pooled_model_hierarchy_mode, "group_only")
        self.assertFalse(pooled_config.pooled_symbol_residual_enabled)

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

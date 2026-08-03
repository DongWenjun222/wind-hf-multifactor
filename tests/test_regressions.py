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
    build_training_sample_weights,
    freeze_active_factor_library_for_run,
    load_active_factor_names,
    load_composite_artifact_manifest,
    run_composite_backtest,
    write_composite_artifact_manifest,
)
from config import BacktestConfig, validate_backtest_config
from framework.factor_library import (
    build_selection_metric,
    conservative_pair,
    get_selection_config_value,
    rank_single_factor_summary,
)
from framework.factors import build_factors, build_single_factor_matrix
from multi_symbol_backtest import (
    collect_multi_symbol_manifest_outputs,
    pipeline_state_matches,
    read_single_factor_summary_for_pruning,
    save_multi_symbol_portfolio,
    save_pipeline_state,
    should_skip_composite_pipeline,
    summarize_active_library,
)
from framework.project_fingerprint import get_default_fingerprint_files
from framework.runtime_utils import write_json_atomic
from single_factor_backtest import run_backtest
from framework.factor_builders.parametric import add_parametric_factors
from trading_signal import (
    get_factor_weight_series,
    get_live_model_predict_index,
    load_cached_factor_inputs,
)


class FrameworkRegressionTests(unittest.TestCase):
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
        self.assertEqual(
            get_selection_config_value(
                config,
                "factor_library_min_selection_win_rate",
                "factor_library_min_test_win_rate",
            ),
            0.5,
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
            library_dir = Path(temp_dir) / "factor_library"
            run_dir = Path(temp_dir) / "runs" / "test_composite"
            output_dir = Path(temp_dir) / "composite_factor"
            library_dir.mkdir(parents=True)
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
            output_dir = Path(temp_dir) / "composite_factor"
            library_dir = Path(temp_dir) / "factor_library"
            output_dir.mkdir(parents=True)
            library_dir.mkdir(parents=True)
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
            output_dir = Path(temp_dir) / "composite_factor"
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
            library_dir = Path(temp_dir) / "factor_library"
            library_dir.mkdir(parents=True)
            pd.DataFrame(columns=["因子"]).to_csv(
                library_dir / "active_factors.csv",
                index=False,
            )
            config = BacktestConfig()
            config.output_dir = temp_dir
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
            library_dir = Path(temp_dir) / "factor_library"
            library_dir.mkdir(parents=True)
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
            library_dir = Path(temp_dir) / "factor_library"
            library_dir.mkdir(parents=True)
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
            detail_dir = Path(temp_dir) / "composite_factor"
            detail_dir.mkdir(parents=True)
            pd.DataFrame({"strategy_net_return": [0.0]}).to_csv(
                detail_dir / "composite_detail.csv"
            )
            save_pipeline_state(config, "composite")

            self.assertFalse(should_skip_composite_pipeline(config))

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

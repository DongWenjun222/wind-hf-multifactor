from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd

from config import BacktestConfig
from factor_library_manager import (
    exclude_factors,
    enrich_factor_identity_records,
    protect_factors,
    resolve_factor_names,
    unprotect_factors,
)
from framework.factor_library import (
    build_factor_library,
    load_existing_factor_library,
    load_manual_factor_exclusions,
    load_manual_factor_protections,
)


class FactorProtectionTests(unittest.TestCase):
    def _config(self, temp_dir: str) -> BacktestConfig:
        config = BacktestConfig()
        config.output_dir = temp_dir
        config.factor_library_storage_format = "pickle"
        config.factor_library_min_predictive_score = None
        config.factor_library_enable_family_quota = False
        config.factor_library_use_value_corr = False
        config.factor_library_use_signal_corr = False
        return config

    @staticmethod
    def _active_master() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "因子编号": [1],
                "因子": ["momentum"],
                "因子标签": ["1_momentum"],
                "因子库状态": ["active"],
                "拒绝原因": [""],
            }
        )

    @staticmethod
    def _weak_summary() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "因子": ["momentum"],
                "初筛有效": [True],
                "训练夏普比率": [-1.0],
                "验证夏普比率": [-1.0],
                "训练累计收益": [-0.10],
                "验证累计收益": [-0.10],
                "训练交易次数": [50],
                "验证交易次数": [20],
            }
        )

    def test_protected_factor_survives_rescreen_until_unprotected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            master = self._active_master()
            protect_factors(
                config,
                ["momentum"],
                "人工确认长期保留",
                master.copy(),
                master.copy(),
                load_manual_factor_protections(config),
            )
            saved_protections = load_manual_factor_protections(config)
            self.assertEqual(int(saved_protections.loc[0, "因子编号"]), 1)
            self.assertEqual(saved_protections.loc[0, "因子标签"], "1_momentum")
            empty = pd.DataFrame()
            self.assertEqual(
                resolve_factor_names(["1"], empty, empty, empty, saved_protections),
                ["momentum"],
            )
            self.assertEqual(
                resolve_factor_names(["1_momentum"], empty, empty, empty, saved_protections),
                ["momentum"],
            )
            protected_active, protected_master, _ = build_factor_library(
                self._weak_summary(),
                pd.DataFrame({"momentum": np.arange(60, dtype="float64")}),
                config,
            )
            self.assertEqual(protected_active["因子"].tolist(), ["momentum"])
            self.assertTrue(bool(protected_active.loc[0, "是否手工保护"]))

            unprotect_factors(
                config,
                ["momentum"],
                protected_active,
                protected_master,
                load_manual_factor_protections(config),
            )
            rebuilt_active, _, _ = build_factor_library(
                self._weak_summary(),
                pd.DataFrame({"momentum": np.arange(60, dtype="float64")}),
                config,
            )
            self.assertTrue(rebuilt_active.empty)

    def test_explicit_exclusion_overrides_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            master = self._active_master()
            active = master.copy()
            protect_factors(
                config,
                ["momentum"],
                "必须保留",
                active,
                master,
                load_manual_factor_protections(config),
            )
            persisted_master = load_existing_factor_library(config)
            persisted_active = persisted_master[persisted_master["因子库状态"].eq("active")].copy()
            exclude_factors(
                config,
                ["momentum"],
                "人工明确剔除",
                persisted_active,
                persisted_master,
                load_manual_factor_exclusions(config),
            )
            self.assertTrue(load_manual_factor_protections(config).empty)
            self.assertEqual(load_manual_factor_exclusions(config)["因子"].tolist(), ["momentum"])

    def test_old_string_governance_columns_accept_numeric_identity_migration(self) -> None:
        old_records = pd.DataFrame(
            {
                "因子编号": pd.Series([""], dtype="string"),
                "因子标签": pd.Series([""], dtype="string"),
                "因子": ["momentum"],
                "手工保护原因": ["旧记录"],
            }
        )
        catalog = self._active_master()
        enriched = enrich_factor_identity_records(old_records, catalog)
        self.assertEqual(int(enriched.loc[0, "因子编号"]), 1)
        self.assertEqual(enriched.loc[0, "因子标签"], "1_momentum")


if __name__ == "__main__":
    unittest.main()

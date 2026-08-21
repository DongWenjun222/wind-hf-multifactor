from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from factor_logic_audit import normalize_review_results, resolve_codex_executable
from framework.factor_logic_review import (
    get_pending_logic_reviews,
    merge_factor_logic_reviews,
)


class FactorLogicReviewTests(unittest.TestCase):
    def _project_root(self, temp_dir: str) -> Path:
        root = Path(temp_dir)
        source = root / "framework" / "factor_builders" / "basic.py"
        source.parent.mkdir(parents=True)
        source.write_text("# version 1\n", encoding="utf-8")
        return root

    def test_completed_matching_review_is_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._project_root(temp_dir)
            active = pd.DataFrame({"因子": ["momentum"]})
            context = get_pending_logic_reviews(active, pd.DataFrame(), project_root=root)
            review = pd.DataFrame(
                {
                    "因子": ["momentum"],
                    "逻辑审查状态": ["approved"],
                    "逻辑审查指纹": [context.loc[0, "逻辑审查指纹"]],
                }
            )
            pending = get_pending_logic_reviews(active, review, project_root=root)
            self.assertTrue(pending.empty)

            (root / "framework" / "factor_builders" / "basic.py").write_text(
                "# version 2\n", encoding="utf-8"
            )
            changed = get_pending_logic_reviews(active, review, project_root=root)
            self.assertEqual(changed["因子"].tolist(), ["momentum"])

    def test_only_matching_fingerprint_is_attached_to_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._project_root(temp_dir)
            active = pd.DataFrame({"因子": ["momentum"], "因子家族": ["basic"]})
            context = get_pending_logic_reviews(active, pd.DataFrame(), project_root=root)
            review = pd.DataFrame(
                {
                    "因子": ["momentum"],
                    "逻辑审查状态": ["approved"],
                    "逻辑类型": ["趋势"],
                    "逻辑审查指纹": [context.loc[0, "逻辑审查指纹"]],
                }
            )
            merged = merge_factor_logic_reviews(active, review, project_root=root)
            self.assertEqual(merged.loc[0, "逻辑审查状态"], "approved")
            self.assertEqual(merged.loc[0, "逻辑类型"], "趋势")
            self.assertEqual(merged.loc[0, "因子家族"], "basic")

    def test_low_confidence_rejection_is_downgraded(self) -> None:
        batch = pd.DataFrame(
            {
                "因子": ["momentum"],
                "因子家族": ["basic"],
                "来源文件": ["framework/factor_builders/basic.py"],
                "逻辑审查指纹": ["fingerprint"],
            }
        )
        normalized = normalize_review_results(
            batch,
            [
                {
                    "factor": "momentum",
                    "status": "rejected",
                    "logic_type": "未知",
                    "logic_explanation": "解释不足",
                    "main_risk": "证据不足",
                    "confidence": 0.60,
                    "code_inspected": True,
                }
            ],
            model_name="test-model",
            min_reject_confidence=0.85,
        )
        self.assertEqual(normalized.loc[0, "逻辑审查状态"], "uncertain")
        self.assertIn("降级待复核", normalized.loc[0, "主要风险"])

    def test_explicit_codex_executable_path_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "codex.exe"
            executable.write_bytes(b"placeholder")
            self.assertEqual(resolve_codex_executable(str(executable)), executable.resolve())


if __name__ == "__main__":
    unittest.main()

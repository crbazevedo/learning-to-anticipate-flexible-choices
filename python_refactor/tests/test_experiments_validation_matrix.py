"""
W10-1 regression tests for validation_matrix.py.

Pins:
  - _flatten_metrics emits dotted-key rows from nested final_metrics
    (closes W9-CARRY-3: pre-W10-1 the writer filtered on top-level
    scalars and wrote 0 rows).
  - WINDOWS['paper'] points to the real 98-CSV paper-window glob
    (closes W8-1-CARRY-1 fully).
"""

import csv
import unittest
from pathlib import Path

from experiments.validation_matrix import WINDOWS, _flatten_metrics
from src.experiments.data_loader import DataLoader


class TestFlattenMetrics(unittest.TestCase):
    """W10-1: nested final_metrics → dotted-key rows."""

    def test_nested_dict_recurses_to_dotted_keys(self):
        nested = {
            "portfolio": {"final_value": 1.4955, "total_return": 0.4955},
            "summary": {"sharpe_ratio": 0.0, "max_drawdown": 0.1},
        }
        rows = _flatten_metrics(nested)
        keys = {k for (k, _) in rows}
        self.assertIn("portfolio.final_value", keys)
        self.assertIn("portfolio.total_return", keys)
        self.assertIn("summary.sharpe_ratio", keys)
        self.assertIn("summary.max_drawdown", keys)
        # Values are floats
        for _, v in rows:
            self.assertIsInstance(v, float)

    def test_flat_dict_passes_through(self):
        flat = {"a": 1, "b": 2.5, "c": 3}
        rows = sorted(_flatten_metrics(flat))
        self.assertEqual(rows, [("a", 1.0), ("b", 2.5), ("c", 3.0)])

    def test_booleans_and_none_excluded(self):
        d = {"x": True, "y": False, "z": None, "ok": 1.5}
        rows = _flatten_metrics(d)
        self.assertEqual(rows, [("ok", 1.5)])

    def test_three_level_nesting(self):
        d = {"L1": {"L2": {"L3": 7.7}}}
        self.assertEqual(_flatten_metrics(d), [("L1.L2.L3", 7.7)])

    def test_empty_dict_returns_empty(self):
        self.assertEqual(_flatten_metrics({}), [])

    def test_real_shape_from_experiment_manager(self):
        """Real shape ExperimentManager emits (sampled from W9-4 smoke-test logs)."""
        real_final_metrics = {
            "algorithm": {"generation": 0, "hypervolume": 0.0, "population_size": 20},
            "portfolio": {"final_value": 1.4955, "concentration": 0.356},
            "summary": {"total_execution_time": 0.69, "final_portfolio_value": 1.4955},
        }
        rows = _flatten_metrics(real_final_metrics)
        # 7 numeric leaves
        self.assertEqual(len(rows), 7)
        # Acceptance criterion: ≥ 5 rows
        self.assertGreaterEqual(len(rows), 5)


class TestPaperWindowGlob(unittest.TestCase):
    """W10-1: WINDOWS['paper'] points at real 98-CSV glob (W8-1-CARRY-1 closure)."""

    def test_paper_window_pattern_uses_legacy_cpp_glob(self):
        self.assertIn("legacy-cpp", WINDOWS["paper"]["asset_files_glob"])
        self.assertIn("table*.csv", WINDOWS["paper"]["asset_files_glob"])

    def test_paper_window_glob_loads_many_assets(self):
        # Resolve relative to python_refactor (where validation_matrix runs from).
        repo_root = Path(__file__).parents[2]
        pattern = repo_root / "python_refactor" / WINDOWS["paper"]["asset_files_glob"]
        # Normalize the '../' prefix
        pattern = pattern.resolve()
        if not pattern.parent.exists():
            self.skipTest(f"paper-window data dir not present at {pattern.parent}")
        loader = DataLoader()
        df = loader.load_asset_data([str(pattern)], date_range={}, assets=[])
        self.assertGreaterEqual(len(df.columns), 90,
                                  f"Expected >=90 paper-window assets, got {len(df.columns)}")

    def test_extended_window_unchanged(self):
        # Backward compat: extended still points at FTSE-updated single-CSV.
        self.assertIn("ftse-updated", WINDOWS["extended"]["asset_files_glob"])


if __name__ == "__main__":
    unittest.main()

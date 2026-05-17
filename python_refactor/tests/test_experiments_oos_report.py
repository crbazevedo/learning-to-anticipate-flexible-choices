"""
W13-3 regression tests for oos_report.

Pure-function tests on the orchestration helpers. The full-pipeline
integration is exercised by the W13-3 wave-close (which produces
docs/OOS-EFHV-REPORT.md from real data) — too slow for unit-test
inclusion.
"""

import unittest

import numpy as np
import pandas as pd

from experiments.oos_report import (
    _aggregate,
    _format_report,
    split_temporal,
)


class TestSplitTemporal(unittest.TestCase):
    def test_basic_split(self):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        train, oos = split_temporal(df, train_frac=0.8)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(oos), 20)
        # Train is the first 80 rows; OOS is the last 20
        self.assertEqual(train["a"].iloc[0], 0)
        self.assertEqual(train["a"].iloc[-1], 79)
        self.assertEqual(oos["a"].iloc[0], 80)
        self.assertEqual(oos["a"].iloc[-1], 99)

    def test_same_asset_count_preserved(self):
        df = pd.DataFrame(np.zeros((50, 7)))
        train, oos = split_temporal(df, train_frac=0.5)
        self.assertEqual(train.shape[1], 7)
        self.assertEqual(oos.shape[1], 7)

    def test_extreme_train_frac(self):
        df = pd.DataFrame(np.zeros((10, 3)))
        # train_frac=1.0: all training, empty OOS
        train, oos = split_temporal(df, train_frac=1.0)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(oos), 0)


class TestAggregate(unittest.TestCase):
    def test_summary_stats(self):
        results = {"S0": [0.1, 0.2, 0.3], "S2": [0.5, 0.6]}
        summary = _aggregate(results)
        self.assertEqual(summary["S0"]["n_seeds"], 3)
        self.assertAlmostEqual(summary["S0"]["mean"], 0.2, places=10)
        self.assertAlmostEqual(summary["S0"]["median"], 0.2, places=10)
        self.assertEqual(summary["S0"]["min"], 0.1)
        self.assertEqual(summary["S0"]["max"], 0.3)
        self.assertEqual(summary["S2"]["n_seeds"], 2)
        self.assertAlmostEqual(summary["S2"]["mean"], 0.55, places=10)

    def test_empty_scenario_returns_nan(self):
        summary = _aggregate({"empty": []})
        self.assertEqual(summary["empty"]["n_seeds"], 0)
        self.assertTrue(np.isnan(summary["empty"]["mean"]))


class TestFormatReport(unittest.TestCase):
    def test_paper_claim_direction_consistent(self):
        summary = {
            "S0": {"n_seeds": 5, "mean": 0.01, "std": 0.001,
                    "median": 0.01, "min": 0.009, "max": 0.011},
            "S2": {"n_seeds": 5, "mean": 0.02, "std": 0.002,
                    "median": 0.02, "min": 0.018, "max": 0.022},
        }
        text = _format_report(
            summary,
            scenarios=["S0", "S2"],
            seeds=[1, 2, 3, 4, 5],
            in_sample_window="paper",
            oos_window="paper[tail20%]",
            n_mc=1000,
        )
        # Headline section present
        self.assertIn("Headline observation", text)
        # Direction = consistent because S2 (0.02) > S0 (0.01)
        self.assertIn("S2 > S0 (consistent)", text)
        # MC E reported
        self.assertIn("1000", text)
        # z_ref reported
        self.assertIn("(0.2, 0.0)", text)
        # Table headers
        self.assertIn("| scenario |", text)

    def test_paper_claim_direction_inconsistent(self):
        summary = {
            "S0": {"n_seeds": 5, "mean": 0.02, "std": 0.001,
                    "median": 0.02, "min": 0.018, "max": 0.022},
            "S2": {"n_seeds": 5, "mean": 0.01, "std": 0.002,
                    "median": 0.01, "min": 0.009, "max": 0.011},
        }
        text = _format_report(
            summary,
            scenarios=["S0", "S2"],
            seeds=[1, 2, 3, 4, 5],
            in_sample_window="paper",
            oos_window="paper[tail20%]",
            n_mc=1000,
        )
        # Direction = inconsistent because S2 (0.01) < S0 (0.02)
        self.assertIn("S2 ≤ S0 (inconsistent)", text)

    def test_single_scenario_no_headline(self):
        summary = {
            "S0": {"n_seeds": 3, "mean": 0.01, "std": 0.001,
                    "median": 0.01, "min": 0.009, "max": 0.011},
        }
        text = _format_report(
            summary,
            scenarios=["S0"],
            seeds=[1, 2, 3],
            in_sample_window="paper",
            oos_window="paper[tail20%]",
            n_mc=100,
        )
        # No headline when S2 missing
        self.assertNotIn("Headline observation", text)


if __name__ == "__main__":
    unittest.main()

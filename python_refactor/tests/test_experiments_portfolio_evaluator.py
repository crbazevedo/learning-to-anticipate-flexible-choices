"""
W11-2 regression tests for portfolio_evaluator._get_asset_returns.

Pins the W10-CARRY-1 fix: when input DataFrame is price-level
(median |value| > 1.5), convert via pct_change before returning.
Returns-shaped input passes through unchanged.
"""

import unittest

import numpy as np
import pandas as pd

from src.experiments.portfolio_evaluator import PortfolioEvaluator


class TestPriceVsReturnsDetection(unittest.TestCase):
    """W11-2: price-level → pct_change; returns pass-through."""

    def setUp(self):
        self.ev = PortfolioEvaluator()

    def test_price_level_input_is_converted_to_returns(self):
        # Index/price levels in the thousands → median magnitude > 1.5.
        prices = pd.DataFrame({
            "AAPL": [150.0, 151.5, 149.0, 152.0],
            "GOOG": [2800.0, 2830.0, 2780.0, 2810.0],
        })
        out = self.ev._get_asset_returns({"assets": prices})
        # Returns shape is one row shorter (first row dropped by dropna).
        self.assertEqual(len(out), 3)
        # Values look like daily returns (|v| < 0.1 typically).
        self.assertTrue((out.abs().max() < 0.1).all())
        # Specific: (151.5 - 150) / 150 = 0.01
        self.assertAlmostEqual(out["AAPL"].iloc[0], 0.01, places=6)

    def test_returns_level_input_passes_through(self):
        # Real return values < 0.2 in magnitude.
        returns = pd.DataFrame({
            "AAPL": [0.01, -0.005, 0.02, -0.01],
            "GOOG": [0.005, 0.01, -0.015, 0.008],
        })
        out = self.ev._get_asset_returns({"assets": returns})
        # No pct_change applied → exact pass-through.
        pd.testing.assert_frame_equal(out, returns)

    def test_returns_sanitation_replaces_inf(self):
        """W10-CARRY-1 root cause: data_loader already produces returns
        but they contain Inf values from pct_change crossing zero-price
        quotes. The W11-2 second layer sanitizes them."""
        dirty = pd.DataFrame({
            "A": [0.01, float("inf"), -0.02, 0.005],
            "B": [-0.005, 0.01, float("-inf"), 0.003],
        })
        out = self.ev._get_asset_returns({"assets": dirty})
        self.assertFalse(np.isinf(out.values).any(), "Inf must be replaced")
        # Inf → 0.0 (no-information treatment).
        self.assertEqual(out["A"].iloc[1], 0.0)
        self.assertEqual(out["B"].iloc[2], 0.0)

    def test_returns_sanitation_clips_implausible_outliers(self):
        """Real FTSE archive has one-day returns of 97999, 42149, 28927
        from data-source quirks. Clip at ±1.0."""
        outliers = pd.DataFrame({
            "A": [0.01, 97999.0, -0.02, 0.005],
            "B": [-0.005, 0.01, -42149.0, 0.003],
        })
        out = self.ev._get_asset_returns({"assets": outliers})
        self.assertLessEqual(out.values.max(), 1.0)
        self.assertGreaterEqual(out.values.min(), -1.0)
        # Specific clips
        self.assertEqual(out["A"].iloc[1], 1.0)
        self.assertEqual(out["B"].iloc[2], -1.0)

    def test_real_paper_window_no_inf_after_sanitation(self):
        """End-to-end: real 98-CSV paper-window passes through
        portfolio_evaluator with all values finite + bounded.

        W13-1 update: data_loader now sanitizes at the source, so
        assets_df arrives Inf-free. The portfolio_evaluator sanitation
        becomes a defensive second layer (no-op in this case). Test
        keeps the post-condition check (output bounded + finite) and
        drops the pre-condition assertion that data has Inf — the
        W11-2 layer would still catch Inf if it ever reached us, but
        a clean source is also a valid input."""
        from pathlib import Path
        from src.experiments.data_loader import DataLoader
        repo_root = Path(__file__).parents[2]
        glob_path = repo_root / "legacy-cpp" / "executable" / "data" / "ftse-original" / "table*.csv"
        if not glob_path.parent.exists():
            self.skipTest(f"paper-window data not present at {glob_path.parent}")
        loader = DataLoader()
        assets_df = loader.load_asset_data([str(glob_path)], date_range={}, assets=[])
        if assets_df.empty:
            self.skipTest("data_loader returned empty")
        # Post-W13-1 + post-W11-2 sanitation: output must be finite + bounded.
        out = self.ev._get_asset_returns({"assets": assets_df})
        self.assertFalse(np.isinf(out.values).any())
        self.assertLessEqual(np.nanmax(out.values), 1.0)
        self.assertGreaterEqual(np.nanmin(out.values), -1.0)

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame()
        out = self.ev._get_asset_returns({"assets": empty})
        self.assertTrue(out.empty)

    def test_threshold_at_boundary_just_above(self):
        # Median magnitude > 1.5 → triggers price-level conversion.
        prices = pd.DataFrame({"x": [2.0, 2.04, 1.98, 2.02]})
        out = self.ev._get_asset_returns({"assets": prices})
        # Converted to returns (3 rows post-dropna), bounded.
        self.assertEqual(len(out), 3)
        self.assertTrue((out.abs() < 0.1).all().all())

    def test_threshold_at_boundary_just_below(self):
        # Median magnitude ≤ 1.5 → pass-through (no conversion).
        # Values < 1.0 so sanitation clip is a no-op.
        returns = pd.DataFrame({"x": [0.5, 0.6, -0.4, 0.3]})
        out = self.ev._get_asset_returns({"assets": returns})
        # Same shape, same values (no clip needed; no conversion).
        pd.testing.assert_frame_equal(out, returns)


if __name__ == "__main__":
    unittest.main()

"""
W14-1 regression tests for walk_forward.

Focuses on period-enumeration arithmetic + aggregation helpers.
The training-loop call is exercised by a single S0-2-period
integration smoke (fast — S0 takes ~0.5s/period).
"""

import unittest

import numpy as np
import pandas as pd

from experiments.walk_forward import (
    aggregate_per_seed,
    enumerate_periods,
    run_walk_forward,
)


class TestEnumeratePeriods(unittest.TestCase):
    def test_basic_enumeration(self):
        # 500 days, 378 train + 50 step → t=1: train=[0,378), oos=[378,428); fits.
        # t=2: train=[50,428), oos=[428,478); fits.
        # t=3: train=[100,478), oos=[478,528); 528 > 500, NO.
        periods = enumerate_periods(n_days=500, train_window_days=378,
                                      step_days=50)
        self.assertEqual(len(periods), 2)
        self.assertEqual(periods[0]["period"], 1)
        self.assertEqual(periods[0]["train_start"], 0)
        self.assertEqual(periods[0]["train_end"], 378)
        self.assertEqual(periods[0]["oos_start"], 378)
        self.assertEqual(periods[0]["oos_end"], 428)
        self.assertEqual(periods[1]["period"], 2)
        self.assertEqual(periods[1]["train_start"], 50)
        self.assertEqual(periods[1]["train_end"], 428)
        self.assertEqual(periods[1]["oos_end"], 478)

    def test_thesis_default_yields_about_24_periods(self):
        # Paper-window data spans ~2600 trading days (2003-2012).
        # With 378-day train + 50-day step:
        # max periods = floor((2600 - 378 - 50) / 50) + 1 = floor(2172/50)+1 ≈ 44.
        # Thesis FTSE has T=24 because they use a SHORTER total span (2006-2012).
        # We verify the formula here for our larger dataset.
        periods = enumerate_periods(n_days=2600, train_window_days=378,
                                      step_days=50)
        # 2172 / 50 = 43.44 → floor = 43; +1 = 44
        self.assertEqual(len(periods), 44)

    def test_train_oos_non_overlap(self):
        periods = enumerate_periods(n_days=1000, train_window_days=300,
                                      step_days=50)
        for p in periods:
            self.assertLess(p["train_end"], p["oos_end"])
            self.assertEqual(p["oos_start"], p["train_end"])

    def test_no_periods_when_train_too_long(self):
        # Train alone is 600 days but data only has 500 days.
        periods = enumerate_periods(n_days=500, train_window_days=600,
                                      step_days=50)
        self.assertEqual(periods, [])

    def test_no_periods_when_oos_doesnt_fit(self):
        # Train=378 fits; but train+step=428 > 400 (no OOS room).
        periods = enumerate_periods(n_days=400, train_window_days=378,
                                      step_days=50)
        self.assertEqual(periods, [])


class TestAggregatePerSeed(unittest.TestCase):
    def test_grand_mean_across_periods(self):
        per_period = [
            {"period": 1, "efhv_mean": 0.001, "efhv_std": 1e-5},
            {"period": 2, "efhv_mean": 0.002, "efhv_std": 1e-5},
            {"period": 3, "efhv_mean": 0.003, "efhv_std": 1e-5},
        ]
        agg = aggregate_per_seed(per_period)
        self.assertEqual(agg["n_periods_ok"], 3)
        self.assertAlmostEqual(agg["grand_mean"], 0.002, places=10)

    def test_excludes_errors_and_nans(self):
        per_period = [
            {"period": 1, "efhv_mean": 0.001},
            {"period": 2, "efhv_mean": float("nan"), "error": "train failure"},
            {"period": 3, "efhv_mean": 0.003},
        ]
        agg = aggregate_per_seed(per_period)
        # Only periods 1 + 3 contribute.
        self.assertEqual(agg["n_periods_ok"], 2)
        self.assertAlmostEqual(agg["grand_mean"], 0.002, places=10)

    def test_all_errors_returns_nan(self):
        per_period = [
            {"period": 1, "efhv_mean": float("nan"), "error": "x"},
            {"period": 2, "efhv_mean": float("nan"), "error": "y"},
        ]
        agg = aggregate_per_seed(per_period)
        self.assertEqual(agg["n_periods_ok"], 0)
        self.assertTrue(np.isnan(agg["grand_mean"]))


class TestRunWalkForwardSmoke(unittest.TestCase):
    """Integration smoke: S0 × 2 periods. Slow-ish (~5s) but pins
    the end-to-end glue between W14-1 and W13-2."""

    def test_s0_few_periods_returns_finite_efhv(self):
        # Load real paper-window data for the smoke (the algorithm
        # needs realistic returns to produce non-degenerate Pareto fronts).
        from pathlib import Path
        from src.experiments.data_loader import DataLoader
        repo_root = Path(__file__).parents[2]
        glob_path = repo_root / "legacy-cpp" / "executable" / "data" / "ftse-original" / "table*.csv"
        if not glob_path.parent.exists():
            self.skipTest(f"paper-window data not at {glob_path.parent}")
        loader = DataLoader()
        full_returns = loader.load_asset_data([str(glob_path)],
                                                date_range={}, assets=[])
        # Use small train + step to keep the smoke fast.
        # n_days=400, train=300, step=50:
        #   t=1: train=[0,300), oos=[300,350); fits
        #   t=2: train=[50,350), oos=[350,400); fits
        #   t=3: train=[100,400), oos=[400,450); NO (450 > 400)
        # → 2 periods.
        full_returns = full_returns.iloc[:400]
        results = run_walk_forward(
            scenario="S0", seed=1,
            full_returns=full_returns,
            train_window_days=300,
            step_days=50,
            n_mc=50,  # small E for speed
            rng=np.random.default_rng(1),
        )
        # Periods enumerated; all should produce finite EFHV.
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertNotIn("error", r)
            self.assertGreater(r["n_pareto"], 0)
            self.assertTrue(np.isfinite(r["efhv_mean"]))


if __name__ == "__main__":
    unittest.main()

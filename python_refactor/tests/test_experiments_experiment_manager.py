"""
W9-1 regression tests for experiment_manager.py.

Pins the fix for `bool(DataFrame)` ambiguity at line 190 (was 189
pre-W9-1): `if asset_data:` on a pandas.DataFrame raises ValueError.
The W9-1 fix replaces with explicit `None or .empty` check and counts
columns (one per asset) instead of rows.
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.experiments.experiment_manager import ExperimentManager


def _make_manager(tmp_path):
    suite_config = {
        "experiment_name": "W9-1-regression",
        "description": "bool(DataFrame) fix",
        "version": "W9-1",
        "timestamp": "2026-05-17T00:00:00Z",
        "output_directory": str(tmp_path),
    }
    return ExperimentManager(suite_config)


class TestBoolDataFrameFix(unittest.TestCase):
    """W9-1: line 190 must not raise ValueError on DataFrame input."""

    def setUp(self):
        # ExperimentManager constructs an output dir; use a per-test tempdir.
        from tempfile import mkdtemp
        from pathlib import Path
        self.tmp = Path(mkdtemp(prefix="w9-1-"))
        self.mgr = _make_manager(self.tmp)

    def _load_with(self, asset_df):
        """Helper: call _load_experiment_data with a mocked data_loader
        whose load_asset_data returns `asset_df` and load_market_data
        returns an empty DataFrame. Exercises the bool(DataFrame) site."""
        self.mgr.data_loader = MagicMock()
        self.mgr.data_loader.load_asset_data.return_value = asset_df
        self.mgr.data_loader.load_market_data.return_value = pd.DataFrame()
        self.mgr.logger = MagicMock()
        cfg = {"data": {"asset_files": [], "date_range": {}, "assets": [],
                          "market_files": []}}
        # Just exercise the path that used to raise.
        return self.mgr._load_experiment_data(cfg)

    def test_non_empty_dataframe_does_not_raise(self):
        df = pd.DataFrame({"AAPL": [1.0, 2.0], "GOOG": [3.0, 4.0]})
        data = self._load_with(df)
        # num_assets logged as column count (2 assets), not row count.
        log_call = self.mgr.logger.log.call_args_list[-1]
        log_payload = log_call[0][3]
        self.assertEqual(log_payload["num_assets"], 2)
        self.assertIs(data["assets"], df)

    def test_empty_dataframe_logs_zero_assets(self):
        df = pd.DataFrame()
        self._load_with(df)
        log_call = self.mgr.logger.log.call_args_list[-1]
        log_payload = log_call[0][3]
        self.assertEqual(log_payload["num_assets"], 0)

    def test_none_logs_zero_assets(self):
        self._load_with(None)
        log_call = self.mgr.logger.log.call_args_list[-1]
        log_payload = log_call[0][3]
        self.assertEqual(log_payload["num_assets"], 0)


if __name__ == "__main__":
    unittest.main()

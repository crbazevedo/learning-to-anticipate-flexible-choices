"""
W9-2 regression tests for data_loader.load_asset_data.

Pins:
  Bug 2: `pivot(columns=None)` → KeyError(None) — fix uses
         columns='asset_id' (tag added per row).
  Bug 3: literal-path-only entries — fix glob-expands entries
         containing wildcard chars.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.experiments.data_loader import DataLoader


def _write_csv(path: Path, dates: list[str], closes: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": dates, "Close": closes}).to_csv(path, index=False)


class TestPivotFix(unittest.TestCase):
    """W9-2 bug 2 — pivot must use columns='asset_id', not None."""

    def test_two_assets_pivot_to_two_columns(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_csv(tmpdir / "AAPL.csv", ["2020-01-02", "2020-01-03", "2020-01-06"], [100.0, 102.0, 101.0])
            _write_csv(tmpdir / "GOOG.csv", ["2020-01-02", "2020-01-03", "2020-01-06"], [1500.0, 1510.0, 1495.0])

            loader = DataLoader()
            df = loader.load_asset_data(
                [str(tmpdir / "AAPL.csv"), str(tmpdir / "GOOG.csv")],
                date_range={},
                assets=[],
            )
            # Two columns, one per asset, named by file stem.
            self.assertEqual(sorted(df.columns.tolist()), ["AAPL", "GOOG"])
            self.assertEqual(len(df), 3)
            # pct_change of [100, 102, 101] → [NaN, 0.02, -0.0098...] → first NaN filled to 0
            self.assertAlmostEqual(df["AAPL"].iloc[1], 0.02, places=4)


class TestGlobExpansion(unittest.TestCase):
    """W9-2 bug 3 — wildcard entries glob-expand."""

    def test_glob_pattern_expands_to_all_matching_csvs(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for name in ["table_001.csv", "table_002.csv", "table_003.csv"]:
                _write_csv(tmpdir / name, ["2020-01-02", "2020-01-03"], [10.0, 11.0])
            # Also an unrelated csv that should NOT match.
            _write_csv(tmpdir / "other.csv", ["2020-01-02", "2020-01-03"], [99.0, 100.0])

            loader = DataLoader()
            df = loader.load_asset_data(
                [str(tmpdir / "table_*.csv")],
                date_range={},
                assets=[],
            )
            self.assertEqual(sorted(df.columns.tolist()),
                              ["table_001", "table_002", "table_003"])

    def test_literal_path_still_works_backward_compat(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_csv(tmpdir / "FTSE_updated.csv",
                        ["2020-01-02", "2020-01-03", "2020-01-06"],
                        [7000.0, 7050.0, 6995.0])
            loader = DataLoader()
            df = loader.load_asset_data(
                [str(tmpdir / "FTSE_updated.csv")],
                date_range={},
                assets=[],
            )
            self.assertEqual(df.columns.tolist(), ["FTSE_updated"])
            self.assertEqual(len(df), 3)


class TestAssetsFilter(unittest.TestCase):
    """`assets` arg restricts and orders the returned columns."""

    def test_assets_filter_restricts_columns(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for name in ["AAPL.csv", "GOOG.csv", "MSFT.csv"]:
                _write_csv(tmpdir / name, ["2020-01-02", "2020-01-03"], [100.0, 101.0])
            loader = DataLoader()
            df = loader.load_asset_data(
                [str(tmpdir / f) for f in ["AAPL.csv", "GOOG.csv", "MSFT.csv"]],
                date_range={},
                assets=["MSFT", "AAPL"],
            )
            # Order preserved; GOOG dropped.
            self.assertEqual(df.columns.tolist(), ["MSFT", "AAPL"])


class TestRealDataDuplicates(unittest.TestCase):
    """W9-4 hotfix to W9-2: real CSVs have duplicate (Date, asset_id)
    tuples (year-end summary rows). Synthetic fixtures missed this;
    smoke-test surfaced it. Pin the dedup behavior."""

    def test_duplicate_dates_in_single_csv_dont_break_pivot(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # CSV has 2 rows for 2020-12-31 (year-end summary + EOD).
            _write_csv(tmpdir / "INDEX.csv",
                        ["2020-01-02", "2020-12-31", "2020-12-31"],
                        [100.0, 110.0, 110.5])
            loader = DataLoader()
            df = loader.load_asset_data(
                [str(tmpdir / "INDEX.csv")],
                date_range={},
                assets=[],
            )
            # Pivot succeeds; one row per unique date.
            self.assertEqual(len(df), 2)
            # 'keep=last' preserves the consolidated-summary value.
            self.assertAlmostEqual(df["INDEX"].loc[pd.Timestamp("2020-12-31")],
                                     (110.5 - 110.0) / 110.0, places=4)

    def test_real_ftse_updated_csv_loads(self):
        repo_root = Path(__file__).parents[2]
        ftse_csv = repo_root / "python_refactor" / "data" / "ftse-updated" / "FTSE_100_20121121_20241231.csv"
        if not ftse_csv.exists():
            self.skipTest(f"FTSE-updated CSV not present at {ftse_csv}")
        loader = DataLoader()
        df = loader.load_asset_data([str(ftse_csv)], date_range={}, assets=[])
        # Smoke: shape sane, no duplicates leaked past pivot.
        self.assertGreater(len(df), 1000)
        self.assertEqual(len(df.index), len(df.index.unique()))


class TestPaperWindowLoad(unittest.TestCase):
    """W8-1-CARRY-1 closure check: paper-window per-asset CSVs load."""

    def test_real_paper_window_glob_loads_many_assets(self):
        # Skip if the dataset isn't present (CI without legacy-cpp/).
        data_dir = Path(__file__).parents[2] / "legacy-cpp" / "executable" / "data" / "ftse-original"
        if not data_dir.exists():
            self.skipTest(f"paper-window dataset not present at {data_dir}")
        loader = DataLoader()
        df = loader.load_asset_data(
            [str(data_dir / "table*.csv")],
            date_range={},
            assets=[],
        )
        # Paper-window: 98 per-asset CSVs (W8-1-CARRY-1 spec).
        self.assertGreaterEqual(len(df.columns), 90,
                                  f"Expected >=90 paper-window assets, got {len(df.columns)}")


if __name__ == "__main__":
    unittest.main()

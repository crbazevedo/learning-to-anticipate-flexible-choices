"""
W17-1: regression tests for the BACKLOG H4 87-asset continuous-trades
filter on the FTSE legacy-cpp universe.

Thesis grounding (verbatim):

§7.2.3 "Artificial and Real-World Datasets", p. 145:
    "All benchmarks provide d = 30 simulated assets for composing
     the portfolios, whereas for the real-world instances we have
     d = 87 for FTSE; d = 30 for DJI; and d = 49 for HSI..."

Operator criterion (verbatim, from session):
    "keep only the assets which had continuous daily closing prices
     trades throughout the full dataset period"
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.data_loader import (
    DataLoader,
    _load_h4_kept_assets,
    _H4_ARTIFACT_PATH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FTSE_GLOB = str(
    REPO_ROOT / "legacy-cpp" / "executable" / "data" / "ftse-original" / "table*.csv"
)


class TestH4Artifact:
    """The asset_universe_87.json artifact exists and has the right shape."""

    def test_artifact_exists(self):
        """The W17-1 EDA result is committed and on disk."""
        assert _H4_ARTIFACT_PATH.exists(), (
            f"Run python_refactor/scripts/h4_asset_universe_eda.py to build "
            f"{_H4_ARTIFACT_PATH}"
        )

    def test_artifact_schema(self):
        """Artifact has the documented top-level keys."""
        with open(_H4_ARTIFACT_PATH) as fh:
            artifact = json.load(fh)
        expected_keys = {
            "criterion", "criterion_verbatim", "thesis_target_d",
            "thesis_target_section", "n_original", "n_kept", "n_dropped",
            "window_start", "window_end", "kept_assets", "dropped_assets",
        }
        assert expected_keys.issubset(set(artifact.keys())), (
            f"missing keys: {expected_keys - set(artifact.keys())}"
        )

    def test_artifact_exact_match_thesis_target(self):
        """n_kept == 87 (thesis §7.2.3 p.145 'd = 87 for FTSE')."""
        with open(_H4_ARTIFACT_PATH) as fh:
            artifact = json.load(fh)
        assert artifact["n_original"] == 98
        assert artifact["n_kept"] == 87
        assert artifact["n_dropped"] == 11
        assert artifact["thesis_target_d"] == 87

    def test_kept_assets_list_length(self):
        """kept_assets list literally has 87 entries."""
        with open(_H4_ARTIFACT_PATH) as fh:
            artifact = json.load(fh)
        assert len(artifact["kept_assets"]) == 87
        assert len(artifact["dropped_assets"]) == 11


class TestH4LoaderHelper:
    """_load_h4_kept_assets reads the artifact + caches."""

    def test_returns_set_of_87(self):
        # Reset cache to force re-load
        from src.experiments import data_loader
        data_loader._h4_kept_assets_cache = None
        kept = _load_h4_kept_assets()
        assert isinstance(kept, set)
        assert len(kept) == 87
        assert all(name.startswith("table") and name.endswith(".csv") for name in kept)


class TestDataLoaderEnforceContinuousTrades:
    """data_loader.load_asset_data honors the new opt-in flag."""

    def test_default_loads_all_98(self):
        """Default behavior unchanged: load all 98 expanded files."""
        loader = DataLoader()
        df = loader.load_asset_data(
            asset_files=[FTSE_GLOB],
            date_range={"start": "2006-11-20", "end": "2012-12-31"},
            assets=[],
        )
        # The pivot uses asset_id (file stem). Some assets may have
        # zero in-window rows after filtering → fewer columns than 98.
        # But the loader should *attempt* all 98.
        # Verify >= 87 (since at least the kept 87 always have rows)
        assert df.shape[1] >= 87

    def test_enforce_continuous_trades_loads_87(self):
        """With enforce_thesis_continuous_trades=True, exactly 87 columns."""
        loader = DataLoader()
        df = loader.load_asset_data(
            asset_files=[FTSE_GLOB],
            date_range={"start": "2006-11-20", "end": "2012-12-31"},
            assets=[],
            enforce_thesis_continuous_trades=True,
        )
        assert df.shape[1] == 87, (
            f"expected 87 columns post-filter, got {df.shape[1]}"
        )

    def test_dropped_assets_absent_when_enforced(self):
        """The 11 dropped assets must not appear as columns when enforced."""
        loader = DataLoader()
        df = loader.load_asset_data(
            asset_files=[FTSE_GLOB],
            date_range={"start": "2006-11-20", "end": "2012-12-31"},
            assets=[],
            enforce_thesis_continuous_trades=True,
        )
        with open(_H4_ARTIFACT_PATH) as fh:
            artifact = json.load(fh)
        dropped_stems = {Path(d["file"]).stem for d in artifact["dropped_assets"]}
        col_set = set(df.columns)
        assert dropped_stems.isdisjoint(col_set), (
            f"dropped assets leaked through: "
            f"{dropped_stems & col_set}"
        )

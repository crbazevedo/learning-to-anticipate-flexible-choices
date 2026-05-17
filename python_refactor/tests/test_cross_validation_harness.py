"""
W18-1 cross-validation harness regression tests.

Tests the harness itself (fixtures + comparison framework), NOT any
specific cross-check. The cross-checks (A risk, B ROI, F TIP, ...)
get their own test files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import via the scripts package
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.fixtures import (
    build_risk_fixture,
    serialize_risk_fixture,
    deserialize_risk_fixture,
)
from scripts.cross_validation.compare import compare_csvs


class TestFixtureDeterminism:
    """Same seed → same fixture."""

    def test_same_seed_same_fixture(self):
        p1, r1 = build_risk_fixture(seed=42)
        p2, r2 = build_risk_fixture(seed=42)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        p1, _ = build_risk_fixture(seed=1)
        p2, _ = build_risk_fixture(seed=2)
        assert not np.array_equal(p1, p2)

    def test_portfolios_on_simplex(self):
        """Each portfolio row sums to ~1 (Dirichlet draw)."""
        portfolios, _ = build_risk_fixture(seed=42, n_portfolios=5, n_assets=10)
        sums = portfolios.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(5), atol=1e-12)
        assert np.all(portfolios >= 0.0)


class TestSerializationRoundtrip:
    """serialize → deserialize is identity."""

    def test_roundtrip(self):
        portfolios, returns = build_risk_fixture(seed=42, n_portfolios=3,
                                                    n_assets=5, n_days=10)
        text = serialize_risk_fixture(portfolios, returns)
        p2, r2 = deserialize_risk_fixture(text)
        np.testing.assert_allclose(portfolios, p2, atol=1e-15)
        np.testing.assert_allclose(returns, r2, atol=1e-15)


class TestCompareFramework:
    """compare_csvs flags violations vs accepts identical inputs."""

    def test_identical_csvs_agree(self, tmp_path: Path):
        df = pd.DataFrame({"portfolio_idx": [0, 1, 2],
                            "risk": [0.01, 0.02, 0.03]})
        cpp_path = tmp_path / "cpp.csv"
        py_path = tmp_path / "py.csv"
        df.to_csv(cpp_path, index=False)
        df.to_csv(py_path, index=False)

        report = compare_csvs(cpp_path, py_path, atol=1e-12)
        assert report["verdict"] == "AGREE"
        assert report["n_compared"] == 3
        assert report["per_column"]["risk"]["n_violations"] == 0

    def test_diverging_csvs_disagree(self, tmp_path: Path):
        cpp_df = pd.DataFrame({"portfolio_idx": [0, 1],
                                "risk": [0.01, 0.02]})
        py_df = pd.DataFrame({"portfolio_idx": [0, 1],
                                "risk": [0.01, 0.025]})  # diverges row 1
        cpp_path = tmp_path / "cpp.csv"
        py_path = tmp_path / "py.csv"
        cpp_df.to_csv(cpp_path, index=False)
        py_df.to_csv(py_path, index=False)

        report = compare_csvs(cpp_path, py_path, atol=1e-9)
        assert report["verdict"] == "DISAGREE"
        assert report["per_column"]["risk"]["n_violations"] == 1
        assert report["per_column"]["risk"]["abs_max"] == pytest.approx(0.005)

    def test_row_count_mismatch_flagged(self, tmp_path: Path):
        cpp_df = pd.DataFrame({"risk": [0.01, 0.02]})
        py_df = pd.DataFrame({"risk": [0.01, 0.02, 0.03]})
        cpp_path = tmp_path / "cpp.csv"
        py_path = tmp_path / "py.csv"
        cpp_df.to_csv(cpp_path, index=False)
        py_df.to_csv(py_path, index=False)

        report = compare_csvs(cpp_path, py_path)
        assert report["verdict"] == "DISAGREE"
        assert "row count differs" in report["markdown"]

    def test_scale_ratio_detected(self, tmp_path: Path):
        """1000x scale difference shows in scale_ratio."""
        cpp_df = pd.DataFrame({"risk": [0.001, 0.002, 0.003]})
        py_df = pd.DataFrame({"risk": [1.0, 2.0, 3.0]})  # 1000x bigger
        cpp_path = tmp_path / "cpp.csv"
        py_path = tmp_path / "py.csv"
        cpp_df.to_csv(cpp_path, index=False)
        py_df.to_csv(py_path, index=False)

        report = compare_csvs(cpp_path, py_path, atol=1e-9)
        assert report["verdict"] == "DISAGREE"
        scale = report["per_column"]["risk"]["scale_ratio_cpp_over_py"]
        assert 0.0005 < scale < 0.0015  # ≈ 1/1000

    def test_markdown_contains_mutual_skepticism_note(self, tmp_path: Path):
        df = pd.DataFrame({"risk": [0.01]})
        cpp_path = tmp_path / "cpp.csv"
        py_path = tmp_path / "py.csv"
        df.to_csv(cpp_path, index=False)
        df.to_csv(py_path, index=False)
        report = compare_csvs(cpp_path, py_path)
        assert "Mutual-skepticism" in report["markdown"]
        assert "neither side is treated as oracle" in report["markdown"]


class TestCPPBuildPrerequisite:
    """W18-1 prerequisite: C++ legacy must build (objs exist)."""

    def test_cpp_objs_exist(self):
        objs_dir = Path(__file__).resolve().parents[2] / "legacy-cpp" / "build" / "objs"
        if not objs_dir.exists():
            pytest.skip(
                "C++ build not yet run; cd legacy-cpp/build && make objs"
            )
        expected = {
            "aprendizado_operadores.o", "kalman_filter.o", "mvtnorm.o",
            "nsga2.o", "operadores_cruzamento.o", "operadores_mutacao.o",
            "operadores_selecao.o", "portfolio.o", "statistics.o", "utils.o",
        }
        actual = {p.name for p in objs_dir.glob("*.o")}
        assert expected.issubset(actual), (
            f"Missing C++ objects: {expected - actual}. "
            "Run `cd legacy-cpp/build && make objs`."
        )

"""
W18-2 cross-check A: risk computation parity tests.

Tests the C++ vs Python disagreement we documented in
docs/CROSS-VALIDATION-A-RISK.md (Python adds sqrt() not in thesis
Eq 7.4). The test verifies the disagreement is exactly the sqrt
relationship — if Python ever fixes the deviation, this test will
flip green-with-fix-in-place.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.fixtures import build_risk_fixture
from src.portfolio.portfolio import Portfolio


CPP_DRIVER = REPO_ROOT / "legacy-cpp" / "build" / "drivers" / "risk_driver"


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


class TestPythonRiskIsSqrtOfThesisFormula:
    """Documents the deviation: Python = sqrt(thesis variance)."""

    def test_python_compute_risk_is_sqrt_of_quadratic_form(self):
        """Python compute_risk = sqrt(u^T Σ u). Thesis says u^T Σ u alone."""
        rng = np.random.default_rng(42)
        n_assets = 10
        cov = rng.standard_normal((n_assets, n_assets))
        cov = cov @ cov.T  # PSD
        weights = rng.dirichlet(np.ones(n_assets))

        P = Portfolio(num_assets=n_assets)
        P.investment = weights

        python_risk = Portfolio.compute_risk(P, cov)
        thesis_eq_7_4_risk = float(weights @ cov @ weights)  # variance

        # The known deviation: Python = sqrt(thesis)
        assert python_risk == pytest.approx(np.sqrt(thesis_eq_7_4_risk), abs=1e-15)

    def test_python_compute_risk_handles_negative_variance(self):
        """The max(variance, 0.0) guard handles numerical degeneracy."""
        n_assets = 5
        # Construct a non-PSD pseudo-covariance to force negative
        cov = -np.eye(n_assets) * 1e-12  # tiny negative diagonal
        P = Portfolio(num_assets=n_assets)
        P.investment = np.full(n_assets, 1.0 / n_assets)
        risk = Portfolio.compute_risk(P, cov)
        # variance would be slightly negative; sqrt(max(., 0)) = 0
        assert risk >= 0.0


@pytest.mark.skipif(
    not CPP_DRIVER.exists(),
    reason=f"C++ risk_driver not built; cd legacy-cpp/build && make drivers/risk_driver"
)
class TestCPPvsPythonExecutionParity:
    """Run both drivers on the same fixture and verify exact sqrt relationship."""

    def test_cpp_outputs_variance_python_outputs_stddev(self, tmp_path: Path):
        # Generate fixture
        from scripts.cross_validation.fixtures import (
            serialize_risk_fixture, deserialize_risk_fixture,
        )
        portfolios, returns = build_risk_fixture(
            seed=42, n_portfolios=5, n_assets=10, n_days=30,
        )
        fixture_text = serialize_risk_fixture(portfolios, returns)

        # Run C++ driver
        cpp_proc = subprocess.run(
            [str(CPP_DRIVER)],
            input=fixture_text,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert cpp_proc.returncode == 0, f"C++ driver failed: {cpp_proc.stderr}"

        # Run Python driver in-process
        from io import StringIO
        from scripts.cross_validation.run_risk import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture_text), stream_out=py_out)

        # Parse
        cpp_df = pd.read_csv(StringIO(cpp_proc.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))

        assert len(cpp_df) == len(py_df) == 5

        # Verify Python = sqrt(C++) within 1e-12
        for i in range(len(cpp_df)):
            cpp_risk = cpp_df.risk.iloc[i]
            py_risk = py_df.risk.iloc[i]
            assert py_risk == pytest.approx(np.sqrt(cpp_risk), abs=1e-12), (
                f"Row {i}: cpp={cpp_risk}, py={py_risk}, "
                f"sqrt(cpp)={np.sqrt(cpp_risk)}, ratio={py_risk/np.sqrt(cpp_risk)}"
            )

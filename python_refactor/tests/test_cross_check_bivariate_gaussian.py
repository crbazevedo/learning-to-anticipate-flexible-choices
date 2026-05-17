"""W19-1: bivariate Gaussian inputs cross-check tests."""
from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.fixtures import build_risk_fixture, serialize_risk_fixture
from src.portfolio.portfolio import Portfolio


CPP_DRIVER = REPO_ROOT / "legacy-cpp" / "build" / "drivers" / "bivariate_gaussian_driver"


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m, prev_c = Portfolio.mean_ROI, Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.covariance = prev_c


class TestPythonEstimators:
    def test_mean_matches_numpy(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=(50, 5))
        mean = Portfolio.estimate_assets_mean_ROI(returns)
        np.testing.assert_allclose(mean, returns.mean(axis=0), atol=1e-15)

    def test_covariance_ddof_1(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=(50, 5))
        mean = Portfolio.estimate_assets_mean_ROI(returns)
        cov = Portfolio.estimate_covariance(mean, returns)
        expected = np.cov(returns.T, ddof=1)
        np.testing.assert_allclose(cov, expected, atol=1e-15)


@pytest.mark.skipif(not CPP_DRIVER.exists(), reason="bivariate_gaussian_driver not built")
class TestCPPvsPythonParity:
    def test_agree_machine_precision(self):
        portfolios, returns = build_risk_fixture(seed=42, n_portfolios=3, n_assets=5, n_days=20)
        fixture = serialize_risk_fixture(portfolios, returns)
        cpp = subprocess.run([str(CPP_DRIVER)], input=fixture, capture_output=True, text=True, timeout=10)
        assert cpp.returncode == 0

        from scripts.cross_validation.run_bivariate_gaussian import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture), stream_out=py_out)

        cpp_df = pd.read_csv(StringIO(cpp.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))
        assert cpp_df.shape == py_df.shape

        # Compare numeric columns row by row
        for col in [c for c in cpp_df.columns if c.startswith("c")]:
            np.testing.assert_allclose(cpp_df[col].to_numpy(),
                                          py_df[col].to_numpy(),
                                          atol=1e-12)

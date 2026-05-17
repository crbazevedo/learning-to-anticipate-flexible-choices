"""W18-3 cross-check B: ROI parity. AGREE within 1e-12."""
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

from scripts.cross_validation.fixtures import (
    build_risk_fixture,
    serialize_risk_fixture,
)
from src.portfolio.portfolio import Portfolio


CPP_DRIVER = REPO_ROOT / "legacy-cpp" / "build" / "drivers" / "roi_driver"


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m = Portfolio.mean_ROI
    prev_c = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.covariance = prev_c


class TestPythonROIMatchesThesisEq74:
    """Python compute_ROI = μ̂^T u verbatim per Eq (7.4)."""

    def test_basic_linear_combination(self):
        n_assets = 5
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n_assets))
        mean_roi_vec = rng.standard_normal(n_assets) * 0.001
        P = Portfolio(num_assets=n_assets)
        P.investment = weights
        py_roi = Portfolio.compute_ROI(P, mean_roi_vec)
        expected = float(weights @ mean_roi_vec)
        assert py_roi == pytest.approx(expected, abs=1e-15)


@pytest.mark.skipif(
    not CPP_DRIVER.exists(),
    reason="roi_driver not built; cd legacy-cpp/build && make drivers/roi_driver",
)
class TestCPPvsPythonROIParity:
    """W18-3 verdict: AGREE within 1e-12."""

    def test_parity_to_machine_precision(self):
        portfolios, returns = build_risk_fixture(
            seed=42, n_portfolios=5, n_assets=10, n_days=30,
        )
        fixture = serialize_risk_fixture(portfolios, returns)

        cpp_proc = subprocess.run(
            [str(CPP_DRIVER)], input=fixture,
            capture_output=True, text=True, timeout=10,
        )
        assert cpp_proc.returncode == 0, f"C++ failed: {cpp_proc.stderr}"

        from scripts.cross_validation.run_roi import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture), stream_out=py_out)

        cpp_df = pd.read_csv(StringIO(cpp_proc.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))
        assert len(cpp_df) == len(py_df) == 5

        for i in range(len(cpp_df)):
            cpp_roi = cpp_df.roi.iloc[i]
            py_roi = py_df.roi.iloc[i]
            assert py_roi == pytest.approx(cpp_roi, abs=1e-12), (
                f"Row {i}: cpp={cpp_roi}, py={py_roi}, diff={abs(cpp_roi - py_roi)}"
            )

"""
W17-5-CARRY-1 closure: integration test for end-to-end λ trace CSV
emit through ExperimentManager._run_algorithm.

The W17-5 production smoke set --lambda-trace-csv-path but the
CSV was not emitted. This test reproduces the path
(ExperimentManager._run_algorithm with a real algorithm + learner
+ data["lambda_trace_csv_path"]) at unit scale to isolate the bug
without paying the 600s walk-forward cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.experiment_manager import ExperimentManager
from src.portfolio.portfolio import Portfolio


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


def _build_minimal_data(tmp_path: Path,
                          lambda_trace_csv_path: str | None = None) -> dict:
    """Build a minimal data dict for ExperimentManager._run_algorithm."""
    rng = np.random.default_rng(42)
    n_days = 50
    n_assets = 5
    returns = rng.normal(loc=0.001, scale=0.01, size=(n_days, n_assets))
    df = pd.DataFrame(returns, columns=[f"a{i}" for i in range(n_assets)])
    data = {"assets": df, "market": pd.DataFrame(), "config": {}}
    if lambda_trace_csv_path is not None:
        data["lambda_trace_csv_path"] = str(lambda_trace_csv_path)
    return data


def _build_minimal_config(use_multi_horizon: bool = True) -> dict:
    """Mirror SCENARIOS' ASMS_mHDM_K3 config shape."""
    return {
        "experiment_name": "trace_emit_smoke",
        "algorithm": {
            "name": "sms_emoa",
            "parameters": {
                "population_size": 10,
                "generations": 3,
                # Keep tiny so the test is fast
            },
        },
        "learning": {
            "enabled": True,
            "use_multi_horizon": use_multi_horizon,
            "parameters": {
                "max_horizon": 2,
                "monte_carlo_samples": 50,
                "window_size": 3,  # K=3
            },
        },
        "data": {},
    }


class TestLambdaTraceCSVEmitsInProduction:
    """W17-5-CARRY-1: trace CSV must emit when path is set via data dict."""

    def test_trace_csv_emitted_when_path_set(self, tmp_path: Path):
        """End-to-end: _run_algorithm with lambda_trace_csv_path → CSV exists."""
        csv_path = tmp_path / "lambda_tip_trace.csv"
        suite_config = {
            "experiment_name": "trace_emit_smoke",
            "description": "W17-5-CARRY-1 reproducer",
            "version": "W17-5-CARRY-1",
            "timestamp": "2026-05-17T00:00:00Z",
        }
        mgr = ExperimentManager(suite_config)
        config = _build_minimal_config(use_multi_horizon=True)
        data = _build_minimal_data(tmp_path, lambda_trace_csv_path=str(csv_path))
        result = mgr._run_algorithm(config, data)
        # Algorithm ran (returned a result)
        assert "pareto_front" in result
        # CSV must exist
        assert csv_path.exists(), (
            f"lambda_tip_trace.csv was not emitted at {csv_path}; "
            f"W17-5-CARRY-1 reproducer fails — flush silently no-ops"
        )

    def test_trace_csv_not_emitted_when_path_missing(self, tmp_path: Path):
        """Default: no path → no CSV write attempted."""
        suite_config = {
            "experiment_name": "no_trace",
            "description": "negative case",
            "version": "W17-5-CARRY-1",
            "timestamp": "2026-05-17T00:00:00Z",
        }
        mgr = ExperimentManager(suite_config)
        config = _build_minimal_config(use_multi_horizon=True)
        data = _build_minimal_data(tmp_path, lambda_trace_csv_path=None)
        result = mgr._run_algorithm(config, data)
        assert "pareto_front" in result
        # No CSV files in tmp_path
        csvs = list(tmp_path.glob("*.csv"))
        assert not csvs, f"unexpected CSVs emitted: {csvs}"

    def test_csv_contents_satisfy_trace_assertions(self, tmp_path: Path):
        """W17-5-CARRY-1: assert the CSV content matches the W17-5 contract.

          - λ^H ∈ [0, 1]
          - λ^K ∈ [0, 1]
          - λ ≈ 0.5 * (λ^H + λ^K) within 1e-9 (Eq 7.16)
          - lambda_k_branch is one of {k0_zero, warmup_traditional, kperiod_sum}
        """
        import csv
        csv_path = tmp_path / "trace.csv"
        suite_config = {
            "experiment_name": "trace_content_assertions",
            "description": "W17-5-CARRY-1 contract test",
            "version": "W17-5-CARRY-1",
            "timestamp": "2026-05-17T00:00:00Z",
        }
        mgr = ExperimentManager(suite_config)
        config = _build_minimal_config(use_multi_horizon=True)
        data = _build_minimal_data(tmp_path, lambda_trace_csv_path=str(csv_path))
        mgr._run_algorithm(config, data)
        assert csv_path.exists()

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) >= 1, "expected at least 1 trace row"

        for r in rows:
            lh = float(r["lambda_h"])
            lk = float(r["lambda_k"])
            lam = float(r["lambda"])
            branch = r["lambda_k_branch"]
            assert 0.0 <= lh <= 1.0, f"λ^H out of range: {lh}"
            assert 0.0 <= lk <= 1.0, f"λ^K out of range: {lk}"
            assert lam == pytest.approx(0.5 * (lh + lk), abs=1e-9), (
                f"Eq 7.16 violated: λ={lam}, expected 0.5*(λ^H + λ^K)={0.5*(lh+lk)}"
            )
            assert branch in {"k0_zero", "warmup_traditional", "kperiod_sum"}, (
                f"unknown branch: {branch}"
            )

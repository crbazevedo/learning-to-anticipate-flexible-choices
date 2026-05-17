"""W18-2 cross-check A: Python driver for Portfolio.compute_risk.

Reads the W18-1 fixture format from stdin; computes risk for each
portfolio using BOTH:
  - the current Python implementation: sqrt(u^T Σ u)  (std-dev)
  - the verbatim-thesis variant:        u^T Σ u       (variance)
… and emits BOTH columns so the comparison can isolate the scale issue.

Usage:
    python -m scripts.cross_validation.run_risk < fixture.csv > py_out.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.fixtures import deserialize_risk_fixture
from src.portfolio.portfolio import Portfolio


def main(stream_in=sys.stdin, stream_out=sys.stdout):
    csv_text = stream_in.read()
    portfolios, returns = deserialize_risk_fixture(csv_text)
    n_assets = portfolios.shape[1]
    n_days = returns.shape[0]

    # Match the C++ driver: sample covariance with ddof=1
    mean_ROI = returns.mean(axis=0)
    centered = returns - mean_ROI
    cov = (centered.T @ centered) / (n_days - 1)

    # Set Portfolio class-state for the Python compute_risk
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = mean_ROI
    Portfolio.covariance = cov
    try:
        stream_out.write("portfolio_idx,risk\n")
        for p_idx in range(portfolios.shape[0]):
            P = Portfolio(num_assets=n_assets)
            P.investment = portfolios[p_idx].copy()
            # Python current implementation (sqrt of variance)
            risk_py = Portfolio.compute_risk(P, cov)
            stream_out.write(f"{p_idx},{risk_py:.17g}\n")
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


if __name__ == "__main__":
    main()

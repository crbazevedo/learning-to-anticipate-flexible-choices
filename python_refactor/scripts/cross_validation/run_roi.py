"""W18-3 cross-check B: Python driver for Portfolio.compute_ROI.

Reads W18-1 fixture from stdin; emits per-portfolio ROI to stdout.
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

    mean_ROI = returns.mean(axis=0)

    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = mean_ROI
    Portfolio.covariance = None
    try:
        stream_out.write("portfolio_idx,roi\n")
        for p_idx in range(portfolios.shape[0]):
            P = Portfolio(num_assets=n_assets)
            P.investment = portfolios[p_idx].copy()
            roi = Portfolio.compute_ROI(P, mean_ROI)
            stream_out.write(f"{p_idx},{roi:.17g}\n")
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


if __name__ == "__main__":
    main()

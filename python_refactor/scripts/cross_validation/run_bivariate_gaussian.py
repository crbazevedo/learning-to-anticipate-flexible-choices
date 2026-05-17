"""W19-1 Python driver: bivariate Gaussian (mean + covariance) inputs."""
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
    text = stream_in.read()
    portfolios, returns = deserialize_risk_fixture(text)
    n_assets = returns.shape[1]

    mean = Portfolio.estimate_assets_mean_ROI(returns)
    cov = Portfolio.estimate_covariance(mean, returns)

    # Emit same wide format as C++ driver
    header = ["row_type", "row_idx"] + [f"c{a}" for a in range(n_assets)]
    stream_out.write(",".join(header) + "\n")
    stream_out.write("mean,0," + ",".join(f"{x:.17g}" for x in mean) + "\n")
    for i in range(n_assets):
        row = [f"{cov[i, j]:.17g}" for j in range(n_assets)]
        stream_out.write(f"cov,{i}," + ",".join(row) + "\n")


if __name__ == "__main__":
    main()

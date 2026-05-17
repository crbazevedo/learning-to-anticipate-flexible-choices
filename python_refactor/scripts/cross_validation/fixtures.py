"""
W18-1 cross-validation fixtures — deterministic input generators.

Each fixture is fully specified by (seed, params) and reproducible.
Output format is plain CSV with stable header so the C++ driver and
the Python driver consume identical data.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import numpy as np


def build_risk_fixture(
    n_portfolios: int = 20,
    n_assets: int = 87,
    n_days: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """W18-2 fixture: portfolios + returns matrix.

    Returns (portfolios, returns) where:
      portfolios: shape (n_portfolios, n_assets), rows sum to 1
      returns:    shape (n_days, n_assets), daily returns ~ N(0.001, 0.01)

    The portfolios are random non-negative rows normalized to unit sum
    (Simplex points), the returns simulate small-cap daily returns
    with mean 0.1% and std 1%.
    """
    rng = np.random.default_rng(seed)
    portfolios = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    returns = rng.normal(loc=0.001, scale=0.01, size=(n_days, n_assets))
    return portfolios, returns


def serialize_risk_fixture(portfolios: np.ndarray, returns: np.ndarray) -> str:
    """Serialize fixture to CSV with header schema:
    HEADER: n_portfolios,n_assets,n_days
    PORTFOLIOS: one row per portfolio (n_assets cols)
    RETURNS: one row per day (n_assets cols)
    Sections separated by ###PORTFOLIOS / ###RETURNS markers.
    """
    out = StringIO()
    n_p, n_a = portfolios.shape
    n_d = returns.shape[0]
    out.write(f"###META\n{n_p},{n_a},{n_d}\n")
    out.write("###PORTFOLIOS\n")
    np.savetxt(out, portfolios, delimiter=",", fmt="%.17g")
    out.write("###RETURNS\n")
    np.savetxt(out, returns, delimiter=",", fmt="%.17g")
    return out.getvalue()


def deserialize_risk_fixture(csv_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of serialize_risk_fixture."""
    lines = csv_text.splitlines()
    # Find section markers
    i_meta = lines.index("###META")
    n_p, n_a, n_d = (int(x) for x in lines[i_meta + 1].split(","))
    i_port = lines.index("###PORTFOLIOS")
    i_ret = lines.index("###RETURNS")
    port_block = "\n".join(lines[i_port + 1 : i_ret])
    ret_block = "\n".join(lines[i_ret + 1 :])
    portfolios = np.loadtxt(StringIO(port_block), delimiter=",")
    returns = np.loadtxt(StringIO(ret_block), delimiter=",")
    if portfolios.ndim == 1:
        portfolios = portfolios.reshape(1, -1)
    if returns.ndim == 1:
        returns = returns.reshape(1, -1)
    return portfolios, returns


def write_risk_fixture_to(path: Path | str, **kwargs) -> Path:
    """Convenience: build + serialize + write fixture to path."""
    portfolios, returns = build_risk_fixture(**kwargs)
    csv_text = serialize_risk_fixture(portfolios, returns)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(csv_text)
    return p


if __name__ == "__main__":
    # CLI: emit a default fixture to stdout
    import sys
    portfolios, returns = build_risk_fixture()
    sys.stdout.write(serialize_risk_fixture(portfolios, returns))

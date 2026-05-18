"""
CLI: Generate synthetic PO(tau, eta) benchmark CSVs.

Produces one CSV per asset (Date,Close format) under
``python_refactor/data/synthetic-po-{tau}-{eta}/``. The output format
is identical to the per-asset files under ``data/ftse-updated/``
(Date,Open,High,Low,Close,Volume,Adj Close are accepted by the
DataLoader; we ship the minimal Date,Close pair which the loader
treats correctly via its ``Close``-only pct_change path).

Reproducibility: the same ``--seed`` must produce bit-identical CSVs.

Usage
-----
    cd python_refactor
    uv run python -m experiments.generate_synthetic_po \\
        --tau 8 --eta 1.0 --seed 42

References
----------
Implements Eqs 31-32 of the paper and Eqs 7.6-7.9 of the thesis
via :mod:`python_refactor.src.data.synthetic_po_generator`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python -m experiments.generate_synthetic_po`` and direct
# script invocation to both resolve src.data.* without depending on
# the parent package being installed.
_REPO_PY = Path(__file__).resolve().parents[1]
if str(_REPO_PY) not in sys.path:
    sys.path.insert(0, str(_REPO_PY))

from src.data.synthetic_po_generator import (  # noqa: E402
    POGeneratorConfig,
    generate_synthetic_po_returns,
    returns_to_close_prices,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic PO(tau, eta) benchmark CSVs.",
    )
    p.add_argument("--tau", type=int, default=8,
                   help="Disruption periodicity (periods between disruptive "
                        "regime changes). Default 8 (low frequency).")
    p.add_argument("--eta", type=float, default=1.0,
                   help="Disruption severity in [0, 1]. Default 1.0 "
                        "(maximum: new P fully replaces previous X).")
    p.add_argument("--d", type=int, default=30,
                   help="Number of synthetic assets. Default 30 "
                        "(thesis section 7.2.3).")
    p.add_argument("--T-periods", dest="T_periods", type=int, default=25,
                   help="Number of investment periods. Default 25 "
                        "(thesis section 7.2.3).")
    p.add_argument("--days-per-period", dest="days_per_period", type=int,
                   default=50,
                   help="Daily-return observations per period. Default 50 "
                        "(thesis page 145-146).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed. Same seed -> bit-identical output.")
    p.add_argument("--mu-lb", dest="mu_lb", type=float, default=-0.005,
                   help="Lower bound for uniform daily-return hyperbox.")
    p.add_argument("--mu-ub", dest="mu_ub", type=float, default=0.005,
                   help="Upper bound for uniform daily-return hyperbox.")
    p.add_argument("--sigma-lb", dest="sigma_lb", type=float, default=0.005,
                   help="Lower bound for per-asset standard deviation.")
    p.add_argument("--sigma-ub", dest="sigma_ub", type=float, default=0.04,
                   help="Upper bound for per-asset standard deviation.")
    p.add_argument("--start-date", dest="start_date", type=str,
                   default="2007-01-01",
                   help="Calendar start date for the synthetic "
                        "business-day index.")
    p.add_argument("--initial-price", dest="initial_price", type=float,
                   default=100.0,
                   help="Anchor close price applied to every asset at t=0.")
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                   help="Output directory. Defaults to "
                        "python_refactor/data/synthetic-po-{tau}-{eta}/")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg = POGeneratorConfig(
        d=args.d,
        T_periods=args.T_periods,
        days_per_period=args.days_per_period,
        tau=args.tau,
        eta=args.eta,
        mu_lb=args.mu_lb,
        mu_ub=args.mu_ub,
        sigma_lb=args.sigma_lb,
        sigma_ub=args.sigma_ub,
        seed=args.seed,
    )

    if args.out_dir is None:
        out_dir = (
            _REPO_PY / "data" / f"synthetic-po-{args.tau}-{args.eta}"
        )
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating PO(tau={cfg.tau}, eta={cfg.eta}) "
        f"with d={cfg.d}, T={cfg.T_periods}, "
        f"days/period={cfg.days_per_period}, seed={cfg.seed} ..."
    )
    returns = generate_synthetic_po_returns(cfg, start_date=args.start_date)
    prices = returns_to_close_prices(returns, initial_price=args.initial_price)

    print(
        f"  Returns shape : {returns.shape}\n"
        f"  Prices shape  : {prices.shape}\n"
        f"  Writing CSVs  -> {out_dir}"
    )

    # One CSV per asset: Date,Close (loader uses Close via pct_change).
    n_written = 0
    for asset in prices.columns:
        df = prices[[asset]].rename(columns={asset: "Close"})
        out_path = out_dir / f"{asset}.csv"
        df.to_csv(out_path, index=True, index_label="Date")
        n_written += 1

    print(f"  Wrote {n_written} per-asset CSV(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

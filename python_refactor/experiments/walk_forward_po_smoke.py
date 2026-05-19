"""W22 PO(8,1.0) walk-forward smoke — validates the breakthrough on
the paper's strongest-signal synthetic dataset.

Reads pre-generated PO(8,1.0) Close-price CSVs from
python_refactor/data/synthetic-po-8-1.0/asset_*.csv (30 assets, 1251
rows), converts to daily returns, then runs the same walk_forward
protocol as walk_forward_report.py.

Defaults to the breakthrough configuration:
- ASMS_mHDM_K3_v2both vs SMS_RDM_K0
- pop=20, gens=30 (the validated production stack)
- n_mc=200
- 5 seeds (extend via --seeds)

Usage:
    cd python_refactor
    uv run python -m experiments.walk_forward_po_smoke \\
        --seeds 1,2,3,4,5 \\
        --output ../results/po_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load_po_returns(po_dir: Path) -> pd.DataFrame:
    """Load all asset_*.csv from po_dir and convert prices → daily returns.

    Returns a DataFrame indexed by date, columns = asset names (asset_00..asset_29),
    values = simple daily returns (price_t / price_{t-1} - 1).
    """
    csvs = sorted(po_dir.glob("asset_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No asset_*.csv files in {po_dir}")
    asset_dfs = {}
    for csv in csvs:
        df = pd.read_csv(csv, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        asset_dfs[csv.stem] = df["Close"]
    prices = pd.DataFrame(asset_dfs)
    # Simple daily returns
    returns = prices.pct_change().dropna()
    return returns


def _run_one(scenario: str, seed: int,
              full_returns_pickle: bytes,
              train_window_days: int, step_days: int,
              n_mc: int) -> dict:
    """ProcessPoolExecutor entry: unpickle returns + run one
    (scenario, seed) walk-forward, return aggregate."""
    import pickle
    from experiments.walk_forward import (aggregate_per_seed, run_walk_forward)
    full_returns = pickle.loads(full_returns_pickle)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    per_period = run_walk_forward(
        scenario=scenario, seed=seed,
        full_returns=full_returns,
        train_window_days=train_window_days,
        step_days=step_days,
        n_mc=n_mc,
        rng=rng,
    )
    wall = time.time() - t0
    agg = aggregate_per_seed(per_period)
    n_ok = sum(1 for r in per_period if "error" not in r)
    n_total = len(per_period)
    print(
        f"[po-smoke] {scenario}/seed{seed}   {n_ok}/{n_total} periods ok  "
        f"grand_mean={agg['grand_mean']:.6f}  ({wall:.1f}s)",
        file=sys.stderr,
    )
    return {"scenario": scenario, "seed": seed, "wall": wall, **agg}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="ASMS_mHDM_K3_v2both,SMS_RDM_K0")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--train-window-days", type=int, default=378)
    parser.add_argument("--step-days", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--po-dir", type=Path,
                          default=Path("data/synthetic-po-8-1.0"))
    args = parser.parse_args(argv)

    scenarios = args.scenarios.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[po-smoke] loading PO data from {args.po_dir}…", file=sys.stderr)
    full_returns = _load_po_returns(args.po_dir)
    print(f"[po-smoke] returns shape: {full_returns.shape}", file=sys.stderr)

    import pickle
    full_returns_pickle = pickle.dumps(full_returns)
    pairs = [(s, sd) for s in scenarios for sd in seeds]
    print(f"[po-smoke] dispatching {len(pairs)} (scenario, seed) pairs over {args.jobs} workers…",
          file=sys.stderr)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_one, s, sd, full_returns_pickle,
                              args.train_window_days, args.step_days,
                              args.n_mc) for s, sd in pairs]
        for f in futs:
            results.append(f.result())
    wall = time.time() - t0
    print(f"[po-smoke] done in {wall:.1f}s", file=sys.stderr)

    # Aggregate
    lines = []
    lines.append("# Walk-Forward OOS Future Hypervolume — PO(8,1.0) synthetic data")
    lines.append("")
    lines.append(f"*PO data: {full_returns.shape[0]} days × {full_returns.shape[1]} assets*")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Train window: {args.train_window_days} days")
    lines.append(f"- Step: {args.step_days} days")
    lines.append(f"- MC scenarios per period (E): {args.n_mc}")
    lines.append(f"- Scenarios: {','.join(scenarios)}")
    lines.append(f"- Seeds: {len(seeds)}")
    lines.append(f"- Wall-clock: {wall:.1f}s")
    lines.append("")
    lines.append("## Aggregated results")
    lines.append("")
    lines.append("| scenario | n_seeds | grand mean Ŝ | std | median | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in scenarios:
        vals = [r["grand_mean"] for r in results
                if r["scenario"] == s and np.isfinite(r.get("grand_mean", float("nan")))]
        if vals:
            arr = np.array(vals)
            lines.append(
                f"| {s} | {len(vals)} | {arr.mean():.6e} | {arr.std(ddof=1) if len(arr) > 1 else 0.0:.6e} "
                f"| {np.median(arr):.6e} | {arr.min():.6e} | {arr.max():.6e} |"
            )
    output.write_text("\n".join(lines))
    print(f"[po-smoke] wrote results to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
W7-1 results aggregator.

Walks a results/ tree produced by validation_matrix.py and aggregates
per-seed metrics into summary statistics per (scenario, window, metric).

Expected input layout:
    results/
    └── <scenario>_<window>_seed<seed>/
        ├── metrics.csv     (long-format: scenario,window,seed,metric,value)
        └── manifest.json   (config + reproducibility receipts)

CLI:
    python -m experiments.aggregate_validation \\
        --input results/ \\
        --output docs/VALIDATION-SUMMARY.csv

Output: summary CSV with one row per (scenario, window, metric):
    scenario, window, metric, mean, std, n_seeds, seeds_used, min, max, median

Plus a JSON sibling for machine consumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def collect_metrics(input_root: Path) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    """Walk input_root and collect (scenario, window, metric) → [(seed, value), ...].

    Skips runs whose manifest.status is not 'ok' (excludes dry-runs and errors).
    """
    bucket: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    skipped = 0
    for run_dir in sorted(input_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "ok":
            skipped += 1
            continue
        if not metrics_path.exists():
            continue
        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["scenario"], row["window"], row["metric"])
                try:
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                bucket[key].append((int(row["seed"]), value))
    if skipped:
        print(f"[aggregate] skipped {skipped} non-ok runs", file=sys.stderr)
    return bucket


def summarise(bucket: dict[tuple[str, str, str], list[tuple[int, float]]]) -> list[dict]:
    """Aggregate per (scenario, window, metric) into summary rows."""
    rows: list[dict] = []
    for (scenario, window, metric), entries in sorted(bucket.items()):
        seeds = sorted(seed for seed, _ in entries)
        values = [v for _, v in entries]
        n = len(values)
        if n == 0:
            continue
        row = {
            "scenario": scenario,
            "window": window,
            "metric": metric,
            "n_seeds": n,
            "seeds_used": ",".join(str(s) for s in seeds),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if n >= 2 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    if not rows:
        output.write_text("scenario,window,metric,n_seeds,seeds_used,mean,std,median,min,max\n")
        return
    fieldnames = ["scenario", "window", "metric", "n_seeds", "seeds_used",
                  "mean", "std", "median", "min", "max"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], output: Path) -> None:
    output.write_text(json.dumps(rows, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W7-1 results aggregator — per-seed → summary statistics.",
    )
    parser.add_argument("--input", type=Path, required=True,
                         help="Root of the results/ tree from validation_matrix runs.")
    parser.add_argument("--output", type=Path, required=True,
                         help="Output summary CSV path (writes sibling .json too).")
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"[error] --input {args.input} is not a directory", file=sys.stderr)
        return 2

    bucket = collect_metrics(args.input)
    if not bucket:
        print(f"[aggregate] no metrics collected from {args.input}", file=sys.stderr)
        write_csv([], args.output)
        return 1

    rows = summarise(bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    write_json(rows, args.output.with_suffix(".json"))
    print(f"[aggregate] wrote {len(rows)} summary rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

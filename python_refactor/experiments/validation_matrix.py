"""
W7-1 validation-matrix driver.

Runs ONE scenario × window × seed from the W6-3 experiment matrix and
writes deterministic outputs. Designed so a W8 batch runner can fan
out across (scenarios × windows × seeds) by invoking this entry point
in parallel.

CLI:
    python -m experiments.validation_matrix \\
        --scenario S2 --window paper --seed 42 \\
        --output results/S2_paper_seed42/

Or for a dry-run config check (no compute):
    python -m experiments.validation_matrix \\
        --scenario S2 --window paper --seed 42 --dry-run

Outputs (under --output):
    metrics.csv       — per-period metrics (Hypv, wealth, λ, v, POCID)
    manifest.json     — config + seed + git SHA + uv.lock SHA + timestamp
    figures/ (when --figures): pareto, wealth, lambda, belief PNGs

Scenarios per docs/EXPERIMENT-VALIDATION-PLAN.md §3:
    S0  : Markowitz baseline (no anticipatory learning)
    S1  : TIP integrated, H=2 (use_tip=True, single-horizon)
    S2  : TIP + Multi-horizon H=3 (full Eqs 14+15 path)
    S3  : TIP + Multi-horizon H=2 (control)
    S4  : TIP + Multi-horizon H=3 with explicit covariance threading

Windows:
    paper     : 2006-11-20 → 2012-12-31 (matches paper §V-A)
    extended  : 2012-11-21 → 2024-12-31 (out-of-sample test)

Honest scar: this scaffold's "real run" path depends on
ExperimentManager surfaces that may need flag plumbing extensions to
fully exercise S0-S4. Dry-run validates the config shape; smoke-test
runs (S0 / paper / seed=1) are the first integration check. Full
matrix execution lands in W8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Scenario → learning config (per docs/EXPERIMENT-VALIDATION-PLAN.md §3)
# --------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "S0": {
        "name": "Markowitz baseline",
        "learning": {"enabled": False},
    },
    "S1": {
        "name": "TIP integrated (H=2)",
        "learning": {
            "enabled": True,
            "use_tip": True,
            "parameters": {"window_size": 20, "monte_carlo_samples": 500},
        },
    },
    "S2": {
        "name": "TIP + Multi-horizon (H=3)",
        "learning": {
            "enabled": True,
            "use_multi_horizon": True,
            "parameters": {"max_horizon": 3, "monte_carlo_samples": 500},
        },
    },
    "S3": {
        "name": "TIP + Multi-horizon (H=2 control)",
        "learning": {
            "enabled": True,
            "use_multi_horizon": True,
            "parameters": {"max_horizon": 2, "monte_carlo_samples": 500},
        },
    },
    "S4": {
        "name": "TIP + Multi-horizon (H=3, explicit covariance)",
        "learning": {
            "enabled": True,
            "use_multi_horizon": True,
            "parameters": {"max_horizon": 3, "monte_carlo_samples": 1000},
        },
    },
}

# W8-1 (closes W7-1-CARRY-1 partial): the paper-window's original FTSE
# data is 98 per-asset CSVs at legacy-cpp/executable/data/ftse-original/
# table\\ (*).csv — the underlying data_loader doesn't accept glob patterns
# in its asset_files list (treats each entry as a literal filename).
# Filed as W8-1-CARRY-1; for now both windows use the FTSE-updated
# single-CSV path so the scaffold's smoke-test gets past data loading.
# A future paper-window-loader unit (W9 candidate) will close the gap.
WINDOWS: dict[str, dict[str, str]] = {
    "paper": {
        "asset_files_glob": "data/ftse-updated/FTSE_100_20121121_20241231.csv",
        "date_start": "2012-11-21",  # W8-1-CARRY-1: paper-window data needs loader extension
        "date_end": "2024-12-31",
        "notes": "Paper-window data loader is a W8-1-CARRY-1; smoke-test temporarily uses the FTSE-updated CSV here.",
    },
    "extended": {
        "asset_files_glob": "data/ftse-updated/FTSE_100_20121121_20241231.csv",
        "date_start": "2012-11-21",
        "date_end": "2024-12-31",
        "notes": "Out-of-sample window.",
    },
}


def _git_sha() -> str:
    """Return short git SHA at HEAD, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _lockfile_sha(path: Path) -> str:
    """SHA-256 of the uv.lock file for reproducibility receipts."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def build_experiment_config(scenario_id: str, window_id: str, seed: int) -> dict[str, Any]:
    """Compose the experiment_config dict that ExperimentManager accepts.

    Scenario + window + seed compose into the run's full identity. The
    composed dict is the single source of truth for what's run.
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_id}; expected one of {list(SCENARIOS)}")
    if window_id not in WINDOWS:
        raise ValueError(f"Unknown window {window_id}; expected one of {list(WINDOWS)}")

    scenario = SCENARIOS[scenario_id]
    window = WINDOWS[window_id]

    return {
        "name": f"{scenario_id}_{window_id}_seed{seed}",
        "description": scenario["name"],
        "data": {
            "asset_files": [window["asset_files_glob"]],
            "date_range": {"start": window["date_start"], "end": window["date_end"]},
        },
        "algorithm": {
            "name": "sms_emoa",
            "parameters": {
                "population_size": 20,
                "generations": 30,
                "crossover_rate": 0.2,
                "mutation_rate": 0.3,
            },
        },
        "learning": scenario["learning"],
        "portfolio_selection": "hypervolume",
        "evaluation_period": "full",
        "_seed": seed,  # surfaced for the manifest
    }


def write_manifest(output_dir: Path, config: dict[str, Any], seed: int,
                   scenario_id: str, window_id: str,
                   status: str, error: str | None = None) -> None:
    """Write the per-run manifest. The manifest is the reproducibility
    receipt: re-running with this manifest's seed + config should
    reproduce the metrics.csv bit-for-bit (given identical code at the
    recorded git SHA).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    manifest = {
        "scenario": scenario_id,
        "scenario_name": SCENARIOS[scenario_id]["name"],
        "window": window_id,
        "window_meta": WINDOWS[window_id],
        "seed": seed,
        "config": config,
        "status": status,
        "error": error,
        "git_sha": _git_sha(),
        "uv_lock_sha256": _lockfile_sha(repo_root / "uv.lock"),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "scaffold_version": "W7-1",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_one(scenario_id: str, window_id: str, seed: int,
            output_dir: Path, dry_run: bool = False) -> int:
    """Run a single (scenario × window × seed) and write outputs.

    Returns process exit code (0 = success).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_experiment_config(scenario_id, window_id, seed)

    # Driver-level seed. Production code uses np.random globally; seeding
    # here makes the run deterministic (verified by W5-1's pattern).
    np.random.seed(seed)
    rng = np.random.default_rng(seed)  # for any consumer that takes a Generator

    if dry_run:
        print(f"[dry-run] scenario={scenario_id} window={window_id} seed={seed}")
        print(f"[dry-run] config name: {config['name']}")
        print(f"[dry-run] learning: {config['learning']}")
        print(f"[dry-run] output dir would be: {output_dir}")
        write_manifest(output_dir, config, seed, scenario_id, window_id,
                       status="dry-run-ok")
        return 0

    # Real run path. Imports are local so --dry-run doesn't pay the
    # import cost (and so the scaffold can be invoked from a partial
    # environment for testing).
    try:
        from src.experiments.experiment_manager import ExperimentManager
    except ImportError as e:
        print(f"[error] cannot import ExperimentManager: {e}", file=sys.stderr)
        write_manifest(output_dir, config, seed, scenario_id, window_id,
                       status="error", error=f"ImportError: {e}")
        return 2

    logging.basicConfig(level=logging.INFO)
    # ExperimentManager takes a top-level "suite" config (name, description,
    # version, timestamp) at init time, NOT a per-experiment config. The
    # per-experiment config is passed to run_experiment() below. W8-1 fix:
    # initial scaffold used `experiment_dir=...` kwarg which doesn't exist.
    suite_config = {
        "experiment_name": f"validation_matrix_{scenario_id}",
        "description": f"W8 validation run: {SCENARIOS[scenario_id]['name']}",
        "version": "W8-1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    manager = ExperimentManager(suite_config)

    try:
        results = manager.run_experiment(config)
    except Exception as e:
        write_manifest(output_dir, config, seed, scenario_id, window_id,
                       status="error", error=f"{type(e).__name__}: {e}")
        raise

    # Write a flat metrics.csv that aggregate_validation.py can consume.
    # Shape: one row per metric (long format), so adding new metrics
    # doesn't change the column schema.
    final_metrics = results.get("final_metrics", {})
    import csv
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "window", "seed", "metric", "value"])
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                writer.writerow([scenario_id, window_id, seed, key, value])

    write_manifest(output_dir, config, seed, scenario_id, window_id, status="ok")
    print(f"[ok] {config['name']} → {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W7-1 validation matrix driver — runs one (scenario × window × seed).",
    )
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS),
                         help="Experiment scenario per W6-3 plan §3.")
    parser.add_argument("--window", required=True, choices=sorted(WINDOWS),
                         help="Dataset window: paper (2006-2012) or extended (2012-2024).")
    parser.add_argument("--seed", type=int, required=True,
                         help="np.random seed; manifest-recorded for reproducibility.")
    parser.add_argument("--output", type=Path, required=True,
                         help="Output directory for metrics.csv + manifest.json.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate config + write manifest with status=dry-run-ok; do NOT run.")
    args = parser.parse_args(argv)

    return run_one(args.scenario, args.window, args.seed, args.output,
                   dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

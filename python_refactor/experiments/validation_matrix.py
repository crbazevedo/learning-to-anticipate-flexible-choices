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

# --------------------------------------------------------------------------
# W15-3 (BACKLOG B2 + H5): SCENARIOS re-keyed to thesis factorial
# {ASMS, SMS} × {mHDM, RDM} × K ∈ {0, 1, 2, 3} per §7.1.1 (p. 140) and
# Fig 7.15 (p. 157). Anticipation horizon H = 2 FIXED per §7.2.3 Eq 7.16
# (p. 146): "λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K)), for which the
# anticipation horizon is H = 2 (one-step-ahead prediction)."
#
# K is the OAL historical-window size (drives λ^K — the squared-KF-
# residual arm of Eq 7.16). For K = 0, the algorithm has no historical
# window, all anticipation rates collapse to λ = 1, and the result is
# equivalent to the myopic SMS baseline (thesis §7.1.1 verbatim).
#
# Legacy S0..S4 aliases preserved so W12/W13/W14 reports + the
# differentiation regression gate at tests/test_scenario_differentiation.py
# remain runnable. S0 ≡ SMS_RDM_K0 (myopic baseline); S2 ≡ ASMS_mHDM_K3
# (paper headline configuration).

_BASELINE_MYOPIC = {"enabled": False}


def _asms_learning_config(K: int, use_multi_horizon: bool = True,
                            use_v2_anticipative_rate: bool = False,
                            max_horizon: int = 2) -> dict:
    """Construct the AnticipatoryLearning constructor kwargs for K > 0.

    H semantics (verified W21-1 code-read):
      - max_horizon=2 → MultiHorizonAnticipatoryLearning loops `for h in
        range(1, 2)` = [1] → single one-step-ahead prediction. Matches
        thesis Eq 7.16 denominator (H-1)=1.
      - max_horizon=3 → loops [1, 2] → one-step + two-step recursive
        prediction via n_step_prediction.kalman_n_step_prediction.

    W20-1 / Reading-E: when use_v2_anticipative_rate=True, the learner
    instance uses v2's monotonic α = 1 - TIP formula (legacy-cpp-v2/
    asms_emoa.cpp:44) instead of Python's default thesis Eq 7.16
    (1/2)(λ^H + λ^K).
    """
    return {
        "enabled": True,
        "use_multi_horizon": use_multi_horizon,
        "parameters": {
            "window_size": K,        # K = OAL historical window
            "max_horizon": max_horizon,  # H per thesis Eq 7.16 (H-1 future steps)
            # NC6 FIX (W22 agent finding 2026-05-18): thesis §7.2.3 p.146
            # specifies "E = 1000, i.e., the risk and return of each
            # candidate portfolio is evaluated from 1000 simulated daily
            # return values." Pre-fix default was 500 (also wrong;
            # validation_matrix overrode to 200 via --n-mc CLI).
            "monte_carlo_samples": 1000,
            "use_v2_anticipative_rate": use_v2_anticipative_rate,  # W20-1 / Reading-E
        },
    }


SCENARIOS: dict[str, dict[str, Any]] = {
    # ─── Thesis-faithful factorial (Fig 7.15 framing) ────────────────────
    "SMS_RDM_K0": {
        "name": "SMS/RDM K=0 — myopic baseline + random DM",
        "learning": _BASELINE_MYOPIC,
        "dm": "RDM",
    },
    # W22 Probe H: pop=30, gens=40 mid-step (thesis spec is pop=200, gens=500)
    "SMS_RDM_K0_pop30gen40": {
        "name": "SMS/RDM K=0 with pop=30, gens=40 (Probe H param sweep)",
        "learning": _BASELINE_MYOPIC,
        "dm": "RDM",
        "algorithm_overrides": {"population_size": 30, "generations": 40},
    },
    "SMS_mHDM_K0": {
        "name": "SMS/mHDM K=0 — myopic baseline + max-Hypv DM",
        "learning": _BASELINE_MYOPIC,
        "dm": "mHDM",
    },
    "ASMS_RDM_K1": {
        "name": "ASMS/RDM K=1 — anticipatory + random DM, 1-period window",
        "learning": _asms_learning_config(K=1),
        "dm": "RDM",
    },
    "ASMS_RDM_K2": {
        "name": "ASMS/RDM K=2",
        "learning": _asms_learning_config(K=2),
        "dm": "RDM",
    },
    "ASMS_RDM_K3": {
        "name": "ASMS/RDM K=3",
        "learning": _asms_learning_config(K=3),
        "dm": "RDM",
    },
    "ASMS_mHDM_K1": {
        "name": "ASMS/mHDM K=1 — anticipatory + max-Hypv DM, 1-period window",
        "learning": _asms_learning_config(K=1),
        "dm": "mHDM",
    },
    "ASMS_mHDM_K2": {
        "name": "ASMS/mHDM K=2",
        "learning": _asms_learning_config(K=2),
        "dm": "mHDM",
    },
    # W22 (operator 2026-05-18): thesis §7.2.3 may specify K=2 historical
    # periods for the FTSE study (not K=3 as W20-1 / W21-1 used). Add
    # K=2 + v2_rate variants to test under Option A closed-form.
    "ASMS_mHDM_K2_v2rate": {
        "name": "ASMS/mHDM K=2 + v2 anticipative-rate (Reading-E, K=2 thesis-faithful)",
        "learning": _asms_learning_config(K=2, use_v2_anticipative_rate=True),
        "dm": "mHDM",
    },
    "ASMS_mHDM_K2_v2both": {
        "name": "ASMS/mHDM K=2 + v2 rate + v2 stability (Reading-E+F-INV at K=2)",
        "learning": _asms_learning_config(K=2, use_v2_anticipative_rate=True),
        "algorithm_overrides": {"use_v2_stability_weighting": True},
        "dm": "mHDM",
    },
    "ASMS_mHDM_K3": {
        "name": "ASMS/mHDM K=3 — PAPER HEADLINE configuration (Fig 7.15)",
        "learning": _asms_learning_config(K=3),
        "dm": "mHDM",
    },
    # W20-1 / Reading-E experimental variant: identical to ASMS_mHDM_K3
    # but with v2 monotonic anticipative-rate formula (α = 1 - TIP) instead
    # of Python's thesis-Eq-7.16 (1/2)(λ^H + λ^K). Per W19-4 finding, v2's
    # formula maintains anticipation at 0.5 in the W17-5 saturation regime
    # (TIP≈0.5) where Python's collapses to 0. If S2_v2rate > S0, Reading E
    # is confirmed and the paper headline replicates with formula adjustment.
    "ASMS_mHDM_K3_v2rate": {
        "name": "ASMS/mHDM K=3 + v2 anticipative-rate formula (Reading-E experiment)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "dm": "mHDM",
    },
    # W21-1 / Reading-F experimental variant (INVERTED per W20-5 correction):
    # identical to ASMS_mHDM_K3_v2rate but ALSO forces SMS-EMOA's stability
    # multiplier to be the v2 effective no-op (1.0) instead of Python's
    # depressing < 1.0 default. v2's stability is declared at asms_emoa.h:18
    # and initialized to 1.0 in 3 constructors but NEVER reassigned anywhere
    # in legacy-cpp-v2/source/*.cpp; the `delta_Si *= stability` lines in
    # asms_emoa.cpp are effective no-ops. Python's
    # `1/(1+pred_error)` and `1/(1+std(weights))` actively depress Δ_S
    # below the bare Gaussian expectation, making ASMS-favorable solutions
    # look less attractive than v2's effective behavior. If combined gap
    # closes ≤ 0, Reading F is confirmed.
    "ASMS_mHDM_K3_v2both": {
        "name": "ASMS/mHDM K=3 + v2 rate AND v2 stability (Reading-F combined experiment)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "algorithm_overrides": {"use_v2_stability_weighting": True},
        "dm": "mHDM",
    },
    # W22 Probe H: pop=30, gens=40 mid-step toward thesis spec (pop=200, gens=500)
    "ASMS_mHDM_K3_v2both_pop30gen40": {
        "name": "ASMS/mHDM K=3 v2both with pop=30, gens=40 (Probe H param sweep)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "algorithm_overrides": {
            "use_v2_stability_weighting": True,
            "population_size": 30,
            "generations": 40,
        },
        "dm": "mHDM",
    },
    # W21-1 / Reading-F isolated variant: v2 stability ONLY (no v2 rate).
    # Isolates the contribution of disabling Python's stability multiplier
    # vs the dominant Reading-E v2-rate effect. For the W21-5 ablation matrix.
    "ASMS_mHDM_K3_v2stab": {
        "name": "ASMS/mHDM K=3 + v2 stability (Reading-F isolated experiment)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=False),
        "algorithm_overrides": {"use_v2_stability_weighting": True},
        "dm": "mHDM",
    },
    # ─── W21-5 ablation matrix scenarios (V5/V6/V7) ─────────────────────
    # V5: v2_rate + sqrt removed (W18-CARRY-1 / Reading-A risk-scale fix).
    # Tests whether returning bare variance per thesis Eq 7.4 (instead of
    # std-dev) closes residual gap after Reading E.
    "ASMS_mHDM_K3_v2rate_noSqrt": {
        "name": "ASMS/mHDM K=3 + v2 rate + sqrt-removed (W21-5 V5)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "algorithm_overrides": {"use_thesis_eq74_risk": True},
        "dm": "mHDM",
    },
    # V6 stub: v2_rate + KF lifecycle flag (full impl deferred). Smoke
    # exists as control; expected to behave identically to v2_rate alone
    # until V6 impl lands.
    "ASMS_mHDM_K3_v2rate_kfV6": {
        "name": "ASMS/mHDM K=3 + v2 rate + KF lifecycle flag (W21-5 V6 stub)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "algorithm_overrides": {"use_v2_kf_lifecycle": True},
        "dm": "mHDM",
    },
    # V7: v2_rate + v2 entropy operators (raise/lower) in mutation suite.
    # Tests whether adding the 2 non-thesis mutation operators (W21-3)
    # closes residual gap.
    "ASMS_mHDM_K3_v2rate_v2entropy": {
        "name": "ASMS/mHDM K=3 + v2 rate + v2 entropy ops (W21-5 V7)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True),
        "algorithm_overrides": {"use_v2_entropy_operators": True},
        "dm": "mHDM",
    },
    # H=3 variant: same as ASMS_mHDM_K3 default but with max_horizon=3
    # (genuine two-step-ahead via recursive KF prediction). Tests whether
    # the paper headline (which may have used H>2) is recoverable with
    # additional anticipation depth.
    "ASMS_mHDM_K3_H3": {
        "name": "ASMS/mHDM K=3 + H=3 two-step-ahead (vs default H=2 one-step)",
        "learning": _asms_learning_config(K=3, max_horizon=3),
        "dm": "mHDM",
    },
    # H=3 with v2_rate: combined two-step horizon + v2 monotonic rate.
    "ASMS_mHDM_K3_H3_v2rate": {
        "name": "ASMS/mHDM K=3 + H=3 + v2 rate (two-step Reading-E)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True, max_horizon=3),
        "dm": "mHDM",
    },
    # W21-5 Phase C kitchen-sink variant: ALL W22 ablation flags ON
    # simultaneously. Tests whether the COMBINED effect of all closed
    # divergences reverses the SMS-favorable direction.
    "ASMS_mHDM_K3_v2_kitchen_sink": {
        "name": "ASMS/mHDM K=3 + ALL W22 ablation flags (kitchen sink for Phase C)",
        "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True, max_horizon=3),
        "algorithm_overrides": {
            "use_v2_stability_weighting": True,
            "use_thesis_eq74_risk": True,
            "use_v2_kf_lifecycle": True,
            "use_v2_entropy_operators": True,
        },
        "dm": "mHDM",
    },
    # ─── Legacy aliases for backward compat with W12-W14 reports ─────────
    "S0": {
        "name": "[legacy alias for SMS_RDM_K0] Markowitz baseline",
        "learning": _BASELINE_MYOPIC,
        "dm": "RDM",
    },
    "S1": {
        "name": "[legacy alias] TIP integrated (H=2 was misnomer; now K=1 ASMS/RDM)",
        "learning": _asms_learning_config(K=1, use_multi_horizon=False),
        "dm": "RDM",
    },
    "S2": {
        "name": "[legacy alias for ASMS_mHDM_K3] PAPER HEADLINE",
        "learning": _asms_learning_config(K=3),
        "dm": "mHDM",
    },
    "S3": {
        "name": "[legacy alias] horizon-ablation control (now ASMS_mHDM_K2)",
        "learning": _asms_learning_config(K=2),
        "dm": "mHDM",
    },
    "S4": {
        "name": "[legacy alias] explicit-cov ablation (now ASMS_mHDM_K3 + 1000 MC)",
        "learning": {**_asms_learning_config(K=3),
                      "parameters": {**_asms_learning_config(K=3)["parameters"],
                                       "monte_carlo_samples": 1000}},
        "dm": "mHDM",
    },
}

# W15-3 (BACKLOG H3): paper-window date_range narrowed to thesis
# §7.2.3 p. 145: "The real-world scenarios are composed of daily
# adjusted close prices collected between 20/11/2006 – 31/12/2012".
# Pre-W15-3 the range was 2003-01-01 → 2012-11-20 (a SUPERSET of the
# thesis range; produced ~44 rolling periods instead of the thesis's
# T=24 for FTSE).
#
# `legacy_paper_2003_2012` is kept as an alias for the pre-W15-3 range
# to preserve backward compat with the W12/W13/W14 reports that used it.
WINDOWS: dict[str, dict[str, str]] = {
    "paper": {
        "asset_files_glob": "../legacy-cpp/executable/data/ftse-original/table*.csv",
        "date_start": "2006-11-20",
        "date_end": "2012-12-31",
        "notes": "Paper window (thesis §7.2.3 p. 145): FTSE daily 20/11/2006 – 31/12/2012, T=24 rolling 50-day-shift periods, d=87 (thesis) vs 98 (our archive — BACKLOG H4).",
    },
    "legacy_paper_2003_2012": {
        "asset_files_glob": "../legacy-cpp/executable/data/ftse-original/table*.csv",
        "date_start": "2003-01-01",
        "date_end": "2012-11-20",
        "notes": "[legacy alias] pre-W15-3 paper range (superset of thesis; ~44 rolling periods). Kept for W12-W14 report reproducibility.",
    },
    "extended": {
        "asset_files_glob": "data/ftse-updated/FTSE_100_20121121_20241231.csv",
        "date_start": "2012-11-21",
        "date_end": "2024-12-31",
        "notes": "Out-of-sample window: FTSE-updated single-CSV, 2012-2024.",
    },
}


def _flatten_metrics(d: dict, prefix: str = "") -> "list[tuple[str, float]]":
    """Recursively flatten a nested metrics dict into [(dotted-key, value), ...].

    W10-1 helper: ExperimentManager returns final_metrics shaped like
    {'algorithm': {...nums...}, 'portfolio': {...nums...}, 'summary': {...}}.
    The pre-W10-1 metrics-writer filtered on top-level scalars and got 0
    rows. This recursion emits 'portfolio.final_value', 'summary.total_return',
    etc., so the analytics layer can consume them.

    Booleans are skipped: in Python `True/False` are int subclasses but
    don't belong in a numeric metric column. None values are skipped.
    """
    out: list[tuple[str, float]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_metrics(v, key))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float)):
            out.append((key, float(v)))
    return out


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

    # W21-1: scenarios may opt-in to extra SMS-EMOA algorithm parameters via
    # the "algorithm_overrides" key (e.g. use_v2_stability_weighting=True
    # for Reading-F experimental test). Default-empty dict merges harmlessly.
    algorithm_overrides = scenario.get("algorithm_overrides", {})
    algorithm_params = {
        "population_size": 20,
        "generations": 30,
        "crossover_rate": 0.2,
        "mutation_rate": 0.3,
        **algorithm_overrides,
    }

    return {
        "name": f"{scenario_id}_{window_id}_seed{seed}",
        "description": scenario["name"],
        "data": {
            "asset_files": [window["asset_files_glob"]],
            "date_range": {"start": window["date_start"], "end": window["date_end"]},
        },
        "algorithm": {
            "name": "sms_emoa",
            "parameters": algorithm_params,
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

    # W10-1 (closes W9-CARRY-3): final_metrics is a NESTED dict
    # ({'algorithm': {...}, 'portfolio': {...}, 'summary': {...}, ...}).
    # The pre-W10-1 writer filtered isinstance(value, (int, float)) on
    # the top level → every value was a sub-dict → 0 rows emitted.
    # Recurse into '<category>.<metric>' rows so the analytics layer
    # has something to chew on. Booleans are explicitly excluded (in
    # Python `True/False` are int subclasses but shouldn't land in a
    # numeric metric column).
    final_metrics = results.get("final_metrics", {})
    import csv
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "window", "seed", "metric", "value"])
        for key, value in _flatten_metrics(final_metrics):
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

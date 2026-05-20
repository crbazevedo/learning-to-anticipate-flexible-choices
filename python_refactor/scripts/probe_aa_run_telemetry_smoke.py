"""W22 Probe AA — CLI smoke runner for the AMFC telemetry analyzer.

Generates SYNTHETIC populations (no real ASMS run; pure synthetic), drives
``select_amfc(..., collect_telemetry=True)`` for ``--num-periods`` periods,
flushes the telemetry, prints the markdown summary, and saves the raw
telemetry to ``results/probe_aa_telemetry_{seed}.json``.

Usage (from repo root):
    uv run python python_refactor/scripts/probe_aa_run_telemetry_smoke.py \
        --num-periods 30 --num-solutions-per-period 6 --seed 42

This is a SMOKE driver only — production ASMS wiring is the operator's
decision per the W22 Probe AA contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the sibling src/ tree importable when running as a plain script.
_HERE = Path(__file__).resolve().parent
_REFACTOR_ROOT = _HERE.parent
sys.path.insert(0, str(_REFACTOR_ROOT))

from src.algorithms.amfc_selector import (  # noqa: E402
    get_amfc_telemetry,
    reset_amfc_telemetry,
    select_amfc,
)
from src.probes.probe_aa_amfc_telemetry_analyzer import (  # noqa: E402
    format_summary_markdown,
    save_telemetry_to_json,
    summarize_telemetry,
)


class _SmokeKalmanState:
    """Minimal KF state that satisfies amfc_selector's reader."""

    def __init__(self, x: np.ndarray, P: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.F = np.eye(len(self.x))


class _SmokePortfolio:
    def __init__(self, roi: float, risk: float, kalman_state) -> None:
        self.ROI = float(roi)
        self.risk = float(risk)
        self.kalman_state = kalman_state


class _SmokeSolution:
    """Minimal Solution-shaped duck-type for select_amfc()."""

    def __init__(
        self,
        roi: float,
        risk: float,
        kalman_state,
        delta_s: float,
        pareto_rank: int = 0,
    ) -> None:
        self.P = _SmokePortfolio(roi, risk, kalman_state)
        self.Delta_S = float(delta_s)
        self.Pareto_rank = int(pareto_rank)


def _generate_synthetic_population(
    n_solutions: int, rng: np.random.Generator
) -> list[_SmokeSolution]:
    """Generate a synthetic Pareto-shaped population of n solutions.

    ROI values are spaced ascending; risk values descend so the front is
    non-dominated. Per-solution forecast variance varies (some certain,
    some uncertain) so the analyzer can discriminate.
    """
    rois = np.linspace(0.001, 0.005, n_solutions)
    risks = np.linspace(0.020, 0.006, n_solutions)
    # Per-solution variance scale: random in [1e-8, 1e-4]; means AMFC will
    # pick low-variance candidates more often than chance.
    var_scales = rng.uniform(1e-8, 1e-4, size=n_solutions)
    sols: list[_SmokeSolution] = []
    for r, rk, vs in zip(rois, risks, var_scales):
        x = np.array([r, rk, 0.0, 0.0])
        P = vs * np.eye(4)
        kf = _SmokeKalmanState(x, P)
        # Delta_S proxy: a noisy current-period rectangle (gives Hv-DM
        # something non-trivial to argmax over).
        delta_s = float(rng.uniform(0.5, 2.0) * r * (0.05 - rk))
        sols.append(_SmokeSolution(r, rk, kf, delta_s))
    return sols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W22 Probe AA — synthetic smoke driver for AMFC telemetry."
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        default=20,
        help="number of synthetic periods (one select_amfc call per period)",
    )
    parser.add_argument(
        "--num-solutions-per-period",
        type=int,
        default=6,
        help="population size per period",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for the numpy Generator (also tags the output JSON)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="output directory (relative to repo root, created if missing)",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    reset_amfc_telemetry()

    for _ in range(args.num_periods):
        pop = _generate_synthetic_population(args.num_solutions_per_period, rng)
        # Pass an independent Generator so AMFC's own MC (if used) is
        # reproducible per-period.
        amfc_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        select_amfc(
            pop,
            horizon=1,
            n_mc=200,
            pareto_only=True,
            rng=amfc_rng,
            collect_telemetry=True,
        )

    telemetry = get_amfc_telemetry()
    summary = summarize_telemetry(telemetry)

    print(format_summary_markdown(summary))

    repo_root = _REFACTOR_ROOT.parent
    results_dir = repo_root / args.results_dir
    out_path = results_dir / f"probe_aa_telemetry_{args.seed}.json"
    save_telemetry_to_json(telemetry, out_path)
    print(f"\nSaved {len(telemetry)} telemetry records → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

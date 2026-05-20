"""W22 Probe Z — stability factor restoration sensitivity (Reading-F revisited).

Hypothesis: Reading-F's flip from per-solution stability_factor =
1/(1 + trace(P)) to stability_factor = 1.0 (use_v2_stability_weighting=True)
lost a load-bearing signal. Quantify, on synthetic Pareto fronts with
known KF traces, how often the argmax flips between the two modes.

Per docs/W22-MASTER-BACKLOG.md Section H. Standalone analyzer; pure synthetic.
NO modifications to shared code paths.

Note: in production code, the DEFAULT is use_v2_stability_weighting=False
(legacy stability factor IS applied). v2 mode is OPT-IN. This probe
characterizes the difference for the operator deciding which to ratify.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def stability_factor_legacy(trace_P: float) -> float:
    """Per-solution stability: 1 / (1 + trace(P))."""
    return 1.0 / (1.0 + max(0.0, trace_P))


def stability_factor_v2() -> float:
    """Reading-F: constant 1.0, ignores KF uncertainty."""
    return 1.0


def apply_stability_legacy(delta_s_values: np.ndarray,
                             trace_Ps: np.ndarray) -> np.ndarray:
    """Multiply each Δ_S by its per-solution stability factor."""
    return delta_s_values * np.array([stability_factor_legacy(t) for t in trace_Ps])


def apply_stability_v2(delta_s_values: np.ndarray,
                         trace_Ps: np.ndarray) -> np.ndarray:
    """Ignore stability (multiply by 1)."""
    return delta_s_values * 1.0


def argmax_legacy(delta_s_values: np.ndarray, trace_Ps: np.ndarray) -> int:
    """Returns winner under legacy stability factor."""
    return int(np.argmax(apply_stability_legacy(delta_s_values, trace_Ps)))


def argmax_v2(delta_s_values: np.ndarray, trace_Ps: np.ndarray) -> int:
    """Returns winner under v2 mode (no stability multiplier)."""
    return int(np.argmax(delta_s_values))


def disagreement_rate(delta_s_samples: np.ndarray,
                        trace_P_samples: np.ndarray) -> float:
    """Run many synthetic comparisons; return fraction where legacy ≠ v2 picks.

    Args:
        delta_s_samples: (n_runs, n_solutions) per-solution Δ_S values
        trace_P_samples: (n_runs, n_solutions) per-solution KF trace values

    Returns:
        Disagreement rate ∈ [0, 1].
    """
    n_runs = delta_s_samples.shape[0]
    n_disagree = 0
    for j in range(n_runs):
        if argmax_legacy(delta_s_samples[j], trace_P_samples[j]) != \
           argmax_v2(delta_s_samples[j], trace_P_samples[j]):
            n_disagree += 1
    return n_disagree / n_runs


def synthesize_population(n_solutions: int, mean_delta_s: float = 1e-4,
                            spread_delta_s: float = 5e-5,
                            mean_trace_P: float = 0.5,
                            spread_trace_P: float = 0.5,
                            rng: Optional[np.random.Generator] = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize per-solution (Δ_S, trace(P)) for one population snapshot."""
    if rng is None:
        rng = np.random.default_rng()
    delta_s = rng.normal(mean_delta_s, spread_delta_s, size=n_solutions)
    delta_s = np.maximum(delta_s, 1e-6)  # keep positive
    trace_P = np.abs(rng.normal(mean_trace_P, spread_trace_P, size=n_solutions))
    return delta_s, trace_P


def sweep_disagreement_vs_trace_spread(n_runs: int = 100,
                                          n_solutions: int = 10,
                                          mean_delta_s: float = 1e-4,
                                          spread_delta_s: float = 5e-5,
                                          mean_trace_P: float = 0.5,
                                          trace_spreads: Optional[list[float]] = None,
                                          rng: Optional[np.random.Generator] = None
                                          ) -> dict[float, float]:
    """For each trace_spread, compute disagreement rate over n_runs synthetic
    populations. Returns dict {trace_spread: rate}."""
    if rng is None:
        rng = np.random.default_rng(0)
    if trace_spreads is None:
        trace_spreads = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    out = {}
    for ts in trace_spreads:
        delta_arr = np.zeros((n_runs, n_solutions))
        trace_arr = np.zeros((n_runs, n_solutions))
        for j in range(n_runs):
            d, t = synthesize_population(
                n_solutions, mean_delta_s, spread_delta_s,
                mean_trace_P, ts, rng,
            )
            delta_arr[j] = d
            trace_arr[j] = t
        out[ts] = disagreement_rate(delta_arr, trace_arr)
    return out


def analyze_stability_tradeoff(n_runs: int = 200, n_solutions: int = 10) -> str:
    """Markdown summary of legacy vs v2 disagreement rate across trace spreads."""
    sweep = sweep_disagreement_vs_trace_spread(
        n_runs=n_runs, n_solutions=n_solutions,
    )
    lines = [
        "## Stability factor (Reading-F) sensitivity",
        "",
        f"n_runs per spread: {n_runs}",
        f"population size: {n_solutions}",
        "",
        "| trace(P) spread | legacy ≠ v2 disagreement rate |",
        "|---|---|",
    ]
    for ts, rate in sweep.items():
        lines.append(f"| {ts:.2f} | {rate:.2%} |")
    lines.append("")
    max_rate = max(sweep.values())
    if max_rate > 0.20:
        lines.append("**Verdict:** stability factor is LOAD-BEARING in high-trace-spread regimes.")
    elif max_rate > 0.05:
        lines.append("**Verdict:** stability factor matters modestly; meaningful at high trace spread.")
    else:
        lines.append("**Verdict:** stability factor near-no-op; Reading-F's flip cost little signal.")
    return "\n".join(lines)

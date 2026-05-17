"""
W14-1 Walk-forward OOS evaluator per thesis §7.2.2.

The thesis evaluates at each rolling investment period t:
  - Train SMS-EMOA on a 1.5-year window of historical returns
  - Extract the trained Pareto-flexible set
  - Observe the next-period MLE Gaussian χ_{t+1} on a 50-day window
  - Compute Ŝ_{t+1} via W13-2 thesis Eqs (7.10)+(7.11)
  - Roll the window forward by 50 business days; repeat

Aggregation across periods (T-1 of them) + seeds (30) gives ~720
observations per scenario — the unit on which paper Table 7.2's
ANOVA is computed.

Critical distinction from W13-3 (single-shot 80/20):
  - At each period t, the algorithm sees ONLY data up to t (the
    anticipation property is activated)
  - The OOS evaluation uses the IMMEDIATELY NEXT period (not a
    distant 20% tail) — matches the thesis's "investment period
    decision" semantic
  - Aggregation has ~24× the statistical power

Defaults from thesis §7.2.3:
  - train_window_days = 378 (~1.5 business years)
  - step_days = 50 (~2.5 months — the rolling-shift granularity)
  - n_mc = 1000 (E in Eq 7.11)
  - z_ref = (0.2, 0.0)^T (handled by compute_oos_efhv default)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from experiments.oos_evaluator import compute_oos_efhv


# ---------------------------------------------------------------------------
# Period enumeration
# ---------------------------------------------------------------------------

def enumerate_periods(n_days: int, train_window_days: int = 378,
                       step_days: int = 50) -> list[dict[str, int]]:
    """Enumerate (train_range, oos_range) tuples for walk-forward.

    Given a return-data length `n_days`, generate all periods t such that:
      train_start = (t-1) * step_days
      train_end   = train_start + train_window_days
      oos_start   = train_end
      oos_end     = oos_start + step_days
    Where oos_end <= n_days. Period index starts at 1 (matching thesis).
    """
    periods: list[dict[str, int]] = []
    t = 1
    while True:
        train_start = (t - 1) * step_days
        train_end = train_start + train_window_days
        oos_start = train_end
        oos_end = oos_start + step_days
        if oos_end > n_days:
            break
        periods.append({
            "period": t,
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
        })
        t += 1
    return periods


# ---------------------------------------------------------------------------
# Per-period training + OOS eval
# ---------------------------------------------------------------------------

def _train_and_extract_pareto(scenario: str, seed: int,
                               train_returns: pd.DataFrame
                               ) -> tuple[list[np.ndarray], int]:
    """Train SMS-EMOA on the train window; extract Pareto weights.

    Mirrors oos_report._run_one_scenario_seed but takes the pre-sliced
    train block instead of loading the full window.
    """
    from experiments.validation_matrix import build_experiment_config
    from src.experiments.experiment_manager import ExperimentManager

    config = build_experiment_config(scenario, "paper", seed)
    np.random.seed(seed)
    suite_config = {
        "experiment_name": f"wf_{scenario}_seed{seed}",
        "description": "W14-1 walk-forward training",
        "version": "W14-1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    mgr = ExperimentManager(suite_config)
    n_assets = train_returns.values.shape[1]
    data = {"assets": train_returns, "market": pd.DataFrame(),
            "config": config.get("data", {})}
    results = mgr._run_algorithm(config, data)
    pareto = results.get("pareto_front", [])
    weights: list[np.ndarray] = []
    for sol in pareto:
        w = getattr(sol.P, "investment", None)
        if w is None:
            continue
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape == (n_assets,):
            weights.append(w_arr)
    return weights, n_assets


def run_walk_forward(scenario: str,
                      seed: int,
                      full_returns: pd.DataFrame,
                      train_window_days: int = 378,
                      step_days: int = 50,
                      n_mc: int = 1000,
                      rng: np.random.Generator | None = None
                      ) -> list[dict[str, Any]]:
    """Walk-forward evaluate one (scenario, seed) across all rolling periods.

    Returns a list of per-period dicts:
      {
        'period': t (1-indexed),
        'train_range': (train_start, train_end),
        'oos_range': (oos_start, oos_end),
        'n_pareto': int,
        'efhv_mean': float (Ŝ_{t+1} from Eq 7.11),
        'efhv_std': float (MC std across the E=n_mc scenarios),
        'error': str (only if training failed; omitted otherwise)
      }
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    periods = enumerate_periods(len(full_returns), train_window_days, step_days)
    results: list[dict[str, Any]] = []
    for p in periods:
        train = full_returns.iloc[p["train_start"]:p["train_end"]]
        oos = full_returns.iloc[p["oos_start"]:p["oos_end"]]
        try:
            weights, n_assets = _train_and_extract_pareto(scenario, seed, train)
        except Exception as exc:
            results.append({**p, "error": f"{type(exc).__name__}: {exc}",
                              "efhv_mean": float("nan"), "efhv_std": float("nan"),
                              "n_pareto": 0})
            continue
        if not weights:
            results.append({**p, "n_pareto": 0,
                              "efhv_mean": 0.0, "efhv_std": 0.0})
            continue
        # OOS slice must have at least 2 rows for cov estimation.
        if len(oos) < 2:
            results.append({**p, "n_pareto": len(weights),
                              "efhv_mean": float("nan"), "efhv_std": float("nan"),
                              "error": "oos_window_too_short"})
            continue
        efhv = compute_oos_efhv(
            pareto_weights=weights,
            oos_returns=oos,
            n_samples=n_mc,
            rng=rng,
        )
        results.append({
            **p,
            "n_pareto": len(weights),
            "efhv_mean": efhv["efhv_mean"],
            "efhv_std": efhv["efhv_std"],
        })
    return results


# ---------------------------------------------------------------------------
# Aggregation helpers (used by W14-2)
# ---------------------------------------------------------------------------

def aggregate_per_seed(per_period_results: list[dict[str, Any]]) -> dict[str, float]:
    """Collapse per-period Ŝ_{t+1} into per-seed grand mean."""
    valid = [r["efhv_mean"] for r in per_period_results
              if "error" not in r and np.isfinite(r["efhv_mean"])]
    if not valid:
        return {"n_periods_ok": 0, "grand_mean": float("nan"),
                "grand_std": float("nan")}
    arr = np.asarray(valid, dtype=float)
    return {
        "n_periods_ok": len(arr),
        "n_periods_total": len(per_period_results),
        "grand_mean": float(arr.mean()),
        "grand_std": float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0,
    }

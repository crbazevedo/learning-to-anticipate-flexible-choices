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

def _select_amfc_index(per_portfolio_efhv: np.ndarray) -> int:
    """W17-4: AMFC selector per thesis §6.4 Eq 6.42.

    Pick the argmax over per-portfolio expected future HV. Ties
    broken deterministically (smallest index wins). Fallback to
    index 0 + warning if all per-portfolio EFHV are 0 / NaN.
    """
    if per_portfolio_efhv.size == 0:
        return 0
    finite_mask = np.isfinite(per_portfolio_efhv)
    if not finite_mask.any() or per_portfolio_efhv[finite_mask].max() <= 0.0:
        import logging
        logging.getLogger(__name__).warning(
            "W17-4: all per-portfolio EFHV are 0/NaN — fallback to index 0 "
            "as u*_{t-1}. Possible degenerate period (Pareto front dominated "
            "by z_ref or all weights produced negative-return / high-variance "
            "predictions)."
        )
        return 0
    # np.argmax breaks ties by smallest index (deterministic).
    return int(np.argmax(per_portfolio_efhv))


def _train_and_extract_pareto(scenario: str, seed: int,
                               train_returns: pd.DataFrame,
                               previous_weights: np.ndarray | None = None,
                               previous_kf_state=None,
                               lambda_trace_csv_path: str | None = None,
                               ) -> tuple[list[np.ndarray], int, list, list]:
    """Train SMS-EMOA on the train window; extract Pareto weights.

    Mirrors oos_report._run_one_scenario_seed but takes the pre-sliced
    train block instead of loading the full window.

    W16-2 (BACKLOG H1): optional ``previous_weights`` threads u*_{t-1}
    into the SMS-EMOA evaluator so the optimizer subtracts thesis
    Table 7.1 transaction costs from the ROI objective per §7.2
    Eqs (7.4)-(7.5). On period 1 / no prior portfolio, pass None →
    no cost subtraction.

    W22-NC8c (2026-05-18): optional ``previous_kf_state`` carries the
    previous period's AMFC portfolio KF state forward. When set,
    sms_emoa._initialize_population uses this as the initial KF state
    for the new period's portfolios (specifically: copies the velocity
    components x[2:4] and the P[2:4, 2:4] block per portfolio, allowing
    velocity learning to persist across walk-forward periods).

    Returns:
        (weights, n_assets, probe_a_records, pareto_kf_states)
        - pareto_kf_states: list of KalmanParams snapshots, one per
          Pareto-front portfolio (parallel to weights). For NC8c
          downstream consumption by run_walk_forward.
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
    # W16-2: thread previous-period implemented portfolio to
    # ExperimentManager → SMS-EMOA via the data dict (so the manager's
    # algorithm setter can call set_previous_weights post-init).
    if previous_weights is not None:
        data["previous_weights"] = np.asarray(previous_weights, dtype=float)
    # W22-NC8c: thread previous period's AMFC KF state for cross-period
    # velocity learning persistence.
    if previous_kf_state is not None:
        data["previous_kf_state"] = previous_kf_state
    # W16-4: thread λ trace CSV path; ExperimentManager flushes per-period.
    if lambda_trace_csv_path is not None:
        data["lambda_trace_csv_path"] = str(lambda_trace_csv_path)
    results = mgr._run_algorithm(config, data)
    pareto = results.get("pareto_front", [])
    weights: list[np.ndarray] = []
    pareto_kf_states: list = []
    # W22 Probe A: optionally also extract per-portfolio KF state +
    # current-period (ROI, risk) for the persistence baseline.
    from src.diagnostics import probe_a_kf_prediction_log as _probe_a
    probe_a_records: list[dict] = []
    for sol in pareto:
        w = getattr(sol.P, "investment", None)
        if w is None:
            continue
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape == (n_assets,):
            weights.append(w_arr)
            # W22-NC8c: capture KF state per Pareto portfolio so
            # run_walk_forward can pick AMFC's KF state for next period.
            kf_for_carry = getattr(sol.P, "kalman_state", None)
            pareto_kf_states.append(kf_for_carry)
            if _probe_a.is_enabled():
                kf = kf_for_carry
                if kf is not None:
                    # x_next post-final-predict = KF's t+1 prediction
                    # ROI is x[0], risk is x[1] per paper Eq 11
                    probe_a_records.append({
                        "weights": w_arr.copy(),
                        "kf_predicted_ROI_t_plus_1": float(kf.x_next[0]),
                        "kf_predicted_risk_t_plus_1": float(kf.x_next[1]),
                        "persistence_ROI_t": float(sol.P.ROI),
                        "persistence_risk_t": float(sol.P.risk),
                        "kf_P_diag": [float(kf.P[i, i]) for i in range(kf.P.shape[0])],
                    })
    return weights, n_assets, probe_a_records, pareto_kf_states


def run_walk_forward(scenario: str,
                      seed: int,
                      full_returns: pd.DataFrame,
                      train_window_days: int = 378,
                      step_days: int = 50,
                      n_mc: int = 1000,
                      rng: np.random.Generator | None = None,
                      lambda_trace_csv_path: str | None = None,
                      use_closed_form_efhv: bool = False,
                      use_closed_form_expectation_efhv: bool = False,
                      use_v2_per_front_efhv: bool = False,
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
    # W16-2 (BACKLOG H1): u*_{t-1} thread across rolling periods.
    # Per thesis convention, the "implemented" portfolio is the
    # argmax-EFHV (or first if EFHV not yet computed) from the
    # previous period's Pareto front. None on period 1 (no prior).
    previous_weights: np.ndarray | None = None
    # W22-NC8c (2026-05-18): carry the AMFC portfolio's KF state across
    # walk-forward periods so velocity learning persists across t boundaries
    # (vs being reset to zero at each fresh period init).
    previous_kf_state = None
    for p in periods:
        train = full_returns.iloc[p["train_start"]:p["train_end"]]
        oos = full_returns.iloc[p["oos_start"]:p["oos_end"]]
        try:
            weights, n_assets, probe_a_records, pareto_kf_states = _train_and_extract_pareto(
                scenario, seed, train,
                previous_weights=previous_weights,
                previous_kf_state=previous_kf_state,
                lambda_trace_csv_path=lambda_trace_csv_path,
            )
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

        # W22 Probe A: for each portfolio in the Pareto front, compute
        # the ACTUAL t+1 ROI/risk by evaluating its weights under the
        # OOS period's MLE (μ̂_{t+1}, Σ̂_{t+1}), then log the
        # (KF prediction, persistence baseline, actual) triple.
        from src.diagnostics import probe_a_kf_prediction_log as _probe_a
        if _probe_a.is_enabled() and probe_a_records:
            oos_arr = oos.values if hasattr(oos, "values") else np.asarray(oos)
            if oos_arr.shape[0] >= 2:
                mu_oos = oos_arr.mean(axis=0)
                cov_oos = np.cov(oos_arr, rowvar=False, ddof=1)
                if cov_oos.ndim == 0:
                    cov_oos = np.array([[float(cov_oos)]])
                for i, rec in enumerate(probe_a_records):
                    w = rec["weights"]
                    actual_roi = float(mu_oos @ w)
                    actual_risk = float(w @ cov_oos @ w)
                    _probe_a.log_period_kf_prediction(
                        scenario=scenario,
                        seed=seed,
                        period_t=int(p["period"]),
                        portfolio_idx=i,
                        weights=w.tolist(),
                        kf_predicted_ROI_t_plus_1=rec["kf_predicted_ROI_t_plus_1"],
                        kf_predicted_risk_t_plus_1=rec["kf_predicted_risk_t_plus_1"],
                        persistence_ROI_t=rec["persistence_ROI_t"],
                        persistence_risk_t=rec["persistence_risk_t"],
                        actual_ROI_t_plus_1=actual_roi,
                        actual_risk_t_plus_1=actual_risk,
                        kf_P_diag=rec["kf_P_diag"],
                    )
        efhv = compute_oos_efhv(
            pareto_weights=weights,
            oos_returns=oos,
            n_samples=n_mc,
            rng=rng,
            use_closed_form=use_closed_form_efhv,
            use_closed_form_expectation=use_closed_form_expectation_efhv,
            use_v2_per_front=use_v2_per_front_efhv,
        )
        # W17-4 (closes W16-2-CARRY-1 + BACKLOG M5 partial): pick the
        # AMFC per thesis §6.4 Eq 6.42 as u*_{t-1} for the NEXT period
        # — argmax over per-portfolio expected single-point HV against
        # z_ref. Replaces W16-2's "first Pareto-front portfolio" proxy.
        from .oos_evaluator import compute_per_portfolio_efhv
        # W22: per-portfolio EFHV should match the chosen Ŝ estimator
        # methodology to keep the AMFC u*_{t-1} selection consistent
        # with the OOS scoring. ANY closed-form Ŝ flag (A, B, or C)
        # triggers the closed-form per-portfolio variant; otherwise
        # bootstrap MC (W14-2 default). Fixes the W17-4 "all
        # per-portfolio EFHV are 0/NaN" fallback that surfaced in
        # Options B/C smokes (per_portfolio_efhv was always MC,
        # producing degenerate values when paired with a closed-form
        # Ŝ estimator).
        per_portfolio_efhv = compute_per_portfolio_efhv(
            pareto_weights=weights,
            oos_returns=oos,
            n_samples=n_mc,
            rng=rng,
            use_closed_form=(use_closed_form_efhv
                              or use_closed_form_expectation_efhv
                              or use_v2_per_front_efhv),
        )
        u_star_idx = _select_amfc_index(per_portfolio_efhv)
        previous_weights = weights[u_star_idx]
        # W22-NC8c: carry the AMFC portfolio's KF state to next period.
        # pareto_kf_states[u_star_idx] is the same Solution-shaped KF state;
        # _initialize_population will use it as initial state for new period.
        if (0 <= u_star_idx < len(pareto_kf_states)
                and pareto_kf_states[u_star_idx] is not None):
            import copy as _copy
            previous_kf_state = _copy.deepcopy(pareto_kf_states[u_star_idx])
        # If AMFC has no KF state (shouldn't happen post-NC7), keep prior.
        results.append({
            **p,
            "n_pareto": len(weights),
            "u_star_idx": int(u_star_idx),
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
        return {"n_periods_ok": 0,
                "n_periods_total": len(per_period_results),
                "grand_mean": float("nan"),
                "grand_std": float("nan")}
    arr = np.asarray(valid, dtype=float)
    return {
        "n_periods_ok": len(arr),
        "n_periods_total": len(per_period_results),
        "grand_mean": float(arr.mean()),
        "grand_std": float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0,
    }

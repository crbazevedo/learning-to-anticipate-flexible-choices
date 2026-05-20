# W22 Probe Q-v1 — Per-Asset AR(1) Return Predictor

Status: **shipped (module + tests; integration deferred)**
Branch: `feat/w22-inspection-backlog`
Module: `python_refactor/src/probes/probe_q_ar1_predictor.py`
Tests: `python_refactor/tests/test_probe_q_ar1.py` (18 tests, all passing)
Specs: `docs/W22-ALT-SIGNAL-PROBES.md` (Probe Q section);
       `docs/W22-MASTER-BACKLOG.md` (P2 row "Probe Q-v1: AR(1) per-asset
       prediction smoke")

This is the v1 (cheapest) variant of Probe Q. v2 (GARCH(1,1) on
volatility, ARMA(1,1) on mean) and v3 (VAR(1) on top-K assets by volume)
are deliberately out of scope.

---

## Hypothesis

**Q-H1** — per-asset AR(1) predictions of next-period return beat the
implicit "no change" baseline that the ROI-level Kalman filter makes:

> r_hat_{k,t+1} = mu_k + rho_k * (r_{k,t} - mu_k)
>
> vs.
>
> r_hat_{k,t+1} = r_{k,t}    (no-change persistence baseline)

**Q-H2** — knowing per-asset next-period predicted returns changes which
assets the portfolio optimizer selects (downstream HV gain).

This module addresses **Q-H1 ONLY**. Q-H2 needs a wiring change into
the mean-variance optimizer and is intentionally a future-operator
decision (see *integration sketch* below).

---

## Mechanism

For each asset k ∈ {0, ..., d-1} independently:

1. Maintain a rolling per-asset window of the last `window_size`
   observations (default 60, matching the Probe Q spec).
2. At fit time: drop non-finite entries, compute window mean `mu_k`,
   centre the series, and estimate the lag-1 autocorrelation `rho_k`
   by OLS on the centred series:

       rho_k = sum_t x_{t-1} * x_t / sum_t x_{t-1}^2

3. Clip `rho_k` to (-0.999, 0.999) for stationarity sanity.
4. Predict:

       r_hat_{k,t+1} = mu_k + rho_k * (r_{k,t} - mu_k)

5. If `window_size < 2`, fewer than 2 finite obs in the window, or the
   centred variance is below `1e-12`: fall back to the no-change
   baseline (per-asset).

---

## Success criteria

Module-level (THIS unit ratifies):

* **C1** — 18/18 unit tests pass (construction, observe, both
  predictors, AR recovery, falsifier-on-synthetic, degenerate cases).
* **C2** — `test_ar_recovers_known_ar_process` recovers a generating
  rho=0.7 within 0.05 on a 4000-step trajectory.
* **C3** — `test_ar_beats_no_change_on_ar_data` shows
  RMSE(AR1) < RMSE(no-change) over 100 one-step-ahead forecasts on a
  synthetic AR(1) process with rho=0.7.

Probe-level (NOT ratified by this unit — deferred):

* **Q-M1** — RMSE of per-asset AR(1) vs no-change on each real regime.
* **Q-M2** — fraction of assets where AR(1) RMSE < no-change RMSE.
* **Q-M3** — final-wealth (or HV) gain of ASMOO with AR(1)-predicted
  mean returns vs ASMOO with historical mean returns (n >= 10 seeds).
* **Q-M4** — Wilcoxon signed-rank p < 0.05 on the gain distribution.

**Q-PASS**: Q-M3 mean >= +5% and Q-M4 p < 0.05 on at least one regime
(FTSE 2015 or NASDAQ 2010).

**Q-FAIL** (falsifier): Q-M2 <= 50% AND Q-M3 mean <= 0% on all regimes.

---

## Expected scars (a-priori, before any real-data run)

These are the failure modes most likely to bite when Q-H1/Q-H2 are
evaluated on real data. Calling them out now so that, if they happen,
we land them as honest scars rather than relitigated mid-flight.

1. **Daily return autocorrelation is tiny in efficient markets.**
   On large-cap equities you should expect |rho| typically in
   [0.02, 0.10]. AR(1) RMSE will be only marginally better than
   no-change; the win rate may sit just above 50%. This is the
   modal outcome and is not a bug.
2. **rho sign flips across regimes.** Crisis periods exhibit
   short-term reversal (rho < 0), calm regimes mild momentum
   (rho > 0). A single global mu/rho per asset will mis-fit at
   regime transitions. The fix would belong in v2 (GARCH+ARMA) or
   v3 (regime-conditional VAR), not v1.
3. **Per-asset AR(1) wins on RMSE but the optimizer LOSES on HV.**
   Mean-variance is sensitive to the COVARIANCE between assets;
   replacing only the mean vector with AR predictions while leaving
   the covariance untouched can change selections in
   variance-destructive ways. This is exactly what Q-H2 falsifies
   for / against — and the most likely path to Q-FAIL even if Q-H1
   holds.
4. **Window-size sensitivity.** `window_size=60` is the spec
   default; the result will be non-trivially different at 30 vs.
   120. A sensitivity sweep belongs in the integration unit, not
   this module.
5. **Numerical degeneracy on holiday-padded series.** Long runs of
   identical (carry-forward) prices give zero centred variance for
   many assets, all of which fall back to the mean — silently. The
   `_VAR_FLOOR` threshold guards against the AR estimate exploding
   but does not flag the data-quality issue upstream.

---

## How to integrate with the portfolio optimizer (sketch only — DO NOT wire in this unit)

The integration is an explicit operator decision because it touches the
ASMOO call-site, and the W22 contract says: *no modifications to shared
code paths from this unit*. The minimal integration path is:

```python
# IN A FUTURE UNIT (NOT THIS ONE):
#   src/experiments/probe_q_v1_integration.py

from src.probes.probe_q_ar1_predictor import AR1AssetPredictor

def predict_mean_returns_for_rebalance(
    historical_returns: np.ndarray,   # shape (T, d)
    window_size: int = 60,
) -> np.ndarray:
    """Return a length-d vector of predicted next-period returns."""
    d = historical_returns.shape[1]
    p = AR1AssetPredictor(d=d, window_size=window_size)
    for t in range(historical_returns.shape[0]):
        p.observe(historical_returns[t])
    return p.predict_next()
```

That vector then replaces the historical mean `mu_hat` input to the
mean-variance objective inside `experiments/walk_forward.py`. The
covariance input stays as the empirical Sigma_hat (replacing covariance
is the job of Probe Q-v2, not v1).

**Three things the integration unit must do that this module
deliberately does NOT:**

1. Hold a per-asset predictor PER rebalance window (the predictor is
   re-fit on the trailing window, not carried across periods).
2. Compare HV / final wealth against the historical-mean ASMOO on
   n >= 10 seeds per regime and emit the Wilcoxon p (Q-M3 / Q-M4).
3. Decide whether to gate Q-H2 on Q-H1 (per-asset win-rate) or run
   both probes simultaneously.

The integration unit also needs its own contract entry, its own
success / falsifier criteria, and (per the operator's W22 discipline)
a paired retro with what-you'd-change captured *before* it runs, not
after.

---

## What this unit does NOT ship

* No wiring into `thesis_aligned_experiment.py`, `walk_forward.py`,
  `oos_evaluator.py`, or any other experiment runner.
* No modification to `sms_emoa.py`, `anticipatory_learning.py`,
  `kalman_filter.py`, or any shared algorithm module.
* No real-data smoke run. The synthetic-AR(1) falsifier test
  (`test_ar_beats_no_change_on_ar_data`) is the ONLY end-to-end
  evidence ratified here.
* No v2 (GARCH/ARMA) or v3 (VAR) variants.
* No telemetry sink (Probe A's pattern). The predictor is pure-state
  and is meant to be driven by whatever harness the integration unit
  authors.

---

## Files touched

| File | Lines | Role |
|---|---:|---|
| `python_refactor/src/probes/__init__.py` | new (12) | package marker; documents the contained-probe discipline |
| `python_refactor/src/probes/probe_q_ar1_predictor.py` | new (~180) | `AR1AssetPredictor` class |
| `python_refactor/tests/test_probe_q_ar1.py` | new (~210) | 18 tests across 5 groups |
| `docs/W22-PROBE-Q-V1-AR1.md` | new (this file) | hypothesis, mechanism, scars, integration sketch |

No other file is modified.

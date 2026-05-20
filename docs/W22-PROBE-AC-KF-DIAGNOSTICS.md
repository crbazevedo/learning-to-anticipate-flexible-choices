# W22 Probe AC — KF NIS/NEES consistency diagnostics

**Status:** module shipped, integration deferred
**Module:** `python_refactor/src/probes/probe_ac_kf_diagnostics.py`
**Tests:** `python_refactor/tests/test_probe_ac_kf_diagnostics.py` (17 passing)
**Backlog row:** `docs/W22-MASTER-BACKLOG.md` Section C row AC

---

## Hypothesis

**AC-H1.** The ASMS Kalman filter is properly tuned **iff** its
Normalized Innovation Squared (NIS) and Normalized Estimation Error
Squared (NEES) statistics lie within their chi-squared confidence
intervals.

Falsification predictions:

| Mis-tuning | Predicted symptom |
|---|---|
| `R_assumed` too small | innovation covariance `S` over-confident → NIS too high (mean > CI upper) |
| `R_assumed` too large | `S` over-inflated → NIS too low (mean < CI lower) |
| `Q_assumed` too small | posterior `P` under-states uncertainty → NEES too high |
| `Q_assumed` too large | filter degenerates toward measurement-only (K → 1, P → R) → NEES drifts toward 1 (band still passes in degenerate 1-D models) |
| `F` mismatched | innovation mean non-zero → averaged NIS biased outside CI |

## Citation

Y. Bar-Shalom, X. R. Li, T. Kirubarajan,
*Estimation with Applications to Tracking and Navigation*, Wiley 2001,
**§5.4 "Consistency of State Estimators"**:

* NIS_t = y_t^T · S_t^{-1} · y_t ~ chi^2(d_meas)
* NEES_t = (x_true − x_hat)^T · P_t^{-1} · (x_true − x_hat) ~ chi^2(d_state)
* For N i.i.d. samples averaged, N·mean ~ chi^2(N·d). The
  two-sided (1 − α) CI on the *average* is
  `[chi2.ppf(α/2, N·d)/N, chi2.ppf(1 − α/2, N·d)/N]`.

## Success criteria

| Criterion | Source of truth |
|---|---|
| Tuned synthetic KF → NIS mean inside 95% CI | `test_nis_consistency_passes_on_tuned_kf` |
| Tuned synthetic KF → NEES mean inside 95% CI (Monte-Carlo across runs) | `test_nees_consistency_passes_on_tuned_kf` |
| Under-tuned R → NIS mean > CI upper | `test_nis_consistency_fails_on_under_tuned_R` |
| Over-tuned R → NIS mean < CI lower | `test_nis_consistency_fails_on_over_tuned_R` |
| Under-tuned Q → NEES mean > CI upper | `test_nees_consistency_fails_on_under_tuned_Q` |
| Under-tuned R wrt NEES → NEES mean > CI upper | `test_nees_consistency_fails_on_under_tuned_R` |
| `chi2_ci` matches `scipy.stats.chi2.ppf` directly | `test_chi2_ci_known_values` |

## Known limitation — over-tuned Q in 1-D constant-state model

In the synthetic 1-D constant-state model (`F = H = 1`), pushing
`Q_assumed → ∞` collapses the steady-state filter to "trust only the
latest measurement" (K → 1, P_post → R_assumed). Once `R_assumed`
matches `R_true` the NEES distribution converges back to chi^2(1) and
the CI test does *not* flag the mis-tuning. This is a model-specific
quirk, not a property of the diagnostic. In production (multi-D state,
constant-velocity F) over-tuned Q does push NEES below the band as
predicted by Bar-Shalom; we document the 1-D caveat here and exercise
over-tuned R against NIS instead.

NEES correlation across time inside one realization is the second
caveat: per Bar-Shalom §5.4.4 the averaged-NEES CI assumes i.i.d.
samples, so the tests use Monte-Carlo averaging across `n_runs = 300`
independent realizations at a fixed timestep.

## Module surface

```python
from src.probes.probe_ac_kf_diagnostics import (
    compute_nis,                 # single-timestep NIS
    compute_nees,                # single-timestep NEES
    chi2_ci,                     # (low, high) for averaged stat
    nis_consistency_test,        # dict {mean, ci_low, ci_high, passes, ...}
    nees_consistency_test,       # same shape
    extract_innovations_from_residual_window,  # (residual, S) -> per-period NIS
)
```

The module is fully self-contained: no imports from
`src.algorithms.kalman_filter`, `src.algorithms.anticipatory_learning`,
or `src.algorithms.sms_emoa`. Per W22 alt-signal probe discipline
(see `python_refactor/src/probes/__init__.py`), the integration that
would call this module from the live ASMS KF loop is a *separate*
operator decision.

## Integration sketch (out of scope for this probe)

To wire this probe into the existing KF without touching shared code,
the future integration step would:

1. Add an opt-in env-var-gated logger
   (`W22_PROBE_AC_LOG_PATH`, mirroring Probe A) at the KF
   `predict_observe` boundary in
   `python_refactor/src/algorithms/kalman_filter.py`.
2. At each KF update emit a record:
   ```json
   {
     "scenario": "...", "seed": ..., "period_t": ...,
     "innovation": [...],          // y_t
     "innovation_cov": [[...]],    // S_t
     "state_cov_posterior": [[...]] // P_{t|t}
   }
   ```
3. A post-run analyzer reads the JSONL, calls
   `extract_innovations_from_residual_window` per (scenario, seed) and
   reports `nis_consistency_test(...)`. NEES requires either
   ground-truth state (synthetic experiments) or a smoothed estimate
   (RTS smoother, future); production use should aggregate across
   seeds → walk-forward periods to approximate Monte-Carlo i.i.d.

## Outputs to look for once integrated

| Symptom | Likely cause | Suggested next probe |
|---|---|---|
| NIS mean ≫ CI on FTSE | `R_assumed` (process variance estimate) too small | recalibrate from realized residuals (Probe AF Mehra) |
| NIS mean ≪ CI | `R_assumed` too large | recalibrate downward |
| NIS bias non-zero | `F = identity` wrong for return series | swap F (random-walk vs constant-velocity, AG) |
| NEES mean ≫ CI | `Q_assumed` too small | bump process noise; revisit constant-velocity assumption |

## Provenance

- Hypothesis admitted via `docs/W22-MASTER-BACKLOG.md` Section C row AC
  ("KF NIS/NEES consistency diagnostics", budget 350).
- Module + tests committed on `feat/w22-inspection-backlog`.
- No shared code paths touched; ships behind no env-var (purely
  importable). Future integration step instruments the KF.

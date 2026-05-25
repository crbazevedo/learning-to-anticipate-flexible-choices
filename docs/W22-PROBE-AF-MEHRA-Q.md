# W22 Probe AF — Mehra (1970) Q-noise estimation from KF innovation autocorrelation

**Status:** module shipped, integration deferred
**Module:** `python_refactor/src/probes/probe_af_mehra_q_estimation.py`
**Tests:** `python_refactor/tests/test_probe_af_mehra_q_estimation.py` (9 passing)
**Backlog row:** `docs/W22-MASTER-BACKLOG.md` Section C row AF

---

## Citation

Mehra, R. K. (1970). **"On the identification of variances and adaptive
Kalman filtering."** *IEEE Transactions on Automatic Control* **15**(2),
175–184. DOI: 10.1109/TAC.1970.1099422.

The seminal paper that established the use of the innovation
autocorrelation sequence ``{C_k}`` for identifying the noise variances
``Q`` and ``R`` of a steady-state linear Kalman filter, alongside the
observability condition that makes the identification well-posed.

## Hypothesis

**AF-H1.** The ASMS Kalman filter is correctly tuned **iff** its
innovation sequence is white — i.e. ``C_k = E[y_t · y_{t+k}^T] = 0`` for
all ``k >= 1`` — AND its observed lag-0 autocorrelation matches the
theoretical expectation ``H · P_pred · H^T + R``.

Falsification predictions:

| Mis-tuning | Predicted symptom |
|---|---|
| ``Q_assumed`` too small | filter under-states process noise, innovations are bigger than expected ⇒ ``C_0_ratio = trace(C_0_obs) / trace(C_0_exp) > 1`` AND lag-1+ autocorrelations are non-zero (under-tuning manifests as a temporal lag-1 bias) |
| ``Q_assumed`` too large | filter over-states process noise, innovations are smaller than expected ⇒ ``C_0_ratio < 1``, lag-1+ autocorrelations also drift |
| ``Q_assumed == Q_true`` | ``C_0_ratio ≈ 1`` AND lag-1+ norms < 30% of ``||C_0_expected||`` ⇒ verdict ``CORRECTLY_TUNED`` |
| ``F`` mismatched | innovation mean drifts; whiteness violated but ``C_0`` ratio may still fall in band ⇒ verdict ``NEEDS_INVESTIGATION`` |

## Methodology

### Step 1 — Collect innovations

A KF run produces, at each timestep ``t``, an innovation
``y_t = z_t - H * x_hat_{t|t-1}``. The probe consumes a list of these
vectors *post-hoc* (no live instrumentation). After dropping a warm-up
window (~50–200 steps depending on filter initial covariance), the
remaining innovations form the empirical estimate of the
**steady-state** innovation process.

### Step 2 — Compute the autocorrelation sequence

For lags ``k = 0, ..., max_lag`` form the unbiased sample
autocorrelation (Mehra 1970, eq. 9):

```
C_k = (1 / (T - k)) * sum_{t=0}^{T-k-1} y_t * y_{t+k}^T
```

The lag-0 slice is symmetrised by construction; higher-lag slices are
in general asymmetric and we leave them so (the magnitude diagnostic
uses the Frobenius norm).

### Step 3 — Compare against the theoretical lag-0

The Kalman filter itself reports ``S_t = H * P_pred * H^T + R`` during
its update step. At steady state this is the matrix that ``C_0`` should
match, so the **ratio test** is:

```
ratio = trace(C_0_observed) / trace(S_theoretical)
```

* ``ratio ≈ 1`` (within ±30%) → magnitude is consistent.
* ``ratio > 1.3`` → observed innovations bigger than expected → most
  likely cause is **under-tuned Q**.
* ``ratio < 0.7`` → observed innovations smaller than expected → most
  likely cause is **over-tuned Q** (filter expects more noise than is
  actually present).

### Step 4 — Whiteness test

For ``k = 1, ..., max_lag`` compute
``||C_k||_F / ||C_0_expected||_F``. If the maximum of these normalised
norms is < 0.30 we declare the innovation sequence **white** and the
filter **correctly tuned** (subject to the ratio test).

If the ratio test passes but the whiteness test fails the verdict is
``NEEDS_INVESTIGATION`` — the magnitude is right but there is residual
temporal structure (often an ``F`` mismatch, sometimes a non-stationary
Q drift the constant-Q assumption cannot capture).

### Step 5 — Simplified Mehra Q estimator (where defined)

Mehra's full algorithm (1970, eqs. 13–17) solves a Lyapunov-style
system for ``Q`` once a steady-state Kalman gain has been recovered
from ``{C_k}``. The full derivation is lengthy; the probe ships a
**simplified** form derived from the steady-state identity

```
S = H * P_pred * H^T + R   ⇒   Q_hat ≈ F * H^+ * (S_obs - H*P_pred*H^T) * (H^+)^T * F^T
```

where ``H^+`` is the Moore-Penrose pseudo-inverse. This is sufficient
for diagnostic purposes when the system is **Mehra-observable**, i.e.
the stacked matrix ``[H; H*F; ...; H*F^{n-1}]`` has full column rank
``n_state``. For our 4-state constant-velocity KF with
``H = [I_2 | 0_2]`` and ``F = [[I_2, dt*I_2]; [0, I_2]]`` this is
satisfied whenever ``dt != 0`` (the velocity becomes observable through
the position dynamics one step later).

When the system is **not** Mehra-observable the estimator returns an
all-NaN matrix and the caller is expected to surface this — it is a
load-bearing diagnostic, not a fallback.

## Success criteria

| Criterion | Source of truth |
|---|---|
| Lag-0 autocorrelation matches the symmetrised outer-product mean | `test_autocorrelation_lag_zero_is_outer_product_mean` |
| White innovations ⇒ lag-k norms ≈ 0 for k ≥ 1 | `test_autocorrelation_white_noise_is_zero_off_diagonal_lag` |
| AR(1)-driven innovations ⇒ lag-1 ≠ 0, ratio ≈ φ | `test_autocorrelation_ar1_innovations_nonzero_lag_one` |
| Expected covariance formula `S = H·P·H^T + R` | `test_expected_cov_formula_matches_HPH_plus_R` |
| Well-tuned KF ⇒ verdict `CORRECTLY_TUNED` | `test_diagnostic_correctly_tuned_on_synthetic_well_tuned` |
| Under-tuned Q ⇒ `UNDER_TUNED_Q` or `NEEDS_INVESTIGATION`, ratio > 1 | `test_diagnostic_under_tuned_on_synthetic_small_q` |
| Over-tuned Q ⇒ `OVER_TUNED_Q` or `NEEDS_INVESTIGATION`, ratio < 1 | `test_diagnostic_over_tuned_on_synthetic_large_q` |
| Q-estimate is PSD on Mehra-observable input | `test_q_estimate_returns_psd_matrix` |
| Q-estimate returns NaN on Mehra-unobservable input | `test_q_estimate_returns_nan_for_unobservable_case` |

## Module surface

```python
from src.probes.probe_af_mehra_q_estimation import (
    innovation_autocorrelation,            # (T innovations, max_lag) -> (max_lag+1, d, d)
    expected_innovation_cov_under_correct_q,  # H, P_pred, R -> S
    q_tuning_diagnostic,                   # innovations, H, P_pred, R -> verdict dict
    simplified_q_estimate,                 # autocorr, F, H, P_pred -> Q_hat (or NaN)
)
```

`q_tuning_diagnostic` returns a dict shaped like:

```
{
  'C_0_observed':              (d, d) ndarray,
  'C_0_expected':              (d, d) ndarray,
  'C_0_ratio':                 float (≈ 1 → tuned),
  'autocorr_norms':            list[float] of length max_lag (Frobenius),
  'normalised_autocorr_norms': list[float] (each / ||C_0_expected||_F),
  'diagnostic_str':            'CORRECTLY_TUNED' | 'UNDER_TUNED_Q'
                                | 'OVER_TUNED_Q' | 'NEEDS_INVESTIGATION',
  'n_samples':                 int (T),
  'max_lag':                   int,
}
```

## Integration sketch (future, separate decision)

Out of scope for this probe — listed here so the operator can pick it
up as a follow-up unit:

1. **Innovation capture.** Instrument the existing KF
   predict/update loop (`python_refactor/src/kalman_filter.py`) to
   append each step's `(innovation_vector, S_matrix, P_pred)` to a
   per-asset ring buffer of fixed length `W` (default 500).
   *No behavioural change* — purely an observability tap.
2. **Post-hoc diagnostic.** After each anticipatory horizon (or
   wave-close in the ASMS-EMOA experiment harness) call
   `q_tuning_diagnostic(buffer, H, P_pred, R)` for each asset and log
   the verdict + `C_0_ratio` to the existing experiment JSON.
3. **Adaptive Q (operator-gated).** If the verdict is
   `UNDER_TUNED_Q` or `OVER_TUNED_Q` on N≥3 consecutive horizons,
   surface a recommendation to scale `Q_assumed` by `C_0_ratio`. This
   is the only behavioural change and requires its own ratification
   step — Probe AF itself ships *only* the diagnostic, not the
   feedback loop.

The W22 backlog row earmarks 500 compute-minutes for this probe; the
diagnostic is cheap (O(T·d²) per call) so the budget will be spent on
horizon-by-horizon scaffolding, not on the Mehra maths.

## Known limitations

* **Warm-up window.** The diagnostic assumes the filter has reached
  steady state. Tests use `n_steps=2500` with a 200-step warm-up drop;
  in production the operator should drop at least one full anticipatory
  horizon (typically 50 steps) before invoking the diagnostic.
* **Stationarity.** The unbiased autocorrelation estimator assumes a
  jointly wide-sense-stationary innovation sequence. Regime-shifts in
  the asset universe (covariance drift, market closures) will inflate
  the apparent autocorrelation and bias the verdict toward
  `NEEDS_INVESTIGATION`. This is a feature (it surfaces non-stationarity)
  not a bug.
* **Simplified Q estimator.** The lifted-pseudo-inverse form is a
  diagnostic, not a production estimator. Full Mehra (1970, §III)
  requires solving a Lyapunov equation for ``Q`` from the recovered
  Kalman gain; ship a separate probe if the operator decides to close
  the adaptive-Q loop.
* **F-mismatch confusion.** A mis-specified ``F`` produces innovations
  with non-zero mean and non-white autocorrelation; the probe will flag
  this as `NEEDS_INVESTIGATION` but cannot disambiguate Q-mistuning
  from F-mistuning. Probe AG (constant-velocity vs random-walk F
  selection) is the companion probe for that question.

---

*Probe AF complements Probe AC (NIS/NEES consistency, Bar-Shalom 2001
§5.4) on the orthogonal question of whether ``Q`` specifically is
mis-tuned. AC catches general miscalibration of the innovation
covariance; AF localises the cause to Q by exploiting the lag-k
autocorrelation structure that R-only miscalibration leaves untouched.*

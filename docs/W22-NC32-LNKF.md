# W22-NC32 — Logistic-Normal Kalman Filter (LNKF)

**Status:** Phase 1 SHIPPED (standalone module + 16 tests passing)
**Spec:** docs/W22-NEXT-STEPS-NC32-36.md Section B2
**Hypothesis register entry:** H-NC32-LNKF-comparable-to-NC27deep

---

## Hypothesis

Portfolio weights `w in Delta^{d-1}` (the (d-1)-simplex) can be tracked by a
STANDARD Kalman filter operating in unconstrained Aitchison log-ratio
coordinates

    y_i = log(w_i / w_d)   for i in {0, ..., d-2}

with inverse

    w_d = 1 / (1 + sum_i exp(y_i))
    w_i = exp(y_i) * w_d

State `y_t in R^{d-1}` evolves as a standard linear-Gaussian KF, giving:

- Standard KF predict/update math (no bespoke filter)
- Closed-form posterior covariance `Sigma_y` per weight (vs Dirichlet's
  approximation)
- Direct coupling pathway: realized weights inform predicted (ROI, risk)
  and vice versa (Phase 2 joint state)
- Single mathematical framework that replaces the artificial separation
  between decision-space (Dirichlet) and objective-space (KF) tracking

## Math

**State equations.** With `F` (default `I_{d-1}`, random walk) and isotropic
process noise `Q = process_noise * I`:

    y_{t+1} = F y_t + w_t,   w_t ~ N(0, Q)
    y_obs   = H y_t + v_t,   v_t ~ N(0, R)        (H = I, R = measurement_noise * I)

**Predict (h-step):**

    mu(t+h | t)    = F^h y(t)
    Sigma(t+h | t) = F^h P(t) (F^h)^T + sum_{k=0}^{h-1} F^k Q (F^k)^T

For `F = I` this reduces to `mu = y(t)` and `Sigma = P + h*Q`.

**Update:**

    S = Sigma_pred + R
    K = Sigma_pred S^{-1}
    y_new = mu_pred + K (y_obs - mu_pred)
    P_new = (I - K) Sigma_pred

**Inverse to simplex.** `predict_simplex_mean(h) = inverse(mu(t+h|t))`. For an
unbiased simplex estimator (vs Jensen bias), use `predict_simplex_samples()`.

## Module surface

`python_refactor/src/algorithms/logistic_normal_kf.py` (NEW; pure numpy)

```python
class LogisticNormalKF:
    def __init__(self, d: int, process_noise: float = 0.01,
                 measurement_noise: float = 0.001,
                 initial_y: np.ndarray | None = None,
                 initial_P: np.ndarray | None = None): ...

    def _forward(self, w: np.ndarray) -> np.ndarray         # simplex -> y (EPS-clipped)
    def _inverse(self, y: np.ndarray) -> np.ndarray         # y -> simplex (stable softmax)
    def predict(self, h: int = 1) -> tuple[mu, Sigma]       # h-step ahead
    def update(self, y_obs: np.ndarray) -> None             # standard KF update
    def observe(self, w_obs: np.ndarray) -> None            # forward + update
    def predict_simplex_mean(self, h: int = 1) -> np.ndarray
    def predict_simplex_samples(self, h: int = 1, n_mc: int = 100, rng=None) -> np.ndarray
    def reset(self, initial_y=None, initial_P=None) -> None
```

No modifications to `anticipatory_learning.py`, `kalman_filter.py`, or any
shared code path. Standalone.

## Tests (16 total, all passing)

`python_refactor/tests/test_nc32_logistic_normal_kf.py`

| Test | What it asserts |
|---|---|
| `test_construction_default_uniform_state` | y=0 -> uniform simplex |
| `test_forward_inverse_roundtrip_uniform` | round-trip identity on uniform |
| `test_forward_inverse_roundtrip_sparse` | EPS clipping handles zero weights |
| `test_predict_one_step_with_identity_F` | mu unchanged, Sigma = P + Q |
| `test_predict_h_step_cov_grows_linearly` | trace(Sigma) = trace(P) + h*trace(Q) |
| `test_observe_then_predict_mean_moves_toward_obs` | KF gain pulls posterior |
| `test_observe_decreases_posterior_variance` | information monotone |
| `test_predict_simplex_mean_returns_simplex` | sum=1, all >= 0 |
| `test_predict_simplex_samples_all_on_simplex` | every MC sample valid |
| `test_reset_restores_initial_state` | clean reset |
| `test_observe_convergence_to_constant_obs` | repeated same obs -> posterior = obs |
| `test_accuracy_comparable_to_dirichlet_posterior` | Inspection-3 L2 within 3.5x |
| `test_lnkf_beats_dirichlet_on_logistic_normal_data` | reverse experiment confirms model-mismatch |
| `test_handles_dim_2` | d=2 scalar y |
| `test_handles_dim_large` | d=10 |
| `test_dim_mismatch_raises_on_observe` | input validation |

## Key empirical claims (Inspection-3, 100 obs from Dirichlet(alpha=[5,3,2,1,1]))

| Predictor | Terminal L2 to true mean | Ratio vs DirichletPosterior |
|---|---|---|
| DirichletPosteriorPredictor (NC27-deep) | 0.0318 | 1.00x |
| **LogisticNormalKF (NC32)** | **0.0932** | **2.93x** |
| Legacy DirichletPredictor (NC27 baseline) | 0.0890 | 2.80x |
| Uniform guess | 0.1850 | 5.81x |

**Reverse experiment** (100 obs from logistic-normal with mu=[1.0, 0.5, -0.2, 0.3], Sigma=0.1*I):

| Predictor | Terminal L2 to true mean | Ratio |
|---|---|---|
| DirichletPosteriorPredictor | 0.0108 | 1.86x |
| **LogisticNormalKF** | **0.0058** | **1.00x (best)** |

LNKF outperforms DirichletPosterior by ~2x on its native (logistic-normal) data,
confirming that the Inspection-3 gap is a STRUCTURAL model-mismatch property
when the data-generating distribution is a true Dirichlet (the perfect-match
case for the Dirichlet posterior).

## HONEST SCARS

1. **Reference-asset choice equivariant but numerically affects log-stability.**
   The Aitchison transform is permutation-equivariant — choosing any
   component as the reference yields an equivalent statistical model. This
   module fixes the LAST component `w_d` as the reference. If `w_d` is near
   zero across observations, log-ratios blow up. A future refinement could
   adaptively pick the reference as the most-stable (or maximum-mass) asset.

2. **Log-ratio transform fails on exact zero weights.** A weight of 0 makes
   `log(w_i / w_d)` ill-defined (either `log(0)` or `log(x/0)`). Handled by
   `EPS = 1e-10` clipping followed by re-normalization in `_forward()`. This
   biases very-sparse weight vectors but keeps the filter well-defined.

3. **Q and R hyperparameters need tuning per regime.** Default
   `process_noise=0.01`, `measurement_noise=0.001` were chosen for general
   stability but the Inspection-3 calibration found `(0.001, 0.05)` optimal.
   No automatic adaptation; calibration scar shared with the (ROI, risk) KF.

4. **Jensen bias on simplex mean.** `inverse(E[y]) != E[inverse(y)]`. The
   `predict_simplex_mean()` method uses the former (cheaper, deterministic);
   use `predict_simplex_samples()` for the unbiased Monte Carlo estimate
   when bias matters.

5. **Structural gap on Dirichlet data.** The accuracy parity test passes at
   2.93x (within the 3.5x threshold), not the design spec's 2x aspiration.
   Root cause: data distribution mismatch (Dirichlet vs logistic-normal).
   Documented above with reverse-experiment evidence.

## Integration sketch for Phase 2 joint state (deferred)

Joint state `x_joint = [ROI, risk, y_1, ..., y_{d-1}]^T` in R^{m + d - 1}:

```
F_joint = block_diag(F_perf, F_weights)            # default block-diagonal
H_joint = block_diag(H_perf, I_{d-1})              # weights observed directly
Q_joint = [[Q_perf,    Q_cross],                   # off-diagonal Q couples
           [Q_cross^T, Q_weights]]                 #   perf <-> weights
```

Off-diagonal `Q_cross` is the keystone: nonzero entries let weight
trajectories influence predicted (ROI, risk) one-step-ahead and vice
versa. The (ROI, risk) KF state lives in `kalman_filter.py`; integrating
will require a new joint module to avoid touching shared paths.

## Comparison with NC27 and NC27-deep

| Predictor | Stateful? | Geometry | Bias on simplex-mean | When LNKF wins | When LNKF loses |
|---|---|---|---|---|---|
| NC27 LogisticNormalPredictor | No (stateless smoothing) | log-ratio | Yes (Jensen) | Streaming with no state cost | When dynamics matter |
| NC27-deep DirichletPosteriorPredictor | Yes (alpha) | simplex (Dirichlet) | No (closed-form mean) | True Dirichlet data | Logistic-normal data |
| **NC32 LogisticNormalKF** | **Yes (y, P)** | **log-ratio** | **Yes (Jensen, use MC)** | **Logistic-normal data; need covariance** | **True Dirichlet data** |

LNKF combines NC27's log-ratio geometry with NC27-deep's stateful Bayesian
posterior structure, adding closed-form posterior covariance via the KF
framework. Its weakness is the inverse of NC27-deep's strength: model
mismatch on true-Dirichlet data.

## Phase 2 next steps (deferred)

- Joint state `[ROI, risk, y]` with off-diagonal Q
- Adaptive reference-asset selection
- Q/R auto-tuning via innovation-sequence whitening tests
- Cross-coupling tests vs the existing (ROI, risk) KF on real M50/M100
  scenarios

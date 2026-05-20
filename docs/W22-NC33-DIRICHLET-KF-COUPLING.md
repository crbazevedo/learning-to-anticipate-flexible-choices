# W22-NC33 — Dirichlet ↔ KF Coupling Module

**Status:** SHIPPED (standalone module + 15/15 tests + this doc)
**Linked:** operator directive 2026-05-20 — "direct linkage and integration between
Dirichlet and KF mechanisms, making their applications less isolated and more
coupled in terms of sequential application and/or param derivations"
**Hypotheses tested:**
- H-NC33a: Dirichlet posterior precision can SCALE KF process noise Q
- H-NC33b: KF residual magnitude can MODULATE Dirichlet concentration_increment

---

## Math

### Direction 1: Dirichlet posterior → KF Q scaling

For a Dirichlet posterior with concentration α (length d):
- Posterior precision = Σα (large → tight posterior)
- Posterior variance per component: Var(X_i) = α_i (Σα - α_i) / (Σα²(Σα+1))
- `posterior_variance_max` = max_i Var(X_i)
- Uniform-prior baseline: `baseline_var = (d-1) / (d² (d·α_prior + 1))`
- Effective Q scaling:
  ```
  scale = min(max_scale, max(1.0, 1 + posterior_variance_max / baseline_var))
  Q_eff = Q_base · scale
  ```

Behavior:
- α at uniform prior (α_i = α_prior): posterior_var ≈ baseline → scale ≈ 2
- α very tight (large Σα): posterior_var → 0 → scale → 1 (no inflation)
- α very diffuse (small Σα): posterior_var > baseline → scale > 2 (up to max_scale)

### Direction 2: KF residual → Dirichlet concentration_increment

Map KF innovation magnitude to per-observation Dirichlet update strength:
```
increment = clamp(||residual||_2 / residual_baseline,
                  min_increment, max_increment)
```

Large innovation (model surprise) → larger increment → faster Dirichlet update.

### Coupled cycle

`coupled_predict_update_cycle(portfolio_obs, objective_obs, objective_pred,
dirichlet_predictor, kalman_state)` runs both directions in sequence:

1. Compute KF residual = objective_obs − objective_pred
2. Map residual → concentration_increment
3. Update Dirichlet posterior with portfolio_obs at that increment
4. Compute Dirichlet posterior precision/variance
5. Return effective Q (scaled by current posterior) + posterior mean + residual telemetry

---

## Module surface

`src/algorithms/dirichlet_kf_coupling.py`:

- `dirichlet_posterior_variance_max(alpha)` — closed-form max Var(X_i)
- `dirichlet_posterior_precision(alpha)` — Σα (posterior precision)
- `baseline_posterior_variance(d, alpha_prior)` — uniform-prior baseline
- `scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior, max_scale)` — Direction 1
- `kf_residual_to_dirichlet_concentration_increment(residual, baseline, min, max)` — Direction 2
- `coupled_predict_update_cycle(...)` — runs both directions in sequence

---

## Falsifiable criteria

All 15 tests in `tests/test_nc33_dirichlet_kf_coupling.py` PASS:

| Test | Verifies |
|---|---|
| `dirichlet_posterior_variance_max_known_value` | closed-form Var for α=[1,1,1] |
| `dirichlet_posterior_variance_max_asymmetric` | closed-form Var for α=[5,3,2] |
| `dirichlet_posterior_precision_is_alpha_sum` | precision = Σα |
| `baseline_posterior_variance_matches_uniform_dirichlet` | baseline formula |
| `scale_Q_at_uniform_prior_is_near_2` | uniform prior → scale ≈ 2 |
| `scale_Q_with_high_concentration_returns_low_scale` | tight → scale → 1 |
| `scale_Q_with_diffuse_posterior_can_exceed_2` | very diffuse → scale > 2 |
| `scale_Q_capped_at_max` | hard cap respected |
| `scale_Q_preserves_matrix_shape` | works on matrix Q (4×4 identity) |
| `residual_to_increment_monotonic` | larger residual → larger increment |
| `residual_to_increment_clamped` | [min, max] respected |
| `residual_to_increment_vector_uses_l2_norm` | vector input → L2 norm |
| `coupled_cycle_returns_all_keys` | dict shape contract |
| `coupled_cycle_high_residual_increases_increment` | Direction 2 verified |
| `coupled_cycle_updates_posterior_mean_toward_obs` | Bayesian update direction correct |

---

## Honest scars

- **Standalone module**: this is COMPOSITION machinery, NOT a wired-in production
  change. The KF and Dirichlet predictors continue to work independently. To
  actually use NC33, a future structural fix would wire `coupled_predict_update_cycle`
  into the production predict-update loop (e.g., in
  `anticipatory_learning.py:predict_anticipative_solution`).
- **Heuristic Q scaling**: the `1 + var/baseline` formula is one defensible
  choice; alternatives (e.g., `exp(precision_ratio)`) deferred. The math is
  in the right ballpark for the operator's intent ("tight posterior → low Q")
  but the exact functional form needs FTSE empirical validation.
- **No bivariate Dirichlet variance**: we use `max_i Var(X_i)` as a scalar summary.
  Full Dirichlet covariance matrix is available but the KF Q is typically scalar-
  or isotropic-scaled, so summarizing to scalar is appropriate.
- **residual_baseline default = 0.01** for the typical ROI/risk scale; operator
  can tune. This is a SCALE parameter, not a tuning knob.
- **min_increment = 0.1, max_increment = 5.0** are heuristics; bound the range
  to avoid runaway updates from outlier residuals (also prevents zero-update
  from tiny residuals).

---

## Composes with

- **NC27-deep** (`DirichletPosteriorPredictor`): the natural Dirichlet predictor
  that provides the posterior α. The wrapper's per-portfolio state is the
  natural place where NC33 would attach.
- **NC32** (`LogisticNormalKF`): an alternative coupling — instead of using
  Dirichlet posterior, use LN-KF's posterior variance directly. NC33's
  scale_Q logic could be lifted to take generic posterior-variance input,
  making it predictor-agnostic.
- **NC36** (analytical TIP): TIP uses KF Σ; if NC33 effectively grows Σ via Q
  inflation, TIP will respond (lower confidence in predictions → higher TIP).

---

## Operator action items

1. **Wire NC33 into production**: modify `anticipatory_learning.py:dirichlet_mean_prediction`
   (or NC27-deep wrapper) to call `coupled_predict_update_cycle` and use the
   returned `effective_Q` for the next KF predict step.
2. **Tune residual_baseline + max_scale** on FTSE: characterize typical KF
   residual magnitude; pick baseline ≈ median residual.
3. **Smoke run on FTSE n=5**: compare wealth between independent (legacy) vs
   coupled (NC33) modes.

---

## Theoretical connection: a unified hierarchical Bayesian view

NC33 implements (informally) a HIERARCHICAL coupling:
- Top level: KF state (ROI, risk) with process noise Q
- Bottom level: Dirichlet posterior over portfolio weights with concentration α
- Cross-level: Q adapts to α (Direction 1); α update rate adapts to KF residual (Direction 2)

This is a step toward a true hierarchical Bayesian filter where the SAME unified
posterior tracks (ROI, risk, weights) jointly — which is what NC32 Phase 2
(joint state KF) would do explicitly. NC33 is the SEQUENTIAL approximation;
NC32 Phase 2 is the JOINT formulation. Both deliver the same coupling intent.

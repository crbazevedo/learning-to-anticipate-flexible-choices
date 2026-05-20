# W22-NC36 — Analytical TIP under bivariate Gaussian

**Status:** SHIPPED (10/10 NC36 tests; 34/34 all TIP tests pass)
**Linked:** Inspection 1 (tip_analytical_conditional), NC31 (Defn 6.1 conditional)
**Hypothesis:** TIP can be computed in closed form under the bivariate Gaussian
forecast model used by the KF, eliminating MC noise entirely.

---

## Math

Per Defn 6.1, with current = (c_ROI, c_risk) OBSERVED and predicted ~ N(μ, Σ):

```
z1 = (c_ROI  − μ_ROI)  / σ_ROI
z2 = (c_risk − μ_risk) / σ_risk
ρ  = Σ_{12} / (σ_ROI · σ_risk)

Φ(z)  = standard normal CDF
Φ_2(z1, z2; ρ) = bivariate normal CDF with correlation ρ

P[current dominates predicted]
  = P[P_ROI < c_ROI ∧ P_risk > c_risk]
  = Φ(z1) − Φ_2(z1, z2; ρ)

P[predicted dominates current]
  = P[P_ROI > c_ROI ∧ P_risk < c_risk]
  = Φ(z2) − Φ_2(z1, z2; ρ)

TIP = 1 − P[c dom] − P[p dom]
```

Bivariate Φ_2 computed via `scipy.stats.multivariate_normal.cdf`.

---

## Benefits over MC

| Aspect | MC (monte_carlo_samples=1000) | Analytical (NC36) |
|---|---|---|
| Variance | ~1/√1000 ≈ 3% | 0 (deterministic) |
| Wall time | O(N_MC · 2) = 2000 ops | O(1) Φ + 1 Φ_2 |
| Determinism | depends on RNG | exact |
| Convergence | asymptotic | exact closed-form |

For Inspection 1's close-means scenario, analytical TIP matches MC(10000) within ±0.05 absolute (test-verified).

---

## Module surface

`src/algorithms/temporal_incomparability_probability.py`:

- `_calculate_tip_analytical_conditional(c_ROI, c_risk, p_ROI, p_risk, predicted_cov)` —
  the closed-form computation
- `_calculate_tip_with_covariance(...)` — dispatches to analytical when
  `W22_NC36_TIP_ANALYTICAL=1` env var is set; otherwise MC path (with optional
  NC31 conditional mode)

---

## Activation

```bash
export W22_NC36_TIP_ANALYTICAL=1
```

Default OFF for backward compatibility.

---

## Falsifiable criteria

All 10 tests in `tests/test_nc36_tip_analytical.py` PASS:

| Test | Verifies |
|---|---|
| `analytical_determinism` | 3 calls → identical result (no MC noise) |
| `analytical_matches_mc_close_means` | analytical vs MC(10000) diff < 0.05 |
| `analytical_degenerate_predicted_cov_falls_back` | σ→0 + dominance → TIP=0 |
| `analytical_degenerate_predicted_cov_non_dominance` | σ→0 + non-dominance → TIP=1 |
| `analytical_predicted_strictly_dominates_low_tip` | TIP < 0.05 |
| `analytical_handles_extreme_correlation` | ρ = ±1 doesn't crash (clamped to ±0.999999) |
| `env_var_switches_to_analytical` | env var routes correctly |
| `env_var_default_uses_mc` | default path is MC |
| `analytical_output_in_clamp_range_when_clamp_active` | composes with NC13b clamp |
| `analytical_with_zero_diff_means_returns_0_5` | analytical symmetry: equal means → TIP = 0.5 exactly |

---

## Honest scars

- Requires `predicted_cov` to be PSD (KF guarantees this; degenerate σ_ROI=0 or
  σ_risk=0 falls back to point-vs-point dominance)
- ρ = ±1 degenerates the bivariate Normal; epsilon clamp keeps ρ ∈ [-0.999999, 0.999999]
- `scipy.stats.multivariate_normal.cdf` is a NEW dependency — was not imported
  previously in this module. Import added at module top.
- Defn 6.1 ONLY (current FIXED; predicted sampled). The "joint sampling"
  alternative (current also sampled) doesn't have a closed-form bivariate
  expression — would require numerical quadrature over the 4-D joint
  distribution. NC36 is Defn-6.1 only.

---

## Composes with

- **NC31** (TIP conditional mode): NC36 is implicitly conditional (current FIXED).
  If NC31 enabled WITHOUT NC36, MC path runs in conditional mode.
- **NC13b** (smooth clamp): NC36 output passes through `_clamp_tip` which honors
  the smooth-clamp env var.
- **All anticipation code paths** that call `calculate_tip(...)` — switching env
  var transparently changes all downstream λ^H computations.

---

## Operator action items

1. Toggle `W22_NC36_TIP_ANALYTICAL=1` and run FTSE n=10 smoke
2. Compare λ^H trace (per-period values) between MC and analytical modes
3. If λ^H values are stable across calls under analytical (vs noisy under MC),
   ratify analytical as default
4. Compose with NC13b smooth clamp + NC31 conditional → all three deterministic
   modes can stack

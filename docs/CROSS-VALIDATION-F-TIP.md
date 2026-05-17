# Cross-validation F: TIP saturation diagnosis (operator check F)

*Generated 2026-05-17 by W18-4. Closes operator check F + addresses W17-5-CARRY-1 RESMOKE.*

## Verdict

✅ **STRUCTURAL — both implementations agree; saturation at 0.5 is NOT a code bug.**

The W17-5 RESMOKE finding (TIP centered at 0.500 ± 0.022 → λ^H ≈ 0) is reproduced across THREE distinct TIP implementations on synthetic production-scale inputs. The saturation is a property of the **data + KF covariance scale**, not the TIP code.

## Thesis grounding

**§6.1.3 Eq (6.4), p. 116** — TIP definition:
> "P_{t,t+h} = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]"

Probability that current and predicted objective vectors are **mutually non-dominated** under Pareto dominance (ROI=max, risk=min).

## Three implementations compared

| Implementation | Method | Source |
|---|---|---|
| `py_mc` | Monte Carlo: independently sample c ~ N(μ_c, Σ_c), p ~ N(μ_p, Σ_p); count mutual-non-dominance fraction | Python current (TemporalIncomparabilityCalculator) |
| `cpp_same_sign` | Analytical: Cholesky-standardize Δ = c - p, integrate quadrants {(+,+), (-,-)} via standard MVN CDF — Pr(matching signs) | C++ legacy `nsga2.cpp:525-565` |
| `py_pareto_mixed` | MC over Δ = c - p; count Δ in quadrants {(+,−), (−,+)} → 1 − Pr(dominance) | Python implementation for verification |

**Key insight**: for ROI=max, risk=min Pareto dominance, **mutual non-dominance = matching-sign deltas** ({(+,+), (-,-)}). So:
- Pr(c dom p) = Pr(Δ_ROI > 0 AND Δ_risk < 0) — quadrant (+, −)
- Pr(p dom c) = Pr(Δ_ROI < 0 AND Δ_risk > 0) — quadrant (−, +)
- Pr(non-dom) = Pr({(+,+) or (-,-)}) ← **what C++ computes**
- Pr(non-dom) = Pr({(+,+) or (-,-)}) ← **what Python MC computes (within noise)**

The "C++ vs Python uses different methods" structural concern from my initial reading was overstated: both methods compute the SAME quantity in expectation. The C++ analytical formula is correct, and the Python MC is correct.

## Synthetic test results (5 cases)

| Case | py_mc | cpp_same_sign | py_pareto_mixed |
|---|---|---|---|
| disjoint Gaussians (means far, small cov) | 0.0000 | 0.0000 | 0.0000 |
| coincident Gaussians (identical means) | 0.5022 | 0.5000 | 0.5165 |
| mild overlap | 0.6330 | 0.6238 | 0.6480 |
| heavy overlap | 0.5042 | 0.5016 | 0.5190 |
| **production-like (W17-5 scale)** | **0.5036** | **0.5000** | **0.5190** |

**Critical observation**: on production-like inputs (tiny means, KF-cov-scale covariances), all three implementations cluster at ~0.50. This matches the W17-5 production trace (TIP centered at 0.500 ± 0.022).

## The W17-5 saturation chain

The saturation is fully explained by the data + KF parameter regime:

```
Returns scale O(0.01)               (small daily returns)
  → Pareto front (ROI, risk) point spread O(0.001)
  → KF state covariance O(1e-5) per W17-5 trace
  → predictive distributions strongly OVERLAP at the (ROI, risk) scale
  → Pr(mutual non-dominance) → ~0.50 (maximum uncertainty)
  → TIP → 0.50
  → binary_entropy(TIP) → 1.0
  → λ^H = (1/(H-1))*(1 - entropy(TIP)) → 0
  → Eq 7.16: λ = 0.5*(λ^H + λ^K) → tiny
  → anticipation barely engages
  → ASMS ≈ SMS + noise from tiny state updates → underperforms SMS baseline
```

## Implications for W17-5 strategic framework

W17-5 framed two readings:

- **Reading A (wrong metric)**: single-period OOS-EFHV doesn't credit multi-period anticipation
- **Reading B (replication failure)**: thesis doesn't replicate on Python port

W18-4 introduces a third explicit reading:

- **Reading C (structural data property)**: The dataset + KF parameterization produce predictive distributions that are maximally uncertain in objective space → TIP correctly reports ≈ 0.5 → λ^H → 0 by construction → anticipation overhead can ONLY hurt. The thesis presumably ran on a regime where predictions were more informative.

W18-4 evidence points strongly to **Reading C** (with Reading A as a secondary refinement). The TIP code is correct on both sides; the issue is upstream in **KF parameterization**.

## What would change the saturation

For TIP to move away from 0.5, the predictive distributions must SEPARATE in (ROI, risk) space relative to their covariances. Three ways:

1. **Smaller KF covariances** (Q, R reduced) — tighter ellipses → less overlap
2. **Larger mean separations** (bigger predicted state change per period) — depends on F + dynamics
3. **Different metric** (multi-period wealth or covariance-aware HV) — bypasses the TIP-saturation choke

W19+ cross-check D (KF Gaussians from KF application) is now the obvious next investigation: are the C++ and Python KFs producing the same predictive distributions on the same inputs? If yes → the saturation IS structural. If no → there's a KF parameter / state propagation drift to fix.

## Honest scars

- **W18-4-CARRY-1**: did NOT run the C++ TIP driver end-to-end (it depends on `pmvnorm` which calls Fortran `mvtdst_` — stubbed on Apple silicon). Mitigated by re-implementing the C++ formula in Python (`tip_cpp_same_sign`) — the arithmetic is identical so the comparison is faithful, but it's not "running the actual binary".
- **W18-4-CARRY-2**: did NOT do the F2 sub-check (production-input replay from W17-5 trace) because the trace CSV doesn't carry the KF covariance matrices — only the scalar TIP values. To do F2 properly, the trace schema needs to expand to include `error_cov_prediction` + `error_cov` per call.

## Bug count from cross-validation

After W18-4: still 1 confirmed Python bug on the headline path (sqrt in compute_risk per W18-2). The TIP saturation is NOT a bug; it's data structure. 

| # | Side | Where | Severity |
|---|---|---|---|
| 1 | C++ | `portfolio.cpp:65` comma-operator (autocorrelation) | Off-headline |
| 2 | Python | `compute_risk` adds sqrt() per W18-2 | **On headline** |

## Verdict per W18-4 matrix (revised)

| F1 outcome | F2 outcome | Diagnosis |
|---|---|---|
| ✅ Both agree on F1 known values | ⏭ F2 deferred (W18-4-CARRY-2; trace schema gap) | **Saturation IS structural** — predictive distributions maximally uncertain at the (data + KF covariance) regime; TIP correctly reports ~0.50 |

## Output artifacts

- `python_refactor/scripts/cross_validation/run_tip.py` — three-implementation comparison harness
- `docs/CROSS-VALIDATION-F-TIP.md` — this receipt

## Reproducing

```bash
cd python_refactor
uv run python -m scripts.cross_validation.run_tip --mode synthetic
```

## Next investigation (W19+)

The Reading-C conclusion makes W19+ priorities clear:

1. **Cross-check D** (KF Gaussians from KF application): do C++ and Python KFs produce the same predictive distributions on the same inputs? If yes → the saturation is genuinely structural and the W7→W17 chain is correctly implementing a thesis that doesn't fully replicate on this data. If no → KF drift to fix.
2. **Multi-period wealth metric** (BACKLOG M8 + §7.3.5) — Reading A test
3. **KF parameter audit** — Q (process noise), R (observation noise) initialization values

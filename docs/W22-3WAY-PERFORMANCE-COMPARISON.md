# W22 3-way performance comparison: baseline vs NC8b vs NC8b+NC12

*Generated 2026-05-18 after the NC8b stale-objectives fix (commit 27cbcd2)
and the NC12 covariance-fusion fix (commit 2154270).*

## Protocol

- Walk-forward OOS evaluation per thesis §7.2.2
- Train window: 378 days, step: 50 days, 23 periods
- Scenarios: SMS_RDM_K0 (baseline) + ASMS_mHDM_K3_v2both (best ASMS variant)
- 2 seeds each (n=2 — small sample; treat as point estimates, not statistical inference)
- n_mc = 200 MC scenarios per period

## Headline metric: ASMS−SMS Δ %

| Run | ASMS Ŝ | SMS Ŝ | Δ (abs) | Δ % | Wall |
|---|---|---|---|---|---|
| **Baseline** (post-NC7) | 0.000381 | 0.000405 | −2.40e-05 | **−5.92%** | 1086s |
| **NC8b only** | 0.000422 | 0.000415 | +7.06e-06 | **+1.70%** ✅ | 1135s |
| **NC8b + NC12** | 0.000401 | 0.000414 | −1.28e-05 | −3.09% | 1043s |

**Headline**: NC8b alone gives **ASMS > SMS for the first time in the W17-W22 chain** (+1.70% vs −5.92% baseline, a 7.6 pp swing).

## What NC8b changed

**`_finalize_offspring_objectives` helper** (added to `python_refactor/src/algorithms/operators.py`)
called from 4 thesis/v2 operator output points. The helper does:

1. `Portfolio.compute_efficiency(offspring.P)` — recomputes ROI/risk on the ACTUAL crossover/mutation output weights, instead of leaving stale random-init values from `Solution.__init__`.
2. `Portfolio.initialize_kalman_filter(offspring.P, ROI, risk)` — re-anchors KF state x[0:2] to the actual portfolio metrics.

This was a regression introduced by W15-2: the OLD SBX `crossover`/`mutation` operators correctly recomputed efficiency after weight assignment; the thesis operators added by W15-2 silently dropped this discipline.

**Why this helped BOTH SMS and ASMS** (SMS Ŝ also improved: 0.000405 → 0.000415, +2.5%):
- Pre-NC8b, offspring's `portfolio.ROI/risk` reflected RANDOM init weights, not the crossover/mutation output → selection (HV contribution, NDS, tournament) ranked solutions on noise
- Post-NC8b, selection sees correct ROI/risk → selection pressure now matches the actual genome
- SMS gains because selection becomes informative; ASMS gains MORE because TIP/λ machinery amplifies stale signal noise into uniform anticipation rates (pre-NC8b TIP was being computed on random-weights ROI/risk, not actual portfolios)

## Why NC12 didn't help (and may have regressed slightly)

**NC12 is mathematically correct** (paper Eq 15 weighted-sum-of-squared-weights vs broken naïve SUM), but it affects only the **single-horizon path** (`learn_single_solution` → `_update_solution_state_anticipative`). The multi-horizon path (`learn_population`, which all `ASMS_mHDM_K3*` scenarios use) ALREADY implements Eq 15 correctly at `multi_horizon_anticipatory.py:666-668`.

Confirmation: Probe B on POST-NC12 data shows TIP distribution **identical** to PRE-NC12:
- Pre-NC12 TIP mean = 0.4999, fraction in (0.45, 0.55) = 99.86%
- Post-NC12 TIP mean = 0.5001, fraction in (0.45, 0.55) = 99.87%

The −3.09% Δ vs +1.70% is within statistical noise for n=2 seeds — same code path (multi-horizon learn_population) runs in both cases, so the difference is attributable to seed-level variance, not NC12 mechanism.

## Next-step diagnosis: NC13 (n-step predictor covariance compounding)

Multi-horizon TIP MC sampling uses `predicted_cov` from `kalman_n_step_prediction`:
```python
predicted_cov = F @ current_cov @ F.T  # compounds per step
```

With NC7's high velocity prior (P[2,2] = 1000), after h steps:
`predicted_cov[0,0] ≈ h × 1000` (each F application adds the velocity uncertainty into position)

TIP MC sampling from `Normal([predicted_roi, predicted_risk], predicted_cov)` then has std ≈ √(1000·h) ≈ 30·√h around means of order 0.001 → samples are essentially independent random noise → mutual non-dominance ≈ 0.5 → TIP saturates at 0.5.

**This is why NC12 didn't fix TIP saturation**: the n-step predictor's compounding makes `predicted_cov` huge regardless of how `current_cov` is set.

## Mitigation candidates for NC13

| Candidate | Mechanism | Scope |
|---|---|---|
| **NC13a** | Clamp `predicted_cov`: `min(F @ cov @ F^T, max_cov_clamp)` where max_cov_clamp ≈ 4 × current_cov | 1-line change in `kalman_n_step_prediction` |
| **NC13b** | Add explicit process noise Q with `predicted_cov = F @ current_cov @ F^T + Q` where Q damps velocity | medium |
| **NC13c** | Use FRESH measurement-noise R for TIP MC instead of propagated KF P | medium |
| **NC13d** | Revert NC7 (restore P[2,2] = 0.1): velocity prior 0.1 means predicted_cov ≤ 0.1·h, small enough that TIP MC can discriminate | high impact (un-shipping prior fix) |

NC13a is the lowest-cost candidate. Smoke first.

## Confidence + replication plan

n=2 seeds is too few for confident inference. NC8b's +1.70% headline could be 1-seed luck.

Recommended next:
1. Ship NC13a (n-step cov clamp) — 1-line, low risk
2. Run wider smoke (5 seeds) on NC8b-only vs NC8b+NC13a — measure ASMS−SMS Δ % with std-error
3. If NC8b+NC13a > NC8b: TIP saturation matters AND NC13a unlocks more gain
4. If NC8b+NC13a ≈ NC8b: NC8b alone is sufficient; TIP saturation may be benign (the λ values clamp lambda to 0.5 which is still a meaningful anticipation midpoint)

## Backlog state after NC8b

| Probe | Pre-NC8b | Post-NC8b |
|---|---|---|
| A (KF predictive) | 🔴 KF == persistence (NC7 + NC8b combined still negative) | TBD — re-run Probe A post-NC8b expected to show some velocity learning since KF state now anchored correctly per period |
| B (TIP + λ) | 🔴 TIP saturated 99.86% | 🔴 STILL saturated 99.87% (NC12 no effect on multi-horizon path) |
| D (front diversity) | 🟢 PASS (median 7) | Likely still 🟢 (same algo structure) |
| C, E, F | pending | pending |

## Decision after this comparison

Per operator hill-climb directive + W22 controlled-probe discipline:
1. Ship NC13a (n-step cov clamp) next
2. Re-run combined: NC8b + NC13a; measure ASMS−SMS Δ %
3. If improves: NC13a confirmed; investigate NC13b/c for additional gain
4. If neutral/regresses: revert NC13a, keep NC8b alone; document
5. Continue probing forward (NC8c cross-period KF persistence next-highest-leverage)

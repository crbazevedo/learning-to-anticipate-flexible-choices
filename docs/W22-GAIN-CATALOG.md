# W22 Gain Catalog — comprehensive matrix of structural fixes + helpers

*Living document, updated as new combinations are tested.*

## Index

1. [Structural fixes (NC7-NC8d)](#structural-fixes-required-precondition-chain)
2. [Helper fixes (NC12-NC22)](#helper-fixes-orthogonal-modulators)
3. [Probe-derived knobs (Probe H, I)](#probe-derived-knobs-pop-gens--cost)
4. [Datasets](#datasets-tested)
5. [Combination matrix (validated)](#combination-matrix-validated)
6. [Untested combinations (next-round targets)](#untested-combinations-next-round)
7. [Mechanism map](#mechanism-map)

## Structural fixes (required precondition chain)

| Fix | Mechanism | Required for | Status |
|---|---|---|---|
| **NC7** P-init harmonization | KF prior covariance `diag([0.1, 0.1, 1000, 1000])` consistent across all entry points | All KF-dependent fixes | ✅ shipped (`e6fc6ba`) |
| **NC8b** `_finalize_offspring_objectives` | Recompute ROI/risk + re-init KF on offspring's ACTUAL weights (closes W15-2 regression) | All selection-quality | ✅ shipped (`27cbcd2`) — **first +1.7 pp at n=2** |
| **NC13a** n-step cov clamp at 1.0 | Breaks positive-feedback P drift loop (NC14) — prevents covariance from exploding via Eq 15 self-reinforcement | NC8c-v2 stability | ✅ shipped (`3d41e91`) |
| **NC8c-v2** carry POSITION (not just velocity) | Cross-period prev_AMFC ROI/risk → first innovation `y = current.ROI − prev_AMFC.ROI` ≠ 0 per portfolio | The breakthrough's load-bearer | ✅ shipped (`e4e54f6`) |
| **NC8d** predict-before-first-update | Cross-terms in P_next enable K[2, 0] ≠ 0 → velocity ACTUALLY learns from innovation | NC8c-v2 effectiveness | ✅ shipped (`e4e54f6`) |

**Chain dependency**: NC7 → NC8b → NC13a → NC8c-v2 + NC8d (the breakthrough).

## Helper fixes (orthogonal modulators)

| Fix | Mechanism | Status |
|---|---|---|
| NC12 | Eq 15 covariance fusion (vs naïve SUM) — mathematically correct but NO production effect for multi-horizon path | ✅ shipped (`2154270`) — neutral |
| NC15 | Per-portfolio λ shrinkage by KF position uncertainty (env-var) | ⚠️ REJECTED at α=1.0 — uniform dampening, not differentiation |
| NC18 | Close-to-prev AMFC selector (env-var W22_DM_SELECTOR=close_to_prev) | ⚠️ REJECTED at λ=0.3 (ASMS -2.51% loss); MIXED at λ=0.5 (n=3 only) |
| NC22 | Closed-form AMFC (`--use-closed-form-efhv`) | ❌ REJECTED — −32% absolute perf (triggers AMFC fallback) |
| NC17 | Replace AMFC with Sharpe DM | ❌ DROPPED per operator (AMFC is thesis core) |

## Probe-derived knobs (pop/gens + cost)

| Knob | Mechanism | Status |
|---|---|---|
| **Probe H** pop=30/gens=40 | 50% more pop, 33% more gens (2× compute per pair) | 🟡 ASMS > SMS p=0.0098 but Δ-of-Δ NS vs baseline (p=0.358) |
| **Probe I** W22_DISABLE_TXN_COST=1 | Remove thesis Table 7.1 transaction cost | 🟢 BIG win at n=10 (+16.11%, p=0.0002); cost asymmetric tax +8.36 pp |
| Combined H+I (pop=30/gens=40 + no-cost) | Both helpers active | 🟡 Hits ASMS ceiling ~0.000494 — helpers don't stack additively |

## Datasets tested

| Dataset | n_assets | T (days) | Periods | Δ % (with cost) | n | Paired p |
|---|---|---|---|---|---|---|
| **FTSE 2006-2012** | 98 (87 filtered) | 1566 | 23 | **+7.50%** | 10 | **0.003** |
| **PO(8, 1.0)** | 30 | 1250 | 17 | **−7.25%** | 10 | 0.15 (NS) |
| **sPO(8, 1.0)-cosine** (NEW) | 30 | 1250 | 17 | TBD | in flight | TBD |

## Combination matrix (validated)

Notation: ✓ = active; baseline = post-NC7 (assumes NC7 always active since it's a precondition).

| Run | NC8b | NC13a | NC8c-v2+NC8d | Probe H | Probe I | NC18 | NC15 | Dataset | n | Δ % | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (post-NC7) |   |   |   |   |   |   |   | FTSE | 2 | −5.92% | — |
| + NC8b | ✓ |   |   |   |   |   |   | FTSE | 2 | +1.70% | — |
| + NC8b + NC12 | ✓ |   |   |   |   |   |   | FTSE | 2 | −3.09% | — |
| + NC8b + NC12 + NC13a | ✓ | ✓ |   |   |   |   |   | FTSE | 3 | −1.61% | — |
| + NC8b + NC12 + NC13a + NC8c | ✓ | ✓ | partial |   |   |   |   | FTSE | 5 | +0.78% | — |
| + NC8b + NC12 + NC13a + NC8c-v2+NC8d | ✓ | ✓ | ✓ |   |   |   |   | FTSE | 10 | **+7.50%** | **0.003** ✅ |
| + Probe H | ✓ | ✓ | ✓ | ✓ |   |   |   | FTSE | 10 | +8.89% | 0.0098 |
| + Probe I | ✓ | ✓ | ✓ |   | ✓ |   |   | FTSE | 10 | **+16.11%** | **0.0002** ✅ |
| + Probe H + Probe I | ✓ | ✓ | ✓ | ✓ | ✓ |   |   | FTSE | 5 | +13.65% | 0.0012 |
| + NC15 α=1.0 | ✓ | ✓ | ✓ |   |   |   | ✓ | FTSE | 5 | +10.73% | — |
| + NC18 λ=0.5 | ✓ | ✓ | ✓ |   |   | ✓ |   | FTSE | 3 | +11.02% noisy | — |
| + NC18 λ=0.3 | ✓ | ✓ | ✓ |   |   | ✓ |   | FTSE | 5 | −2.51% | — |
| + NC22 closed-form | ✓ | ✓ | ✓ |   |   |   |   | FTSE | 5 | +14.20%* | — |
| Production stack | ✓ | ✓ | ✓ |   |   |   |   | PO(8,1.0) | 10 | **−7.25%** | 0.15 NS |

*NC22 Δ% widens due to SMS degrading more; absolute ASMS perf drops −32%.

## Untested combinations (next-round targets)

These would reveal which structural fix is LOAD-BEARING vs SUPPORTING:

| Run | NC8b | NC13a | NC8c-v2+NC8d | Probe H | Probe I | Hypothesis | Predicted impact |
|---|---|---|---|---|---|---|---|
| Ablation A: drop NC13a |  ✓  |   | ✓ |   |   | NC13a may be unnecessary if NC8c-v2 dampens P drift via velocity carry | Test if breakthrough survives without clamp |
| Ablation B: drop NC8b | | ✓ | ✓ |   |   | Test if NC8c-v2 alone is sufficient (without offspring objective fix) | Likely worse — stale objectives feed back into KF |
| Ablation C: NC13a alone (no NC8c-v2) | ✓ | ✓ |   |   |   | Test P drift fix without velocity bootstrap | Should plateau near NC8b alone (+1.7%) |
| Ablation D: NC8c-v2 only (no NC8d) | ✓ | ✓ | partial |   |   | Already tested — gave +0.78% (chicken-and-egg without NC8d) | Confirmed inert |
| Production + NC18 λ=0.7 (helper, AMFC weighted toward EFHV) | ✓ | ✓ | ✓ |   |   | Less stability bias may avoid the λ=0.3 trap | Possibly between baseline +7.5% and λ=0.5's noisy +11% |
| Production + sPO bench | ✓ | ✓ | ✓ |   |   | Does breakthrough generalize to KF-friendly synthetic? | sPO smoke IN FLIGHT — predicted ASMS > SMS |
| Production + Probe I + sPO | ✓ | ✓ | ✓ |   | ✓ |   | Mechanism + cost-free + smooth synthetic | Predicted near sPO + cost-asymmetry margin |
| NC8b + Probe I alone (no NC8c-v2/NC8d) | ✓ |   |   |   | ✓ |   | Does cost-asymmetric tax help even without KF bootstrap? | Test if Probe I is decorrelated from breakthrough |
| Production + Probe H + Probe I + NC18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   | Maximum helper stack | If saturates at ceiling: confirms helpers compete |

## Mechanism map

```
┌─────────────────────────────────────────────────────────────────────┐
│ NC7: P-init harmonization (necessary precondition)                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NC8b: offspring ROI/risk/KF state reflects ACTUAL weights           │
│   ↓                                                                 │
│   ▶ Optimizer sees correct genome-to-fitness mapping                │
│   ▶ Δ ≈ +1.7% (selection quality alone, n=2)                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NC13a: clamp predicted_cov diag ≤ 1.0                               │
│   ↓                                                                 │
│   ▶ Breaks positive-feedback P drift loop (NC14)                    │
│   ▶ P[ROI] 486 → 0.049 (Probe E)                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NC8c-v2 + NC8d: cross-period KF state bootstrap                     │
│   ↓                                                                 │
│   ▶ Each portfolio's KF starts with prev_AMFC POSITION              │
│   ▶ First kalman_update sees y = current.ROI - prev_AMFC.ROI ≠ 0    │
│   ▶ Per-portfolio differentiated anticipative_mean                  │
│   ▶ NDS/HV/tournament sees per-portfolio differentiated objectives  │
│   ▶ Δ +7.50% n=10 p=0.003 (PRODUCTION-FAITHFUL BREAKTHROUGH)        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
        ┌─────────────────┐     ┌─────────────────────┐
        │ Probe H +pop/gens│    │ Probe I -txn cost   │
        │ +1.4 pp marginal │    │ +8.4 pp asymmetric  │
        │ p=0.0098 (Δ-of-Δ │    │ p=0.006 (Δ-of-Δ)    │
        │ p=0.358 NS)      │    │ (NEW asymmetric tax │
        └─────────────────┘     │  finding)           │
                                └─────────────────────┘
                                          │
                                          ▼
                      Combined H+I hits ASMS CEILING
                      at ~0.000494 — helpers don't
                      stack additively
```

## Pending experimental questions

1. **sPO(8,1.0)-cosine bench**: does the breakthrough generalize when KF dynamics are smooth? (IN FLIGHT)
2. **Ablation A** (drop NC13a): is the clamp necessary post-NC8c-v2?
3. **Ablation B** (drop NC8b): is NC8c-v2 sufficient alone?
4. **NC23** (process noise Q): theory says Q > 0 stabilizes KF; current Q = 0
5. **Probe K** (KF F-matrix variants): random walk, AR(1), AR(2) — is constant-velocity the right model?
6. **Probe L** (KF Q tuning sweep): find optimal Q for ASMS
7. **Probe M** (Pareto front shape): is the front concave/convex/linear in (ROI, risk) space?
8. **Probe N** (signal-to-noise ratio): characterize KF innovation vs measurement noise R

## Files added during this session

- `python_refactor/experiments/analyze_probe_{a,b,c,d,e,f,g,j}.py` (8 analyzers)
- `python_refactor/experiments/walk_forward_po_smoke.py` (PO smoke runner)
- `python_refactor/src/data/synthetic_po_generator.py` (PO generator + sPO variant)
- `python_refactor/src/algorithms/n_step_prediction.py` (NC13a clamp)
- `python_refactor/src/algorithms/sms_emoa.py` (NC8c-v2 + NC8d + NC15 + Probe I env)
- `python_refactor/src/portfolio/portfolio.py` (NC8c-v2 carried_position/velocity)
- `python_refactor/src/algorithms/operators.py` (NC8b `_finalize_offspring_objectives`)
- `python_refactor/src/algorithms/multi_horizon_anticipatory.py` (NC15 shrinkage)
- `python_refactor/experiments/walk_forward.py` (NC18 close-to-prev + Probe A log)
- `python_refactor/experiments/validation_matrix.py` (Probe H pop=30/gens=40 scenarios)
- `python_refactor/src/algorithms/anticipatory_learning.py` (NC12 Eq 15)
- `python_refactor/src/algorithms/kalman_filter.py` (NC7 P-init)

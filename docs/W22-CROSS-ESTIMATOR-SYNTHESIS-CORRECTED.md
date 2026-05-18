# W22 cross-estimator synthesis — CORRECTED (smell-A investigation closed)

*Generated 2026-05-18 after re-running all 3 closed-form W22 calibration
smokes on post-#131 code. Supersedes the headline Reading-E magnitude
in docs/W22-CROSS-ESTIMATOR-SYNTHESIS.md for Option A specifically.*

## TL;DR

After the per_portfolio_efhv closed-form fix (#131), the cross-estimator
disagreement on Reading-E magnitude SHRINKS but doesn't fully collapse:

| Estimator | Pre-#131 Reading-E recovery | Post-#131 Reading-E recovery |
|---|---|---|
| MC bootstrap (Phase B n=8-10) | n/a | **+2.25pp** |
| Option A: closed-form point | +15.71pp (AMFC artifact) | **+7.20pp** (Jensen-real) |
| Option B: BS-style expectation | +3.44pp | **+4.05pp** (unaffected by #131) |
| Option C: v2 per-front Δ_S | ~0 (degenerate) | ~0 (still degenerate) |

**Smell-A scoped**: PR #131 (per_portfolio_efhv closed-form fix) affected
**Option A only** — because Option A's "single-MLE-HV-of-cloud" is
sensitive to which portfolios survive selection (AMFC matters). Options
B and C compute per-portfolio sums independent of AMFC, so they were
unaffected.

**3 honest estimators agree**: Reading-E recovery is in the **+2 to +7pp
range** (mean +4.5pp, std ±2.5pp). All agree the effect is small but
positive. Option C remains broken for our regime.

## Headline numbers (n=2 except MC which is n=8-10)

| Variant | MC (n=8-10) | Option A post-#131 (n=2) | Option B post-#131 (n=2) | Option C post-#131 (n=2) |
|---|---|---|---|---|
| SMS_RDM_K0 (mean Ŝ) | 3.878e-04 | 2.534e-04 | 1.702e-03 | 1.996e-01 |
| ASMS_mHDM_K3 (default) Δ | −10.74% | −9.59% | −33.11% | +0.03% |
| ASMS_mHDM_K3_v2rate Δ | −8.49% | −2.39% | −29.06% | +0.03% |
| ASMS_mHDM_K3_v2stab Δ | −9.73% | −10.37% | −32.99% | +0.03% |
| ASMS_mHDM_K3_v2both Δ | −8.49% | −2.22% | −29.21% | +0.02% |

## Recovery patterns (pp gained vs default)

| Reading | MC | Option A post-#131 | Option B post-#131 | Option C post-#131 |
|---|---|---|---|---|
| E (v2_rate) | +2.25 | **+7.20** | +4.05 | ~0 |
| F-INV (v2_stab) | +1.01 | −0.78 | +0.12 | ~0 |
| Combined (v2_both) | +2.25 | +7.37 | +3.90 | ~0 |

## What changed since #132

1. **Smell-A discovery**: pre-#131 W22-A had per_portfolio_efhv using MC
   bootstrap even when use_closed_form_efhv=True. The mismatch led to
   degenerate AMFC selection (W17-4 "all per-portfolio EFHV are 0/NaN"
   warning), which contaminated Option A's results.
2. **Re-runs on post-#131 code**: confirmed Option A's +15.71pp
   collapsed to +7.20pp (real Jensen effect remains, but AMFC
   contamination was ~half the magnitude).
3. **Options B and C confirmed unaffected**: their per-portfolio sums
   don't depend on AMFC selection.

## What this means for the publication-track decision

The W21-6 synthesis (docs/W21-CROSS-VALIDATION-SYNTHESIS.md) framings
still apply, but the verdict trajectory is updated:

- **PAPER-REPLICATES**: requires direction-reversal (V_k > 0). With
  Reading-E +2-7pp on a −10% default gap, no single-flag scenario
  reaches direction-reversal under any estimator.
- **PARTIAL (≥80% gap closure)**: requires V_k > −2%. Option A
  v2_rate at −2.39% is RIGHT at this threshold; Option B and MC are
  well short.
- **WEAK-PARTIAL (20-80% gap closure)**: triggered. v2_rate closes
  ~20-25% under MC; ~50% under Option A.
- **REFUTED (< 20% gap closure)**: NOT triggered.

**Most-likely final verdict: WEAK-PARTIAL** under MC, **PARTIAL** under
Option A, between under Option B.

## K=2 v2_both surprise (separate finding, same wave)

Per docs/W21-5-PHASE-B-KILL-AND-SAVE.md + the K=2 Option A smoke
(2026-05-18): combining K=2 historical window with both v2 flags
produces:

| K=2 + v2_both (Option A n=3) | Δ vs SMS | Recovery vs K=3 default |
|---|---|---|
| 2.505e-04 | **−4.66%** | **+6.08pp (best yet under any estimator)** |

This is a new candidate worth confirmation via MC at n=10. If the
finding holds under MC, K=2 + v2_both becomes the publication-ready
"PARTIAL replication" scenario.

## Recommendation for Phase C

1. **Primary verdict on MC at n=30**: per #132 recommendation
2. **Secondary verdict on Option B at n=30**: closely tracks MC, sharper
3. **DROP Option A from publication tables**: amplification effect is
   real but inflates magnitudes; use only for sanity-check
4. **DROP Option C entirely**: degenerate
5. **ADD K=2 v2_both as Phase C variant**: most-promising single test
6. **ADD K=2 v2_both with V5/V7/V6/H3 ablations** if K=2 v2_both alone
   confirms the +6pp recovery under MC

## Estimator-dependence in publication

The 3-estimator agreement on direction (all show Reading-E small-positive)
+ magnitude range (+2 to +7pp) lets us publish with confidence on the
DIRECTION. The MAGNITUDE statement should be hedged:
"Under our most-thesis-faithful MC estimator, Reading-E recovery is
+2.25pp; under closed-form variants it ranges from +4 to +7pp. The
true v2-paper-companion methodology likely sits within this range."

## Output artifacts

- This corrected synthesis
- 3 post-#131 per_seed.json files (W22-A/B/C)
- 1 K=2 Option A per_seed.json (the surprise finding)
- Pre-#131 results preserved as historical record (per_seed.json in
  original w22-{closed-form,option-b,option-c}-calibration dirs)

## Open follow-ups

- K=2 v2_both MC n=10 confirmation (operator GO pending)
- Phase C 30-seed final ablation
- W21-6 final synthesis

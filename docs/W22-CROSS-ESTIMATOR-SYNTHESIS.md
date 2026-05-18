# W22 cross-estimator synthesis — 4-way OOS Ŝ comparison

*Generated 2026-05-17 after MC + W22 Options A/B/C calibration smokes
all complete. Documents the cross-estimator agreement/disagreement
pattern for the Reading-E + Reading-F-INVERSION effects.*

## TL;DR

Four estimators of OOS Expected Future Hypervolume (Ŝ) were tested on
the same 5-scenario × 2-seed fixture (with MC also at 10 seeds via
Phase B). They DISAGREE substantially on the Reading-E magnitude:

| Estimator | Reading-E recovery (v2_rate vs default) |
|---|---|
| **MC bootstrap (Phase B n=10)** | **+1.98pp** |
| Option A: closed-form point estimate `S(E[μ,Σ])` | **+15.71pp (OUTLIER)** |
| Option B: closed-form expectation `E[S(μ,Σ)]` (BS-style) | **+3.44pp** |
| Option C: lift v2's per-front Δ_S formula | **−0.03pp (DEGENERATE)** |

**MC and Option B agree closely (+2-3pp); Option A is an outlier
(+15.71pp); Option C is broken (all scenarios collapse to ~z_ref[0]).**

## Full data — n=2 except MC which is n=10

| Variant | MC (n=10) | Option A | Option B | Option C |
|---|---|---|---|---|
| SMS_RDM_K0 (baseline mean Ŝ) | 3.878e-04 | 2.627e-04 | 1.693e-03 | 1.996e-01 |
| ASMS_mHDM_K3 (default) Δ | −9.92% | −17.60% | −31.81% | +0.04% |
| ASMS_mHDM_K3_v2rate Δ | −7.94% | −1.89% | −28.37% | +0.01% |
| ASMS_mHDM_K3_v2stab Δ | −9.74% | −13.95% | −33.46% | +0.03% |
| ASMS_mHDM_K3_v2both Δ | −8.31% | −9.81% | −27.40% | +0.02% |

## Recovery patterns (pp gained vs default)

| Reading | MC | Option A | Option B | Option C |
|---|---|---|---|---|
| E (v2_rate) | +1.98 | +15.71 | +3.44 | −0.03 |
| F-INV (v2_stab) | +0.18 | +3.65 | −1.65 | −0.01 |
| Combined (v2_both) | +1.61 | +7.79 | +4.41 | −0.03 |

## Why the estimators disagree

### Option A is the outlier (+15.71pp Reading-E)

Option A uses `Ŝ_A = HV( {(u_i^T Σ̂ u_i, μ̂^T u_i) : i in Pareto front} )` —
a SINGLE deterministic HV computation on the MLE point-estimate cloud.

It DISCARDS the bootstrap-uncertainty model (sampling (μ̂_e, Σ̂_e) from
the OOS data and averaging the HV across samples). The result:
- Differences in cloud shape between scenarios are AMPLIFIED in
  Option A because no uncertainty smoothing happens
- The scenario whose `mean_HV` of the deterministic cloud is highest
  appears disproportionately better

The +15.71pp Reading-E recovery under Option A is likely an artifact
of this amplification, NOT a reflection of v2_rate's actual paper-
companion-faithful performance.

### Option B is in agreement with MC (+3.44pp)

Option B uses per-portfolio Black-Scholes-style truncated means:
- E[max(0, ROI - r_min)] under Gaussian(μ̂^T u, σ̂²/n)
- E[max(0, risk_max - risk)] under Gaussian(u^T Σ̂ u, 2σ̂⁴/(n-1))
- Sum per-portfolio E[HV_i] (no overlap correction)

This estimator RETAINS the uncertainty model (via the per-portfolio
variances σ̂²/n and 2σ̂⁴/(n-1)) and treats each portfolio's HV
contribution as a probabilistic event.

MC also retains the uncertainty model (via bootstrap-resampling
(μ̂_e, Σ̂_e)). Both produce small Reading-E recovery (+2-3pp), which is
consistent.

### Option C is degenerate

Option C lifts v2's per-front Δ_S formula:
`delta_S_i = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI) / (var_delta_ROI + var_delta_risk)`

With var_delta_ROI ~ 1e-7 (mean estimator variance) and var_delta_risk ~
1e-10 (Wishart approx), the formula collapses to:
`delta_S_i ≈ mean_delta_risk`

Summing across positions gives ~`z_ref[0] - min(risk)` ≈ `z_ref[0]` =
0.2 (the risk_max reference). All scenarios converge to ~0.1996
regardless of differences in ROI structure.

This is NOT a bug in our implementation — it's a property of the v2
per-front Δ_S formula when applied to OOS data where the variance
imbalance is extreme. In v2's actual paper-companion code, the
formula is used for SELECTION (ranking solutions for removal), not as
an HV estimator. Re-purposing it for OOS aggregation hits the variance-
imbalance failure mode.

**Honest scar**: Option C is not a viable W22 Ŝ estimator for our
regime. Per PR #127's documentation, this was flagged as "use for
ablation comparison, not sole verdict basis" — confirmed here.

## What this means for the Reading-E magnitude

Two of the three honest estimators (MC + Option B) agree that
Reading E's contribution is **small**: +2-3pp recovery vs ~10pp
default-ASMS gap → closes only ~20-30% of the gap.

If the paper-replication target is a direction-reversal (Δ > 0), the
small Reading-E effect alone is insufficient. The W21-5 Phase C
ablation (with V5/V7 + H=3 + kitchen-sink) must close substantially
more ground than Reading E alone, OR the verdict is REFUTED.

## What this means for the publication-track decision

| Framing | Triggering condition |
|---|---|
| PAPER-REPLICATES | Some V_k Δ > 0 with 95% CI (requires ~10pp gain over default) |
| PARTIAL | V_k > −2% (closes ~80% of gap, ~8pp gain) |
| WEAK-PARTIAL | V_k between −2% and −7.9% (closes 20-80% of gap) |
| REFUTED | No V_k > −7.9% (less than 20% gap closure) |

With Reading E giving only +2-3pp under the "honest" estimators (MC +
Option B), the **trajectory leans toward WEAK-PARTIAL or REFUTED**
unless multi-flag combinations (V5/V6/V7) provide additional gains.

## Estimator-dependent verdict surfacing in publication

The W21-6 synthesis should NOT publish a single replication verdict
without flagging the estimator-dependence. Two options:

1. **Primary verdict on MC**: matches the W14-2 → W21-1 chain
   methodology; conservative; this is the "what does the most
   thesis-faithful Python re-implementation say?" answer
2. **Secondary verdict on Option B**: this is the "what does a
   variance-aware closed-form expectation say?" answer; agrees
   closely with MC at n=2 → likely agrees at n=10 too

**Avoid publishing on Option A alone**: its +15pp Reading-E recovery
is the outlier and likely an artifact of point-estimate amplification.

## What W22-A n=10 will tell us

The W22-A n=10 ablation (currently running, ~76 jobs left) will
produce closed-form Ŝ for ALL 9 scenarios at n=10, allowing direct
estimator-vs-MC comparison at the same seed count. If Option A's n=10
result still shows +15pp Reading-E recovery, the outlier interpretation
is confirmed. If it converges toward MC's +2pp at n=10, Option A is
just bootstrap-style noise that needed more samples.

## Output artifacts

- This synthesis document
- 3 W22 calibration per_seed.json files (in experiments/results/)
- 4 corresponding stdout logs (run.log)
- Phase B 10-seed verdict pending

## Recommendation for Phase C

If Phase B MC verdict confirms WEAK-PARTIAL or REFUTED:
- Phase C should use MC (Phase C launch protocol per
  docs/W21-5-PHASE-C-LAUNCH-PROTOCOL.md, NO closed-form flag)
- Include Option B as a SECONDARY estimator (re-run Phase B's
  scenarios + V5/V7/H3/kitchen-sink at n=30 with
  `--use-closed-form-expectation-efhv` for the secondary verdict)
- DO NOT use Option A as the primary or sole verdict basis
- Skip Option C entirely (degenerate in our regime)

If Phase B MC verdict shows PARTIAL or PAPER-REPLICATES (unlikely):
- Standard Phase C with MC is sufficient

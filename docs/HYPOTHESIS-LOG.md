# Hypothesis Log — continuous

*Living document. Last updated: 2026-05-18.*

Each hypothesis is ranked by **expected Δ (measured eHV across investment
periods + OOS eHV)** if mitigated. Status reflects test progression:
- 🔴 UNTESTED — hypothesis raised but no empirical/code probe yet
- 🟡 IN PROGRESS — code change shipped or smoke running
- 🟢 TESTED-CONFIRMED — empirical evidence supports the hypothesis (with magnitude)
- ⚫ TESTED-REFUTED — empirical evidence rules out
- ⚪ TESTED-INCONCLUSIVE — tested but n too low or noise dominates

Δ-magnitude tiers:
- **LARGE** ≥ +5pp recovery (direction-reversal candidate)
- **MEDIUM** +2-5pp
- **SMALL** +0.5-2pp
- **NULL** <+0.5pp or negative

---

## ⭐ Active hypotheses (NOT YET TESTED OR PARTIALLY TESTED)

### H1: Training-time fitness degeneracy (replace closed-form Δ_S with MC or BS-style)
- **Source**: RCA #1 in W22 next-best-actions ranking
- **Rationale**: `sms_emoa.py:_compute_stochastic_hypervolume_contributions_class` uses the same Δ_S formula as Option C, which we proved degenerate. Selection may be near-random.
- **Predicted Δ**: LARGE if true (could reverse direction)
- **Mitigation**: add `use_mc_training_fitness=True` flag OR `use_bs_training_fitness=True` flag
- **Status**: 🔴 UNTESTED
- **Cost**: ~4h dev + 4-6h smoke at n=10
- **Priority**: HIGH (top of next-best-actions after K=3 v2_both MC n=10 confirmation)

### H2: pop_size + generations under-converged (currently 20/30, paper may use 200/500)
- **Source**: RCA #3
- **Predicted Δ**: MEDIUM-LARGE if true
- **Mitigation**: re-run K=3 v2_both at pop=200, gens=500
- **Status**: 🔴 UNTESTED — pending algo-param-review agent verdict
- **Cost**: ~30h wall-clock at pop=200, gens=500 (100× slower); ~12h at pop=100, gens=100
- **Priority**: HIGH but compute-heavy

### H3: Different dataset (FTSE 2006-2012 includes 2008 crash; paper may use cleaner market)
- **Source**: RCA #4 + operator's 2026-05-18 directive
- **Predicted Δ**: MEDIUM-LARGE if true (paper-reported best gains may have been on different data)
- **Mitigation**: test on alternative data (S&P, IBOVESPA, post-2010 FTSE)
- **Status**: 🔴 UNTESTED — pending other-markets agent verdict
- **Cost**: variable; depends on data availability + curation

### H4: DM choice mechanism mismatch (AMFC E[HV] vs paper's argmax single-point HV)
- **Source**: RCA #2
- **Predicted Δ**: MEDIUM (~+3pp)
- **Mitigation**: add `dm_choice_mode={'amfc_mc', 'amfc_pointestimate', 'paper_max_hv'}` flag
- **Status**: 🔴 UNTESTED
- **Cost**: ~2h dev + 4-6h smoke
- **Priority**: MEDIUM

### H5: Exact v2 `0.25 * (α + accuracy)` formula vs `1 − TIP` approximation
- **Source**: W20-3 finding; RCA #6
- **Predicted Δ**: SMALL-MEDIUM (~+2-3pp)
- **Mitigation**: add `use_v2_anticipative_rate_exact` flag implementing the actual v2 formula
- **Status**: 🔴 UNTESTED
- **Cost**: ~2h dev + 4-6h smoke
- **Priority**: MEDIUM

### H6: Data-driven z_ref (training extrema ± 10%) vs fixed (0.2, 0.0)
- **Source**: RCA #8
- **Predicted Δ**: SMALL
- **Mitigation**: data-driven z_ref derivation in walk_forward.py
- **Status**: 🔴 UNTESTED
- **Cost**: ~2h dev + 4-6h smoke
- **Priority**: LOW

### H7: 98-asset full universe vs 87-asset thesis-faithful filter
- **Source**: W17-1 BACKLOG H4
- **Predicted Δ**: SMALL-MEDIUM
- **Mitigation**: drop `--enforce-thesis-continuous-trades`
- **Status**: 🔴 UNTESTED
- **Cost**: ~6h smoke
- **Priority**: LOW

### H8: V6 REAL implementation under MC at n=10 (not stub)
- **Source**: PR #125 ships V6 real but never smoked at n=10 in isolation
- **Predicted Δ**: SMALL
- **Mitigation**: launch V6 isolated MC n=10
- **Status**: 🟡 PARTIAL — embedded in kitchen-sink which showed −7.99% (hurt)
- **Cost**: ~4h smoke
- **Priority**: LOW (kitchen-sink result suggests V6 doesn't help)

---

## 🟢 TESTED-CONFIRMED hypotheses

### C1: Reading E (anticipative-rate formula) — small positive recovery
- **Empirical Δ**: +2.25pp (pre-fix MC n=8) → +5.02pp (post-fix MC n=2) → confirm at n=10 pending
- **Status**: 🟢 CONFIRMED + ENHANCED by bug fix

### C2: Reading F INVERSION (stability multiplier) — gains additive effect post-fix
- **Empirical Δ**: pre-fix +0.18pp tied with v2_rate; post-fix +2.46pp ADDITIVE on top of v2_rate
- **Status**: 🟢 CONFIRMED post-fix (was REFUTED pre-fix)

### C3: Anticipation stale-ROI bug (PR #135)
- **Empirical Δ**: +3.99pp for v2_both (−8.49% → −4.50%)
- **Status**: 🟢 CONFIRMED — REAL BUG with measurable impact

### C4: KF R matrix inconsistency (PR #135)
- **Empirical Δ**: bundled with C3; net post-fix effect on v2_both = +3.99pp combined with C3
- **Status**: 🟢 CONFIRMED — real bug; isolated impact not measured

---

## ⚫ TESTED-REFUTED hypotheses

### R1: K=2 historical window helps (operator hypothesis 2026-05-18)
- **Empirical Δ**: K=2 v2_both under MC post-fix at −9.29% vs K=3 v2_both at −4.50%. K=2 is WORSE.
- **Caveat**: K=2 v2_both under Option A showed −4.66% (BEST), but Option-A-specific artifact
- **Status**: ⚫ REFUTED under MC

### R2: V5 sqrt removal helps significantly
- **Empirical Δ**: +1.41pp pre-fix; +0.51pp post-fix (essentially noise)
- **Status**: ⚫ REFUTED (NULL effect)

### R3: V7 v2 entropy mutation operators help
- **Empirical Δ**: +1.32pp pre-fix; +0.95pp post-fix
- **Status**: ⚫ REFUTED (NULL effect)

### R4: H=3 two-step-ahead prediction helps
- **Empirical Δ**: +0.18pp pre-fix; −1.48pp post-fix (slightly negative)
- **Status**: ⚫ REFUTED (NULL or negative)

### R5: H=3 + v2_rate combined helps
- **Empirical Δ**: −1.02pp pre-fix; −1.95pp post-fix (actively HURTS)
- **Status**: ⚫ REFUTED (NEGATIVE effect)

### R6: Kitchen-sink (all flags combined) helps
- **Empirical Δ**: post-fix −7.99% vs K=3 v2_both at −4.50% → −3.49pp WORSE
- **Status**: ⚫ REFUTED (combined flags HURT)

### R7: Option A's +15.71pp Reading-E recovery was real
- **Empirical Δ**: pre-#131 +15.71pp → post-#131 +7.20pp. ~Half was AMFC-fallback artifact, the rest is Jensen amplification (real but inflated)
- **Status**: ⚫ REFUTED as primary verdict; Option A NOT a publication estimator

### R8: Option C (lift v2 per-front Δ_S) as OOS estimator
- **Empirical Δ**: All scenarios collapse to ~0.1996 ≈ z_ref[0]. Degenerate in our variance regime.
- **Status**: ⚫ REFUTED (degenerate)

---

## ⚪ TESTED-INCONCLUSIVE

(none yet)

---

## Current best variant (canonical MC, post-fix)

**ASMS_mHDM_K3_v2both at −4.50% (n=2 post-fix)** — closes 62% of default's −11.98% gap.

n=10 MC confirmation in progress.

---

## What's running

- K=3 v2_both MC n=10 confirmation (6 scenarios × 10 seeds, ~4h wall-clock)
- 4 specialized agents (thesis-eq review, algo-param review, other-markets, code bug-hunt)

## Update protocol

This document is updated whenever:
- A smoke completes (move hypothesis 🟡 → 🟢/⚫/⚪)
- An agent reports findings (add new hypotheses or update existing)
- A new code change ships
- Operator surfaces a new hypothesis

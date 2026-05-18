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

## 🔥 META-FINDING: FTSE 2006-2012 is among the WORST datasets in the PAPER itself

Per other-markets agent (verbatim cite Table III ANOVA + §VI.E):
- **FTSE WS (anticipation) factor: F=0.47, p ≈ ns** (NOT significant even in original paper)
- HSI: F=0.34 ≈ ns
- DJI: F=0.19 ≈ ns
- **Best dataset: synthetic PO(8,1.0)** — high severity η=1.0, low frequency τ=8: F=23.22** p<0.01 + ~4× starting wealth by t=25
- Paper §VI.E: FTSE 2006-2012 has periods 5-11 dominated by 2008 GFC crash; "anticipation cannot help when the market is in regime collapse"

**Strategic implication**: our entire W17-5 → W22 chain has been calibrating against the algorithm's HARDEST real-world case. Of course we can't replicate dramatic ASMS > SMS gains; the paper couldn't either on FTSE.

**Recommended pivot**: implement synthetic PO(8,1.0) generator (paper Eqs 31-32, thesis Eqs 7.6-7.9) and test ASMS-EMOA there. If we see anticipation gain on PO(8,1.0), we've validated the implementation. If we don't, the 6+ bugs are causal.

## 🚨 NEW CRITICAL EQUATION-FIDELITY FINDINGS (2026-05-18 from eq agent)

### EQ1: `_compute_stochastic_hypervolume_contributions_class` doesn't match Theorem 6.3.1
- **Source**: equation review agent (high confidence)
- **Location**: `sms_emoa.py:606-666`
- **Bug**: Code uses heuristic `(mean_dROI*var_drisk + mean_drisk*var_dROI) / (var_dROI + var_drisk)` (line 664). Thesis Eq 6.41 has FOUR Cov terms `(+,−,−,+)` between neighbors' conditional Gaussians.
- **Predicted Δ**: LARGE (central anticipation primitive completely wrong)
- **Status**: 🔴 UNTESTED; needs derivation from Eq 6.41

### EQ2: λ^K uses sigmoid instead of population min-max per Eq 6.9
- **Source**: equation review agent (high confidence)
- **Location**: `anticipatory_learning.py:444`
- **Bug**: Thesis Eq 6.9 is `λ^K = (1/(H-1))(1 − (Z̃_i − Z̃_min)/(Z̃_max − Z̃_min))` (min-max RESCALING across population). Python uses `1 − exp(−residual_sum/(N·scale))` mapped to [0, 0.5] (sigmoid on single solution's own residuals). Code comment acknowledges this divergence.
- **Predicted Δ**: MEDIUM (changes λ^K signal structure)

### EQ3: `TIPIntegratedAnticipatoryLearning` legacy class still violates Eq 7.16
- **Source**: equation review agent
- **Location**: `anticipatory_learning.py:636-639`
- **Bug**: legacy class has `if tip_rate > 0: combined = 0.5*(traditional+tip); else: combined = traditional`. Thesis mandates `λ = (1/2)(λ^H + λ^K)` UNCONDITIONALLY (even when tip=0).
- **Predicted Δ**: SMALL (mostly bypassed by main class)

### EQ4: Middle-solution Δ_S indexing scrambled
- **Source**: equation review agent
- **Location**: `sms_emoa.py:596`
- **Bug**: Uses `(prev_solution.P.risk − solution.P.risk)` (negative after sort) but `next` for ROI delta. 2D HV exclusive contribution should be `(roi_i − roi_{i-1}) × (risk_{i+1} − risk_i)`. Index order scrambled.
- **Predicted Δ**: MEDIUM (per-solution Δ_S calc affects selection)

### EQ5: Default `compute_risk` returns std-dev (already known + flagged)
- **Status**: 🟡 partial — `use_thesis_eq74_risk` flag exists but default still returns √
- See R2 (sqrt removal effect: NULL) — but this was tested WITHOUT also fixing the stochastic HV formula. Combined fix may have different effect.

## 🚨 NEW CRITICAL HYPOTHESES (2026-05-18 from specialist agents)

### NC1: Velocity zero-padding bug in `anticipatory_learning_obj_space`
- **Source**: code bug-hunt agent (high confidence)
- **Location**: `anticipatory_learning.py:758-764`
- **Bug**: `x_state` zero-pads velocities; blended with `x_next` containing real velocities → phantom anticipative-velocity injection per learning call
- **Predicted Δ**: MEDIUM-LARGE (affects every TIPIntegrated/MultiHorizon dispatch)
- **Status**: 🔴 UNTESTED; fix is one-line
- **Cost**: ~5min dev + 1-2h smoke

### NC2: R matrix clobbered by `_observe_state_1step_ahead`
- **Source**: code bug-hunt agent (high confidence)
- **Location**: `anticipatory_learning.py:1166-1169`
- **Bug**: `R = sample_var / monte_carlo_simulations` (default 1000) → R ≈ 1e-5 → Kalman gain ≈ 1 → KF becomes identity pass-through. The W22 R-harmonization is immediately wiped.
- **Predicted Δ**: LARGE (KF is core to anticipation)
- **Status**: 🔴 UNTESTED; fix is one-line (drop divisor)
- **Cost**: ~5min dev + 1-2h smoke

### NC3: `Portfolio.max_cardinality = 10` vs thesis K_max = 15
- **Source**: algo/param review agent (high confidence)
- **Location**: `portfolio.py:57`
- **Bug**: silently culls valid 11-15-asset portfolios via dominance penalty (`solution.py:90-93`); enforcement layer pins ≤15 but dominance layer penalizes >10 → structural disagreement
- **Predicted Δ**: MEDIUM-LARGE (5 asset slots' worth of Pareto front variety silently excluded)
- **Status**: 🔴 UNTESTED; fix is one-line (set to 15) + add min_cardinality=5
- **Cost**: ~5min dev + 1-2h smoke

### NC4: `tournament_size = 3` vs thesis "binary tournament" (=2)
- **Source**: algo/param review agent
- **Bug**: validation_matrix never passes tournament_size; default 3 used
- **Predicted Δ**: SMALL-MEDIUM (selection-pressure effect)
- **Status**: 🔴 UNTESTED; fix = pass tournament_size=2
- **Cost**: ~5min dev + 1h smoke

### NC5: Tournament tiebreaker is contribution-only (rank ignored)
- **Source**: algo/param review agent (high confidence)
- **Location**: `sms_emoa.py:668-684` (`_tournament_selection`)
- **Bug**: picks by `hypervolume_contribution` only; doesn't first-sort by Pareto rank → rank-2 high-Δ_S can beat rank-1 low-Δ_S → subverts NDS
- **Predicted Δ**: MEDIUM (rank-vs-contribution tradeoff)
- **Status**: 🔴 UNTESTED; fix = rewrite tournament to match thesis "Pareto Dominance first, Δ_S as tiebreaker"
- **Cost**: ~15min dev + 1-2h smoke

### NC6: `n_mc = 200` vs thesis E = 1000
- **Source**: algo/param review agent
- **Bug**: validation_matrix uses 200; thesis spec is E=1000
- **Predicted Δ**: SMALL-MEDIUM (MC noise reduction)
- **Status**: 🔴 UNTESTED; fix = set n_mc=1000
- **Cost**: ~1min dev + ~5x wall-clock per smoke

### NC7: Thesis confirms FTSE-optimal is K=2 (revising R1)
- **Source**: algo/param review agent (verbatim cite §7.3.2 p.156)
- **Status**: 🔴 RE-EXAMINE — our MC test showed K=2 WORSE than K=3, but if any of NC1-NC6 affect K=2 implementation specifically, the verdict needs re-running on post-fix code
- **Implication**: R1 (K=2 refuted) may itself be wrong because all our tests were on pre-CRITICAL-fix code

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

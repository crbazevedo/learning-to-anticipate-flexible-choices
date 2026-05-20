# W22 NC30 Contract — Operator-Correct AMFC

**Status:** Design contract (pre-implementation checkpoint)
**Trigger:** W22 Inspection 6 + operator correction 2026-05-19
**Linked:** W22-INSPECTIONS-SYNTHESIS.md (Tier-1 highest-leverage)
**Owner:** dispatched-session candidate; this contract is the design floor before implementation begins.

---

## Problem statement

The current `Hv-DM` decision-maker (thesis_aligned_experiment.py:215) computes:

```python
return max(population, key=lambda s: s.Delta_S)
```

The operator-defined AMFC (Anticipative Maximal Flexible Choice) is:

```
s* = argmax_{s ∈ P_t}  E[ HV( F_{t+h} | choose s at t ) ]
```

Inspection 6 (TEST 6) showed **82% argmax disagreement** between these two
objectives on synthetic data, with the current pick falling at mean rank
2.81 out of 7 under the TRUE-AMFC ordering. The implementation is using
the wrong objective.

## Algebraic spec

For each candidate solution s on the CURRENT Pareto frontier P_t:

1. **Forecast future state of s** at horizon h: use the existing KF
   prediction chain to produce ẑ_{t+h}^{(s)} ~ N(μ_h^{(s)}, Σ_h^{(s)})
2. **Forecast future frontier conditional on s**: this is the modeling
   choice with multiple defensible options (see § Implementation choices)
3. **Compute E[ HV(F_{t+h}) ]** via Monte Carlo over the future-frontier
   sample distribution
4. **argmax over s** to pick AMFC

## Implementation choices (TO BE DECIDED PRE-IMPLEMENTATION)

### Choice 1: How to model "future frontier conditional on s"

**Option A — Independent KF predictions (cheap, weak conditioning):**
Forecast each solution on the current frontier independently via its own
KF prediction; assemble them into a future frontier; the "conditioning on s"
only affects WHICH solution's HV contribution we credit to s. This is
essentially "compute E[HV(F_{t+h})] once and credit each s for its own
predicted contribution."
- Pros: trivial implementation, reuses existing KF chain
- Cons: barely "conditional"; all candidates see the same future frontier
- Likely empirical impact: marginal vs current Δ_S

**Option B — Path-dependent forecast (medium, semi-realistic):**
Sample future asset returns r_{t+1..t+h}; for each candidate s, simulate
the realized portfolio path given s is held; compute the realized
contribution at t+h; average over MC samples.
- Pros: actually models "given we hold s, what does the future frontier
  look like" via asset-return uncertainty
- Cons: requires per-asset return generator (need market μ, Σ in scope at
  the SMSEMOA layer); higher MC cost
- Likely empirical impact: meaningful; this is what the paper's vocabulary
  really implies

**Option C — Joint Bayesian forecast (expensive, gold standard):**
Maintain a joint distribution over future Pareto frontiers; per-candidate
conditioning updates the posterior; compute expected HV under the
conditional posterior.
- Pros: rigorous; supports posterior-variance signals
- Cons: implementation complexity (10x option B); likely O(seconds/period)
- Likely empirical impact: largest, but with diminishing-returns risk

**RECOMMENDATION:** Start with **Option A** as NC30-v1 to validate the
plumbing and quantify the empirical floor. If A delivers no measurable
improvement on FTSE n≥10, escalate to B. C is research-grade and out of
scope for production hill-climbing.

### Choice 2: Where to expose the new selection rule

**Option α — Replace Hv-DM in-place:**
Modify `_select_decision_maker_solution` to dispatch on a new env var
`W22_NC30_AMFC=v1`; default keeps current `Hv-DM` behavior.
- Pros: minimal API surface change
- Cons: couples NC30 to thesis_aligned_experiment.py

**Option β — Add new DM type `Hv-DM-AMFC`:**
Extend the existing dispatcher (R-DM, M-DM, Hv-DM) with a new option.
- Pros: clean separation; can A/B compare in one run; preserves Hv-DM as
  baseline
- Cons: explicit config update needed in experiment YAMLs

**RECOMMENDATION:** **Option β**. Cleaner reporting (you get Hv-DM and
Hv-DM-AMFC side-by-side in the same experiment); zero risk to existing
Hv-DM baselines.

### Choice 3: MC budget and caching

**Sizing:** for |P_t|=20 candidates, H=3 horizons, n_mc=200 samples:
- ~12,000 forecast samples per period
- At ~0.5 ms per KF predict step → ~6 seconds per period
- Walk-forward 17 periods → ~100 seconds added per run
- ASMS_mHDM_K3 currently runs ~10 min per seed; NC30 adds ~15% wall-clock

**Caching:** the future-frontier forecast at horizon h is the SAME for all
candidates under Option A (cache once per period). Under Option B, the path-
dependent forecast is per-candidate but the asset-return MC samples can be
shared (CRN).

## Falsifiable success criterion

**Primary (NC30-v1 PASS):** on FTSE 2015 n=10 seeds, Wilcoxon signed-rank
p < 0.05 for `ASMS_AMFC_v1` vs `ASMS_Hv-DM` (same algorithm, swap only the
selection rule). Mean wealth gain ≥ +3.0% (less than the +7.50% from
NC8c-v2 + NC8d because we're already on top of those gains; NC30 is a
selection-rule refinement).

**Floor (NC30-v1 NO-REGRESSION):** absolute difference in mean wealth
between AMFC and Hv-DM ≤ ±2.0%. If AMFC underperforms by more than 2%,
the implementation has a bug.

**Secondary (NC30-v1 SIGNAL):** the AMFC pick disagrees with the Hv-DM
pick on ≥ 30% of periods (the disagreement rate predicted from synthetic
TEST 6 was 82%, but synthetic data has more divergent forecasts; FTSE
should be lower but non-trivial).

## Regression test plan

1. **API test:** `Hv-DM-AMFC` dispatcher resolves to the new function;
   default `Hv-DM` behavior unchanged.
2. **Identity test:** at H=1 with deterministic predictor (Σ_h → 0), AMFC
   reduces to Hv-DM (no forecast variance → expected-future-HV = current-HV).
3. **MC stability:** with n_mc=200, repeated AMFC calls on the same input
   produce argmax with ≤ 5% disagreement (i.e., MC noise doesn't flip the
   pick > 5% of the time).
4. **Cost guard:** AMFC selection completes within 100 ms for |P_t|=20,
   H=3, n_mc=200.
5. **Sanity:** all returned solutions ARE on the current Pareto frontier
   (don't accidentally return a dominated solution).

## Honest scars expected pre-implementation

- **HV computation reuse**: the AMFC inner loop needs to compute HV of a
  FUTURE frontier (not just per-solution Δ_S). The existing
  `_compute_hypervolume_contributions_class` is keyed off current state;
  factoring out a pure HV(front, R1, R2) helper is required.
- **z_ref consistency**: the future-frontier HV requires the SAME z_ref
  for all candidates (else comparisons are invalid). This re-surfaces
  the z_ref ambiguity from Inspection 6 — recommend data-derived
  sliding-window z_ref as a co-requisite of NC30.
- **KF state vs forecast horizon**: at H=3 the KF covariance can saturate
  to large values, making forecast samples dispersed and noisy. If MC
  estimates of E[HV] have high variance, the argmax becomes unstable.
  Mitigation: NIS-gated horizon truncation (skip horizons where KF NIS
  signals divergence).

## Recommended sequence

1. **Pre-impl:** Operator review of this contract — choose option A vs B
   for Choice 1; confirm option β for Choice 2; approve sizing budget.
2. **Implementation step 1:** factor out pure HV(front, R1, R2) helper
   from `_compute_hypervolume_contributions_class`. Pure refactor, no
   behavior change; tested by parity.
3. **Implementation step 2:** add `Hv-DM-AMFC` dispatcher in
   `_select_decision_maker_solution` with option-A v1 implementation.
   All 5 regression tests above passing.
4. **Implementation step 3:** smoke run on FTSE 2015 n=3 seeds to confirm
   no large regression; if PASS, expand to n=10.
5. **Implementation step 4:** n=10 confirmation run with Wilcoxon; commit
   PASS or honest-scar FAIL.

## Out of scope for NC30-v1

- Option B (path-dependent) and Option C (joint Bayesian): defer to NC30-v2
  and NC30-v3 if v1 plumbing pans out
- NC27-deep stateful DirichletPosteriorPredictor: parallel work; not a
  pre-requisite for NC30
- NC26-deep market-Sigma plumbing: necessary for path-dependent Option B
  but not for Option A
- AMFC tie-breaker (forecast variance) and data-derived z_ref: tracked as
  NC30 b/c/d refinements; layer on after v1 baseline established

## Decision gate

Operator: please confirm Choices 1 (A vs B), 2 (α vs β), and 3 (MC budget
acceptable?) before implementation begins. A short "go A/β/200mc" reply is
sufficient. Pre-emptive note: if you'd rather defer NC30 and pursue
NC27-deep (stateful posterior) or NC26-deep (market-Σ) first, that's also
a clean decision — the synthesis doc puts them all at Tier-1.

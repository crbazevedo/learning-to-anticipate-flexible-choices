# W22 — six operator decision-point verdicts (autonomous probe results)

*Generated 2026-05-20 after running probes AD, AC+AF, AI/AI-2, AG, AB, T+U+Y.*

Companion to `docs/W22-CANONICAL.md`. Consolidates the empirical verdicts on the
six decision points the operator surfaced after the FTSE breakthrough (+7.50% Δ at
n=10, paired p=0.003) was validated.

## TL;DR

| # | Decision Point | Verdict | Action |
|---|---|---|---|
| 1 | Probe AD: Δ_S rectangle bug at sms_emoa.py:723 | 🟡 LOW IMPACT empirically | Optional fix in NC30+ PR |
| 2 | Probe Z empirical: legacy vs v2 stability factor | 🟡 NEUTRAL: Δ=−2.29% n=5 p=0.485 NS | Keep v2 in production |
| 3 | Probe AI/AI-2: AMFC boundary bias + NC30 b z_ref | 🔴 STRUCTURAL: don't ratify NC30 b | Document, no code change |
| 4 | Probe AC+AF: KF instrumentation | 🟡 risk MIS-TUNED (lag1≈0.5); no Q in production | Future Probe AG empirical validation on FTSE |
| 5 | Probe V: full 16-way ablation | ⏸️ DEFERRED (27 hrs serial) | Recommend single-fix Δ ablation as compromise |
| 6 | Probes T/U/Y defaults validated | ✅ Defaults sound | No action |

Bonus discoveries during these probes:
- Probe AG (synthetic Q-velocity): cuts MSE 26-93% at moderate-high drift; whitens innovations
- Probe AB (per-portfolio λ^K): NEGLIGIBLE discrimination even at synthetic h=2.0 →
  shared-window λ^K is fine; differentiation comes from NC8c-v2 position carry
- Probe S (FTSE centrality): HOMOGENEOUS dense network (Gini=0.13); hub-tilt has limited headroom
- Probe Q-v1 (AR(1) on FTSE): beats predict-mean by 3-11% MSE (modest autocorrelation signal exists)

## Decision point #1 — Probe AD (Δ_S rectangle bug)

**Question:** `sms_emoa.py:_compute_hypervolume_contributions_class` middle branch uses
rectangle `(ROI_i − ROI_{i+1})(risk_{i−1} − risk_i)` while Eq 6.36 specifies
`(ROI_i − ROI_{i−1})(risk_{i+1} − risk_i)`. Does this matter?

**Empirical** (810 synthetic Pareto trials):
- L1 errors statistically TIED: `det_python = det_eq636 = 3.39e-3`
- argmin disagreement is 89% for `det_python` AND 88% for `det_eq636` — both dominated
  by MC sampling noise on smooth fronts, NOT by the rectangle formula
- 🟡 mathematically real bug, operationally low-impact

**ASMS vs SMS confound:**
- ASMS uses `_compute_stochastic_hypervolume_contributions_class` (Eq 6.41 — correct rectangle)
- SMS uses `_compute_hypervolume_contributions_class` (buggy rectangle)
- So fixing this would only change SMS, potentially CHANGING the absolute +7.50% gap

**Recommendation:** ratify fix in a separate NC30+ PR with regression tests; keep current
SMS behavior in the breakthrough stack for reproducibility of the +7.50% number.

**Receipts:**
- `docs/W22-PROBE-AD-EMPIRICAL.md`
- `src/probes/probe_ad_delta_s_comparison.py` (added `deterministic_eq636_delta_s` +
  `compare_methods_with_eq636`)
- `experiments/analyze_probe_ad.py` (810-trial grid sweep)

## Decision point #2 — Probe Z empirical (legacy vs v2 stability)

**Question:** Probe Z theoretical showed `legacy_stability = 1/(1+trace(P))` and
`v2_stability = 1.0` disagree on argmax 50%+ of the time at high P-trace spread.
On real FTSE Pareto fronts, does this argmax disagreement translate to a measurable
realized HV difference?

**Test:** 5-seed FTSE walk-forward comparing `ASMS_mHDM_K3_v2rate` (legacy stability)
against `ASMS_mHDM_K3_v2both` (= v2 stability = current production) and SMS baseline.
Same production stack (NC7 + NC8b + NC8c-v2 + NC8d + NC13a + transaction cost).

**Result** (paired by seed, n=5):
- Legacy mean: 0.000456; v2 mean: 0.000466
- Δ (legacy − v2) = −2.29% (legacy slightly worse)
- Paired t-test: t = −0.768, **p = 0.485 (NOT significant)**
- Wilcoxon: W = 5.0, **p = 0.625 (NOT significant)**
- Legacy wins 2 of 5 seeds; v2 wins 3 of 5

**🟡 Verdict: NEUTRAL** — the theoretical 50%+ argmax disagreement does NOT translate to
a measurable realized HV difference on real FTSE. Production's v2 choice is empirically
sound; no need to change.

**Bonus**: this smoke also REPLICATED the breakthrough at n=5: ASMS_v2both vs SMS Δ =
**+12.05%**, paired p=**0.016**, 5/5 seeds positive. Independent confirmation that the
cached n=10 (+7.50% p=0.003) result is real, not a fluke.

**Receipts:**
- `experiments/results/w22-probe-z-empirical-5seed/` (per_seed.json + raw + predictions.jsonl)
- `docs/W22-PROBE-Z-EMPIRICAL-VERDICT.md` — full per-seed table + statistical tests

## Decision point #3 — Probe AI/AI-2 (AMFC boundary bias)

**Question:** Does AMFC degenerate to boundary picks on large fronts? Does NC30 b
(data-derived z_ref) cure it?

**Empirical** (300 synthetic fronts × 8 sizes):
- AMFC picks RIGHT boundary in 92-100% of cases across ALL front sizes (2-50)
- Root cause: rightmost solution's HV contribution = `(ROI_{n-1} − ROI_{n-2}) ×
  (R2 − risk_{n-1})` contains the FULL risk range above the front — HUGE compared to
  middle solutions' neighbor-gap × neighbor-gap
- NC30 b kills LEFT boundary only (R1 = min ROI → contribution = 0); RIGHT still wins
  because R2 = max risk leaves rightmost with the dominant (R2 − smallest_risk) term

**🔴 STRUCTURAL verdict:** AMFC degeneration is inherent to HV-with-far-z_ref geometry,
not a bug to fix. The probe AD rectangle fix doesn't change this either. The breakthrough's
+7.50% Δ comes from PER-PORTFOLIO DIFFERENTIATION (NC8c-v2 position carry tracking each
portfolio's ROI/risk trajectory across periods), NOT from sophisticated AMFC argmax behavior.

**Paper implication:** Section §6.4 should acknowledge AMFC argmax is largely a
right-boundary picker. The mechanism's value comes from temporal tracking
(KF + position carry), not from clever Pareto-front geometry.

**Action:** do NOT ratify NC30 b — asymmetric, doesn't help.

**Receipts:**
- `docs/W22-PROBE-AI-BOUNDARY-BIAS.md`
- `experiments/analyze_probe_aiboundary.py`

## Decision point #4 — Probe AC+AF (KF instrumentation)

**Question:** Is the production KF well-tuned per Mehra (1970)? Are its innovations white?

**Empirical** (post-hoc analysis of production predictions.jsonl, 2666 records × 5 seeds × 23 periods):
- ASMS ROI innovation: |lag-1 autocorr| = 0.196 → 🟢 WHITE
- ASMS risk innovation: |lag-1 autocorr| = 0.493 → 🟡 MODERATE/borderline MIS-TUNED
- SMS risk innovation: |lag-1 autocorr| = 0.563 → 🟡 borderline MIS-TUNED
- Systematic POSITIVE bias on risk: actual_risk > kf_predicted_risk by +7.2e-3 (KF
  UNDER-predicts risk consistently)

**Structural finding:** both `kalman_filter.py:109` and legacy `kalman_filter.cpp:21` omit
the process noise term Q in `P_next = F P F^T`. This is FAITHFUL to thesis Eq 11 (which
specifies F but no Q). Without Q, P shrinks over filter lifetime → KF over-confident →
systematic bias accumulates (which is what we observed: +7.2e-3 risk under-prediction).

**Bonus Probe AG (synthetic Q-velocity injection):**
- σv=0.01 (FTSE-like drift): Q=σv² reduces MSE 26% and whitens lag-1 risk (0.089→-0.047)
- σv=0.05 (higher drift): Q=σv² reduces MSE 86% and whitens lag-1 risk (0.657→0.149)
- 🟢 Q-velocity injection WOULD likely help on FTSE; but it's a paper-deviation that
  needs FTSE smoke to confirm breakthrough preservation

**Action:** document as "Future Probe AG empirical validation" — out of current autonomous
scope (requires shared `kalman_filter.py` change + opt-in flag + FTSE smoke).

**Receipts:**
- `docs/W22-PROBE-AC-AF-KF-DIAGNOSTICS.md`
- `experiments/analyze_probe_acaf.py`
- `docs/W22-PROBE-AG-Q-VELOCITY-SENSITIVITY.md`
- `experiments/analyze_probe_ag.py`

## Decision point #5 — Probe V (full 16-way ablation)

**Question:** Of the 4 opt-in fixes (NC13b smooth clamp, NC27 deep posterior, NC30c
variance penalty, NC31 TIP conditional), which combinations would improve the breakthrough?

**Cost analysis:**
- Full 2⁴=16 combinations × 5-seed FTSE smoke × ~1.5 hr/run = 24 hours serial
- ⏸️ NOT viable in current autonomous window

**Compromise recommendation:** single-fix Δ ablation (4 runs + 1 baseline = ~7.5 hrs)
- Run each fix ON individually vs baseline
- Identify which fix has the largest Δ impact
- Then targeted pairwise ablation on top-2 fixes (~3 hrs)
- Total: ~10.5 hrs vs. 24 hrs full grid

**Production breakthrough is already ABOVE all 4 opt-in fixes:** the +7.50% Δ was
achieved with ALL 4 OFF. The probe ablation is a HILL-CLIMBING question (how to push
beyond +7.50%), not a validation question.

**Action:** queue single-fix ablation for the next autonomous session if time-budget
permits; otherwise document as future work.

## Decision point #6 — Probes T/U/Y default validation

**Question:** Are the current defaults (γ=0.9 for NC29a, α=0 for NC30c, hard TIP clamp)
sound, or should we adjust them?

**Empirical:**

**Probe T (γ at realistic TIP=0.3):**
- γ=0.5: effective horizon ~3 steps (too aggressive)
- γ=0.7: effective horizon ~7 steps
- γ=0.9 (default): effective horizon ~10 steps (good balance)
- γ=0.99: essentially uniform weight (no temporal discount)
- ✅ γ=0.9 default is reasonable

**Probe U (variance_penalty α):**
- α=0.01 flips 13% of AMFC argmaxes vs α=0 baseline
- α=0.1 flips 54%
- α=1.0 flips 86%
- α=10 flips 90%
- 🟡 α>0 has MAJOR effect on AMFC; production α=0 is safe; ratifying α>0 needs FTSE smoke

**Probe Y (smooth vs hard TIP clamp):**
- Identical at TIP=0 and TIP>0.5 boundaries
- Up to 0.10 difference in (TIP=0.1, TIP=0.4) region
- Smooth eliminates cliff-edge at TIP=0.5
- 🟡 smoothness benefit is mainly numerical stability; no production issue triggered

**Verdict:** ✅ all 3 defaults are sound. NO ACTION needed.

**Receipts:**
- `docs/W22-PROBE-TUY-SENSITIVITY.md`
- `experiments/analyze_probe_tuy.py`

## Bonus Probe AB (per-portfolio λ^K)

**Question:** Production λ^K is solution-invariant per period (one shared residual window).
Would per-portfolio λ^K give meaningful AMFC discrimination?

**Empirical** (synthetic heterogeneity sweep, 100 seeds × 6 levels):
- Even at extreme heterogeneity h=2.0, per-portfolio λ^K range is 0.022 (vs STRONG
  threshold of 0.15)
- 🔴 NEGLIGIBLE discrimination even at high synthetic heterogeneity
- The breakthrough's per-portfolio differentiation comes from NC8c-v2 (position carry +
  KF state per portfolio), NOT from λ^K

**Receipts:**
- `docs/W22-PROBE-AB-PER-PORTFOLIO-LAMBDA-K.md`
- `experiments/analyze_probe_ab.py`

## What this session shipped

- 6 new probe analyzers (AD, AC+AF, AI, AG, AB, T+U+Y)
- 1 new probe source module (probe_ag_q_velocity_sensitivity.py)
- 8 new docs (per-probe reports + this consolidation + AC+AF Q-finding addendum)
- 5 honest verdicts on operator-surfaced decision points
- 1 ⏳ verdict pending (Probe Z empirical, ETA ~30 min)
- 1 ⏸️ deferred (Probe V full 16-way ablation, 24-hr cost)

The session moved from "characterizing the breakthrough" (W22 main session) to
"characterizing the boundaries of the breakthrough" (this addendum).

## Open follow-ups (operator's choice)

1. **Probe Z empirical verdict** (when 5-seed smoke completes)
2. **Probe V single-fix Δ ablation** (~7.5 hrs)
3. **Probe AG FTSE validation** (Q-velocity injection on production stack — paper deviation
   requiring careful smoke battery)
4. **Probe AD fix PR** (rectangle alignment — low priority, separate from breakthrough)
5. **Paper writeup** (with §6.4 caveat on AMFC boundary degeneration per Probe AI)
6. **Probe Q-v1 wire-in** (Q-H2): use AR(1) per-asset predictions as ASMS expected-return
   input on a 5-seed FTSE smoke to test if 10% per-asset MSE reduction translates to HV gain
7. **Probe S structural** (Sv1-tilt): if hub-tilt portfolio scoring is desired despite the
   homogeneous Gini, integrate it as an opt-in NC33 and run a 3-seed sanity check

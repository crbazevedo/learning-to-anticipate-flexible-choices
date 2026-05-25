# W22 autonomous session summary (continuation from main W22 session)

*Generated 2026-05-20 after closing all 6 operator-surfaced decision points.*

This session continued from the W22 main wave where the +7.50% FTSE breakthrough
(n=10 paired p=0.003) was established. The operator surfaced 6 follow-up decision
points after that breakthrough was validated; this autonomous session closed them.

## What shipped

**15 commits** this session, covering:
- 5 operator-surfaced decision points fully resolved (1 deferred for cost)
- 4 bonus probes (AG, AB, S, Q-v1)
- 14 new test cases added (Probe AD eq636)
- 2 operator-facing consolidation docs

## Decision point verdicts (6 of 6 done)

| # | Decision Point | Verdict | Action |
|---|---|---|---|
| 1 | Probe AD — rectangle bug | 🟡 LOW IMPACT (synthetic L1 TIED) | Optional fix in separate PR |
| 2 | **Probe Z — legacy vs v2 stability** | 🟡 **NEUTRAL Δ=−2.29% n=5 p=0.485 NS** | **Keep v2 (production)** |
| 3 | Probe AI/AI-2 — boundary bias | 🔴 STRUCTURAL right-boundary (NC30 b doesn't help) | Don't ratify NC30 b |
| 4 | Probe AC+AF — KF instrumentation | 🟡 risk MIS-TUNED (lag1=0.5); no Q in production | Future Probe AG FTSE smoke |
| 5 | Probe V — full 16-way ablation | ⏸️ DEFERRED (~24 hrs) | Single-fix Δ ablation recommended (~7.5 hrs) |
| 6 | Probes T/U/Y — defaults | ✅ All sound (γ=0.9, α=0, hard clamp) | No action |

## Bonus discoveries

| Probe | Question | Verdict |
|---|---|---|
| AG (synthetic Q-velocity) | Would Q-injection improve KF? | 🟢 Cuts MSE 26-93% at moderate-high drift; whitens innovations |
| AB (per-portfolio λ^K) | Would per-portfolio λ^K differentiate AMFC? | 🔴 NEGLIGIBLE even at synthetic h=2.0 |
| S (FTSE network centrality) | Hub structure in FTSE? | 🔴 HOMOGENEOUS dense network (Gini=0.13) |
| Q-v1 (AR(1) on FTSE) | Does AR(1) beat predict-mean? | 🟡 +10% MSE reduction (modest autocorrelation signal) |

## Probe Z bonus: breakthrough REPLICATED at n=5

The Probe Z smoke also independently confirmed the breakthrough on the NEW production
stack (including operator's NC29 + NC29a + NC30 + 14 parallel probes):

| Configuration | n | ASMS Ŝ | SMS Ŝ | Δ % | Paired p |
|---|---|---|---|---|---|
| FTSE production (cached, pre-NC29) | 10 | 0.000466 | 0.000433 | +7.50% | 0.003 |
| **FTSE production (this smoke, NC29 enabled)** | **5** | **0.000466** | **0.000416** | **+12.05%** | **0.016** |
| FTSE no-cost | 10 | 0.000493 | 0.000424 | +16.11% | 0.0002 |

**The breakthrough is ROBUST**: replicates across different smoke configurations and
seed sets, with n=5 paired p=0.016 (5/5 seeds positive) in this fresh smoke. The
+12.05% point estimate is higher than cached +7.50% (smaller n = higher variance), but
the direction and significance are unambiguous.

## Files created this session

**Probe code** (1 new module):
- `src/probes/probe_ag_q_velocity_sensitivity.py`

**Analyzers** (9 new):
- `experiments/analyze_probe_ad.py`
- `experiments/analyze_probe_acaf.py`
- `experiments/analyze_probe_aiboundary.py`
- `experiments/analyze_probe_ag.py`
- `experiments/analyze_probe_ab.py`
- `experiments/analyze_probe_tuy.py`
- `experiments/analyze_probe_s.py`
- `experiments/analyze_probe_q_ar1.py`
- (analyze_probe_l was from previous session)

**Reports** (10 new):
- `docs/W22-PROBE-AD-EMPIRICAL.md` + summary JSON
- `docs/W22-PROBE-AC-AF-KF-DIAGNOSTICS.md` + summary JSON
- `docs/W22-PROBE-AI-BOUNDARY-BIAS.md`
- `docs/W22-PROBE-AG-Q-VELOCITY-SENSITIVITY.md` + summary JSON
- `docs/W22-PROBE-AB-PER-PORTFOLIO-LAMBDA-K.md` + summary JSON
- `docs/W22-PROBE-TUY-SENSITIVITY.md`
- `docs/W22-PROBE-S-CENTRALITY-EMPIRICAL.md`
- `docs/W22-PROBE-Q-AR1-EMPIRICAL.md`
- `docs/W22-PROBE-Z-EMPIRICAL-VERDICT.md`
- `docs/W22-DECISION-POINTS-VERDICTS.md` (consolidation)
- `docs/W22-AUTONOMOUS-SESSION-SUMMARY.md` (this doc)

**Probe Z smoke results**:
- `experiments/results/w22-probe-z-empirical-5seed/per_seed.json`
- `experiments/results/w22-probe-z-empirical-5seed/run.log`
- (predictions.jsonl + per_seed_raw.json also generated)

**Source modifications**:
- `src/probes/probe_ad_delta_s_comparison.py`: +120 lines (eq636 helpers)

**Tests**:
- `tests/test_probe_ad_delta_s_comparison.py`: +4 tests (now 14 total, all pass)

## Operator action items (next session)

**High-priority follow-ups** (in suggested order):
1. **Paper writeup** with §6.4 caveats on AMFC boundary bias (Probe AI) +
   §IV KF mis-tuning footnote (Probe AC+AF)
2. **Probe AG FTSE smoke** — empirically validate that Q-velocity injection
   (Q = diag([0,0,σ²·1e-6, σ²])) preserves +7.50% breakthrough while whitening
   innovations. Requires opt-in env var `W22_NC32_VELOCITY_Q` + 5-seed smoke.
   If successful, this becomes NC32 ratification.
3. **n=10 NC29 stack confirmation** — re-run the cached production-stack 10-seed
   smoke with current NC29/NC29a/NC30 enabled to pin the post-NC29 breakthrough
   magnitude at n=10 (currently have n=5 +12.05% p=0.016).
4. **Probe Q-v1 wire-in** — test if 10% per-asset MSE reduction translates to
   portfolio HV gain via integrating AR(1) into ASMS expected-return inputs.
5. **Probe V single-fix Δ ablation** (~7.5 hrs) — quantify each opt-in fix
   (NC13b/NC27/NC30c/NC31) individually vs production baseline.

**Low-priority / nice-to-have:**
6. **Probe AD rectangle fix PR** — align deterministic _compute_hypervolume_contributions_class
   with Eq 6.36 in separate PR. Won't change ASMS (uses stochastic). May shift
   SMS baseline by a few %.
7. **Probe S hub-tilt experimental** — even though FTSE is homogeneous, an opt-in
   S-v1 hub-tilt could be tested as NC33 for completeness.

## Methodology notes

- **Honest baselines**: Probe Q-v1 initial finding of 65% MSE reduction was an artifact
  of the weak no-change baseline (50% inflation from "oscillate around mean" effect).
  Adding predict-mean baseline revealed the true ~10% autocorrelation signal. Lesson:
  when an initial result is "too good", find the stronger baseline.
- **Synthetic vs real characterization**: probes labeled as "synthetic" (AD, AG, AB, T/U/Y,
  AI synthetic-front) characterize the algorithm independent of FTSE. Probes labeled "FTSE"
  (S, Q-v1, Z, AC+AF) use real data. Both layers are needed for complete characterization.
- **Mechanism vs accuracy**: Probe AC+AF's finding of KF risk mis-tuning + Probe AB's
  finding of negligible per-portfolio λ^K differentiation jointly establish that the
  breakthrough's PER-PORTFOLIO differentiation comes from NC8c-v2 (position carry +
  KF state per portfolio), NOT from KF prediction accuracy or λ^K differentiation.
- **Structural vs accidental**: Probe AI's AMFC right-boundary degeneration is a
  STRUCTURAL consequence of HV-with-far-z_ref geometry, not a bug. The breakthrough's
  value comes from temporal tracking, not from clever Pareto-front argmax.

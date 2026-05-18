# W22 finding: NC1-NC6 + EQ1 bundle REGRESSES vs bug-fix-only baseline

*Generated 2026-05-18 after the post-NC1-NC6+EQ1 n=3 confirmation smoke
(6211s wall-clock). The 11+ critical fixes proposed by the parallel
specialist agents were thesis-faithful but empirically harmful on FTSE
2006-2012.*

## TL;DR

The combined application of NC1-NC6 (PR #137) + EQ1 (PR #139) produces
a measurably WORSE result than the bug-fix-only baseline (PR #135
alone):

| Variant | Bug-fix-only (PR #135) | NC1-NC6 + EQ1 | Shift |
|---|---|---|---|
| ASMS_mHDM_K3_v2rate | −6.96% (n=2) | **−7.71%** (n=3) | −0.75pp worse |
| ASMS_mHDM_K3_v2both | **−4.50% (n=2)** ⭐ | **−10.31%** (n=3) | **−5.81pp worse** |
| ASMS_mHDM_K2_v2both | −9.29% (n=2) | −8.49% (n=3) | +0.80pp better |

Best variant after the bundle: −7.71% (lost 3.21pp of recovery vs
bug-fix-only's −4.50%).

## Empirical data

### Smoke run

- Code state: feat/w22-eq1-stochastic-hv-theorem-6-3-1 branch
  (includes PR #135 + PR #137 NC1-NC6 + PR #139 EQ1)
- Scenarios: SMS_RDM_K0, ASMS_mHDM_K3_v2rate, ASMS_mHDM_K3_v2both,
  ASMS_mHDM_K2_v2both
- Seeds: 1-3 per scenario
- MC: n_mc=200 (CLI override; thesis is E=1000)
- Wall-clock: 6211.4s (~1h 43min) at --jobs 3
- Per-seed JSON: `python_refactor/experiments/results/w22-post-eq1-v2rate-v2both-n3/per_seed.json`

### Headline numbers

SMS_RDM_K0 (baseline) n=3 mean = 0.000378984

| Scenario | n=3 mean Ŝ | Δ vs SMS |
|---|---|---|
| SMS_RDM_K0 | 0.000378984 | — |
| ASMS_mHDM_K3_v2rate | 0.000349761 | −7.71% |
| ASMS_mHDM_K3_v2both | 0.000339908 | −10.31% |
| ASMS_mHDM_K2_v2both | 0.000346817 | −8.49% |

## Hypothesis for the regression

### EQ1 truncated Eq 6.41

The EQ1 agent report (PR #139) flagged this honest scar:

> "Cross-pair covariances assumed 0. The C++ legacy computes
> Cov(a,c), Cov(b,c), Cov(b,d) empirically from per-solution
> Monte-Carlo samples (`w_k->P.S.samples[j].roi/risk`). The Python
> refactor stores ONLY the 2×2 Kalman covariance per solution — no
> MC sample buffer. The only non-zero Cov retained is the within-
> solution self Cov(a,d) = KF P[0,1]."

The implemented Δ_S formula is the FULL Eq 6.41 algebraic shape but
with 3 of 4 cross-pair covariance terms set to 0. This produces a
NUMERICALLY DIFFERENT Δ_S signal than either:
- The pre-fix heuristic `(mean_dROI*var_drisk + mean_drisk*var_dROI) / (var_dROI + var_drisk)` (semantically wrong but stable)
- The full Eq 6.41 with C++-style MC sample Cov (semantically correct)

The truncated form combines the worst of both: it's structurally
similar to the correct equation but missing critical regularization.

### NC1-NC6 combined effects

The 6 fixes in PR #137 each have plausibility individually:
- NC1: velocity zero-pad → use full KF state (changes anticipation magnitude)
- NC2: drop R/N divisor (changes Kalman gain by 1000×)
- NC3: max_cardinality 10→15 (expands feasible portfolios by 50%)
- NC4: tournament_size 3→2 (reduces selective pressure)
- NC5: rank-then-Δ_S tiebreaker (changes selection criterion)
- NC6: n_mc 200→1000 (no observable effect since CLI was already overriding)

These all change DIFFERENT aspects of the algorithm. Some may improve
fidelity; others may interact poorly. The combined effect is empirically
−5.81pp for v2_both — worse than any individual fix expected.

The v2_both case specifically: NC5 (rank-then-Δ_S tiebreaker) makes
selection more thesis-faithful but if the underlying Δ_S signal is
broken (EQ1 truncation), this AMPLIFIES the bad signal.

## Recommendation

**Block / revert PR #137 + PR #139.** Keep PR #135 (rebalance + R-init
fixes) as the empirically-best state.

For salvaging the work:

1. **Ship PR #137 NC3 in isolation** (max_cardinality 10→15). This is
   a pure thesis-faithfulness fix with no theoretical interaction.
2. **Hold PR #137 NC1, NC2, NC4, NC5** for individual testing. Each
   should be smoked independently to verify it doesn't regress.
3. **Hold PR #139 EQ1** entirely until we can plumb MC samples through
   `Solution` to enable the full Eq 6.41 with cross-pair Cov (not the
   truncated 1-of-4 version).
4. **Drop PR #137 NC6** (n_mc CLI override exists; default change is
   moot unless the CLI flag is also removed).

## Discipline lesson

The agents proposed thesis-faithful fixes. Their reports correctly
flagged the honest scars (EQ1's truncation, NC4/NC5's interaction
risk). But we bundled all 11 fixes into 2 PRs and smoked the bundle
without isolation tests.

**Going forward**: each agent-proposed fix should be smoked
INDEPENDENTLY at n=2-3 before bundling. If a single fix regresses
vs the bug-fix-only baseline, don't include it. If a bundle regresses
but individual fixes are inconclusive, bisect.

Per operator's new discipline: "any mid experiment run that is giving
us significant negative delta performance against baseline shall be
killed after n=3." Application to this PR-discipline: any fix whose
isolated n=2-3 smoke regresses vs the prior best should NOT be merged.

## Updated best-known state

| Estimator | Best variant | Δ vs SMS | Code state |
|---|---|---|---|
| MC | ASMS_mHDM_K3_v2both | **−4.50%** | PR #135 only (rebalance + R-init fixes) |
| Option A pre-#131 | ASMS_mHDM_K3_v2rate | −1.89% | (estimator artifact; refuted) |
| Option B post-#131 | ASMS_mHDM_K3_v2rate | −29.06% | (different scale; not comparable) |

The bug-fix-only baseline (PR #135) closes 62% of default's −11.98% gap.
That's our publication-defensible state until we can prove individual
NC fixes are net-positive in isolation.

## Output artifacts

- This document
- `python_refactor/experiments/results/w22-post-eq1-v2rate-v2both-n3/per_seed.json`
- `python_refactor/experiments/results/w22-post-eq1-v2rate-v2both-n3/run.log`

## Next actions

1. Operator decides: block PRs #137 + #139 in GitHub UI
2. If bisecting is desired: launch 6 separate n=2 smokes (NC1-only, NC2-only, NC3-only, NC4-only, NC5-only, NC6-only — though NC6 is moot)
3. Pursue PO(8,1.0) baseline as a separate stream (where anticipation works in the paper)
4. EQ2/EQ3/EQ4 deprioritized after EQ1 regression

# W21-1 honest revision — n=2 substantially overstated Reading-E recovery

*Generated 2026-05-17 mid-Phase-B. Documents the n=2 vs n=10 discrepancy
on the W21-1 Reading-E + Reading-F results, before Phase B completes.
Companion to docs/W20-5-CORRECTION-READING-F-INVERSION.md (the prior
honest scar in this wave) and docs/READING-F-EXPERIMENTAL-TEST.md (the
W21-1 receipt being revised).*

## TL;DR

W21-1 (n=2, seeds 1-2) reported:
- ASMS_mHDM_K3 (default): **−11.82%** vs SMS_RDM_K0
- ASMS_mHDM_K3_v2rate: **−4.91%** vs SMS_RDM_K0 → "+6.91pp Reading-E recovery"

Phase B (n=10, in-progress) shows:
- ASMS_mHDM_K3 (default): **−9.92%** vs SMS_RDM_K0 (n=10 baseline)
- ASMS_mHDM_K3_v2rate: **−7.94%** vs SMS_RDM_K0 → "**+1.98pp recovery**"

**Reading-E magnitude was overstated by ~3.5× at n=2.** The directional
finding ("v2 monotonic anticipative-rate formula helps slightly") is
PRESERVED, but the magnitude shrinks dramatically with proper seed
count.

## What happened mechanically

The n=2 sample captured:
1. **Downside outlier for default ASMS**: seeds 1+2 had Ŝ_mean = 0.000327 (lowest 2 of 10), making default ASMS look 3pp worse than it actually is.
2. **Upside outlier for v2_rate**: seeds 1+2 had Ŝ_mean = 0.000353, near the middle of the n=10 distribution (final n=10 mean = 0.000357).

The COMBINATION of both biases produced an apparent +6.91pp delta that
is actually +1.98pp at n=10.

## What this means for the Reading framework

| Reading | W21-1 (n=2) | Phase B (n=10) | Status |
|---|---|---|---|
| E (anticipative-rate formula) | "+6.91pp recovery DOMINANT" | "+1.98pp recovery, modest" | **Revised: directionally correct, small magnitude** |
| F (Δ_S stability inversion) | "REFUTED (combined v2_both -8.12% worse than v2_rate -4.91%)" | TBD (v2_stab + v2_both not yet complete at n=10) | Pending Phase B completion |

The W21-1 conclusion that "Reading E is dominant" is REVISED:
- **Direction is correct**: v2 formula helps slightly (+1.98pp)
- **Magnitude is much smaller** than n=2 suggested
- **Reading E alone does not explain** the residual gap

This means W21-5's keystone ablation (V0..V7 + H=3) becomes MORE
important — no single Reading is the obvious gap-closer, so the
combined effect of multiple closures may be the residual-explainer.

## Cross-validation receipt — re-baselined deltas

Using the n=10 SMS_RDM_K0 baseline mean (0.000387833):

| Scenario | W21-1 Ŝ | W21-1 Δ vs n=2 baseline (0.000371260) | Phase B n=10 Δ vs n=10 baseline | Δ-shift |
|---|---|---|---|---|
| ASMS_mHDM_K3 (default) | 0.000327387 | -11.82% | -9.92% | +1.90pp (less negative) |
| ASMS_mHDM_K3_v2rate | 0.000353020 | -4.91% | -7.94% | -3.03pp (MORE negative) |

The Reading-E recovery shrinks from +6.91pp to +1.98pp = 71% magnitude reduction.

## What this changes for the publication-track decision (W21-6)

The W21-6 synthesis scaffold (docs/W21-CROSS-VALIDATION-SYNTHESIS.md)
listed 3 framings:

1. PAPER-REPLICATES (some V_k > V0)
2. PARTIAL (≥ 80% gap closure but still V_k ≤ V0)
3. REFUTED (none close meaningful gap)

With the n=10 revision, the W21-1 evidence for framing (2) PARTIAL is
weaker than W21-1 indicated:
- Reading E +1.98pp closes only ~20% of the n=10 default-ASMS gap (9.92%)
- Other Readings (D, V5 sqrt, V7 entropy) would need to contribute a
  combined +7.94pp to reach zero — substantially more than W21-1's
  optimistic projection

The Phase B + Phase C 30-seed run will produce the definitive numbers,
but the current trajectory leans more toward framing (3) REFUTED or a
weaker form of (2) PARTIAL than W21-1 suggested.

## What is NOT changed by this revision

- The Reading-F INVERSION refutation from W21-1 (v2_stab combined with
  v2_rate making things WORSE) appears robust — the same direction was
  seen at n=2 for BOTH default seeds. Phase B v2_stab + v2_both data
  (still landing) will confirm.
- The structural code-read findings from W19/W20/W21 cross-checks
  (A through N) are unaffected — those are deterministic comparisons,
  not seed-sensitive.

## Discipline scar pattern

This is the FOURTH honest correction in the W20→W22 chain:
1. **W20-5 fabricated stability formula** (portfolio.cpp:595 didn't exist)
2. **W21-4 "C++ uses SBX"** stale assumption (v2 actually uses uniform)
3. **W21-1 Reading-F INVERSION refuted by smoke** (disabling stability
   doesn't help)
4. **W21-1 n=2 overstated Reading-E recovery by 3.5×** (this document)

The cumulative lesson is the same in each case: **empirical test is the
truth; n=2 is insufficient to distinguish a real effect from seed-specific
noise; preliminary conclusions must be flagged as "n=2 pending n=30
validation" rather than as confirmed verdicts.**

## What W21-1's receipt should have said

> "PRELIMINARY n=2 smoke: ASMS_mHDM_K3_v2rate shows −4.91% vs −11.82%
> for default ASMS (+6.91pp). This is suggestive of a Reading-E recovery
> but **the n=2 sample is insufficient to distinguish a real effect from
> seed-specific outliers** — both the default and v2_rate scenarios have
> seed variance ≥ 2pp at n=2. A Phase B 10-seed run is needed before
> any verdict (PROVEN / PARTIAL / REFUTED). The n=2 directional sign
> is the most that can be concluded from this smoke."

Going forward: any smoke at n < 5 should be labeled "DIRECTIONAL only"
and a higher-n confirmation flagged as REQUIRED before the result feeds
the publication-track decision.

## Output artifacts

- THIS document (the honest revision)
- W21-1 receipt at docs/READING-F-EXPERIMENTAL-TEST.md is NOT amended
  (preserves the original conclusion as documented history); this
  revision supersedes the headline magnitudes
- W21-1 retro at .dfg/retrospectives/W21/W21-1.md likewise preserved

## Next

- Wait for Phase B v2_stab + v2_both to complete (~50 jobs left)
- Aggregate the FULL n=10 verdict per scenario
- Decide whether to launch Phase C (30 seeds) based on Phase B direction
- Update W21-6 synthesis with the revised Reading-E magnitude

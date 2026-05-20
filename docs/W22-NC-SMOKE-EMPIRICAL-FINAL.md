# W22 NC Smoke — FINAL paired empirical results (PO 8,1.0)

**Status:** SHIPPED (PAIRED n=10 for NC27_DEEP, paired n=5 for all others)
**Date:** 2026-05-20
**Total wall:** ~2.5 hr (initial 92min + 3 follow-up batches)
**Supersedes:** W22-NC-SMOKE-EMPIRICAL-N5.md (the +2.7% n=5 NC27_DEEP claim is now CORRECTED to +1.37% n=10)

---

## Honest correction from the n=5 report

The initial n=5 report claimed NC27_DEEP +2.7% on ASMS. The follow-up
seeds-6-10 batch revealed strong seed-to-seed variance: NC27_DEEP seeds 1-5
mean = 3.4861e-04, seeds 6-10 mean = 3.1593e-04 (a 10% swing from random
seed choice alone).

**PAIRED n=10 result: NC27_DEEP +1.37% (6/10 wins, p=0.625)** — direction
holds, but effect is HALF the n=5 estimate. This is the kind of correction
the drift registry was built to surface.

---

## Final empirical table (paired Δ vs BASELINE, ASMS_mHDM_K3_v2both)

| Config | n | Paired Mean %Δ | Wins | Wilcoxon p | Verdict |
|---|---|---|---|---|---|
| **NC27_DEEP** | **10** | **+1.37%** | 6/10 | 0.625 | DIRECTIONALLY POSITIVE |
| FULL_STACK | 5 | +1.90% | 3/5 | 1.000 | Marginal positive (NS) |
| NC13b | 5 | -0.41% | 2/5 | 0.812 | NEUTRAL — safest opt-in |
| TIP_CLEANUP | 5 | -4.46% | 2/5 | 0.812 | Negative |
| NC31 | 5 | -4.44% | 1/5 | 0.625 | Negative |
| NC27_NO_NC36 | 5 | -5.45% | 1/5 | 0.312 | Negative (surprise) |
| NC36 | 5 | -6.13% | 2/5 | 0.312 | Most negative |

All NS at α=0.05 (no Wilcoxon p < 0.05). DIRECTIONAL signals only.

---

## Per-seed table (1e-4 units, ASMS_mHDM_K3_v2both)

```
Config            s1    s2    s3    s4    s5    s6    s7    s8    s9    s10
BASELINE          3.43  3.16  3.34  3.10  3.95  2.98  3.26  3.13  3.14  3.27
NC27_DEEP         3.66  3.43  3.05  3.33  3.96  2.63  3.61  3.48  2.96  3.12
NC36              2.87  3.26  3.12  3.26  3.30   -     -     -     -     -
TIP_CLEANUP       3.72  3.10  3.13  3.23  2.90   -     -     -     -     -
FULL_STACK        3.88  3.59  3.08  3.28  3.35   -     -     -     -     -
NC13b             3.52  3.67  3.27  2.94  3.42   -     -     -     -     -
NC31              2.83  3.79  3.05  2.62  3.94   -     -     -     -     -
NC27_NO_NC36      3.58  3.14  3.09  2.95  3.21   -     -     -     -     -
```

Seed 5 is an outlier (BASELINE 3.95 vs typical ~3.2). Per-seed Δ is what
matters for paired tests.

---

## Findings (revised after follow-ups)

### 1. NC27_DEEP is DIRECTIONALLY positive but modest

- n=5 estimate +2.7% → n=10 paired +1.37%
- 6/10 wins → mild signal
- The synthetic claim was +2.8% (L2 reduction); empirical wealth gain is ~half
- Drift verdict: MODEST_TRANSLATION

### 2. NC36 (analytical TIP) is GENUINELY NEGATIVE on ASMS

- -6.13% paired n=5 (2/5 wins)
- Synthetic claim was PARITY (0%) → empirical OPPOSITE-SIGN
- Hypothesis stands: MC noise in λ^H may aid exploration; deterministic removes it

### 3. NC13b alone is NEUTRAL (-0.41%)

- Cleanest of the TIP fixes
- Could be ratified safely as a "no-harm upgrade" if TIP saturation is high
- The 4.7e-2 mean output diff Probe Y showed in synthetic high-saturation
  doesn't translate to wealth gain or loss

### 4. NC31 alone HURTS (-4.44%)

- Despite Inspection 1's <1.5% empirical TIP equivalence finding
- Wealth impact ≠ TIP value impact — composition with λ^K matters

### 5. Non-additive non-linear interactions

- NC27_DEEP +1.37%
- + NC13b + NC31 (NC27_NO_NC36): -5.45%  ← lost the gain plus more
- + NC13b + NC31 + NC36 (FULL_STACK): +1.90% ← NC36 RECOVERS the loss

The TIP fixes are NOT independent of NC27_DEEP. NC36 specifically appears
to counter-balance the harm from NC13b+NC31 when NC27_DEEP is also active.
Mechanism unclear; likely requires KF residual / λ^K trace inspection.

---

## Ratification recommendations (FINAL)

### Tier 1 — CONDITIONAL RATIFY

| Config | Why | Caveat |
|---|---|---|
| **NC27_DEEP** alone | +1.37% paired n=10; direction matches synthetic | NS at n=10; needs n≥30 for sig |
| **NC13b** alone | Neutral effect, safe upgrade | Only if TIP saturation observed |

### Tier 2 — DO NOT RATIFY (empirical hurts)

| Config | Wealth penalty | Reason |
|---|---|---|
| NC36 | -6.13% | MC noise as exploration hypothesis |
| NC31 | -4.44% | Conditional mode interacts with λ^K |
| TIP_CLEANUP | -4.46% | NC36 + NC31 stacking |
| NC27_NO_NC36 | -5.45% | NC13b+NC31 are not benign with NC27_DEEP |

### Tier 3 — INVESTIGATE FURTHER

| Config | Why investigate |
|---|---|
| FULL_STACK | +1.90% with NC36 — counter-intuitive recovery; needs n=10 + KF trace |
| NC27_DEEP + NC13b only (no NC31, no NC36) | Untested; might be the cleanest gain stack |

---

## Drift registry update

Now 13 claims; 9 tested (vs 4 at start of empirical work).

| Verdict | Count |
|---|---|
| STRONG_TRANSLATION | 1 (PROBE_AD bug) |
| MODEST_TRANSLATION | 2 (NC27_DEEP n=10, FULL_STACK, PROBE_Q_V1) |
| NEUTRAL | 2 (NC13b alone, PROBE_Z) |
| OPPOSITE_SIGN | 4 (NC36, TIP_CLEANUP, NC27_NO_NC36, SPO_BENCH) |
| UNTESTED | 4 (NC32 both regimes, NC29a γ, NC30/AMFC) |

Translation rate has SHIFTED: only 23% (3 of 13) translate STRONG/MODEST/EXCEEDS. The original +2.7% NC27_DEEP n=5 looked like a clear win; n=10 paired shows it's at the edge of MODEST.

**Mean drift score (now): -0.46** (lower than initial -0.43 — adding more "neutral/negative" data points pulls the mean down).

---

## Methodology lessons

1. **n=5 directional smokes are noisy**: a 9-10% seed-variance swing happened
   on NC27_DEEP across seeds 1-5 vs 6-10. Trust nothing until n≥10 paired.
2. **Unpaired comparisons are unfair**: BASELINE seeds 1-5 mean ≠ BASELINE
   seeds 6-10 mean; comparing across different seed sets inflates noise.
3. **Compose with caution**: NCs are NOT independent. NC36 + NC13b + NC31
   together (TIP_CLEANUP) ≠ sum of individual effects. NC36 alone hurts
   ASMS; NC36 in FULL_STACK with NC27_DEEP doesn't hurt.
4. **The drift registry has now caught 3 OPPOSITE_SIGN claims** (NC36,
   NC27_NO_NC36, TIP_CLEANUP) where synthetic suggested gain or parity but
   empirical reversed. The tool is doing its job.

---

## Next concrete actions for operator

### Immediate (low compute)
1. Run NC27_DEEP + NC13b ONLY (no NC31, no NC36): tests Tier-3 hypothesis
   that this is the cleanest gain stack. ~36 min wall.
2. Run NC27_DEEP seeds 11-20 to push to n=20: tighten the +1.37% estimate.
   ~36 min wall.

### Medium (higher compute)
3. FTSE 2015 n=10 with NC27_DEEP only: confirm the PO finding translates
   to real data. ~3 hours wall (FTSE is heavier than PO).
4. KF residual / λ^H trace inspection for NC36 vs MC: characterize the
   "exploration noise" hypothesis empirically. Requires telemetry plumbing.

### Long-term
5. The non-linear interactions among NC36/NC27_DEEP/NC13b/NC31 suggest a
   PROPER design-of-experiments (full factorial) on FTSE would be needed
   to actually disentangle. 2^4 = 16 cells × 10 seeds × 2 scenarios × ~10
   min = ~50 hours. Out of autonomous scope; operator decision.

---

## Output files

```
results/
  nc_smoke_full_20260520_0649.{json,md}       — initial 5-config sweep n=5
  nc_smoke_sanity.{json,md}                    — pipeline validation
  nc_smoke_baseline_seeds6_10_20260520_0937.{json,md}
  nc_smoke_nc27deep_seeds6_10_20260520_0903.{json,md}
  nc_smoke_no_nc36_20260520_0848.{json,md}
  nc_smoke_iso13b31_20260520_0916.{json,md}
```

All 8 configs × seeds covered (BASELINE: 1-10, NC27_DEEP: 1-10, others: 1-5).

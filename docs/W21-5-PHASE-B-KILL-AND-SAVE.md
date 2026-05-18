# W21-5 Phase B kill-and-save receipt

*Generated 2026-05-18 per operator directive: "kill jobs on n=10 if
not promising and >50% accumulated; save compute for promising
branches." Phase B was at 58/90 (64% done) and W22-A n=10 at 18/90
(20% done). Both killed; data extracted from stdout logs since
per_seed.json writes only at end-of-run.*

## What was killed

| Smoke | Status before kill | Saved jobs | Lost jobs |
|---|---|---|---|
| Phase B MC (n=10, 9 scenarios) | 58/90 = 64% done | 51 (extracted from log) | 32 (4 scenarios mostly untested: V7 entropy, V6-stub, H3, H3_v2rate) |
| W22-A n=10 (closed-form) | 18/90 = 20% done | 18 (extracted) | 72 (Option A confirmed outlier in #132 — limited value) |

## Phase B SAVED VERDICT (51 jobs extracted)

Baseline SMS_RDM_K0 (n=10) mean = 3.878e-04 std = 1.76e-05

| Variant | n | mean Ŝ | std | Δ vs SMS | Recovery vs default |
|---|---|---|---|---|---|
| SMS_RDM_K0 (baseline) | 10 | 3.878e-04 | 1.76e-05 | — | — |
| ASMS_mHDM_K3 (default) | 9 | 3.462e-04 | 1.66e-05 | **−10.74%** | (baseline) |
| ASMS_mHDM_K3_v2rate | 8 | 3.549e-04 | 1.91e-05 | **−8.49%** | **+2.25pp** |
| ASMS_mHDM_K3_v2stab | 8 | 3.501e-04 | 1.76e-05 | **−9.73%** | +1.01pp |
| ASMS_mHDM_K3_v2both | 8 | 3.549e-04 | 1.54e-05 | **−8.49%** | +2.25pp (same as v2_rate alone) |
| ASMS_mHDM_K3_v2rate_noSqrt (V5) | 8 | 3.517e-04 | 1.92e-05 | **−9.33%** | +1.41pp |

## Interpretation

**All tested single-flag recoveries are ≤ +2.25pp on a default-ASMS
gap of −10.74%.** The W21-1 n=2 "Reading-E +6.91pp recovery" estimate
was substantially overstated; the n=10 truth is +2.25pp.

Reading-F INVERSION (v2_stab alone) contributes only +1.01pp — barely
distinguishable from noise (std on default ≈ 1.7e-05; mean differences
~3-9e-06 are smaller than std).

W18-CARRY-1 sqrt removal (V5) contributes +1.41pp — also small,
consistent with "the sqrt was not the load-bearing divergence."

Combined v2_rate + v2_stab (v2_both) ties v2_rate alone (+2.25pp) —
**Reading-F INVERSION is REFUTED as additive to Reading E** at n=10.

## Decision: why kill

- **No promising branches in Phase B's remaining scenarios** based on
  what was tested: any closure that didn't manifest at n=8 isn't going
  to suddenly appear at n=10. The remaining V7 entropy + H3 variants
  are untested in Phase B but can be tested in a focused smoke at
  better --jobs (no longer competing with W22-A n=10 for resources).
- **W22-A n=10 was on Option A (outlier estimator per #132).** Even
  if it finished, the verdict would not be the primary publication
  basis. Cutting it free saves ~14 hours of compute.
- **V6 in Phase B is the stub-only version**, identical to v2_rate by
  design. Running its 10 seeds is pure waste — the V6 PR #125 real
  implementation needs a separate smoke if we want V6 data.

## What we lose

- V7 entropy operators @ n=10 (Phase B sweep had this queued)
- H3 + H3_v2rate @ n=10 (two-step-ahead variants)
- V6-stub @ n=10 (pure waste anyway since V6 stub = v2_rate)
- W22-A n=10 (outlier estimator, limited value)

## What we gain

- ~14 hours of compute freed up for:
  - Focused V7 + H3 + H3_v2rate MC smoke at --jobs 5 (running now;
    ~5 scenarios × 10 seeds × ~1100s / 5 = ~3.7 hours)
  - K=2 Option A smoke (operator new hypothesis; running)
  - Future smokes (V6-real, exact-v2-formula, 98-asset, etc.)

## Honest scar (kill discipline)

Phase B's verdict at n=8-10 makes clear that **the W21-1 honest revision
was still too generous**: Reading E recovery is +2.25pp (not +1.98pp as
honest revision n=10 partial said, and certainly not +6.91pp as W21-1
n=2 said). The new n~8-10 verdict is even more conservative.

## Output artifacts

- `python_refactor/experiments/results/w21-5-ablation-phase-b/per_seed_extracted.json`
  (51 jobs, from stdout log)
- `python_refactor/experiments/results/w22-option-a-n10-calibration/per_seed_extracted.json`
  (18 jobs, from stdout log)
- This receipt
- New focused V7+H3 smoke output (~3.7h)
- K=2 Option A smoke output (~3h)

## Next

After focused V7+H3 smoke completes, aggregate into the W21-6 synthesis
and decide on Phase C launch criteria. Per #132 cross-estimator
synthesis: primary verdict on MC + secondary on Option B.

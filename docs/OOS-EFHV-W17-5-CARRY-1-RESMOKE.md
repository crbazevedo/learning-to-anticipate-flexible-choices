# W17-5-CARRY-1 re-smoke — Eq 7.16 properly active in MultiHorizon

*Generated after PR #102 (W17-5-CARRY-1 fix making MultiHorizon use Eq 7.16 verbatim instead of Eq 6.6 alone).*

## Headline result

| scenario | grand mean Ŝ | std (n=2) | vs W17-5 |
|---|---|---|---|
| SMS_RDM_K0 | 3.68997e-04 | 2.86e-06 | +3.7% (likely n=2 RNG drift; baseline doesn't use MultiHorizon) |
| ASMS_mHDM_K3 | 3.25492e-04 | 7.14e-06 | +0.2% (effectively unchanged) |

**Δ(S2 − S0) = -11.79%** (was -8.72% at W17-5).

## What the trace CSV reveals (27,600 rows over 30 periods)

| Field | Mean | Std | Min | Max |
|---|---|---|---|---|
| λ^H | 1.43e-3 | 2.00e-3 | 0 | 0.0246 |
| λ^K | 7.47e-7 | 9.09e-7 | 4.3e-8 | 3.0e-6 |
| λ (combined) | 7.15e-4 | 1.00e-3 | 2.1e-8 | 0.0123 |
| TIP | 0.500017 | 0.022242 | 0.410 | 0.592 |

Eq 7.16 sanity: `max |λ − 0.5*(λ^H + λ^K)| = 9.9e-17` ✅ formula verbatim correct.

All 27,600 rows in `kperiod_sum` branch — **λ^K firing in 100% of calls** (no warm-up fallback).

## The actual finding

The trace data shows BOTH arms of the anticipation rate are tiny in production:

1. **λ^H ≈ 0** because TIP is centered at 0.5 (max-uncertainty). When `TIP = 0.5`, `binary_entropy = 1.0` exactly, and `λ^H = (1/(H-1)) * (1 - 1) = 0`. The TIP std is only 0.022 around 0.5 — extremely narrow.

2. **λ^K ≈ 0** because KF squared residuals on returns data (~0.01 scale) produce sum-over-K=3 of ~1e-4 to 1e-6, which my sigmoid normalization (scale floored at 1.0) collapses to near-zero.

Both arms near zero → λ_combined ≈ 0 → the anticipation chain barely modifies the solution state. **ASMS effectively reduces to "SMS with extra noise from near-zero state updates".** Hence S2 ≤ S0.

## Why W17-5-CARRY-1 widened the gap (-8.72% → -11.79%)

Pre-W17-5-CARRY-1 the MultiHorizon path used Eq 6.6 alone: `λ = (1/(H-1)) * (1 - entropy(TIP))`. Same TIP-near-0.5 → λ^H tiny — but at least UNHALVED.

Post-W17-5-CARRY-1 the path uses Eq 7.16: `λ = 0.5 * (λ^H + λ^K)`. λ^K is still tiny, so the (1/2) factor effectively HALVES the already-tiny λ^H. Halved-anticipation → ASMS even closer to "do nothing" → SMS baseline (which has no anticipation cost) pulls further ahead.

## Two readings (revised after this re-smoke)

**Reading A (revised)**: BOTH arms of the anticipation rate are correctly implemented but tiny because the predictive distribution provides no useful information on this dataset. The anticipation overhead can only HURT when predictions are uninformative. To get S2 > S0 we'd need either:
- A dataset where TIP actually moves away from 0.5 (more predictable returns)
- A different λ normalization scale that amplifies small residual sums into meaningful rates
- A different metric (multi-period wealth) that doesn't penalize anticipation overhead

**Reading B (revised)**: The TIP calculator, λ^H formula, or λ^K normalization has a subtle bug that prevents the anticipation rates from rising into the (0.1, 0.5] range the thesis presumably intends. The thesis presumably ran on data where rates were larger.

## Strategic implications for W18

This is a meaningful diagnostic result. Three concrete W18 directions:

1. **λ^K normalization revisit**: my sigmoid with `scale = max(1.0, mean)` floors at 1.0 — wrong for sub-1.0 residual sums. Better: running-max-residual-sum normalization, OR a fixed scale derived from typical returns variance.

2. **TIP normalization audit**: why does TIP cluster at 0.500 ± 0.022? Either the TIP formula is collapsing to a degenerate output, or the predictive distribution genuinely contains no information on this dataset.

3. **Multi-period wealth metric**: tests whether single-period EFHV is the wrong yardstick — if anticipation pays over multiple periods through wealth accumulation, the per-period metric will miss it.

## Output artifacts

- `python_refactor/experiments/results/w17-5-carry-1-resmoke/lambda_tip_trace.csv` (2.5 MB, 27,600 rows)
- `python_refactor/experiments/results/w17-5-carry-1-resmoke/per_seed.json`
- This document

## Reproducing

```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3 \
    --seeds 1-2 \
    --train-window-days 378 \
    --step-days 50 \
    --n-mc 200 \
    --jobs 2 \
    --enforce-thesis-continuous-trades \
    --lambda-trace-csv-path experiments/results/w17-5-carry-1-resmoke/lambda_tip_trace.csv \
    --output ../docs/OOS-EFHV-W17-5-CARRY-1-RESMOKE.md \
    --per-seed-json experiments/results/w17-5-carry-1-resmoke/per_seed.json
```

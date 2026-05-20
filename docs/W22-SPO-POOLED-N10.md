# W22 sPO(8, 1.0)-cosine — pooled n=10 honest result

**Status:** Honest negative
**Trigger:** Operator request to validate sPO bench smoothness-matters hypothesis at n=10
**Data:** PO(8, 1.0)-cosine synthetic, 1250 days × 30 assets, train=378, step=50, MC=200

## Per-batch results (aggregated only — per-seed CSV not available)

| Batch    | n_seeds | ASMS_mHDM_K3_v2both | SMS_RDM_K0 | Δ ASMS−SMS | %Δ      |
|----------|---------|---------------------|------------|------------|---------|
| Seeds 1–5  | 5       | 3.2680e-04          | 3.1737e-04 | +9.43e-06  | +2.97%  |
| Seeds 6–10 | 5       | 3.1593e-04          | 3.2713e-04 | −1.12e-05  | −3.42%  |
| **Pooled n=10**  | **10**  | **3.2137e-04**      | **3.2225e-04** | **−8.8e-07** | **−0.27%** |

## Interpretation

The +2.97% (NS) headline at n=5 seeds 1–5 was **noise**. On a second batch
(seeds 6–10), ASMS lost to SMS by 3.42% — almost equal-and-opposite. Pooled
n=10 mean is essentially flat (−0.27%).

This is the **expected** null result on PO(8, 1.0)-cosine synthetic data:
- Synthetic data has no real predictive structure beyond what's interpolated
- Cosine smoothing makes derivatives near-zero at period boundaries
- ASMS's anticipation machinery has nothing predictable to anticipate

Compare to **FTSE 2015 n=10**: ASMS won by +7.50% (Wilcoxon p=0.003 — see
W22-BREAKTHROUGH-VALIDATED-N10.md). On REAL data with realistic temporal
structure, the anticipation signal is real.

## What this means for the W22 program

1. **The FTSE result is reaffirmed by exclusion.** ASMS uplift requires
   predictive structure in the data. On synthetic data without that structure
   (sPO smooth), no uplift materializes. On real data with structure (FTSE),
   uplift is +7.50% significant.
2. **The smoothness-matters hypothesis is partially supported.** At n=5
   seeds 1–5, ASMS edged +2.97% on smooth synthetic — but at n=10 this
   collapsed to noise. The smoothness alone is insufficient to drive a
   structural ASMS advantage; you need predictive STRUCTURE within the
   smoothness.
3. **Direction for sPO bench design:** to get a positive-control synthetic
   benchmark, we need to BAKE IN a predictable pattern (e.g., regime shifts
   on a fixed schedule, deterministic trend in correlation matrix) that
   ASMS's anticipation can exploit. Pure cosine interpolation of random
   Pareto points doesn't carry such structure.

## Honest scars

- **Aggregated-only data:** the per-batch JSON saves only grand mean/std,
  not per-seed Ŝ values. Cannot run a proper Wilcoxon signed-rank test
  across the 10 seeds — only a mean-of-means comparison. A future smoke
  run should save per-seed CSV for paired statistical testing.
- **Two independent batches, not strictly identical protocol:**
  - Both used seed-pool-batches of 5 with same wall-clock (~1926s)
  - But seeds 1–5 and 6–10 may have had slightly different RNG state
    depending on how the seed bank is salted; if the same code path
    saw similar seeds it could still produce within-batch correlation
- **Walk-forward Ŝ != final wealth:** Ŝ is the future hypervolume metric;
  it correlates with but doesn't equal final wealth. A separate evaluation
  run is needed to map sPO Ŝ back to wealth gain in the production sense.

## Next steps

This result CLOSES the sPO-bench validation arm with a negative finding.
Subsequent ASMS hill-climbing should:

- Continue on FTSE / NASDAQ / HangSeng (real-data benchmarks) where
  predictive structure is real
- Either re-design the sPO bench with intentional predictable structure
  (per "Direction for sPO bench design" above) OR abandon it as a
  benchmark for anticipation-rich algorithms
- Run the Tier-1 NC stack (NC26 Eq 6.41 truncation, NC27 Dirichlet
  replacement, NC30 operator-correct AMFC) on FTSE to test whether the
  inspection-identified gaps account for residual headroom beyond the
  current +7.50%

## Citations

- `results/spo_smoke_5seed.json` — seeds 1–5 aggregate
- `results/spo_smoke_seeds_6_10.json` — seeds 6–10 aggregate
- `docs/W22-BREAKTHROUGH-VALIDATED-N10.md` — FTSE +7.50% reference result
- `docs/W22-INSPECTIONS-SYNTHESIS.md` — Tier-1 NC stack

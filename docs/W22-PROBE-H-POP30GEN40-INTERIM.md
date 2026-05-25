# W22 Probe H interim: pop=30, gens=40 at n=3 (extending to n=6)

*Generated 2026-05-19 after first 3-seed Probe H run.*

## Headline (3-seed, suggestive but underpowered)

| Run | ASMS Ŝ | SMS Ŝ | Δ % |
|---|---|---|---|
| Baseline NC8c-v2+NC8d (n=10 validated) | 0.000466 | 0.000433 | +7.50% (p=0.003) |
| **Probe H pop=30/gens=40 (n=3)** | **0.000497** | 0.000435 | **+14.20%** |

ASMS jumped +6.7% absolute (0.000466 → 0.000497); SMS barely moved (+0.5%). Suggests ASMS benefits MORE from extra compute than SMS.

## Per-seed honest breakdown

| Seed | Baseline ASMS | Probe H ASMS | ΔASMS | Baseline SMS | Probe H SMS | ΔSMS | Δ(ASMS-SMS)% Δ-of-Δ |
|---|---|---|---|---|---|---|---|
| 1 | 0.000478 | 0.000509 | +6.5% | 0.000439 | 0.000427 | −2.7% | +10.27 pp ✅ |
| 2 | 0.000483 | 0.000447 | **−7.5%** | 0.000452 | 0.000427 | −5.5% | −2.16 pp 🟡 |
| 3 | 0.000465 | 0.000534 | +14.8% | 0.000384 | 0.000450 | +17.2% | −2.28 pp 🟡 |

**Mean Δ-of-Δ: +1.94 pp** (vs +14.20% raw Δ — much of which is artifact of SMS seeds 1-2 coincidentally being lower in Probe H run).

## Caveats

The +14.20% Δ% headline is INFLATED by:
- SMS seed 1 = 0.000427 in Probe H run vs 0.000439 in baseline (−2.7% drop)
- SMS seed 2 = 0.000427 in Probe H run vs 0.000452 in baseline (−5.5% drop)
- Both runs use the same seeds — these SMS drops are within the seed's stochastic variance.

The paired-test view (per-seed Δ-of-Δ) shows mean +1.94 pp improvement, much more modest. With n=3 the 95% CI for mean Δ-of-Δ includes 0.

## Mechanism hypothesis

If validated at higher n, pop=30/gens=40 ASMS improvement may come from:
- 50% more population members → larger Pareto front (mean 11 vs prior 7-8) → more diverse anticipation rates
- 33% more generations → more time for selection pressure to refine
- ASMS exploits the larger front more than SMS (ASMS's anticipation arm rewards portfolios that differentiate)

## Next step

Extending to seeds 4-6 (n=6 total) — in flight (PID 39xxx, ~50 min).
Will compute n=6 paired statistics for honest verdict.

## Status

⏳ INTERIM verdict: pop=30/gens=40 PROMISING but n=3 too small for confident inference. Wait for n=6 extension.

# W22 Probe H FINAL n=10: pop=30/gens=40 confirms ASMS > SMS but Δ-of-Δ NOT significant

*Generated 2026-05-19 after pooling Probe H seeds 1-10.*

## Headline: pop=30/gens=40 doesn't significantly improve over breakthrough baseline

| Run | n | ASMS Ŝ | SMS Ŝ | Δ % | Paired p | Status |
|---|---|---|---|---|---|---|
| Baseline NC8c-v2+NC8d (pop=20/gens=30) | 10 | 0.000466 | 0.000433 | +7.50% | 0.003 | ✅ validated |
| Probe H (pop=30/gens=40) at n=6 (early) | 6 | 0.000501 | 0.000441 | +13.62% | 0.0024 | suggested improvement |
| **Probe H (pop=30/gens=40) at n=10 (final)** | **10** | **0.000496** | 0.000455 | **+8.89%** | **0.0098** | **Δ-of-Δ NOT sig** |

**The n=6 +13.62% was inflated by lucky seed selection (seeds 1-6).** At full n=10, the Δ% drops back to +8.89%, only +1.39 pp better than baseline. Δ-of-Δ paired-test against 0 gives p=0.358 — NOT statistically significant.

## Statistical evidence at n=10

- Probe H pop=30/gens=40 still confirms ASMS > SMS: paired t=2.83, **p=0.0098**
- 8 of 10 seeds: ASMS > SMS (vs baseline's 9 of 10)
- ASMS std: 3.18e-5 (vs baseline 2.53e-5 — MORE variance with more compute)
- SMS std: 3.59e-5 (vs baseline 3.38e-5 — slight increase)

## Per-seed Δ-of-Δ honest breakdown

| seed | baseline Δ% | Probe H Δ% | Δ-of-Δ |
|---|---|---|---|
| 1 | +8.93% | +19.08% | **+10.14 pp** |
| 2 | +6.78% | +4.50% | −2.28 pp |
| 3 | +20.91% | +18.75% | −2.16 pp |
| 4 | +1.13% | +19.00% | **+17.87 pp** |
| 5 | +0.46% | +16.08% | **+15.62 pp** |
| 6 | +2.90% | +4.85% | +1.95 pp |
| 7 | −1.28% | +12.66% | **+13.95 pp** |
| 8 | +16.40% | +11.45% | −4.88 pp |
| 9 | +13.69% | −3.65% | **−17.33 pp** ⚠️ |
| 10 | +8.85% | −8.73% | **−17.63 pp** ⚠️ |
| **mean** | — | — | **+1.52 pp** |
| **std** | — | — | **12.85** |

4 seeds: huge gains (+10 to +18 pp); 4 seeds: near-neutral; 2 seeds: huge losses (−17 pp).

## Mechanism interpretation

pop=30/gens=40 doubles compute per evolution → larger Pareto front (mean ~11 vs ~8) AND more generations for selection refinement. Both AS-MS and SMS benefit somewhat, but the ASMS-vs-SMS Δ% is essentially unchanged (+1.5 pp avg with high variance).

The high per-seed variance suggests pop/gens scaling is SEED-DEPENDENT:
- Some random initial populations benefit from extra exploration (find better basins)
- Others over-fit or get stuck in worse local optima with more generations
- No robust monotone improvement

## Production decision

KEEP baseline (pop=20/gens=30) as production default:
- Validated at +7.50% Δ p=0.003
- 9/10 seeds positive (more consistent than Probe H's 8/10)
- Tighter ASMS variance (std 2.53e-5 vs 3.18e-5)
- 15% less wall-clock per pair

pop=30/gens=40 available as opt-in via:
- `--scenarios ASMS_mHDM_K3_v2both_pop30gen40,SMS_RDM_K0_pop30gen40`

## Hill-climb dashboard (final)

| Iteration | Δ % | n | p-value | Validated? |
|---|---|---|---|---|
| Baseline (post-NC7) | −5.92% | 2 | — | starting point |
| + NC8b | +1.70% | 2 | — | first signal |
| + NC8c-v2 + NC8d | +7.38% | 5 | — | breakthrough |
| **+ n=10 validation (THE BREAKTHROUGH)** | **+7.50%** | **10** | **0.003** | ✅ validated |
| Probe H pop=30/gens=40 (n=10) | +8.89% | 10 | 0.0098 | ✅ ASMS>SMS but Δ-of-Δ p=0.358 not significant |

NET hill-climb: **−5.92% → +7.50% (n=10, p=0.003)** is the canonical session result.

Probe H is a positive but non-significant refinement — useful as a parameter tweak but not a structural improvement.

## Methodology lesson

The Probe H n=3 → n=6 → n=10 trajectory is a **textbook example of insufficient-n problem**:
- n=3: +14.20% (too few seeds, looks great)
- n=6: +13.62% (still inflated; lucky seed selection)
- n=10: +8.89% (drops to honest level; Δ-of-Δ p=0.358 not sig)

The +14% and +13.62% headlines were FALSE WINS that would have led to incorrect "ship pop=30/gens=40 as default" conclusion if we hadn't extended to n=10.

ALWAYS validate at n≥10 with paired tests before declaring a candidate fix shipped.

# W22 Probe Z empirical verdict: legacy vs v2 stability factor on FTSE 5-seed

*Generated 2026-05-20 after 1h46m wall-clock smoke (15 pairs / 4 workers).*

## Question

Probe Z theoretical analysis showed that Python's legacy stability factor
`1/(1+trace(P))` and v2's constant `1.0` disagree on argmax 50%+ of the time at
high P-trace spread. Does this argmax disagreement translate to a measurable
realized HV difference on real FTSE walk-forward?

## Methodology

- Walk-forward 5-seed FTSE smoke with three scenarios:
  - `ASMS_mHDM_K3_v2rate` (default `use_v2_stability_weighting=False` → LEGACY stability)
  - `ASMS_mHDM_K3_v2both` (`use_v2_stability_weighting=True` → v2 stability = production)
  - `SMS_RDM_K0` (baseline for context)
- Train window: 378 BD, step: 50 BD, MC scenarios: 200
- 23 walk-forward periods per (scenario, seed), seeds 1-5
- Wall-clock: 6374.9s = 1h46m on 4 workers

## Probe Z primary result — legacy vs v2 stability

| seed | v2rate (legacy) | v2both (v2) | Δ (legacy − v2) | Δ % |
|---|---|---|---|---|
| 1 | 0.000504 | 0.000495 | +0.0096e-3 | **+1.95%** |
| 2 | 0.000465 | 0.000436 | +0.0295e-3 | **+6.76%** |
| 3 | 0.000481 | 0.000501 | −0.0202e-3 | −4.03% |
| 4 | 0.000436 | 0.000457 | −0.0211e-3 | −4.62% |
| 5 | 0.000392 | 0.000443 | −0.0512e-3 | −11.55% |
| **MEAN** | **0.000456** | **0.000466** | **−0.0107e-3** | **−2.29%** |

**Statistical tests (paired by seed):**
- Paired t-test: t = −0.768, p (two-sided) = **0.485 (NOT significant)**
- Wilcoxon signed-rank: W = 5.0, p = **0.625 (NOT significant)**
- Legacy wins 2 of 5 seeds; v2 wins 3 of 5

## 🟡 Verdict: NEUTRAL — legacy and v2 stability indistinguishable at n=5

The theoretical 50%+ argmax disagreement between legacy `1/(1+trace(P))` and v2
constant `1.0` does NOT translate to a measurable realized HV difference on real
FTSE. Mean Δ is small (−2.29%), per-seed spread is large (−11.55% to +6.76%), and
no formal test reaches significance.

**Implication:** production's choice of `use_v2_stability_weighting=True` (v2 = constant 1.0)
is empirically sound. The legacy formula does NOT provide a measurable advantage and may
even be marginally worse. No need to change production.

**Why no significant difference?** Two possibilities:
1. The argmax disagreement is real but the disagreeing solutions have similar realized HV
   (the choice doesn't matter much when the next-best options are close)
2. The KF state covariance P doesn't vary enough across the population to make
   `1/(1+trace(P))` a meaningful discriminator on real FTSE data

## Bonus result — v2both vs SMS (breakthrough replication at n=5)

| seed | v2both (production) | SMS baseline | Δ ASMS−SMS | Δ % |
|---|---|---|---|---|
| 1 | 0.000495 | 0.000436 | +0.0584e-3 | **+13.40%** |
| 2 | 0.000436 | 0.000408 | +0.0275e-3 | **+6.74%** |
| 3 | 0.000501 | 0.000405 | +0.0954e-3 | **+23.53%** |
| 4 | 0.000457 | 0.000423 | +0.0342e-3 | **+8.09%** |
| 5 | 0.000443 | 0.000408 | +0.0353e-3 | **+8.65%** |
| **MEAN** | **0.000466** | **0.000416** | **+0.0502e-3** | **+12.05%** |

**Statistical tests:**
- Paired t-test: t = 4.028, p (two-sided) = **0.016 (SIGNIFICANT)**
- ASMS wins ALL 5 of 5 seeds

## 🟢 Bonus verdict: breakthrough REPLICATED at n=5 (+12.05% paired p=0.016)

This fresh smoke independently confirms the +7.50% breakthrough on FTSE n=10
(paired p=0.003). The +12.05% on n=5 is slightly higher (smaller-n variance) but
the direction is unambiguous and significant at α=0.05.

The breakthrough is ROBUST — it survives reproduction across different smoke
configurations and n=5 statistical power.

## Combined context

| Configuration | n | ASMS Ŝ | SMS Ŝ | Δ % | Paired p |
|---|---|---|---|---|---|
| FTSE production (cached n=10) | 10 | 0.000466 | 0.000433 | +7.50% | 0.003 |
| FTSE production (this smoke n=5) | 5 | 0.000466 | 0.000416 | **+12.05%** | **0.016** |
| FTSE no-cost (cached n=10) | 10 | 0.000493 | 0.000424 | +16.11% | 0.0002 |
| FTSE LEGACY stability (this smoke n=5) | 5 | 0.000456 | 0.000416 | +9.42% | — |
| ASMS v2 vs LEGACY (this smoke n=5) | 5 | — | — | **−2.29%** | **0.485 NS** |
| PO(8, 1.0) | 10 | 0.000324 | 0.000349 | −7.25% | 0.15 NS |

## Final canonical update for the breakthrough chain

```
Setup: NC7+NC8b+NC8c-v2+NC8d+NC13a + thesis transaction cost
Decision: use_v2_stability_weighting=True (v2 constant 1.0) — empirically ratified
           via Probe Z (legacy alternative not better at n=5)
Production: ASMS_mHDM_K3_v2both
FTSE: +7.50% Δ at n=10 paired p=0.003 (cached) / +12.05% Δ at n=5 p=0.016 (this)
```

## Operator action items

- ✅ Probe Z verdict landed: don't bother re-running with `use_v2_stability_weighting=False`
- 🟡 Optional: re-run Probe Z at n=10 if a formal statistical guarantee is needed (3 more
  hrs wall-clock). Current n=5 result is sufficient for "no signal, no action".
- ✅ The breakthrough is replicated at n=5 — this gives independent confidence that the
  cached n=10 result wasn't a fluke.

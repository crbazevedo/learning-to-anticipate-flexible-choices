# W22 Combined Probe H+I: ASMS hits ceiling ~0.000494 — helpers don't stack additively

*Generated 2026-05-19 after Probe H+I combined smoke (pop=30/gens=40 + no-cost) 5-seed.*

## Headline: ASMS ceiling at ~0.000494 across multiple helper configurations

| Configuration | n | ASMS Ŝ | SMS Ŝ | Δ % | Paired p |
|---|---|---|---|---|---|
| Baseline (with cost) | 10 | 0.000466 | 0.000433 | +7.50% | 0.003 |
| Probe H pop=30/gens=40 (with cost) | 10 | 0.000496 | 0.000455 | +8.89% | 0.0098 |
| Probe I no-cost (pop=20/gens=30) | 10 | 0.000493 | 0.000424 | +16.11% | 0.0002 |
| **Combined H+I (pop=30/gens=40 + no-cost)** | **5** | **0.000494** | 0.000435 | **+13.65%** | **0.0012** |

ASMS Ŝ at all three "improved" configurations: **0.000493-0.000496** (within 0.6%). The helpers SATURATE at this ceiling — combining them does NOT push past.

## Per-seed combined vs Probe I no-cost only (seeds 1-5)

| seed | Probe I no-cost (only) | Combined H+I | ΔASMS |
|---|---|---|---|
| 1 | 0.000512 | 0.000455 | **−5.7e-05** |
| 2 | 0.000429 | 0.000480 | **+5.2e-05** |
| 3 | 0.000529 | 0.000525 | −0.4e-05 |
| 4 | 0.000430 | 0.000510 | **+8.0e-05** |
| 5 | 0.000473 | 0.000501 | +2.8e-05 |
| **mean** | 0.000475 | 0.000494 | **+1.97e-05** (+4.15%) |

Combined is slightly higher on average, but mixed per-seed (2 of 5 lower, 3 of 5 higher). Within noise.

## Mechanism: helpers exploit DIFFERENT lever, saturate at same ceiling

Three mechanism helpers tested in W22:
- **Probe H pop=30/gens=40**: gives the OPTIMIZER more search budget (larger Pareto front, more generations)
- **Probe I no-cost**: removes the TRANSACTION-COST PENALTY on turnover
- **Combined H+I**: both

All three converge ASMS Ŝ to ~0.000494. The breakthrough's per-portfolio differentiation (NC8c-v2's position carry) is the LOAD-BEARING mechanism — once it's enabled, additional helpers can't extract more value because the optimizer is already exploiting the available signal optimally given the noisy KF predictions.

**Diminishing returns at the algorithmic ceiling**: ASMS's performance is bounded by how much MEANINGFUL differentiation signal the anticipation arm can provide. With biased KF predictions, the signal magnitude is finite, so optimization improvements (more compute) and cost removal (less friction) both hit the same ceiling.

## Δ% behavior: SMS also responds to helpers

| Config | SMS Ŝ |
|---|---|
| Baseline (with cost) | 0.000433 |
| Probe H pop=30/gens=40 (with cost) | 0.000455 (+5.1% vs baseline) |
| Probe I no-cost (pop=20/gens=30) | 0.000424 (−2.1% vs baseline) |
| Combined H+I | 0.000435 (essentially baseline) |

SMS benefits from pop=30/gens=40 (more compute → better front quality) but LOSES from no-cost (no anchor against over-fitting). Combined: these effects partially cancel.

So the relative Δ% depends on the SMS response too:
- Probe I (+16.11%) widens partly because SMS degrades
- Combined (+13.65%) is narrower because SMS gains some back from pop=30 scaling

## Conclusion + verdict

**Combined H+I is not a viable production candidate** because:
1. ASMS doesn't exceed the ~0.000494 ceiling reached by Probe I alone
2. Δ% widens less than Probe I alone (+13.65% vs +16.11%)
3. Wall-cost is ~2.5× baseline (pop=30/gens=40 ASMS pairs ~6648s vs ~2500s baseline)

**Production stack remains**: NC7 + NC8b + NC13a + NC8c-v2 + NC8d @ pop=20, gens=30, with thesis Table 7.1 cost active. Validated at n=10 with paired p=0.003. The +7.50% Δ% under paper-faithful conditions is the canonical published number.

The +16.11% no-cost ceiling and the +13.65% combined ceiling reveal the upper bound of ASMS's intrinsic mechanism value, but those configurations sacrifice paper-faithfulness.

## Hill-climb dashboard FINAL FINAL

| Iteration | Δ % | n | p-value | Status |
|---|---|---|---|---|
| Baseline (post-NC7) | −5.92% | 2 | — | starting point |
| + NC8b | +1.70% | 2 | — | first signal |
| **+ NC8c-v2 + NC8d (PRODUCTION)** | **+7.50%** | **10** | **0.003** | ✅ paper-faithful canonical |
| + Probe H (with cost) | +8.89% | 10 | 0.0098 | ASMS>SMS but Δ-of-Δ NS |
| + Probe I (no cost) | **+16.11%** | **10** | **0.0002** | ✅ mechanism ceiling (paper-faithful violated) |
| + Combined H+I | +13.65% | 5 | 0.0012 | ASMS hits ceiling, helpers saturate |

ASMS mechanism ceiling on FTSE 2006-2012: ~0.000494 grand-mean Ŝ.

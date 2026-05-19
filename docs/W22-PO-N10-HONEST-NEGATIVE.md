# W22 PO(8,1.0) n=10: breakthrough does NOT generalize to disrupted synthetic data

*Generated 2026-05-19 after pooling PO seeds 1-10.*

## Headline: ASMS LOSES to SMS on PO(8,1.0) at n=10 (Δ=−7.25%, p=0.15 two-sided)

| Configuration | n | ASMS Ŝ | SMS Ŝ | Δ % | Paired p |
|---|---|---|---|---|---|
| FTSE production-faithful | 10 | 0.000466 | 0.000433 | **+7.50%** | **0.003** |
| FTSE no-cost (mechanism ceiling) | 10 | 0.000493 | 0.000424 | **+16.11%** | **0.0002** |
| **PO(8,1.0)** | **10** | **0.000324** | **0.000349** | **−7.25%** | **0.15** (two-sided) |

3 of 10 seeds positive on PO(8,1.0); Wilcoxon p=0.19.

## Per-seed variance

Huge per-seed spread on PO(8,1.0), with NO clear ASMS advantage:

| seed | ASMS | SMS | Δ % |
|---|---|---|---|
| 1 | 0.000309 | 0.000402 | **−23.13%** |
| 2 | 0.000388 | 0.000332 | +16.87% |
| 3 | 0.000310 | 0.000330 | −6.06% |
| 4 | 0.000322 | 0.000382 | **−15.71%** |
| 5 | 0.000267 | 0.000265 | +0.75% |
| 6 | 0.000294 | 0.000330 | −10.91% |
| 7 | 0.000336 | 0.000374 | −10.16% |
| 8 | 0.000300 | 0.000395 | **−24.05%** |
| 9 | 0.000368 | 0.000325 | +13.23% |
| 10 | 0.000344 | 0.000356 | −3.37% |
| **mean** | 0.000324 | 0.000349 | **−7.25%** |

7 of 10 seeds: ASMS loses to SMS. 3 of 10 seeds: ASMS wins. Pattern is consistent with "anticipation hurts more than helps on disrupted data".

## Mechanism explanation

**PO(8,1.0) generator** (per paper Eqs 31-32 / thesis 7.6-7.9):
- 30 assets, 25 periods × 50 BD = 1250 days
- Disruption every τ=8 periods at η=1.0 (max severity)
- Within an 8-period segment: linear interpolation (smooth dynamics)
- At disruption boundary: parameters JUMP to random new values

**Why ASMS fails on PO(8,1.0)**:
1. NC8c-v2 carries prev_AMFC's POSITION (ROI, risk) and VELOCITY across periods
2. KF velocity learns period-over-period change in expected returns
3. **At a disruption boundary, the change is RANDOM and DISCONTINUOUS** — not a velocity but a step
4. KF velocity captures the wrong signal; predictions diverge from reality
5. The optimizer is mis-steered by the misleading anticipative signal

**Why SMS wins on PO(8,1.0)**:
1. SMS has no anticipation arm → no chance of being mis-steered
2. SMS picks the optimum portfolio for the training window
3. Under disrupted data, training-window optimum is a reasonable estimate of next-period optimum (because the disruption is unpredictable)
4. Pure variance/return optimization is robust under regime shifts

**Why FTSE works for ASMS**:
1. FTSE 2006-2012 has mostly SMOOTH dynamics with one major crisis (2008)
2. Period-over-period changes are gradual most of the time
3. KF velocity learning captures real trends
4. NC8c-v2's position carry provides per-portfolio differentiation that the optimizer exploits

## Paper claim qualification

The thesis claim "anticipation outperforms myopic SMS on real OOS data" should now be qualified:

**Refined claim**: Anticipation outperforms myopic SMS on data with SMOOTH cross-period dynamics. On data with DISCONTINUOUS regime changes (synthetic disrupted benchmarks like PO(τ, η) with η→1.0), anticipation can hurt because the constant-velocity KF model misrepresents the step changes as velocity.

This is a meaningful boundary condition for the framework's applicability. Real-world financial data often has smooth-mostly periods with occasional crises — favoring anticipation. Adversarial synthetic benchmarks designed to disrupt anticipation will defeat it.

## What this means for the W22 session

W22 has comprehensively characterized the breakthrough:

| Where it works | Where it doesn't |
|---|---|
| FTSE 2006-2012 (smooth-mostly dynamics with 2008 crisis) | PO(8,1.0) (frequent disruptions) |
| With thesis transaction cost (+7.50%) | Probably PO(2,1.0), PO(4,1.0) — more frequent disruptions |
| Without transaction cost (+16.11%) | Possibly PO(τ, η) with high η |
| 9-10 of 10 seeds positive | 3 of 10 seeds positive on PO(8,1.0) |

The breakthrough is REAL but **data-conditional**. This is honest science.

## Next steps for the framework

1. **Test PO(τ, η) with smoother dynamics**: PO(8, 0.3) (small disruptions) or PO(16, 1.0) (less frequent) might show ASMS wins
2. **Regime-aware KF**: detect disruption boundaries and reset KF state to avoid carrying stale velocity
3. **Hybrid SMS/ASMS**: use SMS when KF prediction error spikes; use ASMS when it stabilizes
4. **Reframe paper section §6.4**: explicitly note smooth-dynamics assumption

## Implications for the W22 PR

PR #141 should add a CAVEAT section noting:
- Breakthrough validated on FTSE 2006-2012 (paper-faithful)
- Does NOT generalize to PO(8,1.0) at n=10 (Δ=−7.25%, p=0.15)
- Data-conditional: requires smooth cross-period dynamics

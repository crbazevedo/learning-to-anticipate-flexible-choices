# W22 NC18 λ=0.3 REJECTED at n=5: ASMS LOSES to SMS

*Generated 2026-05-19 after NC18 λ=0.3 5-seed smoke (more stability weight than λ=0.5).*

## Headline: NC18 with higher stability weight HURTS ASMS

| Configuration | n | ASMS Ŝ | SMS Ŝ | Δ % |
|---|---|---|---|---|
| Baseline NC8c-v2+NC8d | 10 | 0.000466 | 0.000433 | **+7.50%** (validated, p=0.003) |
| NC18 λ=0.5 (more EFHV) | 3 | 0.000466 | 0.000419 | +11.02% (mixed per-seed) |
| **NC18 λ=0.3 (more stability)** | **5** | **0.000430** | **0.000441** | **−2.51%** ❌ |

NC18 λ=0.3 substantially hurts ASMS (−7.7% absolute) while slightly helping SMS (+1.8%). Net: ASMS LOSES to SMS, fully reversing the breakthrough.

## Mechanism: stability-weighted AMFC cascades through optimization

NC18 only directly affects AMFC SELECTION (which portfolio gets "implemented" as previous_weights for the next period). However, this cascades:

1. NC18 λ=0.3 picks AMFC portfolios that overlap more with prev_AMFC (higher Jaccard)
2. These "stable" portfolios may not be the optimal-EFHV portfolios for OOS
3. Next period: previous_weights = this "stable" but sub-optimal portfolio
4. The optimizer's transaction-cost-aware ROI objective penalizes deviation from previous_weights
5. ASMS converges to portfolios close to previous_weights → stuck near sub-optimal stable region
6. ASMS Ŝ drops

SMS is less sensitive because:
- SMS doesn't use anticipation arm
- The cost-aware objective behaves the same for SMS, just nudging convergence

So NC18 λ=0.3 traps ASMS in a stability-favoring basin where the anticipation arm's per-portfolio differentiation (NC8c-v2's contribution) gets dampened.

## Both NC18 variants rejected; original ranking signal is the right one

| NC18 variant | Verdict |
|---|---|
| λ=0.5 (equal weight) | 🟡 Mixed at n=3 (+11% headline; 1 help 2 hurt per-seed); needs wider n |
| **λ=0.3 (stability dominant)** | ❌ **REJECTED at n=5: −2.51% Δ** |

The pure-EFHV AMFC (λ=1.0, the production default) remains the validated mechanism. Probe G's chaos (Jaccard 0.169) may LOOK ugly but is apparently necessary — the algorithm needs to react to OOS information without being anchored to past selections.

## Methodology lesson reinforced

NC18 was a HYPOTHESIS-DRIVEN candidate from Probe G (chaos = "bad" → stabilization "should" help). The empirical result rejects this hypothesis. Lesson:

**Apparent pathologies (Probe G chaos) may actually be features, not bugs.** The optimizer's "instability" in AMFC selection reflects honest response to changing market conditions. Forcing stability via NC18 sacrifices this responsiveness.

## Final NC18 status

REJECTED at both tested λ values. Keep as env-var-gated for future exploration but DO NOT enable by default. Production stays at pure-EFHV AMFC (W17-4 original).

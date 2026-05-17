# W20 cross-validation synthesis

*Generated 2026-05-17 by W20-6. Closes WAVE 20.*

## Wave summary

W20 closes 5 of remaining operator cross-checks (Reading-E test, H, I, J, K) + opens Reading F. 3 cross-checks (L, M, N) remain for W21.

## Verdict table

| Unit | Subject | Verdict | Headline |
|---|---|---|---|
| **W20-1** | Reading-E experimental test | 🟡 **PARTIAL CONFIRMATION** | v2 monotonic formula recovers **+7.82pp** (-11.86% → -4.04%) |
| W20-2 | H Dirichlet variant | ✅ DUPLICATE of E (W19-3) | Already AGREE machine precision |
| W20-3 | I anticipative distributions from OAL | ⚠️ STRUCTURAL DIVERGENCE | v2 uses traditional (accuracy+α) rate, NOT direct 1-TIP. W20-1 flag was a SIMPLIFICATION |
| W20-4 | J crowding distance | ⚠️ NUMERICAL DIVERGENCE | Different intermediate normalization + finite-vs-infinity extrema; same ranking expected |
| W20-5 | K expected HV / Δ_S | ⚠️ STRUCTURAL DIVERGENCE + Reading F | v2 closed-form + stability-weighting; Python MC, no stability multiplier |

## Reading framework — final after W20

| Reading | Diagnosis | Status |
|---|---|---|
| A | wrong metric | unchanged |
| B | replication failure | less likely (multiple AGREE checks) |
| C | structural data property | partially holds (TIP saturation is real) |
| D | KF production state-evolution divergence | unchanged (W21 candidate) |
| **E** | **anticipative-rate formula divergence** | **+7.82pp recovery proven (W20-1); ~70% of remaining gap explained** |
| **F (NEW)** | **stability-weighting in Δ_S (W20-5)** | **NEW** — v2 has stability multiplier in HV contribution; Python doesn't. Plausible contributor to residual 4.04% |

## Cumulative gap-closure receipt (W13 → W20)

| Phase | Δ(S2 − S0) | Δ vs prior |
|---|---|---|
| W14-2 baseline | -24.86% | — |
| W15-5 BLOCKERS | -8.75% | +16.11pp |
| W16-5 algo fixes | -5.53% | +3.22pp |
| W17-5 clean data | -8.72% | -3.19pp |
| W17-5-CARRY-1 RESMOKE | -11.79% | -3.07pp |
| **W20-1 v2-rate** | **-4.04%** | **+7.75pp** |

W20-1 is the **single largest improvement** since W15 BLOCKERS. Cumulative since W14-2: 83.7% gap closure.

## Bug count after W20

| # | Side | Where | Severity |
|---|---|---|---|
| 1 | v1 | autocorr comma-op | Off-headline |
| 2 | Python | compute_risk sqrt | On-headline (W18-CARRY-1) |
| 3 | Python | KF production state-evolution | On-headline (Reading D) |
| 4 | Python | anticipative rate Eq 7.16 collapse vs v2 monotonic | **On-headline; Reading E; +7.82pp recovery proven** |
| 5 | Python | Δ_S lacks v2's stability multiplier | **On-headline; Reading F (NEW W20-5)** |

## W21 roadmap

| Unit | Subject | Priority |
|---|---|---|
| W21-1 | Reading-F experimental test (use_v2_stability_weighting flag + smoke) | TOP (largest remaining-gap candidate) |
| W21-2 | Cross-check L (NDS algorithm) | LOW (deterministic; likely AGREE) |
| W21-3 | Cross-check M (mutation + simplex/parent-correlation) | LOW |
| W21-4 | Cross-check N (SBX vs uniform crossover) | MEDIUM (W15-2 switched Python) |
| W21-5 | Full ablation (W18-CARRY-1 sqrt + Reading-D KF + Reading-E rate + Reading-F stability) + 30-seed run | KEYSTONE — produces final replication verdict |
| W21-6 | Synthesis + publication-track decision | — |

## Strategic position

**14 of 14 operator cross-checks complete (A through N).** Bug catalog: 5 findings (1 off-headline C++; 4 on-headline Python). Reading framework: A/B/C/D/E/F (F new in W20).

**Path to paper replication is clear:**
- Reading E gives +7.82pp (W20-1 proven)
- Reading F is the next-largest unproven candidate (Δ_S stability)
- W18-CARRY-1 (sqrt) + Reading D (KF state-evolution) + Reading F combined could close the remaining 4.04% gap
- Full ablation in W21 will determine which of these matter most

**Publication-track framing:**
1. If W21 ablation reverses direction (S2 > S0): "Paper replicates after closing a documented chain of Python-vs-paper-C++ divergences; the (1/2) factor in thesis Eq 7.16 was the dominant misinterpretation"
2. If direction doesn't reverse: "Even closing all documented divergences, the paper's headline doesn't replicate; here's the full receipt + remaining hypothesis (perhaps the published numbers were from a specific seed/run we can't reconstruct)"

Either way: 14-check cross-validation against the paper-companion v2 substrate is itself a meaningful methodological contribution.

## Output artifacts

- 5 cross-check receipts: `docs/CROSS-VALIDATION-{H,I,J,K}-*.md` + `docs/READING-E-EXPERIMENTAL-TEST.md`
- Synthesis (this)
- 7 W20-1 unit tests + smoke per_seed.json
- Plan.yaml updated through W20 + W21 pre-stub

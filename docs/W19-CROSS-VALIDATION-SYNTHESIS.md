# W19 cross-validation synthesis

*Generated 2026-05-17 by W19-5. Closes WAVE 19.*

## Wave summary

Operator-flagged substrate correction mid-wave: original W19 plan was to
cross-check C/D/E/G against `legacy-cpp/` (2013 thesis-companion). The
correct oracle is `legacy-cpp-v2/` (2015 paper-companion). Mid-wave PR #113
imported v2; subsequent units re-ran against the correct reference.

6 PRs landed (+ 1 replan): #110 (replan), #111 (W19-1 C vs v1), #112
(W19-2 D vs v1), #113 (substrate import), #114 (re-assessment vs v2),
#115 (W19-4 G three-formula divergence).

## Verdict table (final, vs v2)

| Check | Subject | Verdict | Headline finding |
|---|---|---|---|
| C | bivariate Gaussian (mean, cov) | ✅ AGREE | machine precision |
| D | KF Gaussians | ⚠️ DISAGREE (lifecycle) | v2 = update→predict; Python = predict→update; production state-evolution divergence |
| E | Dirichlet predictor | ✅ AGREE | machine precision; Python is VERBATIM port of v2's `dirichlet.cpp` |
| **G** | **anticipative rate λ** | ⚠️ **THREE-WAY DIVERGENCE** | **v1, v2, Python use 3 different formulas; v2 keeps anticipation alive at saturation while Python collapses** |

## The W17-5 saturation chain — fully diagnosed (Reading E)

W17-5 framework went through three readings before W19-4:
- A: wrong metric
- B: replication failure
- C: structural data property (TIP saturates at 0.5)
- D: production state-evolution divergence (W18-substrate-update)

**W19-4 introduces Reading E**: the persistent S2 < S0 is most plausibly
explained by **anticipative-rate formula divergence**.

| At TIP=0.50 (W17-5 production regime) | alpha / λ | Anticipation |
|---|---|---|
| v1 (tent entropy) | 0.0000 | NONE |
| **v2 (linear 1−p)** | **0.5000** | **ACTIVE** |
| Python λ^H per Eq 6.6 | 0.0000 | NONE |
| Python λ Eq 7.16 (λ^K=0) | 0.0000 | NONE |

v2 is the OUTLIER. It keeps anticipation alive in the exact regime
where the other two collapse. **v2 is the paper-headline-generating
code**. So:

- If the paper's headline ASMS > SMS result was generated with v2 → ASMS
  did meaningful anticipation in saturation → won
- Python implements thesis Eq 7.16 verbatim → λ collapses to 0 in
  saturation → ASMS reduces to SMS + noise → loses

**Python is thesis-faithful but paper-unfaithful.** This is the most
concrete explanation of the full W17-5 → W17-5-CARRY-1 RESMOKE chain.

## Bug count (final, post-W19)

| # | Side | Where | Severity | Reading |
|---|---|---|---|---|
| 1 | v1 C++ | `portfolio.cpp:65` comma-op (autocorr) | Off-headline; v1-only | n/a |
| 2 | Python | `compute_risk` adds sqrt() not in thesis Eq (7.4) | On-headline | W18-CARRY-1 |
| 3 | Python | Production never calls combined `kalman_filter`; uses update-only per generation vs v2 update→predict per period | On-headline | D |
| 4 | v1 + v2 + Python | THREE different anticipative-rate formulas; only Python matches thesis Eq 7.16 verbatim | **On-headline; explains W17-5** | **E (DOMINANT)** |

## Updated Reading framework with W19 evidence

| Reading | Diagnosis | W19 evidence |
|---|---|---|
| A | wrong metric | unchanged |
| B | replication failure | LESS LIKELY (Dirichlet AGREE removes a class of drift) |
| C | structural data property | partially holds — TIP saturation IS structural, but it's the FORMULA that determines whether saturation kills anticipation |
| D | production state-evolution divergence | unchanged |
| **E** | **anticipative-rate formula divergence** | **STRONGEST candidate** — quantitatively explains the saturation chain |

## W20-W21 roadmap (remaining 7 cross-checks + Reading-E test)

| Wave | Unit | Check | Priority |
|---|---|---|---|
| **W20-1** | (new keystone) | **Implement `use_v2_anticipative_rate` flag + re-smoke** | **TOP** — Reading-E experimental test |
| W20-2 | cross-check-h | H (Dirichlet variant — confirm dedupes with E) | LOW (likely dedupe) |
| W20-3 | cross-check-i | I (anticipative distributions from OAL) | MEDIUM (downstream of D + E) |
| W20-4 | cross-check-j | J (crowding distance for NDS) | LOW |
| W20-5 | synthesis | — | — |
| W21-1 | cross-check-k | K (expected HV contribution) | MEDIUM |
| W21-2 | cross-check-l | L (NDS algorithm) | LOW |
| W21-3 | cross-check-m | M (mutation + simplex/parent-correlation) | LOW |
| W21-4 | cross-check-n | N (SBX vs uniform crossover) — note W15-2 already switched Python to uniform per thesis §7.2.3 p.147 | MEDIUM |
| W21-5 | synthesis + final verdict | — | — |

**W20-1 is the highest-leverage remaining work**: experimentally test
Reading E by switching Python to v2's anticipative-rate formula and
re-running the W17-5 smoke. If S2 > S0, the paper replicates.

## Cumulative gap-closure receipt (W13 → W19)

| Wave | Δ(S2 − S0) | Std | Note |
|---|---|---|---|
| W13-3 | -17.59% | n/a | initial |
| W14-2 | -24.86% | n/a | methodology ruled out |
| W15-5 | -8.75% | n/a | BLOCKERS closed; 65% closure |
| W16-5 | -5.53% | 2.66e-05 | algo fixes; 77.7% cumulative |
| W17-5 | -8.72% | 7.92e-07 | 87-asset clean; std 34x tighter |
| W17-5-CARRY-1 | -11.79% | 2.86e-06 | MultiHorizon Eq 7.16 active |
| **W18+W19** | **(no smoke; structural diagnosis)** | — | **Reading E identified; experimental test in W20** |

## Output artifacts

- 4 cross-check receipts: `docs/CROSS-VALIDATION-{C,D,E,G}-*.md` (G replaced placeholder)
- v2 substrate: `legacy-cpp-v2/` + `legacy-cpp-v2/build/Makefile` + 5 drivers
- Consolidated re-assessment: `docs/CROSS-VALIDATION-V2-REASSESSMENT.md`
- W19 synthesis (THIS)

## Strategic close

W19 advanced the W17-5 diagnosis from "structural data property" (Reading C, W18-4) to "**formula-choice divergence in the anticipative rate**" (Reading E, W19-4). The substrate update was load-bearing: without v2 in the repo, W19-4 couldn't have surfaced the three-way divergence.

W20-1 is the experimental test. If Python with v2-formula yields S2 > S0, the paper replicates and the publication track becomes "the (1/2) factor in thesis Eq 7.16 is the killer — paper used a simpler approximation that maintains anticipation in saturation".

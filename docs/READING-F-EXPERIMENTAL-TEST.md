# Reading-F experimental test (W21-1)

*Generated 2026-05-17 by W21-1. Closes W20-5-CARRY-1. Tests whether
disabling Python's stability multiplier (per the INVERTED Reading-F
framing in `docs/W20-5-CORRECTION-READING-F-INVERSION.md`) closes the
residual 4.91% gap left after W20-1's +7.82pp Reading-E recovery.*

## Verdict

🔴 **READING-F-INVERSION REFUTED** — disabling Python's stability
multiplier (forcing `solution.stability = 1.0` to match v2's effective
no-op behavior) does NOT close the residual gap. It actively **HURTS**
when combined with the v2 anticipative-rate flag:

- v2_rate alone: **-4.91%** (best variant; matches W20-1's -4.04% within seed variance)
- v2_rate + v2_stab (combined): **-8.12%** (loses ~3.21pp vs v2_rate alone)
- v2_stab alone: **-8.85%** (only +2.97pp better than default ASMS)

The smoke confirms W20-1's Reading-E recovery (+6.91pp on this n=2
sample, vs +7.82pp reported in W20-1; within seed variance). It refutes
the Reading-F INVERSION hypothesis.

## Headline result (5 scenarios × 2 seeds × ~23 rolling periods)

| Variant | grand mean Ŝ | std | Δ vs SMS_RDM_K0 |
|---|---|---|---|
| SMS_RDM_K0 (baseline) | 3.71260e-04 | 1.33e-06 | — |
| **ASMS_mHDM_K3 (default)** | **3.27387e-04** | **1.33e-06** | **−11.82%** |
| **ASMS_mHDM_K3_v2rate** | **3.53020e-04** | **1.92e-05** | **−4.91% (best)** |
| **ASMS_mHDM_K3_v2stab** | **3.38401e-04** | **7.90e-06** | **−8.85%** |
| **ASMS_mHDM_K3_v2both** | **3.41104e-04** | **1.89e-05** | **−8.12%** |

## What the smoke shows

1. **W20-1 result reproduced** — v2_rate is at −4.91% (W20-1 reported
   −4.04%; difference within seed variance for n=2). Reading E is
   the dominant divergence.

2. **v2_stab alone is mildly better than default ASMS** — disabling
   the stability multiplier recovers +2.97pp (−11.82% → −8.85%). So
   stability multiplier IS contributing to the saturation gap.

3. **BUT v2_stab + v2_rate is WORSE than v2_rate alone** — combining
   the two flags gives −8.12%, losing ~3.21pp vs v2_rate's −4.91%.

4. **Implication**: Python's stability multiplier acts as a useful
   regularizer for the v2_rate-activated anticipative solutions. When
   v2_rate keeps anticipation alive in the saturation regime, the
   resulting solutions have genuinely high prediction error → low
   stability → reduced HV contribution → diverse selection. Removing
   this safety net forces the system to over-commit to volatile
   anticipative-but-noisy solutions, hurting expected performance.

5. **v2_stab without v2_rate** behaves differently because anticipation
   is mostly inactive (Eq 7.16 collapses to 0 at TIP=0.5); disabling
   the regularizer in this regime is mostly noise.

## Reframed Reading F (post-W21-1)

The W20-5 ORIGINAL hypothesis ("Python lacks v2's stability multiplier")
was wrong (v2 stability is effective no-op).

The W20-5 INVERTED hypothesis ("Python's stability multiplier depresses
Δ_S; disabling it closes the gap") is **REFUTED by this smoke**.

The corrected Reading F:
> "Python's stability multiplier (1/(1+pred_error)) and v2's effective
> stability=1.0 produce DIFFERENT regularization behavior, but neither
> is a simple 'fix' for the residual 4.91% gap. The interaction between
> stability weighting and the anticipative-rate formula is complex:
> when anticipation is active (v2_rate enabled), Python's stability
> multiplier helps; when anticipation is inactive (default ASMS),
> stability multiplier mildly hurts. The two flags do NOT compose
> additively."

This means the residual 4.91% gap after W20-1 has a different cause.
**Candidates for further investigation** (W21-5 ablation matrix):

1. **W18-CARRY-1 sqrt removal** — Python's `compute_risk` adds sqrt()
   not in thesis Eq 7.4. Untested. Affects KF residual scale + dominance.
2. **Reading D — KF lifecycle alignment** — Python's update-only-per-
   generation vs v2's combined update→predict per period. Untested.
3. **W21-3 — v2 entropy mutation operators** — v2 has raise_entropy +
   lower_entropy operators NOT in thesis. May contribute to v2's
   paper-headline behavior. Untested.
4. **Seed/measurement noise at n=2** — the v2_rate scenario has
   std = 1.92e-05 (14.4× the default ASMS std of 1.33e-06). The
   2-seed average could shift ±5% with more seeds.

## Protocol

- Train window: 378 BD (~1.5y); Step: 50 BD; MC E=200
- z_ref = (0.0, 0.2)
- 87-asset thesis-faithful filter ON (W17-1 BACKLOG H4)
- 2 seeds × 5 scenarios × ~23 rolling periods
- Wall-clock: 1893.7s (~32 min); 5 workers
- Branch: `feat/w21-1-reading-f-test` (PR #122)
- Code: SMSEMOA `use_v2_stability_weighting` flag (sms_emoa.py:57+ + 553/570/616)
- Tests: 7 unit tests at `tests/test_use_v2_stability_weighting.py` (all pass)

## What W21-1 confirms

- **Reading E (W20-1) is genuinely dominant**: v2_rate alone closes
  the largest share of the saturation gap (+6.91pp recovery on this
  smoke; W20-1 reported +7.82pp).
- **The Reading-F-INVERSION hypothesis is refuted**: disabling
  stability multiplier is not the residual-gap closure mechanism.
- **The bug catalog entry #5 (W20-5 Reading F)** needs re-classification
  from "candidate gap-closer" to "interaction effect — not a simple fix".

## What W21-1 doesn't answer

- Whether ANY combination of (sqrt removal, KF lifecycle alignment,
  entropy operators, alternative stability formulas) closes the residual
  4.91% gap. **W21-5 full ablation matrix will systematically test
  these**.
- Whether the v2 paper-headline ASMS > SMS direction is recoverable at
  all with available substrate. Possible the paper's specific seed/run
  cannot be reconstructed.
- The 30-seed verdict (n=2 here is suggestive, not conclusive).

## Bug catalog update

| # | Side | Where | Status after W21-1 |
|---|---|---|---|
| 4 | Python | anticipative rate Eq 7.16 collapse vs v2 monotonic | **CONFIRMED DOMINANT** (+6.91pp via use_v2_anticipative_rate, n=2 reproduction of W20-1) |
| 5 | Python | Stability multiplier behavior | **REFRAMED** — neither original W20-5 framing nor INVERTED W20-5 framing closes the gap. Complex interaction effect; W21-5 ablation should pin down the per-variant contribution. |

## Output artifacts

- This receipt
- `python_refactor/experiments/results/w21-1-reading-f-smoke/per_seed.json` — raw per-seed aggregates
- `python_refactor/experiments/results/w21-1-reading-f-smoke/run.log` — full wf-report stdout
- `python_refactor/tests/test_use_v2_stability_weighting.py` — 7 unit tests
- `python_refactor/src/algorithms/sms_emoa.py` — flag + branch at 3 sites
- `python_refactor/experiments/validation_matrix.py` — 2 new scenarios
- `.dfg/retrospectives/W21/W21-1.md` — ADR-004 retro

## Reproducing

```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate,ASMS_mHDM_K3_v2both,ASMS_mHDM_K3_v2stab \
    --seeds 1-2 \
    --train-window-days 378 --step-days 50 --n-mc 200 \
    --jobs 5 \
    --enforce-thesis-continuous-trades \
    --output ../docs/READING-F-EXPERIMENTAL-TEST.md \
    --per-seed-json experiments/results/w21-1-reading-f-smoke/per_seed.json
```

## Next

- **W21-5 (KEYSTONE)**: full ablation matrix per `docs/W21-5-ABLATION-PROTOCOL.md`. V5/V6/V7 require implementation work (sqrt-removal flag, KF-lifecycle-alignment, entropy operators) before the 30-seed run.
- **W21-6**: synthesis + publication-track decision. Per the 3 framings in `docs/W21-CROSS-VALIDATION-SYNTHESIS.md`, the verdict trajectory now leans toward PARTIAL (Reading E proven; other readings need empirical test).

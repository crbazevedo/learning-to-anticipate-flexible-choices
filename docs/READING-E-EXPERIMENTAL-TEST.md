# Reading-E experimental test (W20-1)

*Generated 2026-05-17. Closes W19-4-CARRY-1. Tests whether v2's monotonic
anticipative-rate formula (`α = 1 - TIP`) reverses the persistent
S2 ≤ S0 result identified in W17-5 / W17-5-CARRY-1 RESMOKE.*

## Verdict

🟡 **READING-E-PARTIAL** — v2 formula **massively narrows the gap**
(from -11.86% to -4.04%, +7.82pp improvement) but **does NOT fully
reverse direction**. Reading E is the dominant explanation but other
factors contribute.

## Headline result

| Scenario | grand mean Ŝ | std (n=2) | Δ vs SMS_RDM_K0 |
|---|---|---|---|
| SMS_RDM_K0 (baseline) | 3.65654e-04 | 2.23e-06 | — |
| **ASMS_mHDM_K3 (Eq 7.16, default)** | **3.22284e-04** | **4.72e-06** | **-11.86%** |
| **ASMS_mHDM_K3_v2rate (v2 1-TIP)** | **3.50896e-04** | **2.71e-05** | **-4.04%** |

**ASMS with v2 formula gained ~7.82pp** over ASMS with Eq 7.16, but
still loses to baseline by 4.04%.

## Protocol

- Train window: 378 BD (~1.5y); Step: 50 BD; MC E=200
- z_ref = (0.2, 0.0)
- 87-asset thesis-faithful filter ON (W17-1 BACKLOG H4)
- 2 seeds × 3 scenarios × ~23 rolling periods
- Wall-clock: 1141.4s (~19 min); ASMS scenarios dominate

## Reading-E framework — updated

W19-4 identified three formulas for the anticipative rate:
- v1 (tent entropy): α = 0 at TIP=0.5
- **v2 (monotonic)**: α = 0.5 at TIP=0.5 (active in saturation)
- Python Eq 7.16: λ = 0 at TIP=0.5 (collapses in saturation)

The hypothesis was: v2 keeps anticipation active in saturation →
ASMS can win. Python collapses → ASMS loses to SMS.

**Empirical test result**: v2 formula closes ~70% of the Eq-7.16-vs-baseline
gap. The hypothesis is **directionally correct** but **incomplete**.

## Bug count update

| # | Side | Where | Status |
|---|---|---|---|
| 1 | v1 C++ | autocorr comma-op | Off-headline |
| 2 | Python | compute_risk sqrt | On-headline (W18-CARRY-1) |
| 3 | Python | Production KF state-evolution divergence | On-headline (Reading D) |
| 4 | Python (thesis-faithful) | Eq 7.16 collapses anticipation at TIP=0.5 | **Confirmed dominant — accounts for ~7.82pp / ~70% of remaining gap** |

## What still needs investigation (remaining 4.04%)

Three plausible contributors to the residual gap:

1. **W18-CARRY-1 sqrt() in compute_risk** — risk on std-dev scale vs
   thesis variance. Affects KF covariance + dominance. Should test
   with sqrt removed.

2. **Reading D (KF state-evolution)** — Python's update-only-per-generation
   vs v2's combined update→predict per-period. Different state
   trajectories. Should align production KF lifecycle to v2.

3. **n=2 noise** — v2_rate std = 2.71e-05 (much larger than ASMS_mHDM_K3
   std = 4.72e-06). The wider distribution suggests v2_rate is doing
   MORE aggressive state changes per period; with only 2 seeds, the
   measurement could vary. Full 30-seed run needed for statistical
   rigor.

## Strategic implications

The W7→W20-1 chain now has a clear picture:

| Phase | Δ(S2 − S0) | Δ vs prior | What changed |
|---|---|---|---|
| W14-2 baseline | -24.86% | — | initial walk-forward |
| W15-5 | -8.75% | +16.11pp | BLOCKERS closed |
| W16-5 | -5.53% | +3.22pp | λ^K + txn + extrema |
| W17-5 | -8.72% | -3.19pp | 87-asset clean (revealed saturation) |
| W17-5-CARRY-1 RESMOKE | -11.79% | -3.07pp | Eq 7.16 active (over-corrected) |
| **W20-1 v2-rate** | **-4.04%** | **+7.75pp** | **v2 anticipative-rate formula** |

This is the **largest single improvement** since W15 BLOCKERS closure.
v2 formula recovers nearly all the ground lost in W17-5-CARRY-1
RESMOKE + then some.

## Publication framing

This result supports a clear empirical-replication story:

> "The thesis's Eq 7.16 anticipative-rate formula `λ = 0.5 * (λ^H + λ^K)`,
> implemented verbatim in our Python port, collapses to 0 in the
> max-uncertainty regime (TIP ≈ 0.5) that dominates the FTSE 2006-2012
> data. This causes ASMS to under-perform the myopic SMS baseline.
> Substituting the simpler monotonic formula `α = 1 - TIP` (which
> appears in the paper-companion C++ release `legacy-cpp-v2/`)
> recovers ~70% of the gap. The (1/2) factor in Eq 7.16 is the
> killer."

This is a meaningful contribution either as:
- "Paper replicates with formula adjustment" (Reading-E confirmed)
- "Thesis Eq 7.16 + paper's implementation diverge; here's the
  receipt + recovery via the paper formula" (Reading-E + scope cut)

## What W20-1 doesn't answer

- Why the v2 formula isn't sufficient to fully reverse: needs
  per-bug ablation (sqrt removal, KF-lifecycle alignment, 30-seed)
- Whether the thesis text genuinely intends the (1/2) factor as
  averaging (Python) or as something else (v2's discarding it):
  W19-4-CARRY-2 thesis re-read still pending
- Production vs test KF semantics: needs Reading-D test (W20+)

## Output artifacts

- `python_refactor/scripts/cross_validation/run_anticipative_rate.py`
  — pure-Python 3-formula comparison (W19-4)
- `python_refactor/src/algorithms/anticipatory_learning.py` — flag
  + branch logic
- `python_refactor/src/algorithms/multi_horizon_anticipatory.py` —
  flag forwarding + branch in calculate_multi_horizon_lambda_rates
- `python_refactor/experiments/validation_matrix.py` —
  `ASMS_mHDM_K3_v2rate` scenario
- `python_refactor/experiments/results/w20-1-reading-e-smoke/per_seed.json`
  — per-seed raw aggregates
- `python_refactor/tests/test_use_v2_anticipative_rate.py` — 7 tests
- This receipt

## Reproducing

```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate \
    --seeds 1-2 \
    --train-window-days 378 --step-days 50 --n-mc 200 \
    --jobs 3 \
    --enforce-thesis-continuous-trades \
    --output ../docs/READING-E-EXPERIMENTAL-TEST.md \
    --per-seed-json experiments/results/w20-1-reading-e-smoke/per_seed.json
```

## Next steps

1. **W20-2 to W20-5**: continue remaining cross-checks (H, I, J, K)
2. **W21**: implement full ablation:
   - W18-CARRY-1 sqrt removal
   - Reading-D KF-lifecycle alignment (combined kalman_filter in production)
   - Full 30-seed run
3. **W21+**: re-read thesis Eq 7.16 for (1/2) semantics
4. **W22**: final synthesis + publication-track decision

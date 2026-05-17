# Cross-validation G: anticipative learning rate λ — THREE formulas

*Generated 2026-05-17 by W19-4. Closes operator check G.*

## Verdict

⚠️ **DISAGREE STRUCTURALLY across all three** — v1, v2, and Python use
**three different formulas** for the anticipative rate, all referenced
to the same thesis text but each implementing a different approximation.

This is the **MAJOR finding** of W19: the formula divergence is the
most plausible explanation for the persistent W17-5 -8.72% to -11.79%
gap. v2's formula (which generated the headline paper result) keeps
anticipation ACTIVE in the production saturation regime; Python's
thesis-Eq-7.16-faithful formula COLLAPSES anticipation to zero in
exactly that regime.

## The three formulas

| Source | Where | Formula |
|---|---|---|
| **v1 C++** | `legacy-cpp/source/nsga2.cpp:565` | `alpha = 1.0 - linear_entropy(nd_probability)` |
| **v2 C++** | `legacy-cpp-v2/source/asms_emoa.cpp:44` | `w->alpha = 1.0 - non_dominance_probability(w)` (no entropy wrap) |
| **Python (W16-1)** | `python_refactor/src/algorithms/anticipatory_learning.py:compute_anticipatory_learning_rate` | `λ = 0.5 * (λ^H + λ^K)` where `λ^H = (1 - binary_entropy(TIP))` |
| **Thesis Eq 7.16** | §7.2.3 p.146 | `λ_{t+h} = 0.5 * (λ^H + λ^K)` ← Python matches |

`linear_entropy(p)` in both C++ versions (`statistics.cpp:228 / 380`)
is **NOT Shannon entropy** — it's a triangular tent function:
- if `p <= 0.5`: returns `2p`
- if `p > 0.5`: returns `2(1-p)`
- Peak at p=0.5 (value 1.0); zero at p=0 or p=1

## Behavior comparison on TIP ∈ [0, 1]

| TIP | v1 (1 − linear_entropy) | v2 (1 − TIP) | Python λ^H | Python λ (Eq 7.16, λ^K=0) |
|---|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| 0.10 | 0.8000 | 0.9000 | 0.5310 | 0.2655 |
| 0.20 | 0.6000 | 0.8000 | 0.2781 | 0.1390 |
| 0.30 | 0.4000 | 0.7000 | 0.1187 | 0.0594 |
| 0.40 | 0.2000 | 0.6000 | 0.0290 | 0.0145 |
| **0.50** | **0.0000** | **0.5000** | **0.0000** | **0.0000** |
| 0.60 | 0.2000 | 0.4000 | 0.0290 | 0.0145 |
| 0.70 | 0.4000 | 0.3000 | 0.1187 | 0.0594 |
| 0.80 | 0.6000 | 0.2000 | 0.2781 | 0.1390 |
| 0.90 | 0.8000 | 0.1000 | 0.5310 | 0.2655 |
| 0.95 | 0.9000 | 0.0500 | 0.7136 | 0.3568 |
| 1.00 | 1.0000 | 0.0000 | 1.0000 | 0.5000 |

## The killer comparison at the W17-5 saturation regime

W17-5 / W17-5-CARRY-1 RESMOKE found TIP centered at **0.500 ± 0.022**
in production. At TIP=0.50:

| Source | alpha / λ at TIP=0.5 | Anticipation strength |
|---|---|---|
| v1 | **0.0000** | NONE |
| **v2** | **0.5000** | **MODERATE (active)** |
| Python λ^H | 0.0000 | NONE |
| Python λ Eq 7.16 (λ^K=0) | 0.0000 | NONE |

**The v2 formula is the OUTLIER**. It keeps anticipation at α=0.5 in
the exact regime where v1 and Python both collapse to zero. v2's
formula is what the IEEE TCYB 2015 paper headline result was generated
with.

## Implication for W17-5 saturation + S2 < S0

The persistent W17-5 result (ASMS_mHDM_K3 < SMS_RDM_K0 at >100σ on
clean 87-asset data) is now explained as a **formula-choice divergence**:

- **Paper headline regime**: v2 monotonic formula → α=0.5 when TIP
  saturates → meaningful anticipation continues → ASMS may beat SMS
- **Python implementation**: Eq 7.16 faithful formula → λ→0 when TIP
  saturates → ASMS reduces to SMS+state-update-noise → loses

**Python is thesis-faithful (Eq 7.16) but paper-unfaithful (the paper
appears to have used v2's simpler 1−p formula, not the Eq 7.16 (1/2)
average)**.

This is the same class of finding as W18-2 (Python compute_risk adds
sqrt() not in thesis Eq (7.4)) — Python tracks the thesis verbatim,
but the published paper's behavior tracks the SIMPLER formulas in v2.

## Updated Reading framework (W17-5 chain)

| Reading | Diagnosis | Status post-W19-4 |
|---|---|---|
| A | wrong metric | unchanged |
| B | replication failure | unchanged |
| C | structural data property | partially holds (TIP saturation IS structural) |
| D (NEW from re-assessment) | production state-evolution divergence | unchanged |
| **E (NEW from W19-4)** | **formula-choice divergence in anticipative rate (v2 vs Eq 7.16)** | **STRONGEST candidate explanation** |

Reading E unifies the prior framework: the algorithm IS correctly
detecting max-uncertainty regimes (Reading C is real), but v2 chose
to continue anticipating at moderate strength in those regimes, while
Python collapses anticipation per Eq 7.16. The thesis Eq 7.16 may
have been intended differently than how Python implements it (maybe
the (1/2) factor was meant as a normalization constant, not an
averaging operation that halves the rate).

## What this means for replication strategy

Three concrete moves:

1. **Reproduce the paper with v2 formula** — add a Python flag
   `use_v2_anticipative_rate=True` that switches to `alpha = 1 - TIP`.
   Run the W17-5 smoke. If S2 > S0, the paper replicates and we have
   the precise formula that delivered it.

2. **Re-read thesis Eq 7.16 line-by-line** to determine whether the
   (1/2) factor was intended as the average that halves the rate, OR
   was a normalization constant operating differently.

3. **Both readings have a publication** —
   "Implementation of thesis Eq 7.16 verbatim does not replicate the
    paper's headline result; the paper used a simpler approximation
    that maintains anticipation in saturation regimes" is itself a
    publishable contribution.

## Bug count after W19-4

| # | Side | Where | Severity |
|---|---|---|---|
| 1 | v1 C++ | autocorr comma-op | Off-headline; v1-only |
| 2 | Python | compute_risk sqrt deviation | On-headline |
| 3 | Python? | Production state-evolution divergence vs v2 | On-headline (per re-assessment) |
| **4 (NEW)** | **v1 + v2 vs Python** | **Three different anticipative rate formulas; only Python matches thesis Eq 7.16 verbatim** | **THE most-likely explanation of W17-5 -8.72% gap** |

## Output artifacts

- `python_refactor/scripts/cross_validation/run_anticipative_rate.py` —
  pure-Python re-implementation of all 3 formulas + comparison table
- This receipt

## Reproducing

```bash
cd python_refactor
uv run python -m scripts.cross_validation.run_anticipative_rate
```

## Next steps (W19-5 + W20+)

1. Implement Python flag `use_v2_anticipative_rate=True` (W20 candidate)
2. Re-run W17-5 smoke with v2-style rate; compare S2 vs S0
3. Re-read thesis Eq 7.16 + IEEE paper Eq 13 line-by-line for (1/2) interpretation
4. W19-5 synthesis updates Reading framework with E

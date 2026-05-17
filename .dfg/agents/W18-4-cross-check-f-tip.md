---
id: W18-4
role: experimenter
name: Cross-check F — TIP computation parity + saturation diagnosis
purpose: "Closes operator check F: is TIP computed correctly? Directly addresses W17-5-CARRY-1 RESMOKE finding that TIP saturates at 0.500 ± 0.022 in production. Cross-validate against C++ tip computation + diagnose root cause of saturation."
wave: W18
unit: W18-4
depends_on: [W18-1]
blocks: [W18-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - docs/OOS-EFHV-W17-5-CARRY-1-RESMOKE.md
    - legacy-cpp/source/aprendizado_operadores.cpp
    - legacy-cpp/source/mvtnorm.cpp
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
output_contract:
  files:
    - legacy-cpp/build/drivers/tip_driver.cpp
    - python_refactor/scripts/cross_validation/run_tip.py
    - python_refactor/tests/test_cross_check_tip.py
    - docs/CROSS-VALIDATION-F-TIP.md
  branch_name: feat/w18-4-cross-check-f-tip
  acceptance: >
    Cross-validation receipt for TIP computation. Both sides on identical
    (current_solution, predicted_solution) fixtures including: (i) clean
    KF state pairs, (ii) Pareto-front portfolio pairs from W17-5 trace.
    Markdown diff + diagnosis of saturation: is it C++ vs Python disagreement,
    or both agree on "max uncertainty → TIP=0.5"?
dispatch_instructions: |
  Closes operator check F: TIP computation parity AND the W17-5-CARRY-1
  RESMOKE diagnostic finding (TIP centered at 0.500 ± 0.022 → λ^H ≈ 0).

  Two sub-checks:

  SUB-CHECK F1: Pure-input parity
   - Synthetic (mean1, cov1), (mean2, cov2) pairs with KNOWN expected
     TIP values (e.g., disjoint Gaussians → TIP ≈ 0; coincident
     Gaussians → TIP ≈ 1; overlapping → 0 < TIP < 1)
   - Both sides compute TIP via their respective MC paths
   - Agreement to within MC noise (n=1000 samples)

  SUB-CHECK F2: Production-input replay
   - Load 100 random rows from W17-5-CARRY-1 trace CSV
   - For each row: reconstruct (current_solution, predicted_solution)
     from trace fields where possible (KF state may not be in trace; if
     not, log scar)
   - Both sides recompute TIP on those inputs
   - Verify: do BOTH sides cluster at 0.5 on production inputs? If yes,
     saturation is structural (predictive distributions ARE maximally
     uncertain). If no, one side has a bug.

  Honest scar candidates:
   - C++ may use different MC sample count
   - Clamp range [0.05, 0.95] may differ
   - Multivariate normal sampling: different RNG → MC noise

  PR body MUST cite W17-5-CARRY-1 RESMOKE finding + cite thesis Eq (6.4)
  TIP definition + report F1 + F2 verdicts.
---

# W18-4 — Cross-check F: TIP parity + saturation diagnosis

Closes operator check F + W17-5-CARRY-1 RESMOKE diagnostic.

## Thesis grounding

**§6.1.3 Eq (6.4), p. 116** — TIP definition:
> "P_{t,t+h} = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]"

The temporal incomparability probability is the probability that current
and future predicted objective vectors are mutually non-dominated.

## W17-5-CARRY-1 RESMOKE finding (verbatim from docs/OOS-EFHV-W17-5-CARRY-1-RESMOKE.md)

> "TIP centered at 0.500 ± 0.022 ... binary_entropy(TIP=0.5) = 1.0
>  → λ^H = (1/(H-1)) * (1 - 1) = 0"

This widened the Δ(S2 − S0) gap from -8.72% to -11.79%. The root cause
diagnosis is the operator-mandated cross-check for W18.

## Two sub-checks

### F1: Pure-input parity (synthetic Gaussians with known TIP)

Generate 5 test cases:
1. Disjoint distributions (mean separation ≫ std) → expected TIP ≈ 0
2. Coincident distributions → expected TIP ≈ 1
3. Mild overlap → expected TIP in (0, 0.5)
4. Heavy overlap → expected TIP in (0.5, 1)
5. Maximal entropy case → expected TIP ≈ 0.5

Both C++ and Python compute TIP. Compare to expectations + each other.

### F2: Production-input replay

Sample 100 rows from `python_refactor/experiments/results/w17-5-carry-1-resmoke/lambda_tip_trace.csv`. Reconstruct inputs (current vs predicted state vectors + KF covariances if available). Re-run TIP through both sides. Verify saturation reproduces (i.e., both sides give ~0.5 on the same inputs).

## Verdict matrix

| F1 outcome | F2 outcome | Diagnosis |
|---|---|---|
| Both agree on F1 known values | Both saturate on F2 production | Structural: predictive distributions ARE max-uncertain on this data |
| Both agree on F1 | Sides diverge on F2 | Production-data reconstruction may differ; deeper investigation needed |
| Sides disagree on F1 | (any) | TIP code has parity bug; determine which side is wrong via thesis Eq (6.4) |

If F1+F2 both agree on saturation → the answer is "predictive distributions
are maximally uncertain, which is a structural property of the data + the
KF predictive model on this dataset, NOT a TIP bug".

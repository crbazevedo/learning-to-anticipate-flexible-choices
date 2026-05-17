---
id: W5-2
role: bug-fixer
name: Thread covariance updates (paper Eq 15) through MultiHorizon.learn_population
purpose: "Closes W4-2-CARRY-1. Extends mean-only Eq 14 to full Gaussian convex combo: Σ_combined = (1 - Σλ)^2 * Σ_t + Σ λ_h^2 * Σ_{t+h}."
wave: W5
unit: W5-2
depends_on: []
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/tests/test_multi_horizon_anticipatory.py
output_contract:
  files:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/tests/test_multi_horizon_anticipatory.py
  branch_name: feat/w5-2-multihorizon-covariance-threading
  acceptance: >
    learn_population threads covariance: when solution.P.kalman_state is
    not None, computes anticipatory covariance per paper Eq (15)
    generalized to multi-horizon: Σ_combined = w_0^2 · Σ_t + Σ w_h^2 · Σ_{t+h}
    where w_0 = (1 - Σλ) and w_h = λ_h. Writes back to
    solution.P.kalman_state.P[:2, :2]. Degrades gracefully when
    kalman_state is None (skip covariance update). New test asserts
    covariance is updated AND remains positive-semidefinite. All
    existing 25 multi_horizon tests continue to PASS.
dispatch_instructions: |
  ## What this contract does NOT authorise

  - Touching anticipatory_learning.py.
  - Modifying apply_anticipatory_learning_rule (it stays mean-only;
    covariance threading is wrapped around it in learn_population).

  ## Implementation

  1. Extend learn_population to compute Σ_combined when kalman_state
     is present:
       - Build current_cov from solution.P.kalman_state.P[:2, :2]
       - For each h, get predicted_cov from
         predicted_solution.P.kalman_state.P[:2, :2] if predicted
         _solution.P.kalman_state exists, else fall back to current_cov
       - Compute: Σ_combined = (1-Σλ)^2 · current_cov + Σ λ_h^2 · predicted_cov_h
       - Write back to solution.P.kalman_state.P[:2, :2]
  2. New test test_w5_2_covariance_threaded_through_learn_population:
       - Build a MockSolution WITH kalman_state (P matrix is PSD)
       - Run learn_population
       - Assert solution.P.kalman_state.P[:2, :2] eigenvalues all >= 0
         (positive-semidefinite preserved)
       - Assert covariance changed (non-trivial update)
---

# W5-2 — Covariance threading per paper Eq (15)

Closes W4-2-CARRY-1.

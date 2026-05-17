---
id: W13-2
role: tooling-author
name: Out-of-sample future hypervolume evaluator (thesis Eqs 7.10+7.11)
purpose: "New module computing OOS Future Average Hypervolume per Azevedo thesis §7.2.2 Eqs 7.10+7.11. Pure functional API; no I/O coupling."
wave: W13
unit: W13-2
depends_on: [W13-1]
blocks: [W13-3]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/THESIS-INDEX.md
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - python_refactor/experiments/oos_evaluator.py
    - python_refactor/tests/test_experiments_oos_evaluator.py
  branch_name: feat/w13-2-oos-evaluator
  acceptance: >
    oos_evaluator.py exposes compute_oos_efhv(pareto_weights, oos_returns,
    n_samples, z_ref, rng) returning mean + std + samples. Uses bootstrap
    over held-out returns to obtain E sample (μ̂_e, Σ̂_e) pairs per thesis
    Fig 7.2. Hypervolume of N-point Pareto cloud computed in 2D
    (risk-min, return-max) against z_ref = (0.2, 0.0). ≥ 5 regression
    tests including a hand-computed HV case + a deterministic-future case
    (zero variance over bootstraps) + E=1000 stability check.
dispatch_instructions: |
  Pure-function module. Imports numpy + pandas only. No side effects.

  Public API:
    fit_future_state(returns_df) → (mu, sigma)
    evaluate_portfolio_under_future(weights, mu, sigma) → (risk, return)
    hypervolume_2d(points, z_ref) → float
    compute_oos_efhv(pareto_weights, oos_returns, n_samples=1000,
                     z_ref=(0.2, 0.0), rng=None) → dict

  Thesis fidelity:
    - z_ref = (0.2, 0.0)^T (risk_max=0.2, return_min=0.0) per §7.2.3
    - E default 1000 per §7.2.3
    - Bootstrap resampling matches Fig 7.2 spirit (E independent
      sample-stats realizations from held-out distribution)

  What NOT to do:
    - Don't import experiment_manager or sms_emoa (heavy + circular risk).
    - Don't write to disk; W13-3 owns the orchestration layer.
    - Don't reimplement the algorithm — evaluate trained portfolios only.
---

# W13-2 — OOS future hypervolume evaluator

Implements thesis Eqs 7.10+7.11. Pure-function API.

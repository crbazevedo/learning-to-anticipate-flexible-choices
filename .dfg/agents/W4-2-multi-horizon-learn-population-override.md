---
id: W4-2
role: bug-fixer
name: Override MultiHorizon.learn_population to drive multi-horizon machinery
purpose: "Closes W1-3-CARRY-3. Adds MultiHorizonAnticipatoryLearning.learn_population that drives apply_anticipatory_learning_rule (paper Eq 14) instead of the parent's single-horizon path."
wave: W4
unit: W4-2
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
    - python_refactor/src/algorithms/anticipatory_learning.py
output_contract:
  files:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/tests/test_multi_horizon_anticipatory.py
  branch_name: feat/w4-2-multi-horizon-learn-population-override
  acceptance: >
    MultiHorizonAnticipatoryLearning overrides learn_population. The
    override (a) iterates each solution, (b) computes multi-horizon
    lambda rates via existing calculate_multi_horizon_lambda_rates,
    (c) generates predicted states per horizon, (d) applies
    apply_anticipatory_learning_rule (paper Eq 14) to update the
    solution's [ROI, risk] state, (e) tags the solution with
    solution.multi_horizon_applied = True for verifiability.
    New regression test asserts the override is invoked + the tag
    is set. All 22 existing multi_horizon tests + new test PASS.

dispatch_instructions: |
  ## What this contract does NOT authorise

  - Touching anticipatory_learning.py (parent class is left alone).
  - Touching files outside output_contract.

  ## Implementation

  1. multi_horizon_anticipatory.py: add a `learn_population` method on
     MultiHorizonAnticipatoryLearning. Body:
       - Iterate each solution in population.
       - Initialize state estimates if needed (call parent's
         _observe_state_1step_ahead for setup).
       - Get multi-horizon lambda rates via
         self.calculate_multi_horizon_lambda_rates(solution, self.max_horizon).
       - Skip if no rates (H<2 case).
       - Build the list of predicted_state vectors for h=1..H-1.
         For each h, use self._generate_predicted_solution(solution, h)
         then convert its [ROI, risk] to a 2D state vector.
       - Build current_state = np.array([solution.P.ROI, solution.P.risk]).
       - Call self.apply_anticipatory_learning_rule(current_state,
         predicted_states, lambda_rates) → anticipatory_state.
       - Update solution.P.ROI = anticipatory_state[0],
         solution.P.risk = anticipatory_state[1].
       - Tag solution.multi_horizon_applied = True.
       - Store via self.store_historical_population(population) at end.

  2. test_multi_horizon_anticipatory.py: add a regression test
     `test_w4_2_learn_population_drives_multi_horizon` that calls
     learn_population on the mock_solution + asserts
     `mock_solution.multi_horizon_applied == True` after the call.
     Test against current_time > 0 so the loop actually runs.

  3. Verify all existing multi_horizon tests still PASS.
---

# W4-2 — MultiHorizon learn_population override

Closes W1-3-CARRY-3.

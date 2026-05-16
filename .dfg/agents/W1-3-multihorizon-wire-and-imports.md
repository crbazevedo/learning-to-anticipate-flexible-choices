---
id: W1-3
role: bug-fixer
name: Wire MultiHorizonAnticipatoryLearning + fix advanced-module imports
purpose: >
  Two coherent fixes in one unit. (a) Convert 4 advanced modules from
  the broken `from algorithms.X` import style to relative `from .X`
  imports — closing the bulk of the remaining 16 pre-pr pytest
  collection errors as a mechanical side effect. (b) Make
  MultiHorizonAnticipatoryLearning ExperimentManager-compatible by
  having it inherit from AnticipatoryLearning (so it has
  learn_single_solution / learn_population / set_learning interfaces)
  and adding a `learning.use_multi_horizon` flag in ExperimentManager.
  Plus one new test pinning paper Eq (14) — the multi-horizon convex
  combo — against a 3-horizon hand-computed case.
wave: W1
unit: W1-3
depends_on: ['W1-2']
blocks: ['W1-4']
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf  # §IV-A.2 Eq (14) — the multi-horizon convex combination
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py  # 5 broken `from algorithms.X` imports + standalone class needs inheritance
    - python_refactor/src/algorithms/belief_coefficient.py  # 2 broken imports
    - python_refactor/src/algorithms/sliding_window_dirichlet.py  # 0 broken imports (verified by grep); included for completeness
    - python_refactor/src/algorithms/enhanced_n_step_prediction.py  # 4 broken imports
    - python_refactor/src/experiments/experiment_manager.py  # use_multi_horizon flag, added via replan unit-W1-3-20260516T225709Z
  may_read:
    - docs/VISION.md
    - .dfg/agents/W1-2-tip-wire-into-experiment-manager.md  # the parallel pattern for the flag wiring
output_contract:
  files:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/src/algorithms/belief_coefficient.py
    - python_refactor/src/algorithms/sliding_window_dirichlet.py
    - python_refactor/src/algorithms/enhanced_n_step_prediction.py
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/tests/test_eq14_integration.py
  branch_name: feat/w1-3-multihorizon-wire-and-imports
  acceptance: >
    1. Every `from algorithms.X` import in the 4 src/algorithms modules
       is rewritten to `from .X`. Grep returns 0 hits after the unit
       lands.

    2. MultiHorizonAnticipatoryLearning inherits from AnticipatoryLearning
       (single-line class-signature change). Its __init__ calls
       super().__init__() WITHOUT positional miswiring (the pattern
       W1-2 fixed in TIPIntegratedAnticipatoryLearning). All existing
       attributes (max_horizon, tip_calculator, n_step_predictor,
       dirichlet_model, prediction_history, lambda_rates_history) are
       preserved.

    3. ExperimentManager._run_algorithm honours a new
       `learning.use_multi_horizon: true` flag (sibling to use_tip).
       When set, instantiates MultiHorizonAnticipatoryLearning with
       parameters from learning.parameters (constructor accepts
       max_horizon + monte_carlo_samples — narrower than the base class,
       same shape as TIPIntegratedAnticipatoryLearning).

    4. New test python_refactor/tests/test_eq14_integration.py asserts
       paper Eq (14): for a 3-horizon hand-computed case with known
       current_state, predicted_states, and lambda_rates, the result
       of `apply_anticipatory_learning_rule` matches the analytically
       computed `(1 - Σλ) * current + Σ λ_h * predicted_h` to
       numerical precision. Test imports use the W1-2 pattern
       (`from src.algorithms.X`).

    5. All 18 KF tests + 9 existing TIP integration tests + 2 W1-2
       tests + any pre-existing MultiHorizon / belief_coefficient
       tests continue to pass.

    6. Pre-pr collection-error count drops materially (target:
       16 → ≤ 12 errors, depending on which test files were blocked
       solely by these source-file imports).
dispatch_instructions: |
  ## What this contract authorises

  Touching exactly the 6 files in output_contract.files. No other
  files; no other surfaces.

  ## What this contract does NOT authorise

  - Equation-level tests for TIP / λ / belief coefficient — W1-4 scope.
  - Removing the [0.05, 0.95] clamp in temporal_incomparability_probability.py — W1-4 scope.
  - Modifying compute_anticipatory_learning_rate's signature — W1-2 settled that.
  - Translating Portuguese — W1-5 settled docs.
  - Building a pyproject.toml / uv lock — separate packaging unit.
  - Refactoring imports in any test file other than the new test_eq14_integration.py
    (test file import-style is W1-2's pattern; further test-file refactor
    is a separate housekeeping unit).
  - Refactoring imports in src/experiments/thesis_aligned_experiment.py
    (3 broken imports; queued for a separate housekeeping unit because
    thesis_aligned_experiment isn't reached by any orchestrator and
    fixing its imports without wiring it changes nothing).

  ## Required reading sequence (per operator directive 2026-05-16
  sub-papercut #16)

  1. docs/paper.pdf §IV-A.2 Eq (14) — confirm the canonical form:
     ẑ_t | ẑ_{t+1:t+H-1} ≡ ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} (ẑ_{t+h} | ẑ_t − ẑ_t)
     which equivalently is:
     (1 − Σ_{h=1}^{H-1} λ_{t+h}) ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} ẑ_{t+h}
  2. multi_horizon_anticipatory.py lines 13-18 (broken imports) and
     35-105 (class signature + Eq 14 implementation at line 66-105).
  3. belief_coefficient.py lines 13-15 (broken imports).
  4. sliding_window_dirichlet.py — verify it's already clean
     (grep `from algorithms.X` returns nothing) and there's no
     hidden import-style violation deeper in the file.
  5. enhanced_n_step_prediction.py lines 13-17 (broken imports).
  6. experiment_manager.py lines 219-237 (the W1-2-installed
     if-chain for use_tip; the use_multi_horizon flag goes in
     the same chain).

  ## Implementation order

  1. multi_horizon_anticipatory.py:
     a. Convert 5 `from algorithms.X` → `from .X`.
     b. Change class signature: `class MultiHorizonAnticipatoryLearning(AnticipatoryLearning):`.
     c. Add `from .anticipatory_learning import AnticipatoryLearning` import.
     d. Modify __init__ to call `super().__init__(window_size=20)` (or
        another sensible default) AFTER setting the multi-horizon-specific
        attributes. Important: use keyword form to avoid the W1-2 super-bug
        pattern.

  2. belief_coefficient.py: convert 2 `from algorithms.X` → `from .X`.

  3. sliding_window_dirichlet.py: verify clean; if no changes, the
     file is still in output_contract.files for traceability (a 0-line
     diff is acceptable).

  4. enhanced_n_step_prediction.py: convert 4 `from algorithms.X` → `from .X`.

  5. experiment_manager.py: extend the W1-2 if-chain:
        if learning_config.get('use_multi_horizon', False):
            from ..algorithms.multi_horizon_anticipatory import (
                MultiHorizonAnticipatoryLearning,
            )
            learning = MultiHorizonAnticipatoryLearning(
                **learning_config.get('parameters', {})
            )
        elif learning_config.get('use_tip', False):
            ...
        else:
            ...
     Order: multi_horizon first because it has TIP built-in
     (composes upward); use_tip second; default base last.

  6. tests/test_eq14_integration.py (NEW):
     - Use `from src.algorithms.multi_horizon_anticipatory import MultiHorizonAnticipatoryLearning`.
     - Single class TestPaperEq14MultiHorizonConvexCombo with at
       least one test:
         test_apply_anticipatory_learning_rule_matches_paper_eq14
       Inputs:
         current_state = np.array([1.0, 0.5])  # ROI=1.0, risk=0.5
         predicted_states = [np.array([1.2, 0.4]), np.array([1.4, 0.3])]  # H=3
         lambda_rates = [0.2, 0.3]  # Σλ = 0.5; (1-Σλ) = 0.5
       Expected:
         expected = 0.5 * current_state + 0.2 * predicted_states[0] + 0.3 * predicted_states[1]
                  = 0.5 * [1.0, 0.5] + 0.2 * [1.2, 0.4] + 0.3 * [1.4, 0.3]
                  = [0.5+0.24+0.42, 0.25+0.08+0.09]
                  = [1.16, 0.42]
       Assert np.testing.assert_allclose(result, expected, atol=1e-9).
       Docstring cites paper Eq (14).

  ## Verification before commit

  - `grep -rn "from algorithms\." python_refactor/src/algorithms/` returns 0 hits.
  - In `.venv-w1` from python_refactor with PYTHONPATH=.:
    `python -m pytest tests/test_kalman_filter.py tests/test_tip_integration.py tests/test_eq14_integration.py tests/test_multi_horizon_anticipatory.py tests/test_belief_coefficient.py tests/test_sliding_window_dirichlet.py tests/test_enhanced_n_step_prediction.py -v`
    All previously-passing tests remain passing; new test passes;
    previously-blocked tests now collect (some may surface real
    pre-existing failures unrelated to W1-3 — flag those as inherited
    carries for W1-4 or a follow-up unit, do NOT fix them in W1-3).
  - `dfg validate` / `dfg index --verify` / `dfg substrate check` /
    `make ci`: all OK.

  ## Commit discipline (per operator directive 2026-05-16 #3)

  First commit on the feat/w1-3-multihorizon-wire-and-imports branch
  = THIS contract file alone (P3). Implementation commits come AFTER
  the contract is on the branch.

  ## Honest scars to surface in retro

  - MultiHorizonAnticipatoryLearning inheriting from AnticipatoryLearning
    means it auto-gains learn_single_solution / learn_population that
    DON'T actually use the multi-horizon machinery. The wiring is
    "live" in the sense that ExperimentManager can instantiate the
    class, but the algorithm's run loop still goes through the
    parent class's single-horizon path. A deeper integration unit
    is needed to make the multi-horizon machinery actually drive
    the learn_population loop.
  - Some pre-existing tests (test_anticipatory_learning.py,
    test_correspondence_mapping.py, etc.) may have OTHER bugs
    surface now that their imports collect cleanly. Those are NOT
    introduced by W1-3 — they are pre-existing bugs the audit
    didn't enumerate. Document any newly-visible failures as
    inherited-carry-now-visible.
  - thesis_aligned_experiment.py has 3 broken `from algorithms.X`
    imports but is NOT in scope (operator decision: fixing its
    imports without wiring it from any orchestrator changes
    nothing observable). Queued for a future cleanup unit.
---

# W1-3 — Wire MultiHorizon + fix advanced-module imports

## The single sentence

11 broken `from algorithms.X` imports across 4 modules are the bulk of
the remaining pre-pr collection errors. Fix them mechanically. While
in that surface, make MultiHorizonAnticipatoryLearning inherit from
AnticipatoryLearning so the ExperimentManager use_multi_horizon flag
has somewhere to land.

## Paper canon (Eq 14)

> ẑ_t | ẑ_{t+1:t+H-1} ≡ ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} (ẑ_{t+h} | ẑ_t − ẑ_t)         (14)
>
> — Azevedo & Von Zuben (2015), §IV-A.2, p. 4

Equivalent algebraic form (after distributing):
`(1 − Σλ) ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} ẑ_{t+h}` — exactly what
`apply_anticipatory_learning_rule` already implements at lines 66-105
of `multi_horizon_anticipatory.py`.

## Why this matters

After W1-3 lands, pre-pr's pytest gate should drop from 16 collection
errors to a much smaller number (target ≤ 12; possibly lower depending
on transitive blocking). And the audit's "wired but unreachable"
finding for MultiHorizon is closed at the wiring level — though the
deeper "actually use multi-horizon in the run loop" integration
remains a future unit.

## What this unit deliberately does NOT do

- Touch test files other than the new test_eq14_integration.py.
- Touch src/experiments/thesis_aligned_experiment.py (3 broken
  imports there, but it's never reached by an orchestrator;
  separate cleanup unit).
- Modify any algorithm's run loop to USE the multi-horizon
  machinery — that's a follow-up integration unit.
- Remove the [0.05, 0.95] TIP clamp — W1-4 scope.

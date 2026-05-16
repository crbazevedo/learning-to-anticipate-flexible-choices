---
id: W1-2
role: bug-fixer
name: Wire TIP into the live ExperimentManager path
purpose: >
  Make the published λ in the live anticipatory-learning path actually
  use the TIP-derived arm (paper Eq 13) instead of falling back to the
  KF-residual-only path. The audit found that every advanced module is
  implemented + unit-tested but unreachable from `ExperimentManager` /
  `run_experiments.py` / `main.py`. This unit closes the keystone gap:
  (a) fixes the silent positional-arg miswire in
  `TIPIntegratedAnticipatoryLearning.__init__`; (b) threads
  `tip_calculator` + `predicted_solution` from `self` through
  `anticipatory_learning_obj_space` into `compute_anticipatory_learning_rate`
  so paper Eq 14 / 7.16 actually fires the TIP arm in the live run;
  (c) teaches `ExperimentManager` to instantiate
  `TIPIntegratedAnticipatoryLearning` when the experiment config has
  `learning.use_tip: true`; (d) adds one baseline experiment in
  `run_experiments.py` that sets the flag end-to-end; (e) adds an
  integration test asserting the TIP arm is non-zero when use_tip=true.
wave: W1
unit: W1-2
depends_on: ['W1-1']
blocks: ['W1-3']
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf  # §IV-A.1 Eq (13) — λ^(H); §IV-A Eq (12) — TIP definition
    - python_refactor/src/algorithms/anticipatory_learning.py  # line 124-148 base __init__; line 214-263 compute_anticipatory_learning_rate; line 298-317 TIPIntegrated.__init__ (the bug at 315); line 429-462 anticipatory_learning_obj_space (the call site that needs threading)
    - python_refactor/src/algorithms/temporal_incomparability_probability.py  # line 37-75 calculate_tip — its API + what predicted_solution needs
    - python_refactor/src/experiments/experiment_manager.py  # line 197-225 _run_algorithm + learning instantiation
    - python_refactor/run_experiments.py  # line 120-155 create_learning_experiments — the config shape
    - python_refactor/tests/test_tip_integration.py  # existing test surface to extend
  may_read:
    - docs/VISION.md
    - thesis_codebase_analysis.md
    - .dfg/agents/W1-1-kalman-state-canonical.md  # paper Eq (11) canonical ordering — relevant for predicted_solution's kalman_state.x indices
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/run_experiments.py
    - python_refactor/tests/test_tip_integration.py
  branch_name: feat/w1-2-tip-wire-into-experiment-manager
  acceptance: >
    1. TIPIntegratedAnticipatoryLearning.__init__ correctly calls
       super().__init__(window_size=window_size) — no positional
       miswiring that previously set learning_rate=window_size.

    2. anticipatory_learning_obj_space, when self has a tip_calculator
       attribute (i.e., self is a TIPIntegratedAnticipatoryLearning
       instance), constructs a predicted_solution shadow from the KF
       prediction (anticipative_portfolio.P.kalman_state.x_next +
       P_next) and passes both tip_calculator + predicted_solution to
       compute_anticipatory_learning_rate.

    3. ExperimentManager._run_algorithm honours an experiment-config
       flag `learning.use_tip: true` (sibling to `learning.enabled`)
       and instantiates TIPIntegratedAnticipatoryLearning instead of
       the base AnticipatoryLearning.

    4. run_experiments.py contains at least one new experiment config
       with `learning.enabled: true` AND `learning.use_tip: true` and
       parameters compatible with TIPIntegratedAnticipatoryLearning's
       constructor.

    5. New integration test in test_tip_integration.py asserts that
       (a) the super().__init__ bug is fixed (TIPIntegratedAnticipatory
       Learning(window_size=10).base_learning_rate is NOT 10.0), and
       (b) compute_anticipatory_learning_rate, when called with a
       non-None tip_calculator + predicted_solution, returns a
       combined rate that REFLECTS the TIP arm (i.e., differs from
       _compute_traditional_learning_rate on the same inputs).

    6. The 4 existing TestPaperEq11Canonical tests in test_kalman_filter.py
       continue to pass (no regression in W1-1's deliverable).

    7. test_tip_integration.py existing tests do not regress.
dispatch_instructions: |
  ## What this contract authorises

  Touching exactly the 4 files in output_contract.files. No other
  files; no other surfaces.

  ## What this contract does NOT authorise

  - Refactoring advanced-module imports (`from algorithms.X` → `from .X`).
    That's W1-3.
  - Adding equation-level tests for TIP / λ / belief coefficient.
    That's W1-4.
  - Touching kalman_filter.py or sms_emoa.py (W1-1's surface).
  - Touching docs (W1-5's surface).
  - Wiring MultiHorizonAnticipatoryLearning. That's W1-3.

  ## Required reading sequence (per operator directive 2026-05-16
  sub-papercut #16)

  1. docs/paper.pdf §IV-A.1 Eq (13) — λ^(H) = (1/(H-1))[1 - H(TIP)]
     and §IV-A Eq (12) — TIP definition.
  2. anticipatory_learning.py — verify the diagnostic understanding:
     - line 124-148: parent __init__ signature
     - line 214-263: compute_anticipatory_learning_rate (TIP arm
       EXISTS at lines 257-262 — it just never fires because callers
       don't pass tip_calculator)
     - line 298-317: TIPIntegratedAnticipatoryLearning.__init__; the
       super() call at line 315 silently miswires window_size into
       learning_rate
     - line 429-462: anticipatory_learning_obj_space; line 460 is
       the call site that drops tip_calculator on the floor
  3. temporal_incomparability_probability.py line 37-75 — confirm
     calculate_tip's API: needs current_solution AND predicted_solution
     with .P.ROI, .P.risk, .P.kalman_state.P[:2,:2]
  4. experiment_manager.py line 197-225 — the instantiation site
     that always uses base AnticipatoryLearning today
  5. run_experiments.py line 120-155 — confirm the config schema
     for adding a new use_tip experiment
  6. test_tip_integration.py — confirm existing test patterns

  ## Implementation order

  1. Patch TIPIntegratedAnticipatoryLearning.__init__ (anticipatory_
     learning.py line 315) — use keyword arg:
        super().__init__(window_size=window_size)
     Verify with grep that no other miswired super() calls exist
     in the file.

  2. Add a small helper method on AnticipatoryLearning (NOT on the
     subclass — keeps the wiring point at the base class so the
     attribute-check in step 3 works for both classes):
        def _build_predicted_shadow(self, current: Solution) -> Solution
     that constructs a transient Solution-like with .P.ROI from
     kalman_state.x_next[0], .P.risk from x_next[1] (paper Eq 11
     canonical ordering), .P.kalman_state with P[:2,:2] = P_next[:2,:2].

  3. Modify anticipatory_learning_obj_space (line 460) to detect
     self.tip_calculator (via getattr(self, 'tip_calculator', None)),
     build the predicted shadow, and pass both to
     compute_anticipatory_learning_rate as keyword arguments.

  4. Modify experiment_manager.py _run_algorithm (line 219-224)
     so that:
        if learning_config.get('use_tip', False):
            from ..algorithms.anticipatory_learning import (
                TIPIntegratedAnticipatoryLearning
            )
            learning = TIPIntegratedAnticipatoryLearning(
                **learning_config.get('parameters', {})
            )
        else:
            from ..algorithms.anticipatory_learning import AnticipatoryLearning
            learning = AnticipatoryLearning(
                **learning_config.get('parameters', {})
            )
     Note: TIPIntegratedAnticipatoryLearning's constructor accepts
     only window_size + monte_carlo_samples (per its current
     signature). The use_tip baseline experiment in step 5 must
     supply parameters compatible with this narrower surface.

  5. Add one new experiment in run_experiments.py with:
        "learning": {
            "enabled": True,
            "use_tip": True,
            "parameters": {
                "window_size": 20,
                "monte_carlo_samples": 500
            }
        }
     and add it to the returned list from create_learning_experiments().

  6. Add new integration tests in test_tip_integration.py:
     - test_super_init_bug_fixed: assert
       TIPIntegratedAnticipatoryLearning(window_size=10).base_learning_rate
       is NOT 10.0 (the bug would have made it so).
     - test_tip_arm_fires_when_threaded: construct a base AnticipatoryLearning
       and a TIPIntegratedAnticipatoryLearning; for the SAME inputs, call
       compute_anticipatory_learning_rate WITHOUT tip_calculator (base)
       vs WITH tip_calculator + predicted_solution (TIP-integrated).
       Assert the two rates differ — that's the proof the TIP arm fires.

  ## Verification before commit

  - In a fresh venv with numpy+pytest installed:
    `cd python_refactor && python -m pytest tests/test_tip_integration.py tests/test_kalman_filter.py -v`
    All previously-passing tests continue to pass; new tests pass.

  - Grep for any remaining `super().__init__(window_size)` or other
    positional super() calls in anticipatory_learning.py — should be
    zero.

  - Grep for `compute_anticipatory_learning_rate(` in anticipatory_
    learning.py — every call site should either be the new TIP-aware
    invocation or have a documented reason to not pass tip_calculator.

  ## Commit discipline (per operator directive 2026-05-16 #3)

  First commit on the feat/w1-2-tip-wire-into-experiment-manager
  branch = THIS contract file alone (P3). Implementation commits
  come AFTER the contract is on the branch.

  ## Honest scars to surface in retro

  - `TIPIntegratedAnticipatoryLearning`'s constructor signature is
    narrow (window_size + monte_carlo_samples) and ignores most of
    the base class's parameters (learning_rate, prediction_horizon,
    state_observation_frequency, error_threshold, learning_type,
    adaptive_learning). This may surface as a friction when a user
    tries to enable TIP AND tune learning_rate; a constructor
    harmonisation unit might be warranted later.
  - The "predicted shadow" helper builds a transient Solution with
    a minimal kalman_state. If TIP MC sampling exercises code paths
    that touch other Solution attributes (P.investment, etc.), the
    shadow may not be a sufficient stand-in. Document this scar
    and flag for runtime testing with a real experiment.
  - The `[0.05, 0.95]` clamp in `_calculate_tip_with_covariance`
    (TIP module) hides degenerate dominance and is NOT addressed in
    this unit (it's W1-4 scope). Document so W1-4 doesn't have to
    re-discover it.
---

# W1-2 — Wire TIP into the live ExperimentManager path

See YAML frontmatter for the structured contract. Free-form prose
below is operator-readable context.

## The single sentence

`compute_anticipatory_learning_rate` already supports the TIP arm.
Nothing in the live path passes it the inputs to fire it. This unit
threads the inputs.

## Paper canon

> λ^(H)_{t+h} = (1/(H-1)) [1 − H(p_{t,t+h})]                    (13)
>
> — Azevedo & Von Zuben (2015), §IV-A.1, p. 4

where p_{t,t+h} is the TIP from paper Eq (12), and H is the binary
entropy. When TIP ≈ 0.5 (maximum uncertainty), the arm contributes 0;
when TIP is near 0 or 1 (high predictability), the arm contributes
~1/(H-1). The live algorithm without this wiring uses only the
KF-residuals arm (λ^(K), thesis-only / paper §IV-A.1 narrative). After
this unit, the published λ is the (1/2)(λ^(H) + λ^(K)) blend per
thesis Eq 7.16 — explicitly flagged as thesis-only-extension in W1-5's
reconciliation table.

## Why this unit matters

This is the keystone of W1. Without it, the entire advanced-anticipatory-
learning machinery (TIP calculator, belief coefficient, multi-horizon,
Dirichlet) is dead code reachable only from unit tests. After this lands,
`run_experiments.py --use-tip` (effectively) produces λ values whose TIP
arm is provably non-zero — the proof the audit's "wired but divergent"
finding is closed.

## What this unit deliberately does NOT do

- Translate Portuguese narrative anywhere.
- Touch kalman_filter.py / sms_emoa.py (W1-1's surface).
- Wire MultiHorizonAnticipatoryLearning (W1-3 scope).
- Add equation-level tests for TIP-vs-known-analytical-Gaussians
  (W1-4 scope).
- Remove the [0.05, 0.95] clamp in TIP (W1-4 scope).

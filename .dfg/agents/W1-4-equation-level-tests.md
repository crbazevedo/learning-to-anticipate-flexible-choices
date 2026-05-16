---
id: W1-4
role: bug-fixer
name: Equation-level tests for TIP / λ / belief coefficient; remove silent clamp
purpose: >
  Close the audit's "tests assert bounds, not equations" finding for the
  three core paper equations the live algorithm depends on after W1-2 +
  W1-3 wiring landed. Specifically: (a) add at least two known-analytical
  Gaussian regression tests for TIP (paper Eq 12) — disjoint and identical
  — that pin the calculator against analytical truth rather than bounds;
  (b) add a binary-entropy parametric test for λ (paper Eq 13) covering
  H(0)=0, H(1)=0, H(0.5)=1; (c) make the silent [0.05, 0.95] TIP clamp
  opt-out-able via a constructor argument so the disjoint test can
  actually see TIP near 0; (d) add a known-Gaussian regression test for
  the belief coefficient (paper Eq 20 / current code's Eq 6.30). Also
  fix the W1-3-CARRY-1 import-style issue in both touched test files
  (they're already in scope per output_contract).
wave: W1
unit: W1-4
depends_on: ['W1-3']
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf  # §IV-A Eq (12) TIP; §IV-A.1 Eq (13) λ^(H); binary entropy form
    - python_refactor/src/algorithms/temporal_incomparability_probability.py  # 3 sites of the [0.05, 0.95] clamp (lines 119, 161, 193); binary_entropy at 195-211; rate calc at 213-230
    - python_refactor/tests/test_temporal_incomparability_probability.py  # current bounds-only tests + broken `from algorithms.X` imports
    - python_refactor/tests/test_belief_coefficient.py  # current bounds-only tests + broken imports
  may_read:
    - docs/VISION.md
    - thesis_codebase_analysis.md
    - .dfg/agents/W1-2-tip-wire-into-experiment-manager.md
    - .dfg/agents/W1-3-multihorizon-wire-and-imports.md
output_contract:
  files:
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
    - python_refactor/tests/test_temporal_incomparability_probability.py
    - python_refactor/tests/test_belief_coefficient.py
  branch_name: feat/w1-4-equation-level-tests
  acceptance: >
    1. `TemporalIncomparabilityCalculator.__init__` accepts a new
       `clamp_range: tuple[float, float] | None = (0.05, 0.95)`
       parameter (default preserves pre-W1-4 behaviour; `None` disables
       the clamp entirely). All 3 internal clamp sites (lines 119, 161,
       193 pre-W1-4) call a single private helper that honours the
       constructor setting.

    2. NEW class `TestPaperEq12TIPKnownAnalytical` in
       test_temporal_incomparability_probability.py with ≥ 2 tests:
         * test_tip_near_zero_for_disjoint_gaussians: with
           clamp_range=None, two Gaussians whose means differ by
           > 10·σ in both objectives produce TIP < 0.1
           (Monte-Carlo tolerance).
         * test_tip_near_half_for_identical_gaussians: two identical
           Gaussians produce TIP near 0.5 ± Monte-Carlo tolerance.

    3. NEW class `TestPaperEq13LambdaBinaryEntropy` in
       test_temporal_incomparability_probability.py: parametric
       binary-entropy table — H(0)=0, H(1)=0, H(0.5)=1 — and the
       λ form `(1/(H-1))*(1-H(tip))` for at least H=2 and H=3.

    4. NEW class `TestW1_4_BeliefCoefficientKnownGaussians` in
       test_belief_coefficient.py: belief coefficient v near 1 when
       TIP is near 0 or 1 (predictable); v near 0.5 when TIP is near
       0.5 (unpredictable). Use whatever Belief* class exists; if the
       belief module clamps too, mirror the constructor-opt-out pattern.

    5. Both test files' broken `from algorithms.X` imports are
       refactored to `from src.algorithms.X` (W1-2 / W1-3 pattern),
       closing 2 more of the .venv-w1 collection-error count
       (W1-3-CARRY-1 partial closure).

    6. All pre-existing tests in scope continue to pass; existing
       bounds tests remain unchanged. New tests add value, do not
       replace.
dispatch_instructions: |
  ## What this contract authorises

  Touching exactly the 3 files in output_contract.files.

  ## What this contract does NOT authorise

  - Wiring MultiHorizon's `apply_anticipatory_learning_rule` into the
    learn_population loop (W1-3-CARRY-3 / future integration unit).
  - Touching any source file beyond temporal_incomparability_probability.py.
  - Touching test_kalman_filter / test_tip_integration / test_eq14_integration
    (already in good shape from earlier units).
  - Translating Portuguese narrative.
  - Building a pyproject.toml / uv lock.
  - Refactoring imports in test_multi_horizon_anticipatory.py /
    test_belief_coefficient.py beyond what W1-4's own assertions need
    (the belief_coefficient test file IS in scope; the multi_horizon
    one is NOT).

  ## Required reading sequence (per operator directive 2026-05-16
  sub-papercut #16)

  1. docs/paper.pdf §IV-A Eq (12) — TIP definition; §IV-A.1 Eq (13)
     — λ^(H) form; binary entropy discussion.
  2. python_refactor/src/algorithms/temporal_incomparability_probability.py
     full file scan — confirm the 3 clamp sites and the binary_entropy
     + calculate_anticipatory_learning_rate_tip surfaces.
  3. python_refactor/tests/test_temporal_incomparability_probability.py
     — read existing tests so the new class doesn't duplicate.
  4. python_refactor/tests/test_belief_coefficient.py — read existing
     tests so the new class doesn't duplicate; confirm what API the
     belief module exposes (BeliefCoefficientCalculator etc.).

  ## Implementation order

  1. temporal_incomparability_probability.py:
     a. Add `clamp_range: Optional[tuple[float, float]] = (0.05, 0.95)`
        kwarg to `__init__`. Store as `self.clamp_range`.
     b. Add private helper:
          def _clamp_tip(self, tip: float) -> float:
              if self.clamp_range is None:
                  return tip
              return max(self.clamp_range[0], min(self.clamp_range[1], tip))
     c. Replace the 3 sites of `return max(0.05, min(0.95, tip))` with
        `return self._clamp_tip(tip)`.
     d. Docstring updates: explain that clamp_range=None disables the
        clamp (intended for equation-level regression testing).

  2. test_temporal_incomparability_probability.py:
     a. Fix imports: `from src.algorithms.temporal_incomparability_probability import ...`
        (drop sys.path hack).
     b. Add TestPaperEq12TIPKnownAnalytical class with 2 tests:
        * test_tip_near_zero_for_disjoint_gaussians: build
          TemporalIncomparabilityCalculator(monte_carlo_samples=2000,
          clamp_range=None). Build current + predicted with
          mean_diff = 1.0 in ROI, 1.0 in risk; covariance = 0.01 * I
          for each. Assert TIP < 0.1.
        * test_tip_near_half_for_identical_gaussians: same calculator,
          identical means + covariances. Assert 0.3 < TIP < 0.7
          (loose bound; MC variance is real).
     c. Add TestPaperEq13LambdaBinaryEntropy class:
        * test_binary_entropy_table: assert H(0)=0, H(0.5)=1, H(1)=0,
          and H(0.25) ≈ 0.811 (known value).
        * test_lambda_form_eq13_for_h_equals_2_and_3: for H=2,
          calc.calculate_anticipatory_learning_rate_tip(0.5, 2) == 0.0
          (max entropy); calc.calculate_anticipatory_learning_rate_tip(0, 2) == 1.0
          (zero entropy). Repeat for H=3 with the (1/(H-1)) normalisation.

  3. test_belief_coefficient.py:
     a. Fix imports: `from src.algorithms.belief_coefficient import ...`
     b. Inspect the belief module's API (BeliefCoefficientCalculator etc.)
        to determine the right call surface for the test.
     c. Add TestW1_4_BeliefCoefficientKnownGaussians class:
        * test_belief_high_when_predictable: TIP=0 → v should be 1.0
          (or whatever the high-confidence value is per current
          implementation; cite the equation).
        * test_belief_low_when_unpredictable: TIP=0.5 → v should be
          0.5 (or the documented low-confidence value).

  ## Verification before commit

  - In `.venv-w1` from python_refactor with PYTHONPATH=.:
    `python -m pytest tests/test_temporal_incomparability_probability.py
    tests/test_belief_coefficient.py tests/test_eq14_integration.py
    tests/test_kalman_filter.py tests/test_tip_integration.py
    tests/test_sliding_window_dirichlet.py -v`
    All previously-passing tests continue to pass; new tests pass.
  - `dfg validate` / `dfg index --verify` / `dfg substrate check` /
    `make ci`: all OK.

  ## Commit discipline (per operator directive 2026-05-16 #3)

  First commit on feat/w1-4-equation-level-tests = THIS contract alone (P3).

  ## Honest scars to surface in retro

  - The clamp_range default preserves pre-W1-4 behaviour even though
    the audit found the clamp "hides degenerate dominance." Making
    the clamp opt-out via constructor (rather than removing it
    outright) keeps behavioural compatibility but doesn't fix any
    LIVE callers — they all still pass the default. Documenting as
    scar.
  - The belief_coefficient module's API may not perfectly mirror
    TIP's: it might already accept TIP as input (treating TIP-based
    confidence directly), or it might require a more complex setup.
    Adapt the test as needed; document any adaptation in retro.
  - MC tolerance for the disjoint-Gaussians test (TIP < 0.1 with
    clamp_range=None) is empirical. If MC noise occasionally exceeds
    that bound, the test may flake. Use a fixed numpy random seed
    + 2000 samples to keep variance bounded.
---

# W1-4 — Equation-level tests for TIP / λ / belief coefficient

## The single sentence

The audit found: "Out of ~16 test files, exactly **one** asserts a
thesis equation against a hand-computed value." W1-4 raises that count
by adding equation-level tests for TIP (paper Eq 12), λ (paper Eq 13),
and the belief coefficient (paper Eq 20) against known-analytical
Gaussian cases. Plus an opt-out for the silent [0.05, 0.95] clamp so
the disjoint test can see TIP near 0.

## Paper canon

> p_{t,t+h} = Pr[ẑ_t, ẑ_{t+h}|ẑ_t are mutually incomparable]    (12)
>
> λ^(H)_{t+h} = (1/(H−1)) [1 − H(p_{t,t+h})]                    (13)
>
> v_{t+1} = 1 − (1/2) H(p_{t−1,t})                              (20)
>
> — Azevedo & Von Zuben (2015), §IV-A / §IV-A.1, p. 4–5

## What this unit deliberately does NOT do

- Wire MultiHorizon's apply_anticipatory_learning_rule into the
  learn_population run loop (W1-3-CARRY-3; future integration unit).
- Make the clamp removal mandatory (constructor opt-out is the
  middle path — pin the analytical behaviour without breaking
  existing callers).
- Refactor imports in test files outside W1-4's output_contract
  (deferred to the test-file-housekeeping unit).

---
id: W1-1
role: bug-fixer
name: Canonicalize Kalman state-vector ordering to paper Eq (11)
purpose: >
  Align the Kalman Filter state vector across kalman_filter.py, sms_emoa.py,
  and anticipatory_learning.py to the canonical ordering specified by
  paper Eq (11) — z_t^+ = (z_{1,t}...z_{m,t}, ż_{1,t}...ż_{m,t}). For the
  m=2 portfolio case (objectives = ROI, risk) this is
  [ROI, risk, ROI_velocity, risk_velocity]. Eliminate the silent miswiring
  at sms_emoa.py:499 where x_next[2] (which is ROI_velocity under sms_emoa's
  own F matrix) is read as future_risk. Add an equation-level test that
  asserts a known KF predict-update yields the expected x_next[1] (future
  risk) under the canonical ordering.
wave: W1
unit: W1-1
depends_on: []
blocks: ['W1-2']
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf  # §IV-A Eq (11) — canonical state-vector ordering
    - python_refactor/src/algorithms/kalman_filter.py  # existing interleaved-ordering F + reads
    - python_refactor/src/algorithms/sms_emoa.py  # lines 152-186 (paper-canonical F + H + R) and lines 497-499 (the bug)
    - python_refactor/src/algorithms/anticipatory_learning.py  # lines 883-884, 1103-1104 (paper-canonical reads)
    - python_refactor/tests/test_kalman_filter.py  # existing test surface to extend
  may_read:
    - docs/VISION.md
    - thesis_codebase_analysis.md  # for the (now-to-be-reconciled) thesis-numbered KF discussion
output_contract:
  files:
    - python_refactor/src/algorithms/kalman_filter.py
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_kalman_filter.py
  branch_name: feat/w1-1-kalman-state-canonical
  acceptance: >
    All three Python files use the paper-canonical state-vector ordering
    [ROI, risk, ROI_velocity, risk_velocity] (matching paper Eq 11 with m=2).
    The F matrix in kalman_filter.py is rewritten to match the canonical
    ordering. sms_emoa.py:499 reads x_next[1] (not x_next[2]) for future_risk.
    anticipatory_learning.py reads (already canonical) verified unchanged.
    A new test in test_kalman_filter.py asserts a known KF predict-update
    against an analytical case derived from Eq (11); the test docstring cites
    Eq (11) by section. All pre-existing tests in test_kalman_filter.py
    continue to pass.
dispatch_instructions: |
  ## What this contract authorises

  Touching exactly the 4 files in output_contract.files. No other files;
  no other surfaces.

  ## What this contract does NOT authorise

  - Wiring TIP, multi-horizon, or any anticipatory-learning advanced
    module into ExperimentManager (that's W1-2 / W1-3).
  - Doc translation or rewrites (that's W1-5).
  - Reformatting / black / ruff sweeps unrelated to the actual edits.

  ## Required reading sequence (per operator directive 2026-05-16
  sub-papercut #16 — "must_read means BOTH emit Read events AND actually
  open + scan the files")

  1. docs/paper.pdf §IV-A Eq (11) — the canonical state-vector specification
  2. python_refactor/src/algorithms/kalman_filter.py — full file, confirm
     the F matrix at lines 117-122 encodes the interleaved
     [ROI, ROI_vel, risk, risk_vel] ordering
  3. python_refactor/src/algorithms/sms_emoa.py lines 152-186 — confirm the
     OVERRIDE F matrix at lines 169-174 encodes the paper-canonical
     [ROI, risk, ROI_vel, risk_vel] ordering
  4. python_refactor/src/algorithms/sms_emoa.py lines 497-499 — confirm the
     bug: future_risk = x_next[2] reads ROI_velocity, not risk
  5. python_refactor/src/algorithms/anticipatory_learning.py lines 883-884
     and 1103-1104 — confirm these already read paper-canonical indices
     (x[0]=ROI, x[1]=risk)

  ## Implementation order

  1. Rewrite kalman_filter.py:
     - `__post_init__` comment: `# [ROI, risk, ROI_velocity, risk_velocity]`
     - `initialize_kalman_matrices()`: F matrix to paper-canonical form
       (move the velocity columns to positions 2,3 instead of 1,3)
     - `create_kalman_params()`: x = `[initial_roi, initial_risk, 0.0, 0.0]`
     - `get_portfolio_state()`: return `x[0], x[1]`
     - `get_portfolio_prediction()`: return `x_next[0], x_next[1]`
     - Update docstrings to cite paper Eq (11) explicitly
  2. Patch sms_emoa.py:499: `future_risk = ... x_next[1]` (NOT `[2]`)
  3. Verify anticipatory_learning.py needs no changes (its reads are
     already paper-canonical).
  4. Add a new test in test_kalman_filter.py:
     `test_kalman_state_vector_matches_paper_eq11_canonical_ordering`
     - Build a KalmanParams via create_kalman_params(initial_roi=1.0, initial_risk=0.5)
     - Set a known velocity in x[2]=0.1 (ROI_velocity) and x[3]=-0.05 (risk_velocity)
     - Run kalman_prediction
     - Assert x_next[0] ≈ 1.1 (ROI + ROI_velocity)
     - Assert x_next[1] ≈ 0.45 (risk + risk_velocity) — this is the
       canonical "future risk" the prior code was misreading
     - Assert x_next[2] ≈ 0.1 (ROI_velocity preserved under constant-velocity F)
     - Assert x_next[3] ≈ -0.05 (risk_velocity preserved)
     - Docstring cites paper §IV-A Eq (11)

  ## Verification before commit

  - `cd /Users/crbazevedo/Documents/Professional/research/learning-to-anticipate-flexible-choices`
  - `cd python_refactor && python -m pytest tests/test_kalman_filter.py -v`
  - `cd python_refactor && python -m pytest tests/ -k "kalman or sms_emoa or anticipatory_learning" -v`
  - All passing.

  ## Commit discipline (per operator directive 2026-05-16 #3)

  First commit on the feat/w1-1-kalman-state-canonical branch = THIS
  contract file alone (P3 — contract-first). Implementation commits come
  AFTER the contract is on the branch.

  ## Honest scars to surface in retro

  - Confirm whether `kalman_filter.py`'s default `__post_init__`
    fallback (line 43) is ever actually used in a live code path — if not,
    note as dead-code candidate for W2.
  - Confirm whether other call sites (besides sms_emoa.py:499) read
    `x_next[2]` expecting future_risk — grep before claiming this is
    the only instance.
---

# W1-1 — Canonicalize Kalman state-vector ordering to paper Eq (11)

See YAML frontmatter for the structured contract. Free-form prose below
is operator-readable context.

## The single sentence

Three files disagree about whether the Kalman state vector is
`[ROI, risk, ROI_vel, risk_vel]` (paper Eq 11, m=2) or
`[ROI, ROI_vel, risk, risk_vel]` (interleaved). The paper is canon.
Align all three to the paper.

## The bug surface

`sms_emoa.py` documents the paper-canonical ordering (line 157) and
encodes a paper-canonical F matrix (lines 169-174), but reads
`x_next[2]` as future risk on line 499 — which is `ROI_velocity` under
that F matrix. Every "future risk" the SMS-EMOA hypervolume calculation
uses downstream is therefore reading ROI velocity instead of risk.

## Why this matters

The script narrates (and the paper formalizes) a "future risk halo"
that the algorithm uses to anticipate decisions. If the halo is
actually computed off ROI velocity, the anticipation is steering by
the wrong signal, silently. Every benchmark result downstream of this
is suspect.

## Paper-canon citation

> z_t^+ = (z_{1,t} ⋯ z_{m,t}  ż_{1,t} ⋯ ż_{m,t})^T            (11)
>
> — Azevedo & Von Zuben (2015), §IV-A, p. 4

For m=2 in the portfolio case: `(z_1, z_2, ż_1, ż_2) = (ROI, risk, ROI_vel, risk_vel)`.

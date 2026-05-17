---
id: W17-2
role: code-fixer
name: Wire record_kf_residual into Kalman update path (W16-1-CARRY-1)
purpose: "Closes W16-1-CARRY-1. The λ^K formula is structurally correct per Eq 6.9 but in production self._kf_residual_window is never populated. Wire record_kf_residual into the Kalman update path so λ^K actually fires (vs warm-up fallback)."
wave: W17
unit: W17-2
depends_on: []
blocks: [W17-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/kalman_filter.py
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/kalman_filter.py
    - python_refactor/tests/test_lambda_k_production_firing.py
  branch_name: feat/w17-2-record-kf-residual-wiring
  acceptance: >
    record_kf_residual is called from the Kalman update path (either via
    a returned innovation from kalman_update OR via a hook in the
    AnticipatoryLearning per-solution loop) so that self._kf_residual_window
    becomes non-empty after enough periods. ≥ 3 regression tests covering:
    (1) post-update buffer is non-empty for K>0; (2) buffer respects
    K-period truncation; (3) λ^K branches to the non-warm-up path.
dispatch_instructions: |
  Closes W16-1-CARRY-1.

  Background. W16-1 shipped the λ^K formula per Eq 7.16 + Eq 6.9 but
  W16-1's retro flagged that in production self._kf_residual_window
  is never populated because nothing calls record_kf_residual. As a
  result λ^K falls back to the warm-up "traditional rate" branch,
  not the K-period normalized-residual-sum branch.

  Surgical workflow:

  1. AUDIT FIRST: read kalman_filter.kalman_update + the per-solution
     loop where it's called (likely _observe_state_1step_ahead or
     similar in anticipatory_learning). Determine the cleanest hook:
       (a) kalman_update returns innovation → caller records
       (b) kalman_update accepts an optional callback
       (c) caller computes innovation explicitly post-update

     Option (a) is least invasive — kalman_update already mutates
     state in place; adding a return value is additive.

  2. IMPLEMENT (whichever option per audit):
       - kalman_update returns innovation_squared_sum (scalar)
       - anticipatory_learning sums the per-solution / per-objective
         innovation² across the population for the period, then calls
         self.record_kf_residual(sum)
       - The accumulation happens ONCE per period (not per generation,
         not per solution) so the K-period semantic is preserved

  3. EXPOSURE: add a per-call trace field `lambda_k_branch ∈
     {'k0_zero', 'warmup_traditional', 'kperiod_sum'}` so W17-5
     can directly assert which branch fired.

  4. Tests at python_refactor/tests/test_lambda_k_production_firing.py:
       - simulate K=3 + 4 periods + record_kf_residual after each →
         _kf_residual_window has length 3 (truncated correctly)
       - verify lambda_k_branch == 'kperiod_sum' (not warmup)
       - K=0 → record_kf_residual is no-op → buffer stays empty →
         lambda_k_branch == 'k0_zero'

  What NOT to do:
    - Don't change the λ^K formula itself (W16-1 nailed that).
    - Don't change the trace CSV format (W16-4); just add the
      lambda_k_branch field as a NEW column (backward-compat:
      old readers tolerate extra cols).
    - Don't touch sms_emoa.py / operators / etc.
    - Don't add txn cost / AMFC selector (W17-4).

  PR body MUST echo Eq 6.9 (λ^K definition) + note the carry's
  pre-W17 behavior explicitly per BACKLOG §6.
---

# W17-2 — Wire record_kf_residual into Kalman update path

Closes W16-1-CARRY-1 (residual-buffer population in production).

## Thesis grounding

**§6.1.4 Eq (6.9), p. 119** — λ^K_{t+h} as normalized sum of KF
squared residuals over the historical window of size K periods.

**§7.1.1 (p. 140)** — K ∈ {0, 1, 2, 3}. K=0 → no historical window →
record_kf_residual must be a no-op.

**§7.2.3 Eq (7.16), p. 146** — λ = (1/2)(λ^H + λ^K). The W16-1
implementation is correct on paper; this unit makes it fire in
production.

## Why this matters

W16-1 retro: "the buffer is only populated if the caller explicitly
calls record_kf_residual — which neither the ExperimentManager nor
walk_forward.py does today. So in practice λ^K falls back to the
'warm-up traditional rate' branch, not the 'K-period residual sum'
branch."

The 3.22pp improvement W16-5 saw was therefore mostly attributable
to W16-2 (txn costs) and W16-3 (extrema), not W16-1 per se. Wiring
record_kf_residual lets W17-5 measure W16-1's *actual* production
contribution + decompose cleanly via trace.

## Files to touch

- `python_refactor/src/algorithms/kalman_filter.py` — kalman_update
  (likely returns innovation² scalar)
- `python_refactor/src/algorithms/anticipatory_learning.py` —
  per-solution / per-period accumulation + record_kf_residual call
- `python_refactor/tests/test_lambda_k_production_firing.py` — NEW;
  ≥ 3 tests

## Acceptance

- After K periods of record_kf_residual calls, _kf_residual_window
  has the expected length (truncated to K)
- λ^K computation branches to the non-warm-up path
- lambda_k_branch field surfaced in trace
- ≥ 3 regression tests

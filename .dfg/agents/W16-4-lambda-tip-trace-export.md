---
id: W16-4
role: code-fixer
name: Per-call λ + TIP trace export for W16-1 verification (BACKLOG M3 + M4 partial)
purpose: "Closes BACKLOG M3 + M4 (partial — instrumentation scope). Export per-call (period, generation, solution_rank, λ^H, λ^K, λ, TIP) trace as CSV so W16-5 verifier can confirm both λ arms fire in production."
wave: W16
unit: W16-4
depends_on: [W16-1]
blocks: [W16-5]
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
    - python_refactor/experiments/walk_forward.py
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/experiments/walk_forward.py
    - python_refactor/tests/test_lambda_tip_trace.py
  branch_name: feat/w16-4-lambda-tip-trace-export
  acceptance: >
    Per-call (period, generation, solution_rank, λ^H, λ^K, λ, TIP)
    rows accumulated during anticipatory rate computation. Exported as
    `lambda_tip_trace.csv` alongside the existing metrics.csv in each
    experiment-output directory. ≥ 2 regression tests covering the
    trace row schema + the CSV emit.
dispatch_instructions: |
  Closes BACKLOG: M3 + M4 (partial — trace export scope only; full
  Fig 7.4-7.13 rendering is W17 / W18).

  Background. W16-1 implements λ^K consumption. W16-5 needs to
  empirically confirm both λ arms fire in a real walk-forward run.
  Without a trace, the smoke can only measure aggregate HV delta;
  with the trace, we can verify the λ formula is operating as
  designed before accepting the smoke result.

  This unit ships MINIMAL instrumentation — just the (period, generation,
  solution_rank, λ^H, λ^K, λ, TIP) tuple per anticipatory-learning
  invocation. Full per-portfolio diagnostic plots are W17 / W18 (the
  M-item instrumentation wave).

  Surgical changes:

  1. In anticipatory_learning.py, augment compute_anticipatory_learning_rate
     to append a row to self._trace_rows: a list of dicts with keys
     {period, generation, solution_rank, lambda_h, lambda_k, lambda,
     tip}. Add a flush method to write self._trace_rows to a CSV.

  2. In walk_forward.py, after each period's optimization completes,
     call flush_trace to persist that period's rows to
     `<results_dir>/lambda_tip_trace.csv` (append mode).

  3. Tests:
     - Trace row shape: each row has all 7 keys and correct dtypes.
     - CSV emit: invoke a minimal AnticipatoryLearning, populate trace,
       flush, read back, assert row count == calls.

  Defer to W17 / W18:
    - Full per-portfolio per-period λ plots (Figs 7.4-7.13)
    - 1 - E[TIP] reduction over ranks (the figure shape)
    - Per-seed aggregation + bootstrap CI

  What NOT to do:
    - Don't compute the figures (defer to W18).
    - Don't add coherence/POCID/wealth trace (M6-M8 → W17).
    - Don't touch sms_emoa.py beyond what's needed to pass period+
      generation into the rate-computation call.
    - Don't break the trace if W16-1's λ^K isn't yet wired (record
      λ^K=NaN if missing; W16-5 will catch via the verifier).

  PR body MUST cite TH Eq (6.4) (TIP) + Eq (7.16) (λ) verbatim per
  BACKLOG §6.
---

# W16-4 — Per-call λ + TIP trace export

Closes BACKLOG.md items: **M3** + **M4** (partial — trace export scope
only; full Fig 7.4-7.13 rendering is W17 / W18).

## Thesis grounding

**§7.3.1 "Estimated Confidence Over the Stochastic Pareto Frontiers
(SPFs)", p. 148** — verbatim:
> "We recall that, according to the OAL rules proposed in chapter 6,
>  the available KF predictive objective distributions of portfolios
>  associated with higher predictive confidence are more intensely
>  incorporated into the resulting anticipatory distributions...
>  Moreover, recall that rank 1 portfolios correspond to the highest
>  risk/return, whereas a rank 20 one corresponds to the lowest
>  risk/return in the population."

**§6.1.3 Eq (6.4), p. 116** — TIP (Temporal Incomparability Probability)
definition.

**§7.3.1 Figs 7.4-7.13, pp. 150-155** — show `1 - E[TIP]` per portfolio
rank, averaged over all periods, for each (instance, K, DM) combination.
This is the visualization downstream of the trace.

**§7.2.3 Eq (7.16), p. 146** — λ = (1/2)(λ^H + λ^K). The trace records
all 3 scalars per call so W16-5 can confirm both arms fire after W16-1.

## Why this contributes to closing the 9% gap

This is verifier-infrastructure for W16-1 + W16-5. Without the trace,
W16-5 can only measure aggregate HV delta and infer whether λ^K is
firing — high inference noise. With the trace, we can directly assert
"both λ^H AND λ^K are nonzero in production for K>0 scenarios" and
distinguish "λ^K firing but didn't help" from "λ^K silently zero".

## Files to touch

- `python_refactor/src/algorithms/anticipatory_learning.py` —
  self._trace_rows + flush method
- `python_refactor/experiments/walk_forward.py` — call flush after
  each period
- `python_refactor/tests/test_lambda_tip_trace.py` — NEW; ≥ 2 tests

## Acceptance

- Trace rows have schema {period, generation, solution_rank,
  lambda_h, lambda_k, lambda, tip}
- CSV `lambda_tip_trace.csv` emitted in each experiment-output dir
- ≥ 2 regression tests
- Works even if W16-1 cycle-1 left λ^K=NaN (then trace shows NaN
  honestly)

---
id: W16-1
role: code-fixer
name: Implement λ^K consumption in compute_anticipatory_learning_rate (BACKLOG H2 + W15-3-CARRY-1)
purpose: "Closes BACKLOG H2 + W15-3-CARRY-1 consumption half. Verify both λ^H and λ^K arms fire per thesis Eq 7.16; implement λ^K (normalized KF squared-residual sum) if missing. The W16 keystone — highest-probability lever for closing the remaining 9% gap."
wave: W16
unit: W16-1
depends_on: []
blocks: [W16-4, W16-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/src/algorithms/kalman_filter.py
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_lambda_k_consumption.py
  branch_name: feat/w16-1-lambda-k-consumption
  acceptance: >
    compute_anticipatory_learning_rate (or its callers) consume self.window_size
    (K) to drive the λ^K arm per thesis Eq 7.16. λ = (1/2)(λ^H + λ^K) is the
    canonical formula; verify or implement. ≥ 4 regression tests covering:
    K=0 (λ^K=0, only λ^H fires), K>0 (both arms fire), squared-residual
    accumulation correctness, balanced-tension semantic.
dispatch_instructions: |
  Closes BACKLOG: H2 + W15-3-CARRY-1 consumption half.

  Surgical changes (in priority order):

  1. AUDIT FIRST. Read compute_anticipatory_learning_rate in
     anticipatory_learning.py. Determine:
       - Is λ_t+h computed as a single scalar or as (1/2)(λ^H + λ^K)?
       - Is λ^K (squared KF residual sum) computed at all?
       - Does the function consume self.window_size?
     Document the audit verdict in the PR description before any code change.

  2. IF λ^K is missing (most likely per W15-5 retro hypothesis): implement
     it per thesis Eq 7.16 + Eq (6.9). The formula uses normalized sum of
     KF squared residuals over the historical window of size K. Required
     state per portfolio: a deque/array of past KF prediction errors
     of length ≤ K. On each invocation, accumulate (measurement - prediction)^T
     (measurement - prediction); normalize to [0,1] sigmoid-style; combine
     with λ^H via the (1/2) average.

  3. IF self.window_size is already consumed but the formula is wrong:
     fix the formula match thesis Eq 7.16 verbatim.

  4. Add per-call trace export (just λ^H + λ^K + λ scalars per call, NOT
     the full M3 instrumentation — that's W17). Sufficient for W16-4
     verifier to confirm both arms fire.

  What NOT to do:
    - Don't touch sms_emoa.py — only anticipatory_learning.py + tests.
    - Don't refactor the constructor signature (W15-3 + W15-5 already
      fixed window_size kwarg threading).
    - Don't add txn costs to HV (W16-2).
    - Don't implement extrema preservation (W16-3).
    - Don't ship the M-item instrumentation (W17).

  PR body MUST echo the thesis Eq 7.16 verbatim + the audit verdict
  (was λ^K missing or just wrong?) per BACKLOG §6 grounding discipline.
---

# W16-1 — Implement λ^K consumption per thesis Eq 7.16

Closes BACKLOG.md items: **H2** (λ formula completeness) + **W15-3-CARRY-1**
consumption half (constructor wired in W15-5 but consumption may be absent).

## Thesis grounding

**§7.2.3 Bayesian Tracking Parameters, Eq (7.16) (p. 146)** — verbatim:
> "λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K)), for which the anticipation
>  horizon is H = 2 (one-step-ahead prediction). The anticipation rate of
>  each portfolio is thus determined not only by the estimated temporal
>  incomparability probability between the current and the predictive
>  objective distribution (λ_{t+h}^(H)), but also by the observed
>  normalized sum of KF squared residuals (λ_{t+h}^(K))."
>
> "The motivation for taking the average (the 1/2 factor) of the two
>  aforementioned self-adjustment strategies for setting λ_{t+h} can be
>  explained by the intuition of providing a *balanced tension* in the
>  OAL rule between:
>  - The desire of *trusting* in decision paths that *were shown* to
>    lead to higher predictable consequences *in the past*
>    (λ^{K}_{t+h} in Eq (6.9)); and
>  - The desire of *trusting* in decision paths that *are estimated* to
>    lead to higher predictable consequences *in the future*
>    (λ^{H}_{t+h} in Eq (6.6))."

**§6.1.3 Eq (6.6), p. 117** — defines `λ_{t+h}^(H)` (TIP-based, future-looking).

**§6.1.4 Eq (6.9), p. 119** — defines `λ_{t+h}^(K)` (squared-KF-residual,
past-looking). Specifically: normalized sum of KF squared residuals over
the historical window of size K periods.

**§7.1.1 Investigated Algorithmic Variants (p. 140)** — K ∈ {0, 1, 2, 3}.
For K = 0, λ^K = 0 by construction (no historical window).

## Why this is the W16 keystone

W15-5 retro analysis: the remaining -8.75% gap (S2 < S0) after closing
all 4 BLOCKERS most likely traces to λ^K not consuming K. If only λ^H
fires (TIP-driven future-looking arm), the OAL rule has *unbalanced*
tension — only future trust, no past-validation. This degrades tracking
quality, which directly degrades anticipatory HV estimation.

## Files to touch

- `python_refactor/src/algorithms/anticipatory_learning.py` — audit +
  fix target. compute_anticipatory_learning_rate likely needs the λ^K
  arm.
- `python_refactor/src/algorithms/multi_horizon_anticipatory.py` —
  W15-5 wired window_size kwarg into __init__; verify K propagates
  to the rate computation.
- `python_refactor/src/algorithms/kalman_filter.py` — squared residual
  accumulation may live here; if so, expose getter.
- `python_refactor/tests/test_lambda_k_consumption.py` — NEW; ≥ 4
  regression tests per acceptance criteria.

## Acceptance

- compute_anticipatory_learning_rate consumes self.window_size (K)
- λ = (1/2)(λ^H + λ^K) is the formula (verbatim Eq 7.16)
- K=0 collapses to λ = (1/2)(λ^H + 0) — only future arm fires
- K>0 both arms fire — λ^K is normalized sum of KF squared residuals
  over last K periods
- ≥ 4 regression tests covering the cases
- Per-call trace logging of (λ^H, λ^K, λ) sufficient for W16-4 to
  verify both arms fire in production

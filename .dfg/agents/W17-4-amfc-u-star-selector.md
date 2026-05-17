---
id: W17-4
role: code-fixer
name: AMFC selector for u*_{t-1} per thesis Eq 6.42 (W16-2-CARRY-1 + BACKLOG M5 partial)
purpose: "Closes W16-2-CARRY-1. Replace 'first Pareto-front portfolio' proxy with the argmax-EFHV (AMFC) per thesis §6.4 Eq 6.42 as the u*_{t-1} carried across rolling periods."
wave: W17
unit: W17-4
depends_on: []
blocks: [W17-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/experiments/walk_forward.py
    - python_refactor/experiments/oos_evaluator.py
output_contract:
  files:
    - python_refactor/experiments/walk_forward.py
    - python_refactor/experiments/oos_evaluator.py
    - python_refactor/tests/test_amfc_u_star_selector.py
  branch_name: feat/w17-4-amfc-u-star-selector
  acceptance: >
    walk_forward.run_walk_forward selects u*_{t-1} as the argmax-EFHV
    portfolio from the previous period's Pareto front per thesis Eq 6.42,
    NOT the "first portfolio" proxy. Selection is auditable via
    per-period u*_{t-1} index trace. ≥ 3 regression tests covering:
    (1) selection picks argmax-EFHV from the front; (2) tie-breaking
    deterministic; (3) ties / NaN-EFHV fall back to first-portfolio
    with logged warning.
dispatch_instructions: |
  Closes W16-2-CARRY-1 + partial BACKLOG M5.

  Background. W16-2 ships txn-cost integration with u*_{t-1} =
  "first Pareto-front portfolio" as a proxy. Thesis §6.4 Eq 6.42
  specifies the AMFC (Anticipated Maximal Flexible Choice) as the
  argmax over predicted future hypervolume.

  Surgical workflow:

  1. In walk_forward.run_walk_forward's rolling loop, after
     compute_oos_efhv runs for a period, use the per-portfolio
     EFHV contributions (NOT just the mean Ŝ) to select the
     argmax-EFHV portfolio as u*_{t-1} for the NEXT period.

  2. compute_oos_efhv currently returns {efhv_mean, efhv_std}; may
     need to extend to also return per-portfolio EFHV array OR add
     a separate per-portfolio EFHV computation. Choose the lightest
     touch — likely extending oos_evaluator.

  3. Tie-breaking: deterministic via portfolio_index (smallest wins);
     log warning if multiple portfolios tied at argmax.

  4. Fallback: if all per-portfolio EFHV are NaN / 0 (degenerate
     period), fall back to weights[0] (W16-2 proxy) with a logged
     warning.

  5. Per-period u*_{t-1} index trace: add field `u_star_idx` to
     per-period result dict; useful for W18 diagnostics.

  6. Tests at python_refactor/tests/test_amfc_u_star_selector.py:
       - synthetic 5-portfolio Pareto front with known argmax-EFHV →
         selector picks it
       - tie-break determinism (two equal-EFHV portfolios → pick
         smallest index)
       - all-NaN-EFHV fallback → weights[0] + logged warning

  What NOT to do:
    - Don't change SMS-EMOA / operators / anticipatory_learning.
    - Don't change the txn-cost computation (W16-2 stays).
    - Don't add full AMFC pseudocode (per §6.4 with KF-state
      propagation); the OOS-EFHV argmax over the Pareto front is
      sufficient for u*_{t-1} selection in the walk-forward driver.
    - Don't refactor compute_oos_efhv into a class — keep functional.

  PR body MUST echo §6.4 Eq 6.42 verbatim per BACKLOG §6.
---

# W17-4 — AMFC selector for u*_{t-1} per thesis Eq 6.42

Closes W16-2-CARRY-1 + partial BACKLOG M5.

## Thesis grounding

**§6.4 "Estimating the Anticipated Maximal Flexible Choice", Eq (6.42),
p. 133** — verbatim:
> "AMFC selects, from the available essential set of Pareto-flexible
>  decision candidates Û_t^{N*}, the portfolio u_t^{(i*)*} which is
>  predicted to lead to the maximum future expected hypervolume:
>  u_t^{(i*)*} = argmax_{u_t^{(i)*} ∈ Û_t^{N*}} E[Hypv(Ẑ_{t+1}^{(i)*})]"

**§7.3.6 "Average Predicted Future Hypv Along the Evolved SPFs",
p. 171** — verbatim:
> "We also assess whether the EMFCs predicted to lead to maximal future
>  Hypv are indeed those associated with the highest projected future
>  Hypv among the population members."

## Why this matters

W16-2-CARRY-1: the current implementation uses "first Pareto-front
portfolio" as u*_{t-1}. This is arbitrary and can drift the txn-cost
penalty signal: the optimizer is told it's rebalancing from a
portfolio that may not be the one actually selected as the
implemented decision.

Fixing this means the txn-cost penalty (W16-2) penalizes the RIGHT
rebalancing — from the period's actual AMFC, not from an arbitrary
front member.

## Files to touch

- `python_refactor/experiments/walk_forward.py` — AMFC selector
  + per-period trace field
- `python_refactor/experiments/oos_evaluator.py` — likely needs
  per-portfolio EFHV exposed
- `python_refactor/tests/test_amfc_u_star_selector.py` — NEW; ≥ 3 tests

## Acceptance

- u*_{t-1} = argmax-EFHV portfolio per Eq 6.42
- Per-period u_star_idx trace
- Tie-break deterministic
- All-NaN fallback to first-portfolio + warning
- ≥ 3 tests

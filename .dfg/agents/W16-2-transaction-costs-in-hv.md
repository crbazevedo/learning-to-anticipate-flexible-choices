---
id: W16-2
role: code-fixer
name: Integrate transaction costs into anticipatory HV objective (BACKLOG H1)
purpose: "Closes BACKLOG H1. Subtract h(u_t, u*_{t-1}) (brokerage fees per thesis Table 7.1) from anticipatory ROI estimate before computing HV, so the optimizer sees and avoids high-churn portfolios per thesis §7.2 Eqs (7.4)-(7.5)."
wave: W16
unit: W16-2
depends_on: []
blocks: [W16-5]
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
    - python_refactor/src/config/thesis_parameters.py
    - python_refactor/src/portfolio/portfolio.py
    - python_refactor/src/portfolio/portfolio_evaluator.py
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - python_refactor/src/portfolio/portfolio.py
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/experiments/walk_forward.py
    - python_refactor/tests/test_transaction_costs_in_hv.py
  branch_name: feat/w16-2-transaction-costs-in-hv
  acceptance: >
    Transaction-cost function h(u_t, u*_{t-1}) integrated into the
    anticipatory ROI computed by the SMS-EMOA evaluator. Brazilian
    SEC fee schedule per thesis Table 7.1 (p. 144) implemented (proportional
    + fixed by traded-value bracket). Previous-period implemented portfolio
    (u*_{t-1}) threaded into the walk-forward driver so the optimizer
    sees churn costs. ≥ 4 regression tests covering the fee schedule
    + the optimizer-sees-cost integration.
dispatch_instructions: |
  Closes BACKLOG: H1.

  Background. The current code (per BACKLOG H1) declares
  TRANSACTION_COST_RATE=0.001 in thesis_parameters.py but only uses
  it in the wealth-evaluation path (portfolio_evaluator), NOT in the
  optimization objective. The optimizer is therefore BLIND to churn:
  it can pick an EMFC whose realized future Hypv is eaten by trading
  fees, and never know.

  Surgical changes:

  1. Add a `compute_transaction_cost(weights_new, weights_prev, portfolio_value)`
     helper to portfolio.py implementing thesis Table 7.1 brackets:
        traded_value = |weights_new - weights_prev| * portfolio_value, per asset
        sum brokerage fee per bracket per asset
        return total_fee / portfolio_value (i.e., cost as fraction of wealth)

  2. Thread previous-period implemented portfolio u*_{t-1} into the
     SMS-EMOA call site. In walk_forward.py rolling loop, capture the
     implemented portfolio from period t-1 and pass it through to
     run_optimization (default to equal-weight on first period).

  3. In sms_emoa._evaluate_solution (or _compute_objectives), subtract
     transaction cost from the ROI objective:
        ROI_net = ROI - h(u_t, u*_{t-1})
        solution.objectives = [ROI_net, risk]
     (Keep `non_robust_ROI` as gross-of-cost for reporting; only the
     objective passed to NDS + HV is net.)

  4. Tests:
     - Fee schedule brackets match thesis Table 7.1 within rounding
     - Zero churn → zero cost
     - Optimizer sees the cost: identical 2-portfolio pop where one
       requires high churn from u*_{t-1} — verify the lower-churn one
       wins HV contribution
     - Walk-forward chain preserves u*_{t-1} across periods

  What NOT to do:
    - Don't touch anticipatory_learning.py (W16-1).
    - Don't touch the operators (W15-2 — those are the right ones).
    - Don't touch extrema preservation (W16-3).
    - Don't enable txn costs in the wealth-evaluation path (already
      done elsewhere; just integrate into HV objective).

  PR body MUST echo thesis Eq (7.4)-(7.5) + Table 7.1 verbatim per
  BACKLOG §6 grounding discipline.
---

# W16-2 — Integrate transaction costs into anticipatory HV objective

Closes BACKLOG.md items: **H1**.

## Thesis grounding

**§7.2 Eqs (7.4)-(7.5), p. 142** — verbatim:
> "Following the AS-MOO notation,
>  z_t|u*_{t-1} = g(u_t, χ_t) + h(u_t, u*_{t-1}),
>  where:
>  g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T   (7.4)
>  and
>  h(u_t, u*_{t-1}) = (0, h(u_t, u*_{t-1}))^T            (7.5),
>  in which h is a cost function representing all incurring
>  transaction fees and commissions."

The objective vector includes the h-cost component; the AS-MOO solver
sees it in `z_t = g(u_t, χ_t) + h(u_t, u*_{t-1})`. Critically, h enters
the OPTIMIZATION step, not just the wealth-evaluation step.

**§7.2.3 "Brokerage Fees", Table 7.1 (p. 144)** — Brazilian Securities
Commission fee schedule used by the thesis:

| Traded value | Proportional Fee | Fixed Fee |
|---|---|---|
| < 135.07 | 0.0% | 2.70 |
| ≥ 135.08 and < 498.62 | 2.0% | 0.00 |
| ≥ 498.63 and < 1,514.69 | 1.5% | 2.49 |
| ≥ 1,514.70 and < 3,029.38 | 1.0% | 10.06 |
| ≥ 3,029.39 | 0.5% | 25.21 |

**§5.2.4 "Time-Linkage Formulation", p. 105** — formal derivation of
why h enters the objective at the OPTIMIZATION step, not just the
wealth-evaluation step.

## Why this contributes to closing the 9% gap

Without txn costs in HV: the anticipatory rule may keep flipping the
"best" EMFC every period because the optimizer cannot distinguish
high-churn flexible choices from low-churn ones. The resulting
realized future hypervolume (out-of-sample) is then eroded by costs
the optimizer never modeled. This is consistent with the W15-5
direction (S2 still < S0) — the anticipation IS happening but it's
implementing portfolios that look better in the objective space than
they actually realize.

## Files to touch

- `python_refactor/src/portfolio/portfolio.py` — add
  `compute_transaction_cost(weights_new, weights_prev, portfolio_value)`
- `python_refactor/src/algorithms/sms_emoa.py` — subtract cost from ROI
  in _evaluate_solution
- `python_refactor/experiments/walk_forward.py` — thread u*_{t-1}
  through the rolling loop
- `python_refactor/src/config/thesis_parameters.py` — Table 7.1
  brackets as constants
- `python_refactor/tests/test_transaction_costs_in_hv.py` — NEW

## Acceptance

- Brazilian SEC fee schedule per Table 7.1 implemented (≥ 4 brackets)
- Optimizer sees the cost (ROI_net = ROI - h in NDS objective)
- u*_{t-1} threaded across periods (equal-weight on period 1)
- ≥ 4 regression tests
- `non_robust_ROI` preserved as gross-of-cost for reporting (only
  objective vector uses net)

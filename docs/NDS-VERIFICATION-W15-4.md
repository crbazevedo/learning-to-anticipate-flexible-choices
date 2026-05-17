# NDS Verification — BACKLOG M15 closure (W15-4)

*Authored 2026-05-17. Closes BACKLOG M15.*

## Verdict

✅ **MATCHES THESIS — no code change required.**

Non-dominated sorting in the Python port operates over **deterministic
means** (`solution.P.ROI`, `solution.P.risk`), exactly as thesis §6.5.1
prescribes. No deviation found.

## Audit evidence

### Code site 1: `_evaluate_solution` (sms_emoa.py:255-266)

```python
def _evaluate_solution(self, solution: Solution, data: Dict[str, Any]):
    """Evaluate solution and update objectives."""
    portfolio = solution.P
    if hasattr(solution.P, 'kalman_state') and solution.P.kalman_state is not None:
        from .kalman_filter import kalman_update
        measurement = np.array([portfolio.ROI, portfolio.risk])
        kalman_update(solution.P.kalman_state, measurement)
    # Store objectives
    solution.objectives = [portfolio.ROI, portfolio.risk]
    ...
```

`portfolio.ROI` and `portfolio.risk` are set by `Portfolio.compute_efficiency`
(portfolio.py:264-274):

```python
@classmethod
def compute_efficiency(cls, portfolio: 'Portfolio'):
    portfolio.ROI = portfolio.non_robust_ROI = cls.compute_ROI(portfolio, cls.mean_ROI)
    portfolio.risk = portfolio.non_robust_risk = cls.compute_risk(portfolio, cls.covariance)
    ...
```

Where:
- `cls.mean_ROI` = `np.mean(returns_data, axis=0)` (portfolio.py:107) —
  empirical **mean** vector
- `cls.covariance` = `(centered_data.T @ centered_data) / (n - 1)`
  (portfolio.py:166) — empirical **mean** covariance matrix
- `compute_ROI(portfolio, mean_ROI) = weights @ mean_ROI` — scalar **mean** ROI
- `compute_risk(portfolio, cov) = sqrt(weights @ cov @ weights)` — scalar
  **mean** risk (well, std-dev of mean covariance; but it's deterministic
  given weights + cov)

So `solution.P.ROI` and `solution.P.risk` are deterministic functions
of the weight vector and the empirical-mean parameters. **No stochastic
distribution enters the dominance comparison.**

### Code site 2: `_dominates` (sms_emoa.py:377-394)

```python
def _dominates(self, solution1: Solution, solution2: Solution) -> bool:
    """Check if solution1 dominates solution2."""
    roi1, risk1 = solution1.P.ROI, solution1.P.risk
    roi2, risk2 = solution2.P.ROI, solution2.P.risk
    ...
```

Direct use of `solution.P.ROI` / `solution.P.risk` — the deterministic
means. Standard Pareto dominance: better-or-equal in both objectives + 
strictly better in at least one.

### Code site 3: `_fast_non_dominated_sort` (sms_emoa.py:359-374)

Calls `_dominates` pairwise. Sorts purely on the deterministic-mean
basis. No KF-state or stochastic-distribution awareness in the dominance
relation itself.

## Thesis grounding (verbatim, §6.5.1 p. 134)

> "we argue that, since we are already able to handle the estimated
>  uncertainty (see section 6.1) and to combine the learned predictive
>  correlation into the computation of E[Δ_S] (see section 6.3), there
>  is little need to incorporate uncertainty awareness directly into
>  the dominance relation. Therefore, our proposed non-dominated sorting
>  procedures are executed in terms of the deterministic Pareto Dominance
>  **over the estimated means of the random objective vectors**"

Cross-reference:
- **Theorem 6.5.1 (p. 134)**: stochastic dominance theorem providing the
  theoretical justification for the deterministic-mean shortcut.
- **§6.6 contribution #3 (p. 136)**: "It showed a way to incorporate
  uncertainty awareness in the computation of the expected anticipatory
  S-Metric contributions"
- **Pseudocode 8 lines 8-9 (p. 135)**: explicit ANTICIPATORY REDUCE
  / NDS class assignment over the mean vectors `{ẑ_t^(i)} ← E[ẑ_t^(i)
  | ẑ_{t+1:t+H-1}^(i)]`.

## Summary

| Code | Thesis | Match? |
|---|---|---|
| `_dominates` uses `solution.P.ROI, solution.P.risk` (deterministic means) | §6.5.1 NDS over **estimated means** | ✅ |
| `Portfolio.compute_efficiency` populates means from `mean_ROI` + `covariance` (empirical means) | §6.1 + §6.3 (uncertainty handled separately) | ✅ |
| `_fast_non_dominated_sort` runs pairwise `_dominates` to assign Pareto ranks | §6.5.1 dominance-over-means + Pseudocode 8 | ✅ |

**Closes BACKLOG M15** — no code change required.

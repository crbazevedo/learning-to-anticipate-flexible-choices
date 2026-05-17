# Cross-validation K — expected HV contribution / stochastic Δ_S

*Generated 2026-05-17 by W20-5. Closes operator check K.*

## Verdict

⚠️ **STRUCTURAL DIVERGENCE** — v2 uses a **closed-form** Gaussian
expectation while Python uses **Monte Carlo** sampling. Both estimate
the same quantity in expectation but via different methods. v2 also
multiplies by a `stability` factor that Python does NOT.

## Code-level audit

### v2 `compute_stochastic_Delta_S_front_id` (legacy-cpp-v2/source/asms_emoa.cpp:380+)

For each portfolio i in the Pareto front (sorted by ROI):
```cpp
struct stochastic_params {
    conditional_mean_ROI = ROI
    conditional_var_ROI = (1 - corr²) * var_ROI    // Gaussian-conditional
    conditional_mean_risk = risk
    conditional_var_risk = (1 - corr²) * var_risk
}

// Per portfolio i (interior), via mean_delta_product (line 297):
mean_delta_ROI  = sw_1.conditional_mean_ROI - sw_2.conditional_mean_ROI
mean_delta_risk = sw_0.conditional_mean_risk - sw_1.conditional_mean_risk
var_delta_ROI   = sw_1.conditional_var_ROI + sw_2.conditional_var_ROI
var_delta_risk  = sw_0.conditional_var_risk + sw_1.conditional_var_risk

delta_S = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI)
          / (var_delta_ROI + var_delta_risk)

// Multiply by stability:
Pareto_front[i]->Delta_S = delta_S * Pareto_front[i]->stability
```

This is a CLOSED-FORM Gaussian expectation using:
- `E[XY] = E[X]·E[Y] + cov(X,Y)` expansion
- Conditional variance under partial-info Gaussian assumption
- Stability-weighted (per-portfolio robustness multiplier)

For first and last portfolios, additional terms use `cov(samples)` directly
(line 418-421, 442-444).

### Python `_compute_expected_future_hypervolume` (sms_emoa.py:739+)

```python
total_hypervolume = 0.0
for _ in range(1000):       # MC samples
    future_front = []
    for solution in pareto_front:
        kalman_prediction(solution.P.kalman_state)
        future_roi = x_next[0]
        future_risk = x_next[1]
        temp = Solution(...)
        temp.P.ROI = future_roi
        temp.P.risk = future_risk
        future_front.append(temp)

    # 2D HV against z_ref = (R1, R2):
    sorted_front = sorted(future_front, key=ROI)
    sample_hv = sum( (roi[i] - prev_roi) * (R2 - risk[i])
                      for i in front )
    total_hypervolume += sample_hv

return total_hypervolume / 1000
```

Pure Monte Carlo. No stability multiplier. 1000 samples per call.

## Differences

| Aspect | v2 | Python |
|---|---|---|
| Method | Closed-form Gaussian expectation | 1000-sample Monte Carlo |
| Cost per call | O(N) for N portfolios | O(1000 × N) |
| Determinism | Deterministic given inputs | Stochastic (depends on KF random sampling) |
| Stability multiplier | YES (per-portfolio) | NO |
| Assumes Gaussian conditional? | YES | YES (via KF predict) |
| Returns per-portfolio Δ_S? | YES (one per front member) | NO (single aggregate HV) |

## Behavioral expectation

In expectation, both should converge to E[Σ HV_per_portfolio] under
Gaussian KF predict. But:
- v2's `stability` factor down-weights unstable portfolios → effectively
  a robustness-aware HV (favors stable Pareto-front members)
- Python's MC gives an UNWEIGHTED expected HV

The v2 stability-weighting could meaningfully differ from Python's:
unstable portfolios with high HV contribution get penalized in v2 but
fully credited in Python. In W17-5 production where MANY portfolios
have similar HV contributions, the weighting might matter.

## v2 `stability` formula recap (from portfolio.cpp:595)

```cpp
stability = 1.0 / (1.0 + pow(ROI_unseen - P.ROI, 2.0)
                       + pow(risk_unseen - P.risk, 2.0))
```

So stability ∈ (0, 1]; max=1 when prediction matches observed perfectly.

## Verdict per W18 matrix

⚠️ **DISAGREE STRUCTURALLY** — different methods (closed-form vs MC) +
v2 has stability weighting that Python lacks.

In expectation (large MC samples; ignoring stability), both should
agree. But the stability multiplier is a NEW DIFFERENCE that may
contribute to the W17-5 gap.

## NEW finding for W17-5 chain

The v2 stability-weighting in Δ_S is a previously-undocumented
divergence:
- v2's selection pressure favors STABLE Pareto-front members (low
  prediction error)
- Python's selection pressure ignores stability

In a saturation regime (high prediction uncertainty), v2's
stability-weighting would systematically prefer the most-stable
portfolios — which may correlate with anticipation-friendly portfolios
(those whose KF predictions consistently match observations).

This is potentially another contributing factor (call it **Reading F**):
**stability-weighting in Δ_S** that v2 has and Python doesn't.

## Bug count update

| # | Side | Where | Severity |
|---|---|---|---|
| 5 (NEW) | Python | `_compute_expected_future_hypervolume` lacks v2's stability multiplier | On-headline (potential contributor to remaining 4.04% gap after W20-1) |

## Verdict per W18 matrix

⚠️ **DISAGREE STRUCTURALLY** with potential behavioral impact (Reading F candidate).

## Reproducing

Code-read only. Execution-level cross-check would require building
`expected_hv_driver.cpp` against v2 + Python equivalent with
stability injection. Deferred to W21.

## Next steps

- W21+: implement `use_v2_stability_weighting=True` Python flag
- Re-run W17-5 smoke with W20-1 v2-rate + W21 v2-stability combined
- Quantify Reading F contribution to remaining gap

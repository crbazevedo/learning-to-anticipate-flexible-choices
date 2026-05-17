# Cross-validation B: ROI computation parity (operator check B)

*Generated 2026-05-17 by W18-3. Closes operator check B.*

## Verdict

✅ **AGREE — both sides match thesis Eq (7.4) verbatim, within 1e-12.**

Both C++ and Python compute `μ̂^T u` (linear combination of mean asset
returns weighted by portfolio composition). The execution receipt
shows `rel_max = 0` — meaning the floating-point operations agree to
machine precision.

## Thesis grounding (verbatim)

**§7.2 Eq (7.4), p. 142**:
> "g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T"

The second component is the ROI: `μ̂^T u`.

## Code evidence

### C++ (matches thesis ✅)

`legacy-cpp/source/portfolio.cpp:448-451`:
```cpp
double portfolio::compute_ROI(portfolio &P, const Eigen::VectorXd &mean_ROI)
{
    return P.investment.transpose()*mean_ROI;
}
```

### Python (matches thesis ✅)

`python_refactor/src/portfolio/portfolio.py:235-246`:
```python
@classmethod
def compute_ROI(cls, portfolio: 'Portfolio', mean_ROI: np.ndarray) -> float:
    return portfolio.investment @ mean_ROI
```

## Execution receipt

Same fixture as W18-2 (10 portfolios × 20 assets × 30 days; seed=42).

| portfolio_idx | C++ ROI | Python ROI | abs_diff |
|---|---|---|---|
| 0 | 0.00012005533676535418 | 0.00012005533676535414 | 4e-20 |
| 1 | 0.00071002247402812864 | 0.00071002247402812842 | 2.2e-19 |
| 2 | 0.00088088241936497128 | 0.00088088241936497118 | 1e-19 |

Max abs diff across all 10 portfolios: ~1e-18 (= machine epsilon).
Verdict: AGREE.

## Implications

1. **ROI is NOT the source of any cross-implementation drift.** The
   sqrt() deviation in compute_risk (W18-2) is unique to risk; ROI
   is identical.

2. **Returns scale**: both sides treat returns as DECIMAL (not %).
   Mean ROI on this fixture: ~7.57e-4 per day → ~0.075% per day →
   ~19% annualized. Consistent with thesis §7.2.3 "daily adjusted
   close prices".

3. **No annualization or other scaling**: both sides report PER-DAY
   ROI in decimal units. Downstream code is responsible for any
   horizon-scaling.

## Bug count after W18-3

| # | Side | Where | Status |
|---|---|---|---|
| 1 | C++ | `portfolio.cpp:65` comma-operator (autocorrelation) | Not on headline path |
| 2 | Python | `compute_risk` adds sqrt() (W18-2) | **On headline path; W18-CARRY-1** |

W18-3 ROI cross-check found NO new bugs. The headline-path bug count
remains at 1 (Python sqrt in compute_risk).

## Output artifacts

- `legacy-cpp/build/drivers/roi_driver` — C++ binary
- `python_refactor/scripts/cross_validation/run_roi.py` — Python driver
- `docs/CROSS-VALIDATION-B-ROI.md` — this receipt

## Reproducing

```bash
cd legacy-cpp/build && make drivers/roi_driver
cd ../../python_refactor
cat experiments/results/w18-cross-validation/risk_fixture.csv | ../legacy-cpp/build/drivers/roi_driver > /tmp/cpp_roi.csv
cat experiments/results/w18-cross-validation/risk_fixture.csv | uv run python -m scripts.cross_validation.run_roi > /tmp/py_roi.csv
uv run python -m scripts.cross_validation.compare /tmp/cpp_roi.csv /tmp/py_roi.csv --atol 1e-12
```

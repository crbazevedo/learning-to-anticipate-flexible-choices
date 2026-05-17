# Cross-validation A: risk computation parity (BACKLOG operator check A)

*Generated 2026-05-17 by W18-2. Closes operator check A.*

## Verdict

🔴 **DISAGREE — PYTHON deviates from thesis Eq (7.4) by introducing sqrt().**

The Python port computes `sqrt(u^T Σ u)` (standard deviation), while the
C++ reference and thesis Eq (7.4) compute `u^T Σ u` (variance, the
quadratic form). The ratio is EXACTLY `py = sqrt(cpp)` within 1e-15.

## Thesis grounding (verbatim)

**§7.2 Eq (7.4), p. 142**:
> "g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T"

The first component is the variance (quadratic form). The thesis is
unambiguous: no sqrt() in the formula.

## Code evidence

### C++ (matches thesis ✅)

`legacy-cpp/source/portfolio.cpp:453-456`:
```cpp
double portfolio::compute_risk(portfolio &P, const Eigen::MatrixXd &covariance)
{
    return P.investment.transpose()*covariance*P.investment;
}
```

### Python (deviates ❌)

`python_refactor/src/portfolio/portfolio.py:249-262`:
```python
@classmethod
def compute_risk(cls, portfolio: 'Portfolio', covariance: np.ndarray) -> float:
    variance = portfolio.investment @ covariance @ portfolio.investment
    return np.sqrt(max(variance, 0.0))  # Ensure non-negative
```

The docstring even says "Compute portfolio risk (**standard deviation**)" —
confirming the Python port deliberately deviates from the thesis text.

## Execution receipt

Fixture: 10 portfolios × 20 assets × 30 days of synthetic returns
(seed=42, Dirichlet portfolios, Gaussian returns ~ N(0.001, 0.01)).

First 5 portfolios:

| portfolio_idx | C++ risk (variance) | Python risk (std-dev) | sqrt(C++) | ratio (Python / sqrt(C++)) |
|---|---|---|---|---|
| 0 | 6.449844e-06 | 2.539654e-03 | 2.539654e-03 | 1.000000 |
| 1 | 5.817764e-06 | 2.412004e-03 | 2.412004e-03 | 1.000000 |
| 2 | 1.196924e-05 | 3.459658e-03 | 3.459658e-03 | 1.000000 |
| 3 | 6.676441e-06 | 2.583881e-03 | 2.583881e-03 | 1.000000 |
| 4 | 9.276113e-06 | 3.045671e-03 | 3.045671e-03 | 1.000000 |

**Ratio = 1.000000 for all rows** → confirms `py = sqrt(cpp)` exactly.

## Impact analysis

### On the W17-5 -8.72% gap

The risk is the second component of the SMS-EMOA objective vector
`[ROI, risk]` used in dominance comparisons + HV computation. With
the sqrt() applied:

1. **HV reference point** is `z_ref = (0.2, 0.0)` — risk_max = 0.2.
   On variance scale (C++): ~1e-5 typical → far below 0.2 → ENTIRE
   front fits inside HV box.
   On std-dev scale (Python): ~3e-3 typical → still below 0.2 →
   front also fits.
   So z_ref = 0.2 was chosen to accommodate std-dev (Python), NOT
   variance (C++).

2. **Dominance ordering** is invariant under monotonic transform
   (sqrt is monotonic) → Pareto fronts are the SAME set under
   variance vs std-dev. So dominance + non-dominated sorting is
   unaffected.

3. **HV contribution** is sensitive to the metric scale. Areas grow
   when one axis is squashed (std-dev compresses high-variance
   outliers; variance amplifies them). HV contributions may differ
   materially.

4. **TIP & KF**: the Kalman state uses `[ROI, risk]` as the
   measurement vector. If risk is std-dev (Python) vs variance (C++),
   the KF covariance estimates are computed on different scales →
   different residual magnitudes → different λ^K (W17-5 finding!) →
   different anticipation strength.

### Hypothesis on the TIP saturation finding (W17-5-CARRY-1 RESMOKE)

The W17-5 trace showed `λ^K mean = 7.47e-7` because KF residuals on
**std-dev-scale risk** are tiny. If we switch to variance-scale risk,
the residuals would be 6 orders of magnitude SMALLER (because
sqrt(1e-5) = 3e-3 → variance residual ~3e-3 squared ~ 1e-5; std-dev
residual on 3e-3 scale is itself ~3e-4 → squared ~ 1e-7).

So the std-dev choice actually MAKES residuals smaller, NOT larger.
Switching to variance might make λ^K SMALLER not larger — needs
re-investigation.

## Verdict per W18-2 matrix

| Outcome | Reading |
|---|---|
| AGREE | n/a |
| DISAGREE-PYTHON-WRONG | **CURRENT VERDICT** — Python deviates from thesis Eq (7.4) by adding sqrt(); C++ matches thesis verbatim |
| DISAGREE-CPP-WRONG | n/a |
| DISAGREE-AMBIGUOUS | n/a |

## Honest scar: the deviation may be intentional

Pre-existing Python comments imply the std-dev choice was deliberate
(docstring says "standard deviation"). Possible reasons:
- Plotting / interpretation: std-dev shares units with ROI
- Numerical stability: smaller absolute values
- Match a *different* version of the thesis (paper vs PhD thesis may
  differ; verify)

**But the thesis Eq (7.4) text is unambiguous** — variance, not
std-dev. Either:
1. Fix Python to match thesis (rationalize ROI/risk in std-dev space
   downstream)
2. Document the deliberate deviation + verify downstream code
   accommodates it correctly

## Carry-forward for W18-5 + W19+

- **W18-CARRY-1** (THIS finding): Python `compute_risk` adds sqrt() not in thesis Eq (7.4).
  Decide: fix to match thesis, OR document deliberate deviation + verify
  downstream consistency.
- W18-CARRY-2 (filed in W18-1): C++ `portfolio::sample_autocorrelation` has
  comma-operator bug (`(v - avg, 2.0)*(v - avg, 2.0) = 4.0`). Not on the
  ASMS/SMS headline path, but documented.

## Output artifacts

- `legacy-cpp/build/drivers/risk_driver` — C++ binary (Apple silicon)
- `python_refactor/scripts/cross_validation/run_risk.py` — Python driver
- `python_refactor/experiments/results/w18-cross-validation/risk_fixture.csv` — shared fixture (seed=42, 10×20×30)
- This document — the receipt

## Reproducing

```bash
# Build C++ driver (once)
cd legacy-cpp/build && make drivers/risk_driver

# Generate fixture
cd ../../python_refactor && uv run python -c "from scripts.cross_validation.fixtures import write_risk_fixture_to; write_risk_fixture_to('experiments/results/w18-cross-validation/risk_fixture.csv', seed=42, n_portfolios=10, n_assets=20, n_days=30)"

# Run both drivers
cat experiments/results/w18-cross-validation/risk_fixture.csv | ../legacy-cpp/build/drivers/risk_driver > /tmp/cpp_risk.csv
cat experiments/results/w18-cross-validation/risk_fixture.csv | uv run python -m scripts.cross_validation.run_risk > /tmp/py_risk.csv

# Compare
uv run python -m scripts.cross_validation.compare /tmp/cpp_risk.csv /tmp/py_risk.csv --atol 1e-9
```

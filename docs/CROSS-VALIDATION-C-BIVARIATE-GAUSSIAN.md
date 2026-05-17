# Cross-validation C: bivariate Gaussian inputs (mean + covariance)

*Generated 2026-05-17 by W19-1. Closes operator check C.*

## Verdict

✅ **AGREE** within 1e-12 (machine precision).

Both C++ and Python compute identical mean vectors and covariance
matrices from the same fixture. No drift in the Gaussian-input
construction step.

## Code evidence

### C++

`legacy-cpp/source/portfolio.cpp:298-309`:
```cpp
Eigen::VectorXd portfolio::estimate_assets_mean_ROI(const Eigen::MatrixXd &returns_data)
{
    return returns_data.colwise().mean();
}

void portfolio::estimate_covariance(const Eigen::VectorXd &mean_ROI, const Eigen::MatrixXd &returns_data, Eigen::MatrixXd &covariance)
{
    covariance.resize(portfolio::available_assets_size,portfolio::available_assets_size);
    covariance = (returns_data.rowwise() - mean_ROI.transpose()).transpose()*(returns_data.rowwise() - mean_ROI.transpose())/(returns_data.rows()-1.0);
}
```

### Python

Mean (numpy default) + ddof=1 sample covariance — matches.

## Execution receipt

Fixture: 20-asset × 30-day synthetic returns (W18 standard fixture).

| Output | Shape | abs_max diff | rel_max diff | Verdict |
|---|---|---|---|---|
| mean vector | (20,) | < 1e-17 | 0 | AGREE |
| covariance matrix | (20, 20) | < 1e-17 | 0 | AGREE |

## Implications

**Upstream of cross-check D (KF)**: the inputs feeding the Kalman
filter ARE identical on both sides. So any KF disagreement in W19-2
will be in the KF itself, not in its inputs.

## Bug count after W19-1

Unchanged: 1 C++ off-headline + 1 Python on-headline (from W18).

## Reproducing

```bash
cd legacy-cpp/build && make drivers/bivariate_gaussian_driver
cd ../../python_refactor
cat experiments/results/w18-cross-validation/risk_fixture.csv | ../legacy-cpp/build/drivers/bivariate_gaussian_driver > /tmp/cpp_bg.csv
cat experiments/results/w18-cross-validation/risk_fixture.csv | uv run python -m scripts.cross_validation.run_bivariate_gaussian > /tmp/py_bg.csv
uv run python -m scripts.cross_validation.compare /tmp/cpp_bg.csv /tmp/py_bg.csv --atol 1e-12
```

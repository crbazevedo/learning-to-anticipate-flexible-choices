"""
W18-1 cross-validation framework.

Mutual-skepticism comparison: same input fixture fed to BOTH the C++
legacy reference (compiled via legacy-cpp/build/Makefile) and the
Python port; outputs compared with abs+rel tolerance + scale ratio.

Per operator directive: "There is the possibility of the C++ reference
implementation also being wrong, so do not merely treat it as the
oracle." → comparison flags ANY discrepancy without privileging either
side.

Per-check usage pattern:

1. Generate fixture (deterministic given seed):
       from scripts.cross_validation.fixtures import build_risk_fixture
       fixture_csv = build_risk_fixture(seed=42)

2. Run C++ driver:
       legacy-cpp/build/drivers/risk_driver < fixture.csv > cpp_out.csv

3. Run Python equivalent:
       python -m scripts.cross_validation.run_risk < fixture.csv > py_out.csv

4. Compare:
       from scripts.cross_validation.compare import compare_csvs
       report = compare_csvs(cpp_path, py_path, atol=1e-9, rtol=1e-9)

The output `report` is a structured dict + markdown receipt suitable
for the docs/CROSS-VALIDATION-<CHECK>.md format.
"""

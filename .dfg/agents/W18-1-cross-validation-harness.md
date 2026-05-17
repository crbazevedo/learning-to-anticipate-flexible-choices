---
id: W18-1
role: code-fixer
name: Cross-validation harness scaffold (C++ build + per-check driver pattern + fixtures)
purpose: "Build a reproducible cross-validation framework: same input fixtures fed to BOTH the C++ legacy reference and the Python port; outputs compared with tolerance. Closes W18 prerequisite for A-N checks per the operator directive."
wave: W18
unit: W18-1
depends_on: []
blocks: [W18-2, W18-3, W18-4]
governance_tier: VT1
sized: L
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - legacy-cpp/README.md
    - legacy-cpp/source/portfolio.cpp
    - legacy-cpp/source/main.cpp
    - python_refactor/src/portfolio/portfolio.py
output_contract:
  files:
    - legacy-cpp/build/Makefile
    - legacy-cpp/build/README-BUILD-ADAPTER.md
    - python_refactor/scripts/cross_validation/__init__.py
    - python_refactor/scripts/cross_validation/fixtures.py
    - python_refactor/scripts/cross_validation/compare.py
    - python_refactor/tests/test_cross_validation_harness.py
  branch_name: feat/w18-1-cross-validation-harness
  acceptance: >
    Cross-validation harness scaffold ships with: (1) legacy-cpp/build/
    Makefile that builds the C++ source on macOS Apple clang 17 + C++14
    + Eigen 3.4 via documented adapter patches; (2) fixed input fixtures
    (synthetic portfolios + returns matrix + seed) reproducible across
    runs; (3) Python comparison framework with configurable tolerance +
    scale-check; (4) per-check C++ driver pattern with example driver
    proving the framework works; (5) ≥ 3 regression tests for the
    harness itself.
dispatch_instructions: |
  Closes W18 prerequisite for the 14 cross-checks (A-N) per operator
  directive in session.

  The operator's caveat: "There is the possibility of the C++ reference
  implementation also being wrong, so do not merely treat it as the
  oracle." → harness must support mutual-skepticism reporting; both
  sides log their result; the comparison flags discrepancies without
  privileging either side.

  Surgical workflow:

  1. C++ BUILD ADAPTER (legacy-cpp/build/Makefile):
     - Adapter patches needed for modern toolchain (already validated):
       a. Eigen include path: `<Eigen/Eigen/Dense>` → `<Eigen/Dense>`
       b. Narrowing conversion: `2.0*portfolio::window_size` → `2u*portfolio::window_size`
       c. `-std=c++14` (random_shuffle removed in C++17)
       d. `-DEIGEN_DONT_VECTORIZE` for stability on Apple silicon
     - Document patches in legacy-cpp/build/README-BUILD-ADAPTER.md
       as PURELY mechanical build adapters (no algorithm change).
     - DO NOT modify legacy-cpp/source or legacy-cpp/headers directly —
       use sed in the build target to create build/src_patched + build/
       headers_patched.

  2. PER-CHECK DRIVER PATTERN (legacy-cpp/build/drivers/):
     - Each cross-check C++ driver: includes the relevant header, reads
       input fixture from stdin (CSV / JSON), runs the function under
       test, emits result to stdout as CSV with stable header.
     - One example driver (e.g., risk_driver.cpp) demonstrates pattern.
     - Makefile target: `make driver-<name>` builds the driver.

  3. PYTHON COMPARISON FRAMEWORK (python_refactor/scripts/cross_validation/):
     - fixtures.py: generates fixed-seed synthetic portfolios + returns
       matrices; serializes to CSV. Same fixture file consumed by both
       sides.
     - compare.py: takes 2 result CSVs (cpp_out, py_out), compares
       columns with abs-tol + rel-tol + scale-ratio; emits a markdown
       diff report. Does NOT privilege either side; flags ANY discrepancy
       for human review.

  4. EXAMPLE END-TO-END (W18-2 will fully use this):
     - Generate fixture: scripts/cross_validation/fixtures.py
     - Run C++ driver: ./build/risk_driver < fixture.csv > cpp_risk.csv
     - Run Python: scripts/cross_validation/run_python_risk.py < fixture.csv > py_risk.csv
     - Compare: scripts/cross_validation/compare.py cpp_risk.csv py_risk.csv
       --tolerance 1e-9 --output diff.md

  5. TESTS (python_refactor/tests/test_cross_validation_harness.py):
     - Fixture generation is deterministic (same seed → same CSV)
     - Compare framework: identical inputs → empty diff
     - Compare framework: tolerance-violating inputs → flagged
     - Build adapter: cpp source patched correctly (file count check)

  Honest scars to expect:
   - Eigen 5.x is too modern for legacy code; pinned to 3.4.0 (download
     in Makefile or vendored as .tar.gz adapter dep)
   - Build tested only on Apple silicon + clang 17 + Eigen 3.4
   - C++ random number streams use std::rand by default; cross-check
     must seed BOTH sides explicitly for parity

  What NOT to do:
   - Don't modify legacy-cpp/source/* or legacy-cpp/headers/* directly
   - Don't run any actual cross-check (those are W18-2..W18-4 units)
   - Don't decide A vs B (mutual-skepticism reporting only)
   - Don't ship Reading A or B determination (W18-5 synthesis)

  PR body MUST cite the operator's mutual-skepticism caveat verbatim
  and document the 4 mechanical build adapters per BACKLOG §6.
---

# W18-1 — Cross-validation harness scaffold

Closes W18 prerequisite for the 14 cross-checks (A-N) per the operator
directive: "we need to investigate everything... There is the possibility
of the C++ reference implementation also being wrong, so do not merely
treat it as the oracle."

## Operator directive (verbatim, from session)

> "We need to investigate everything. Agree with starting with option 2.
>  Additional checks: A through N... Check the C++ implementation,
>  not only reading the code, but running it isolated for same set of
>  portfolios/assets data/period, etc. ... There is the possibility of
>  the C++ reference implementation also being wrong, so do not merely
>  treat it as the oracle."

## Mutual-skepticism reporting

Neither the C++ legacy nor the Python port is treated as oracle. The
harness:
1. Logs both sides' results verbatim
2. Flags ANY discrepancy beyond configured tolerance
3. Reports discrepancies as findings requiring human review (Reading A
   = Python bug; Reading B = C++ bug; Reading C = numerical noise
   tolerable)

## C++ build adapters (4 mechanical, no behavior change)

1. **Eigen include path**: `<Eigen/Eigen/Dense>` → `<Eigen/Dense>`
   (legacy code vendored Eigen at non-standard path; modern install
   exposes `Eigen/Dense`)
2. **Narrowing conversion**: `2.0*portfolio::window_size` →
   `2u*portfolio::window_size` (Eigen 3.4 + C++14 disallow implicit
   `double → int` for matrix indices)
3. **`-std=c++14`** (legacy uses `std::random_shuffle` removed in C++17)
4. **`-DEIGEN_DONT_VECTORIZE`** (stability on Apple silicon)

All patches preserve algorithm semantics. Documented at
`legacy-cpp/build/README-BUILD-ADAPTER.md`.

## Per-check driver pattern

Each W18-2..W18-N+ check will follow this pattern:
```
legacy-cpp/build/drivers/<check>_driver.cpp        # C++ harness
python_refactor/scripts/cross_validation/run_<check>.py  # Python harness
python_refactor/scripts/cross_validation/<check>_fixture.csv  # Shared input
docs/CROSS-VALIDATION-<CHECK>.md                   # Receipt
```

## Files

- `legacy-cpp/build/Makefile` — NEW; build target per cross-check driver
- `legacy-cpp/build/README-BUILD-ADAPTER.md` — NEW; documents adapters
- `python_refactor/scripts/cross_validation/{__init__,fixtures,compare}.py` — NEW
- `python_refactor/tests/test_cross_validation_harness.py` — NEW; ≥ 3 tests

## Acceptance

- C++ build succeeds (10 of 10 .o files; one example driver linked)
- Python compare framework: identical inputs → 0 discrepancies
- Python compare framework: violating inputs → discrepancies flagged
- ≥ 3 regression tests
- Both sides demoed end-to-end with example check
- Build adapter doc with operator's caveat echoed

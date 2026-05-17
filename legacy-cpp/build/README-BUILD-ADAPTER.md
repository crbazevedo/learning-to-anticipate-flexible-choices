# legacy-cpp/build — W18-1 build adapter

This directory provides a reproducible build of the frozen C++ legacy
implementation (`legacy-cpp/source/` + `legacy-cpp/headers/`) on a
modern toolchain (Apple silicon, Apple clang 17, C++14, Eigen 3.4).

The C++ source is read-only — we never modify `legacy-cpp/source/` or
`legacy-cpp/headers/` directly. Instead, this Makefile creates a
patched copy in `src_patched/` and `headers_patched/` and applies
four PURELY MECHANICAL build adapters before compilation.

## Operator caveat (verbatim)

> "There is the possibility of the C++ reference implementation also
>  being wrong, so do not merely treat it as the oracle."

The build adapters below are documented mechanical patches required
ONLY for the modern toolchain to ingest 2013-era code. None of them
changes algorithm semantics. The cross-validation harness (W18-2..)
applies mutual-skepticism comparison — both sides report; neither is
privileged.

## The four build adapters

### 1. Eigen include path

| Legacy | Adapter |
|---|---|
| `#include <Eigen/Eigen/Dense>` | `#include <Eigen/Dense>` |

The legacy code vendored Eigen at a non-standard path (`Eigen/Eigen/Dense`
inside the vendored copy). Modern Eigen installs expose `Eigen/Dense`
at the standard top-level include.

Sed pattern: `s|<Eigen/Eigen/|<Eigen/|g`

### 2. Double → unsigned int narrowing

| Legacy | Adapter |
|---|---|
| `mat(t - (tr_period - 2.0*portfolio::window_size), i)` | `mat(t - (tr_period - 2u*portfolio::window_size), i)` |

Eigen 3.4 (with C++14) refuses implicit `double → Index` conversions
for matrix-element accessors. The legacy code uses `2.0 *
portfolio::window_size` (double) as a matrix index in 17 sites; the
adapter changes the constant to `2u` (unsigned int) so the resulting
expression matches Eigen's `Index` type.

Sed pattern: `s|2\.0\*portfolio::window_size|2u*portfolio::window_size|g`

### 3. `-std=c++14` (NOT c++17)

The legacy code uses `std::random_shuffle` which was deprecated in
C++14 and **removed** in C++17. We compile with `-std=c++14` to
preserve the symbol. If a future port wants C++17, the call sites
need to migrate to `std::shuffle` with an explicit RNG.

Sed pattern: none (compiler flag only).

### 4. `-DEIGEN_DONT_VECTORIZE`

Apple silicon (M-series) NEON vectorization in Eigen 3.4 has known
edge cases. Disabling vectorization gives stable, predictable
floating-point output across runs — critical for cross-validation
parity (any vectorization-driven reordering would muddy comparisons
even when the algorithm is identical).

If a future investigation needs vectorized C++ output for speed,
remove this flag and document any resulting parity drift.

## Build

```bash
cd legacy-cpp/build

# One-time: download Eigen 3.4.0 (≈ 2 MB tarball; cached)
make eigen

# Apply build adapters to source + headers (idempotent)
make patches

# Compile all 10 source files to objs/*.o
make objs

# Build per-check cross-validation drivers (W18-2+)
make drivers
```

## Build receipt (2026-05-17, Apple M-series + clang 17)

Verified: all 10 source files compile to .o:

```
aprendizado_operadores.o  kalman_filter.o      mvtnorm.o
nsga2.o                   operadores_cruzamento.o  operadores_mutacao.o
operadores_selecao.o      portfolio.o          statistics.o
utils.o
```

## Adding a new cross-check driver

Per the W18-1 driver pattern, each cross-check ships:

```
legacy-cpp/build/drivers/<check>_driver.cpp      # C++ harness
python_refactor/scripts/cross_validation/run_<check>.py  # Python harness
python_refactor/scripts/cross_validation/<check>_fixture.csv  # Shared input
docs/CROSS-VALIDATION-<CHECK>.md                  # Receipt
```

The C++ driver:
1. Reads input fixture from stdin (CSV)
2. Constructs the relevant data structures (portfolio, KF state, etc.)
3. Calls the function under test (e.g., `portfolio::compute_risk`)
4. Emits the result to stdout as CSV with stable header

The Python driver mirrors this exactly so the comparison framework
can diff the two CSV outputs.

## Cross-platform notes

This Makefile is tested on:
- **macOS 14+ (Apple silicon)** with Apple clang 17 + Eigen 3.4 + Boost (via Homebrew)

For other platforms (Linux, x86_64, etc.) the includes may differ:
- Linux: replace `BOOST_INC = /opt/homebrew/include` with `/usr/include`
- Older gcc: may not need `-DEIGEN_DONT_VECTORIZE`

The build is deliberately stripped-down — no CMake, no autoconf —
because the adapter is a single-shot bring-up tool for cross-validation,
not a long-lived build system. Frozen C++ stays frozen.

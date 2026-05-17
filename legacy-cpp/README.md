# legacy-cpp — the 2013-era C++ ASMOO implementation (PRE-PAPER)

⚠️ **NOT the cross-validation oracle.** This directory preserves the
**2013-era** C++ codebase that pre-dates the paper-companion release.
For cross-validation use [`../legacy-cpp-v2/`](../legacy-cpp-v2/) — the
**2015-era paper-companion release** that the Python port mirrors.

This directory accompanies the PhD thesis *Learning to Anticipate
Flexible Trade-off Choices in Multiobjective Combinatorial Optimization
Under Catastrophic Information Losses* (Azevedo, 2014).

The 2015 IEEE TCYB paper:

> **Azevedo, C. R. B., & Von Zuben, F. J.** (2015). *Learning to Anticipate
> Flexible Choices in Multiobjective Combinatorial Optimization Under
> Catastrophic Information Losses.* IEEE Transactions on Cybernetics.

was published with a refined post-thesis C++ codebase that's vendored at
`../legacy-cpp-v2/`. THAT is the cross-validation reference. Use it.

The actively-maintained, publication-track implementation is the Python
package under `../python_refactor/`. **Use that for any new work.** This
directory is kept for historical provenance only.

## Differences vs `legacy-cpp-v2/`

- This (2013) version LACKS `dirichlet.cpp` (the Dirichlet predictor
  module). Python's `DirichletPredictor` was built against the 2015
  v2 code where Dirichlet exists.
- This version's `Kalman_filter()` calls `Kalman_prediction()` BEFORE
  `Kalman_update()`. v2 reverses the order.
- This version uses Portuguese filenames (`aprendizado_operadores`,
  `operadores_*`). v2 uses English.
- `nsga2.cpp` here → `asms_emoa.cpp` in v2.

W18 cross-validation work initially used THIS directory as the oracle
and found false-positive structural divergences (missing Dirichlet,
"different" algorithms). PR #(forthcoming) imports `legacy-cpp-v2/`
and re-runs cross-checks against the correct reference.

## Layout

```
legacy-cpp/
├── source/       — C++ source files (.cpp)
├── headers/      — C++ headers (.h)
└── executable/
    ├── *.dll     — Windows boost runtime (mgw46-mt-1_52)
    └── data/
        └── ftse-original/  — original FTSE asset price CSVs (98 files)
```

## Status

- **Frozen.** No further C++ development is planned.
- **Build artifacts** (the boost DLLs) are kept solely to document the
  original Windows MinGW-w64 toolchain (g++ 4.6, Boost 1.52, ca. 2013).
- The FTSE CSVs here are the **original** dataset; an updated FTSE
  dataset lives at `../python_refactor/data/ftse-updated/`.

## Why we kept it

Three reasons:

1. **Provenance.** This is what was actually run for the experiments
   reported in the IEEE Transactions on Cybernetics paper.
2. **Cross-validation.** When the Python refactor's behaviour is questioned
   on a specific equation, the C++ source here is the original reference.
3. **History.** The C++ predates the Python refactor by roughly a decade.

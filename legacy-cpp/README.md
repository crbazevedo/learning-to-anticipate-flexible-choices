# legacy-cpp — the original C++ ASMOO implementation

This directory preserves the original C++ codebase ("ASMOO") that accompanied
the PhD thesis *Learning to Anticipate Flexible Trade-off Choices in
Multiobjective Combinatorial Optimization Under Catastrophic Information Losses*
(Azevedo, 2014) and the IEEE Transactions on Cybernetics paper that distilled
its main contributions:

> **Azevedo, C. R. B., & Von Zuben, F. J.** (2015). *Learning to Anticipate
> Flexible Choices in Multiobjective Combinatorial Optimization Under
> Catastrophic Information Losses.* IEEE Transactions on Cybernetics.

The actively-maintained, publication-track implementation is the Python
package under `../python_refactor/`. **Use that for any new work.** This
directory is kept for provenance and to enable cross-checking the
Python implementation against the original C++ behaviour where that is
ever in question.

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

# legacy-cpp-v2 — the 2015-era ASMOO C++ reference

This directory preserves the **2015-era** C++ implementation that
accompanies the IEEE TCYB 2015 paper. It is the **authoritative
upstream reference** for the Python port (`../python_refactor/`).

## Provenance

- **Source**: `Downloads/crbazevedo-anticipatory-learning-asmoo-6643c92`
  (GitHub repo `crbazevedo/anticipatory-learning-asmoo`, commit `6643c92`)
- **Dated**: 10 March 2015 (file mtimes); ASMOO post-thesis revision
- **License**: GPL (see `LICENSE`)
- **Upstream README**: see `UPSTREAM-README.md`

## Why this is the authoritative reference (not `../legacy-cpp/`)

The repo previously vendored only `legacy-cpp/` — a 2013-era thesis-companion
codebase. Cross-validation work in W18 / W19 (PRs #105-#112) initially
treated `legacy-cpp/` as the oracle but found apparent structural divergences
(missing Dirichlet module, old `nsga2.cpp` naming, etc.). The operator then
flagged that the **post-thesis 2015 release** at the path above is the
correct reference, and the Python port was built against it.

Both versions are kept:

| Tree | Era | Use |
|---|---|---|
| `legacy-cpp/` | 2013 (thesis-companion, pre-paper) | Historical provenance; do NOT use as oracle |
| `legacy-cpp-v2/` (this) | 2015 (paper-companion, with Dirichlet) | **Cross-validation oracle for the Python port** |

## What v2 adds vs v1

1. **`dirichlet.cpp` + `dirichlet.h`** — explicit Dirichlet prediction module
   that Python's `DirichletPredictor` mirrors (`dirichlet_mean_prediction_vec`,
   `dirichlet_mean_map_update`, `dirichlet_variance`).
2. **`asms_emoa.cpp`** (replaces `nsga2.cpp`) — refined algorithm with
   ASMS-EMOA-specific orchestration.
3. **English filenames** — `learning_operators` (was `aprendizado_operadores`),
   `mutation_operators` (`operadores_mutacao`), `selection_operators`
   (`operadores_selecao`), `crossover_operators` (`operadores_cruzamento`).
4. **KF lifecycle reversed** — v2's `Kalman_filter()` calls
   `Kalman_update()` BEFORE `Kalman_prediction()` (v1 had predict→update).
   Also `Kalman_params` now exposes `error` (running residual norm)
   and `window_size` (for the K-period λ^K consumption).
5. **Portfolio refinement** — `samples_per_portfolio`, `current_period`,
   `periodicity`, `brokerage_commission`, `sequence_mean_covar` —
   structural support for paper §7 protocol.
6. **`alpha = 1 - non_dominance_probability(w)`** (no `linear_entropy()`
   wrap; v1 had `alpha = 1 - linear_entropy(nd_probability)`) — different
   anticipative-rate formula.
7. **`compute_efficiency` invokes `observe_state` (KF integration)** —
   tighter coupling between portfolio evaluation and KF state tracking.

## Layout

```
legacy-cpp-v2/
├── source/                    11 .cpp files (3,350 LOC total)
├── headers/                   11 .h files
├── LICENSE                    GPL upstream
├── UPSTREAM-README.md         Original repo README
└── README.md                  THIS file
```

## Building

W18-1's `legacy-cpp/build/Makefile` was written against `legacy-cpp/`
sources. A `legacy-cpp-v2/build/` Makefile is planned for the
post-substrate-update cross-checks (subsequent units after the import).

## Cross-validation impact (re-assessment pending)

After this substrate import, all prior cross-checks must be re-run
against v2:

| Check | W18/W19 verdict (vs v1) | Re-assessment needed |
|---|---|---|
| A risk (W18-2) | DISAGREE-PYTHON-WRONG (Python adds sqrt) | v2 also has `u^T Σ u` (no sqrt) — verdict should hold |
| B ROI (W18-3) | AGREE | should still hold (same formula) |
| C bivariate (W19-1) | AGREE | should still hold |
| D KF Gaussians (W19-2) | AGREE | **MAY FLIP** — v2 reverses KF lifecycle (update→predict vs predict→update) |
| E Dirichlet (W19-3) | (pending) | v2 HAS Dirichlet — full cross-check now possible |
| F TIP (W18-4) | STRUCTURAL (Reading C) | v2 has same formula; verdict should hold |
| G λ end-to-end (W19-4) | (pending) | v2 uses `1 - p` (not `1 - entropy(p)`); structural divergence vs Python |

W18 / W19 synthesis docs will be updated with re-assessment receipts
in subsequent units.

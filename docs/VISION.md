# learning-to-anticipate-flexible-choices — Vision & Trajectory

*Authored 2026-05-16 during dfg-harness bootstrap of the renamed repo
(formerly `crbazevedo-phd-research`). Anchored on the paper PDF at
`docs/paper.pdf` and the code-vs-paper adherence audit dated 2026-05-16.*

---

## §1 — What this is

A Python implementation of *Learning to Anticipate Flexible Choices in
Multiple Criteria Decision-Making Under Uncertainty* (Azevedo & Von Zuben,
IEEE Transactions on Cybernetics, April 2015,
DOI:[10.1109/TCYB.2015.2415732](https://doi.org/10.1109/TCYB.2015.2415732)).
The paper proposes the Anticipatory Stochastic Multi-objective Optimization
(AS-MOO) model — a methodology for sequential decision-making in dynamic and
noisy environments where preferences cannot be reliably elicited up-front,
and so flexibility (postponable preference specification) is itself the
objective. The implementation tracks trade-off distributions over time via
a Kalman Filter (objective space) and a Dirichlet Dynamical model (search
space), self-adjusts anticipation rates from a Time Incomparability
Probability (TIP), and integrates both via the ASMS-EMOA algorithm.

**Canonical long-form source.** The IEEE paper is a condensed version of
chapters 5–7 of the originating PhD thesis: *Anticipation in Multiple
Criteria Decision-Making Under Uncertainty* (Azevedo, UNICAMP 2014, advisor
Prof. Fernando José Von Zuben, DOI [10.47749/T/UNICAMP.2012.938003](https://doi.org/10.47749/T/UNICAMP.2012.938003)).
The thesis is available locally at `docs/Azevedo_CarlosRenatoBelo_D.pdf` and
indexed (chapter map, OOS protocol Eqs 7.10–7.11, parameters, future research
directions, key references) at [`docs/THESIS-INDEX.md`](THESIS-INDEX.md). When
the thesis and the IEEE paper disagree, the thesis wins — it has the
unabridged derivations and the explicit OOS evaluation protocol.

**Identity sentence.** *The reference open-source implementation of the
AS-MOO methodology, faithful to the paper PDF, reproducible from a pinned
environment, and usable by the broader MOO / MCDM research community.*

## §2 — The problem it solves

Two distinct problems:

1. **The research problem** (the paper's): how to maximize a decision
   maker's future freedom of action when preferences will only crystallize
   later and the objective functions themselves drift. Solved by treating
   flexibility as a stochastic-dynamic Pareto frontier and using OAL
   (Online Anticipatory Learning) to self-adjust how much weight to give
   to predicted futures vs. current observations.
2. **The codebase problem** (current state): every theoretical construct
   from the paper exists in `python_refactor/src/algorithms/` as a tested
   module, **but the advanced modules are not wired into the live
   execution path**. `main.py` / `run_experiments.py` route through
   `ExperimentManager`, which only knows the base (non-anticipatory)
   `AnticipatoryLearning` class. TIP, sliding-window Dirichlet, multi-
   horizon learning rule, and the thesis-aligned experiment driver are
   all dead code reachable only from unit tests.

## §3 — Empirical receipts

| Claim | Value | Provenance |
|---|---|---|
| Python LoC (excl. tests) | 16,356 | `wc -l` of `python_refactor/src/**/*.py` at HEAD `03e26dc` (2026-05-16 dfg audit) |
| Test functions | 287 across 17 files | grep `^def test_` in `python_refactor/tests/*.py` (2026-05-16 dfg audit) |
| Test files asserting paper equations (not bounds-only) | **1 of 17** (`test_sliding_window_dirichlet.py`) | 2026-05-16 dfg audit §"Tests vs claims" |
| Modules wired into the live `ExperimentManager` path | 0 of 6 advanced modules | 2026-05-16 dfg audit §"Faithful but un-wired" |
| Paper page count | 14 (incl. Editor cover letter pages) | `docs/paper.pdf` ToC inspection 2026-05-16 |
| Paper equations (Section IV — OAL) | (11)–(18) | paper §IV-A, IV-A.1, IV-A.2, IV-B, IV-B.1 |
| Most-recent code push | 2025-10-19 | `gh api repos/crbazevedo/learning-to-anticipate-flexible-choices` 2026-05-16 |

## §4 — Current surface

```
learning-to-anticipate-flexible-choices/
├── docs/                                         (new)
│   ├── VISION.md                                 (this file)
│   ├── paper.pdf                                 (canon — IEEE paper, 1.9 MB)
│   ├── Azevedo_CarlosRenatoBelo_D.pdf            (canon long-form — PhD thesis, 17.8 MB, UNICAMP 2014)
│   ├── THESIS-INDEX.md                           (chapter map + OOS protocol + parameters + future works)
│   ├── ANALYTICS-PLAN.md                         (analytics-layer spec, W7-2)
│   ├── EXPERIMENT-VALIDATION-PLAN.md             (validation matrix spec, W6-3)
│   └── VALIDATION-RESULTS.md                     (template, W7-3)
├── legacy-cpp/                                   (renamed from ASMOO/, frozen)
│   ├── source/             — 11 .cpp
│   ├── headers/            — 10 .h
│   └── executable/
│       ├── *.dll           — 3 Windows boost runtime
│       └── data/ftse-original/  — 98 CSVs
├── python_refactor/                              (active development)
│   ├── src/
│   │   ├── algorithms/     — 16 modules (6 of them dead-code per audit)
│   │   ├── portfolio/      — portfolio + asset + statistics
│   │   ├── experiments/    — ExperimentManager (live) + thesis_aligned (dead)
│   │   └── config/         — thesis_parameters + experiment_config
│   ├── experiments/        — 7 top-level scripts, overlapping scope
│   ├── tests/              — 17 files, 287 functions
│   └── data/ftse-updated/  — current dataset (8.9 MB)
├── 100_percent_adherence_backlog.md              (PT — translation queue)
├── HVDM_OPTIMIZATION_PLAN.md                     (mixed EN/PT — translation queue)
├── repository_analysis.md                        (stale per audit — overdue cleanup)
├── thesis_codebase_analysis.md                   (EN, equation-level analysis)
└── README.md                                     (in python_refactor/, not at root)
```

## §5 — Audience

Three tiers, in priority order:

1. **Future Carlos** (the maintainer) — needs to extend the paper without
   re-discovering the codebase every six months.
2. **MOO / MCDM researchers** — need to (a) reproduce the paper's
   experiments from a single command, (b) compare their own algorithms
   against ASMS-EMOA on a fair baseline, (c) cite the canonical
   implementation rather than re-implementing from the equations.
3. **Industrial practitioners** in dynamic portfolio optimization or
   sequential decision-making under uncertainty — need a tested,
   installable Python package with a clear API.

## §6 — Visible vector

*Inferred from commit history + the merged "100% Thesis Adherence" PR
(commit `69b4aab`, 2025-09-07), not from a stated roadmap.*

The repo's trajectory over the past 6 months has been: build out the
theoretical-completeness layer (TIP, Dirichlet, multi-horizon, belief
coefficient, correspondence mapping) as standalone modules with unit
tests, while leaving the integration into `ExperimentManager` for
later. Documents written along the way (`thesis_codebase_analysis.md`,
`100_percent_adherence_backlog.md`) claim 85% adherence to the thesis,
but the audit shows adherence is high at the **module** level and
near-zero at the **executed** level. The vector is therefore: *finish
the bridge*. Make the live experiment driver use what's already built.

## §7 — Near-term horizon (next 4–8 weeks)

Six themes, in priority order:

| Theme | Why |
|---|---|
| **T1. Reconcile to paper canon** | Strike thesis-numbered equation citations (6.x) in favour of paper-numbered (11)–(18) throughout docs and tests. Single canonical reference. |
| **T2. Wire the advanced modules** | TIP, multi-horizon, belief coefficient, sliding-window Dirichlet must actually run when an experiment requests them. |
| **T3. Equation-level test coverage** | At least one test per paper equation that asserts against a hand-computed analytical case, not just bounds. |
| **T4. Reproducibility** | Single `make experiments` that pins seeds, reproduces every paper figure, with a manifest of what each script produces. |
| **T5. Packaging discipline** | Migrate from `requirements.txt` to `pyproject.toml`, fix `sys.path` hacks, support `pip install -e .`. |
| **T6. Doc translation + cleanup** | The 3 PT-flavored docs to EN; delete the stale `repository_analysis.md`. |

## §8 — Far-term horizon (next 6–12 months)

| Theme | Why |
|---|---|
| **F1. Public release** | Move repo back to public + cite-able with a stable DOI (Zenodo). |
| **F2. Methodology paper** | A short follow-up describing the reproducible implementation + the substrate that enforces its discipline (composes with dfg-harness). |
| **F3. Methodology generalization** | Carlos's stated direction — generalize this codebase's reproducibility discipline beyond MOO into a template applicable to other research codebases. |
| **F4. Extension experiments** | New experiments enabled by faithful wiring: multi-horizon vs. one-step ASMS, A/B between belief-coefficient on/off, portfolio drift regimes. |

## §9 — Risks the plan must absorb

| Risk | Mitigation |
|---|---|
| The paper says one thing, the code does another, and tests pass because tests assert the code, not the paper. | Every test must cite a paper equation by section + number; reviewers check the citation, not just the assertion. |
| The 3 analysis docs already on disk are themselves out of date — partly stale, partly translated, partly using thesis numbering. Updating them costs as much as authoring them did. | Treat the analysis docs as **deprecation candidates**, not sources of truth. The paper is canon. The audit is the working ledger. |
| `legacy-cpp/` is preserved for provenance but cannot be built on a modern macOS toolchain (MinGW Windows binaries). Cross-validation against C++ behaviour is therefore impractical. | Don't make cross-validation against C++ a gate. Use paper equations as the spec; use `legacy-cpp/` only for "what was actually run for the 2015 paper" provenance. |
| Wiring the advanced modules requires touching `experiment_manager.py`, which has implicit tests scattered across the experiment scripts. Regressions are easy. | Add an integration test per advanced-module wiring that runs a tiny experiment end-to-end and asserts the live path uses the new module. |
| The translation queue (3 PT docs) is bounded but tedious. Risk: dragged out, never finished. | Schedule it as a single unit (W1-5b in W2 or W3); treat un-translated docs as un-published. |

## §10 — Composition with dfg-harness

This repo is the first non-`dfg-harness` repo Carlos governs with
methodology-as-code. The harness gives this codebase: contract-first
unit authoring, dual-critic gates before merge, paired plan.yaml ↔
replan-accept discipline, ADR-004-framed retros that survive
compaction, the `dfg pre-pr` battery as a local gate before every
push, and the §amendment-trigger pattern for codifying recurring pain
into substrate. In return, the codebase tests the *generality* of the
harness against a non-Click / non-CLI / scientific-Python repo —
empirically validating ADR-019 §amendment-trigger discipline on a
codebase whose unit tests routinely exercise numerical equations.

The harness's claim of methodology-as-code is empirically falsifiable
here in a way it isn't inside `dfg-harness` itself. If after wave-1 the
substrate-first discipline doesn't visibly improve the velocity-vs-
correctness frontier, that's a `dfg-harness` finding worth recording
as such.

---

*End of VISION.md. Updates to this file must pair with a
`RePlanProposed`/`RePlanAccepted` event per ADR-007.*

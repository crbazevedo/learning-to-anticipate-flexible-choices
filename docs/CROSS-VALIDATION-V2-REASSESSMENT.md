# Cross-validation re-assessment vs legacy-cpp-v2 (post-substrate-update)

*Generated 2026-05-17 after substrate-update PR #113 (import of 2015-era ASMOO C++).*

## TL;DR — verdict table

| Check | vs v1 (W18/W19) | vs v2 (this) | New finding |
|---|---|---|---|
| A risk (W18-2) | DISAGREE: Python adds sqrt() | DISAGREE: same — Python adds sqrt() | Holds: v1 risk == v2 risk (variance, no sqrt); Python deviation confirmed |
| B ROI (W18-3) | AGREE | AGREE | No drift |
| C bivariate (W19-1) | AGREE | AGREE | No drift |
| **D KF (W19-2)** | **AGREE w/ Python combined kalman_filter** | **DISAGREE** (lifecycle reversed) | v2 = update→predict; v1 = predict→update; Python kalman_filter matches v1, NOT v2 |
| **E Dirichlet (W19-3)** | "C++ has no Dirichlet" (WRONG diagnosis vs v1) | **AGREE machine precision** | v2 HAS dirichlet.cpp; Python is verbatim port; agreement to 1e-16 |
| F TIP (W18-4) | STRUCTURAL (Reading C) | STRUCTURAL (v2 same formula) | Holds: both v1 and v2 use Cholesky-standardized normal_cdf sum |
| G λ end-to-end | (pending) | **Three different formulas** | v1: 1-linear_entropy(p); v2: 1-p; Python: 0.5*(λ^H + λ^K) per Eq 7.16 |

## Production-call analysis (key for interpreting D)

| Side | When `kalman_filter` (combined) called | When `kalman_update` alone | When `kalman_prediction` alone |
|---|---|---|---|
| v1 production (`portfolio.cpp:699`) | YES (`Kalman_filter()` predict→update) | — | YES (`Kalman_prediction()` after final period — line 705) |
| v2 production (TBD; uses combined `Kalman_filter()` update→predict) | YES (`Kalman_filter()` update→predict) | — | — |
| Python production (`sms_emoa._evaluate_solution:318`) | NO | YES (per-solution per-generation) | YES (only inside MC stochastic-HV sampling — `sms_emoa.py:754`) |

**Production-state divergence**:
- v2 production state evolution = update→predict cycle
- Python production state evolution = update-only per-generation (no combined predict-update)

So even though Python's COMBINED `kalman_filter` test matches v1's behavior, the PRODUCTION state-evolution pattern is structurally different from BOTH v1 and v2 — Python never calls the combined function in the live ASMS-EMOA loop.

This is a more subtle finding than the original W19-2 "AGREE" implied. The W19-2 verdict needs to be qualified:
- ✅ Python's `kalman_filter` agrees with v1's `Kalman_filter` (predict→update)
- ❌ Python's `kalman_filter` DISAGREES with v2's (update→predict)
- ⚠️ Python production never calls `kalman_filter`; uses update + prediction separately

## Reading-C status (W18-4 finding) — UPDATED

W18-4 concluded the TIP saturation was structural (data + KF parameter property), based on:
- 3 TIP implementations agree → not a TIP bug ✅
- v1 KF identical to Python KF → not a KF coding bug (re: v1) ✅

**With v2 as the correct oracle**, the picture is more nuanced:
- TIP formula in v2 is same as v1 — verdict still holds
- KF predict + update PRIMITIVES are identical across v1, v2, Python (just lifecycle order differs)
- BUT: production state-evolution patterns differ — v2 production uses update→predict per period; Python production uses update-only per generation. Different effective state trajectories on same data!

This adds a NEW hypothesis for the saturation: production state-evolution mismatch may produce different KF covariance trajectories → different TIP regimes → different λ. Worth investigating in W19-4/W19-5 once production-state-trace comparison is built.

## Bug count (updated)

| # | Side | Where | Severity |
|---|---|---|---|
| 1 | v1 C++ | `portfolio.cpp:65` comma-op (autocorr) | Off-headline; only in v1 |
| 2 | Python | `compute_risk` adds sqrt() not in thesis Eq (7.4) | On-headline; W18-CARRY-1 |
| 3 | Python? | Production never calls combined `kalman_filter`; uses update-only per generation vs v2's update→predict per period | **NEW** — may explain part of W17-5 saturation |

## Updated Reading framework

| Reading | Diagnosis | W18-substrate evidence |
|---|---|---|
| A | wrong metric (single-period EFHV vs multi-period wealth) | unchanged |
| B | replication failure | less likely after substrate update — Dirichlet AGREE removes a class of suspected drift |
| **C** | structural data property (TIP saturation = max-uncertainty regime) | confirmed by v2 cross-check; TIP/KF primitives all parity |
| **D** (NEW) | **production state-evolution divergence** (Python update-only vs v2 update-predict per period) | **Plausible — needs W19-5 production-trace cross-check** |

## What W18 / W19 prior receipts need correcting

- `docs/CROSS-VALIDATION-A-RISK.md` — verdict holds, but cite "vs v2" (was vs v1)
- `docs/CROSS-VALIDATION-B-ROI.md` — verdict holds, cite v2
- `docs/CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md` — verdict holds, cite v2
- `docs/CROSS-VALIDATION-D-KF-GAUSSIANS.md` — **verdict needs revision** per the v2 lifecycle order finding
- `docs/CROSS-VALIDATION-F-TIP.md` — verdict holds, cite v2; add the production state-evolution caveat
- `docs/W18-CROSS-VALIDATION-SYNTHESIS.md` — update with this re-assessment

(Future units can ship these as housekeeping updates. The substrate-update PR + this consolidated receipt are the actionable artifacts for now.)

## W19-3 receipt (Dirichlet — first true cross-check)

The Python `DirichletPredictor.dirichlet_mean_prediction_vec` and `dirichlet_mean_map_update` are **verbatim ports** of v2's `dirichlet.cpp` functions. Cross-execution on identical fixture (5 portfolios × 5 assets × 5 cases) shows agreement to 1e-16 absolute (machine precision).

| Function | abs_max diff | rel_max diff |
|---|---|---|
| `dirichlet_mean_prediction_vec` | 1.11e-16 | 3.96e-16 |
| `dirichlet_mean_map_update` | 1.11e-16 | 5.07e-16 |

No bugs in either side. The W19-3 contract's premise ("C++ has no Dirichlet") was based on the wrong substrate (v1); the correct premise is "v2 has Dirichlet that Python mirrors verbatim".

## W19-4 still pending: anticipative rate (G)

Three formulas need direct comparison:

| Source | Formula |
|---|---|
| v1 `nsga2.cpp:565` | `alpha = 1.0 - linear_entropy(nd_probability)` |
| v2 `asms_emoa.cpp:44` | `w->alpha = 1.0 - non_dominance_probability(w)` (no entropy wrap) |
| Python `compute_anticipatory_learning_rate` (post-W16-1) | `λ = 0.5 * (λ^H + λ^K)` where λ^H per Eq 6.6 multi-horizon |
| Thesis Eq 7.16 | `λ_{t+h} = 0.5 * (λ^H + λ^K)` |

Python matches the thesis Eq 7.16 verbatim. v1 and v2 use DIFFERENT formulas, neither of which matches the thesis Eq 7.16 directly. **The C++ legacy uses a simplified formula (Eq 6.6-like, without the (1/2) λ^H + λ^K combination)**.

This is a major finding for W19-4: Python implements the thesis verbatim, v1/v2 don't. The empirical question is whether the simpler v2 formula behaves better on this data despite being thesis-non-faithful.

## Output artifacts

- `legacy-cpp-v2/build/drivers/{risk_driver_v2, kf_driver_v2, bivariate_gaussian_driver_v2, dirichlet_driver}.cpp` — NEW drivers against v2
- `legacy-cpp-v2/build/drivers/mvtnorm_stub.cpp` — extern "C" pmvnorm stub for Apple silicon
- `legacy-cpp-v2/build/Makefile` — v2 build adapter
- `python_refactor/scripts/cross_validation/run_dirichlet.py` — Python driver
- This document — consolidated re-assessment receipt

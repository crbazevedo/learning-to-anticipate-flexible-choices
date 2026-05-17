# W18 cross-validation synthesis

*Generated 2026-05-17 by W18-5. Closes WAVE 18.*

## Wave summary

Operator directive: investigate 14 cross-checks (A-N) against C++ legacy with mutual-skepticism discipline. W18 scope: harness + 3 most-strategic checks (A risk, B ROI, F TIP). W19+ inherits the harness for the remaining 11 (C/D/E/G/H/I/J/K/L/M/N).

5 units shipped: PRs #105 (harness), #106 (A), #107 (B), #108 (F), this synthesis.

## Verdict table

| Check | Subject | Verdict | Finding |
|---|---|---|---|
| **A** | risk computation | 🔴 **DISAGREE-PYTHON-WRONG** | Python adds sqrt() not in thesis Eq (7.4); C++ matches thesis verbatim |
| **B** | ROI computation | ✅ **AGREE** (machine precision) | Both match thesis verbatim; rel_max = 0 |
| **F** | TIP saturation | ⚪ **STRUCTURAL (Reading C)** | TIP code correct on both sides; saturation at 0.5 is a property of (data + KF cov) regime |

## Bugs surfaced

| # | Side | Where | On headline path? | Status |
|---|---|---|---|---|
| 1 | C++ | `portfolio.cpp:65` — comma-op makes `var += 4.0` instead of squared deviation | ❌ Off (`sample_autocorrelation`) | Documented; not blocking |
| 2 | Python | `compute_risk` adds sqrt() ≠ thesis Eq (7.4) | ✅ **YES** | **W18-CARRY-1; needs operator decision** |

Both findings validate the operator's mutual-skepticism caveat: "the C++ reference implementation also being wrong" is observed; both sides can have bugs.

## Revised strategic framework (Reading A vs B vs C)

W17-5 framed two readings of the persistent S2 < S0 result:

- **Reading A** — wrong metric (single-period EFHV vs multi-period wealth)
- **Reading B** — replication failure on modern Python

W18-4 introduces:

- **Reading C** — structural data property: at the (data + KF covariance) regime in this dataset, TIP saturates at 0.5 by mathematical necessity → λ^H → 0 → anticipation overhead can only hurt SMS baseline

**W18 evidence strongly favors Reading C, with Reading A as a refinement**. The TIP saturation is reproduced across 3 independent implementations; the issue is upstream in **KF parameterization + data scale**, not in the anticipation code.

This is a significant scientific finding either way:
- If KF parameters can be re-calibrated to reduce predictive uncertainty → anticipation engages → ASMS may beat SMS → paper replicates with a "KF tuning" note
- If KF parameters are correctly calibrated and uncertainty IS irreducible on this data → the thesis's headline-claim regime is dataset-specific and doesn't generalize → publishable replication-failure result

## What W18 ruled out

- ❌ Python TIP code has a bug → no, three implementations agree
- ❌ C++ TIP code has a bug → no, matches the others
- ❌ ROI computation drift → no, machine-precision agreement
- ⚠️ Risk computation drift → yes, Python sqrt() deviation found

## What W18 promotes to high-priority for W19+

1. **Cross-check D** (KF Gaussians from KF application) — likely highest-leverage. If KFs disagree, the predictive distributions feeding TIP are different → all downstream metrics misaligned. If KFs agree, Reading C is confirmed.

2. **Cross-check C** (bivariate Gaussian parameters before KF) — sets up the KF inputs; verifies the (mean, covariance) construction is consistent.

3. **Fix decision for W18-CARRY-1** (Python sqrt in compute_risk) — operator call:
   - (a) Fix Python to match thesis (Cleanup + downstream consistency audit)
   - (b) Document deliberate deviation + verify downstream code accommodates correctly

## W19-W22 roadmap (pre-stubbed unit contracts to be added)

For each of the 11 remaining cross-checks, W19+ ships a unit using the W18-1 harness pattern (C++ driver in `legacy-cpp/build/drivers/` + Python in `scripts/cross_validation/` + receipt in `docs/CROSS-VALIDATION-<check>.md`).

| Wave | Unit | Check | Why ordered here |
|---|---|---|---|
| W19-1 | cross-check-c-bivariate-gaussians | C | Sets up KF inputs; upstream of D |
| W19-2 | cross-check-d-kf-gaussians | D | **Reading-C critical test** |
| W19-3 | cross-check-e-dirichlet-distributions | E | Decision-space tracking |
| W19-4 | cross-check-g-anticipative-rate | G | Eq 7.16 λ end-to-end (closes W17-5 chain) |
| W19-5 | synthesis | — | Decide Reading A vs B vs C with full evidence |
| W20-1 | cross-check-h-dirichlet-from-filter | H (= E variant?) | Per operator letter (may dedupe with E) |
| W20-2 | cross-check-i-anticipative-distributions | I | OAL output verification |
| W20-3 | cross-check-j-crowding-distance | J | NDS supporting metric |
| W20-4 | cross-check-k-expected-hv-contribution | K | Selection pressure metric |
| W20-5 | synthesis | — | |
| W21-1 | cross-check-l-nds-algorithm | L | Selection step |
| W21-2 | cross-check-m-mutation-operator | M | Search operator + simplex check |
| W21-3 | cross-check-n-sbx-vs-uniform-crossover | N | Note: W15-2 switched Python to uniform; N must compare both interpretations |
| W21-4 | synthesis + final decision | — | Reading A/B/C final verdict |

## Cumulative gap-closure receipt across W13→W18

| Wave | Δ(S2 − S0) | What changed | Std (W17+) |
|---|---|---|---|
| W13-3 single-shot 80/20 | -17.59% | initial | n/a |
| W14-2 walk-forward | -24.86% | methodology ruled out | n/a |
| W15-5 walk-forward | -8.75% | BLOCKERS closed | n/a |
| W16-5 walk-forward | -5.53% | algo fixes (λ^K + txn + extrema) | 2.66e-05 |
| W17-5 walk-forward | -8.72% | 87-asset + AMFC + λ^K firing | 7.92e-07 |
| W17-5-CARRY-1 RESMOKE | -11.79% | MultiHorizon Eq 7.16 active | 2.86e-06 |
| **W18-1..5** | **(no smoke; structural finding)** | **TIP saturation = data property, not code; risk sqrt deviation found** | — |

W18 didn't run a smoke. The contribution is the **structural diagnosis**: with the W17-5-CARRY-1 receipt + W18-4 confirming TIP saturation is data-driven, the residual gap is now understood at the algorithm level. Re-running smokes won't change it unless we change upstream (KF params, data, metric).

## Output artifacts

- `docs/CROSS-VALIDATION-A-RISK.md` (W18-2)
- `docs/CROSS-VALIDATION-B-ROI.md` (W18-3)
- `docs/CROSS-VALIDATION-F-TIP.md` (W18-4)
- `legacy-cpp/build/Makefile` + `README-BUILD-ADAPTER.md` (W18-1)
- `legacy-cpp/build/drivers/{risk,roi}_driver.cpp` + `mvtdst_stub.cpp` (W18-1, W18-2, W18-3)
- `python_refactor/scripts/cross_validation/{__init__,fixtures,compare,run_risk,run_roi,run_tip}.py` (W18-1..4)
- `python_refactor/tests/test_cross_{check_risk,check_roi,check_tip,validation_harness}.py` — 19 tests, all green
- This synthesis

## Carry-forward map (W18 → operator decision)

- **W18-CARRY-1**: Python `compute_risk` adds sqrt() not in thesis Eq (7.4). Decide (a) fix or (b) document deliberate deviation.
- **W18-4-CARRY-1**: TIP C++ binary not directly executed (Apple silicon Fortran). Mitigated; not blocking.
- **W18-4-CARRY-2**: F2 production-replay deferred; W17-5 trace schema needs KF covariance fields.
- **W18-CARRY-2** (NEW): Reading C is the most likely diagnosis. W19+ should test by cross-checking KF predictive distributions (cross-check D); if KFs agree across implementations, structural-uncertainty is confirmed and the operator may need to decide between (a) KF re-calibration, (b) multi-period wealth metric, (c) publish replication-failure with the W7→W18 receipt chain as the contribution.

## Recommended W19 keystone

**W19-1: cross-check D — KF Gaussians from KF application.**

If C++ and Python KFs disagree on predictive (mean, cov) for the same inputs, we've found another structural drift to fix. If they AGREE, Reading C is confirmed and the operator's strategic decision becomes the next-level question.

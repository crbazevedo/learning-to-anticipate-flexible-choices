# W22-9 — FTSE NC27-deep n=10 verdict: REGRESSION

**Wave unit:** W22-9 (FTSE empirical confirmation of W22-7 cross-instance positive)
**Date:** 2026-05-21 (computed late-night 2026-05-20)
**Verdict:** **NC27_DEEP REGRESSES FTSE by −1.11% (NS)** — opposite of PO cross-instance
**Recommendation:** **CANCEL W23-1 default flip.** Keep NC27_DEEP as OPT-IN only.
**Thesis anchor:** §6.2 Eq 6.7 Dirichlet posterior; §7.2.2 FTSE protocol

---

## Results

FTSE 2015 walk-forward, paired n=10 seeds, n_mc=200, ASMS_mHDM_K3_v2both
scenario:

| Metric | Value |
|---|---|
| BASELINE grand mean Ŝ | 4.701e-04 |
| NC27_DEEP grand mean Ŝ | 4.649e-04 |
| Paired mean Δ | **−1.11%** |
| Wins (seed-by-seed) | 6/10 |
| Wilcoxon p | 0.9219 |
| Wall time (each config) | ~100min |

**Per-seed paired Δ% on FTSE ASMS:**

| seed | BASELINE | NC27_DEEP | Δ | %Δ | direction |
|---|---|---|---|---|---|
| 1 | 4.85e-04 | 4.77e-04 | −7.87e-06 | −1.62% | ↓ |
| 2 | 4.29e-04 | 4.64e-04 | +3.53e-05 | +8.24% | ↑ |
| 3 | 4.60e-04 | 4.75e-04 | +1.53e-05 | +3.32% | ↑ |
| 4 | 4.38e-04 | 4.60e-04 | +2.23e-05 | +5.09% | ↑ |
| 5 | 3.98e-04 | 4.07e-04 | +8.74e-06 | +2.20% | ↑ |
| 6 | 4.86e-04 | 5.04e-04 | +1.74e-05 | +3.57% | ↑ |
| 7 | 4.80e-04 | 5.14e-04 | +3.34e-05 | +6.95% | ↑ |
| 8 | 4.57e-04 | 4.27e-04 | −3.04e-05 | −6.65% | ↓ |
| 9 | 5.05e-04 | 4.23e-04 | −8.18e-05 | **−16.20%** | ↓ |
| 10 | 5.62e-04 | 4.98e-04 | −6.46e-05 | **−11.49%** | ↓ |

**Critical pattern**: 6 wins (mostly small, +2.2% to +8.2%) BUT 4 losses
include two TAIL-RISK events (−16.20% and −11.49%). The Wilcoxon ranks
loss magnitudes against win magnitudes; the 2-sided test gives p=0.9219
(NS). Direction is NEGATIVE on mean.

---

## Theory check: PO →FTSE divergence

This is the **synthetic-vs-empirical drift the registry was designed to catch**.
The 3-tier translation chain:

1. **Synthetic (Inspection 3)**: NC27-deep claim 2.8× tighter L2 on Dirichlet data ✓
2. **PO synthetic (W22-4..7)**: All 4 instances POSITIVE paired (+2.65%, +2.19%, +2.82%, +8.93%) ✓
3. **FTSE real data (W22-9)**: **REGRESSION −1.11% with 2 tail-risk seeds losing >10%** ✗

The PO synthetic generates Dirichlet-distributed weights with smooth dynamics.
FTSE has:
- Real market regime shifts (2014-2015 commodity crash, etc.)
- Asymmetric volatility (fat-tailed asset returns)
- Cross-asset correlations that change over time

Hypothesis for the tail-risk losses: NC27-deep's aggressive Bayesian update
(α += observation per period) causes the posterior to OVER-COMMIT to recent
observations. When the market shifts regime, the posterior is "stuck" at
the last regime's weight distribution and produces bad predictions for
2-3 periods. The legacy DirichletPredictor's exponential smoothing is
more conservative — it doesn't fully commit to recent observations and
recovers faster from regime shifts.

This is consistent with seeds 9 and 10 (where the random initialization
+ market-data interaction triggered a regime-shift scenario that NC27-deep
mishandled).

---

## Updated drift registry verdict

NC27_DEEP empirical now: **REGIME-DEPENDENT**
- Translation Synthetic → PO: **STRONG** (4/4 positive)
- Translation PO → FTSE: **OPPOSITE_SIGN** (−1.11% mean, 6/10 wins with tail-risk)
- Translation Synthetic → FTSE: **OPPOSITE_SIGN**

---

## Decision matrix for W23-1 (final)

Per W22-9 evidence:

| Condition | Action |
|---|---|
| FTSE paired Δ < −1% | **CANCEL W23-1** ✗ |
| Tail-risk losses > 10% in 2/10 seeds | **CANCEL W23-1** ✗ |

**FINAL VERDICT**: **CANCEL W23-1**. NC27_DEEP stays as OPT-IN env var.

Updated `THESIS_CITATIONS.md` should note: NC27-deep is "regime-dependent;
positive on smooth synthetic Dirichlet dynamics; negative on real FTSE
data due to over-aggressive Bayesian commitment to recent observations
causing tail-risk losses during regime shifts."

---

## What's left for ratification

For NC27_DEEP to become a production default, we'd need ONE of:
1. **A regime-detection front-end** that reverts to legacy DirichletPredictor
   when regime-shift signals appear
2. **A tunable concentration_increment** (currently fixed at 1.0) that the
   operator can dial down to make the Bayesian update less aggressive
3. **A hybrid update** that blends Bayesian posterior (NC27-deep) with
   exponential smoothing (legacy) as a function of forecast variance
4. **Empirical evidence on a SECOND real-data benchmark** (NASDAQ, HangSeng)
   where NC27-deep wins; would suggest the FTSE regression is dataset-specific
   rather than a fundamental issue

For now, NC27_DEEP is the BEST opt-in NC candidate (only one with positive
PO cross-instance evidence) but does NOT clear the bar for production
default flip.

---

## Audit trail

- Commit: pending
- Per-seed JSON: `results/W22-9_ftse_baseline.json`, `results/W22-9_ftse_nc27deep.json`
- Markdown summaries: `results/W22-9_ftse_baseline.md`, `results/W22-9_ftse_nc27deep.md`
- Computation: ad-hoc Python in chat (sets paired Wilcoxon stats)
- CHANGES_LOG entry 6 to be added (REVISES Entry 5 which recommended PROCEED)
- Drift registry update: NC27_DEEP now REGIME_DEPENDENT verdict
- Wave state: W22-9 → DONE; W23-1 → CANCELLED

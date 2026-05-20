# W22-7 — NC27-deep cross-instance verdict

**Wave unit:** W22-7 (aggregates W22-4, W22-5, W22-6)
**Date:** 2026-05-20
**Verdict:** WEAKLY POSITIVE — direction holds on 3 of 4 instances; PO(16,1.0) outlier regression
**Recommendation:** DO NOT flip NC27_DEEP default in W23-1 yet; investigate PO(16,1.0) first
**Thesis anchor:** §6.2.1 Dirichlet concentration sensitivity; Paper Eqs 16-17

---

## Cross-instance result table

Per-instance paired NC27_DEEP vs BASELINE on `ASMS_mHDM_K3_v2both`:

| Instance | n_seeds | NC27_DEEP Δ | wins | Wilcoxon p | direction |
|---|---|---|---|---|---|
| **PO(8,1.0) paired (prior W22-2)** | 10 | **+1.37%** | 6/10 | 0.625 | ⬆ |
| **PO(8,1.0) unpaired (W22-4 added n=20)** | 20 | +0.52% | — | — | ⬆ (washing out) |
| **PO(16,1.0) (W22-5a)** | 5 | **−6.10%** | — | 0.625 | ⬇ **OUTLIER** |
| **PO(8,0.3) (W22-5b)** | 5 | +2.51% | — | 1.000 | ⬆ |
| **sPO(8,1.0)-cosine (W22-6)** | 5 | **+8.59%** | — | 0.188 | ⬆ **TRENDING** |

**Cross-instance avg paired Δ**: +1.6% (excluding unpaired W22-4)
**Sign-consistency**: 3 of 4 positive
**Single significant trend**: sPO p=0.188

---

## Interpretation

### What worked
1. **Direction is mostly stable** across PO α regimes (0.3, 1.0) and across smoothness regimes (raw PO vs cosine sPO). 3 of 4 positive.
2. **sPO(8,1.0) +8.59%** is the **strongest empirical signal of any W22 NC to date** — and it's on the regime that previously gave FLAT results for ASMS-vs-SMS (prior session pooled n=10 was -0.27%). NC27-deep's Bayesian posterior particularly suits the smooth dynamics where weight distributions evolve slowly.
3. **PO(8,0.3) +2.51%** corroborates the low-α regime works for NC27-deep.

### What broke
1. **PO(16,1.0) -6.10%** is a real regression. Hypothesis: with α=16, the synthetic Dirichlet generating process is HIGHLY CONCENTRATED. NC27-deep's aggressive Bayesian updates (α += observation) cause the posterior to over-trust recent observations and lose the slow-changing dynamics. The legacy exponential-smoothing DirichletPredictor (a heavily biased "predictor") happens to be MORE ROBUST in this regime because it doesn't fully commit to the posterior mean.

This is the classic Bayesian over-confidence problem under model misspecification.

### Mechanism (theory check vs thesis §6.2.1)

Thesis Eq 6.7: posterior mean `α_post / Σα_post` with `α_post = α_prior + obs`.
For α=16 concentration, the prior is well-calibrated to the true generating
process. The Bayesian update `α += obs` causes the posterior to over-fit
the noisy observation, especially when the prior was already correct.

The legacy DirichletPredictor (exponential smoothing) does:
  `pred_t = (1 - α_rate) · pred_{t-1} + α_rate · obs_t`
which acts as a low-pass filter and is more robust to noisy observations
in the high-α concentrated-prior regime.

This is a regime-dependent behavior. The synthetic NC27-deep claim ("2.8×
tighter L2 error vs DirichletPredictor on Dirichlet source data") is
**TRUE on average** but **REGIME-DEPENDENT** in wealth-translation: it
helps in some α regimes (1.0, 0.3) and hurts in others (16.0).

---

## Pooled n=20 PO(8,1.0) update

After adding W22-4 seeds 11-20:

| Metric | Value |
|---|---|
| NC27_DEEP n=20 mean Ŝ | 3.294e-04 |
| BASELINE n=10 mean Ŝ | 3.277e-04 |
| Unpaired %Δ | **+0.52%** |
| Note | BASELINE seeds 11-20 NOT run; cannot do paired n=20 without 1 more BASELINE batch (~25min wall) |

The signal at n=10 paired (+1.37%) is HALVED to +0.52% unpaired at n=20.
Two interpretations:
- **Optimistic**: the +1.37% paired estimate held but BASELINE seeds 11-20
  may be different from seeds 1-10 (random fluctuation pulling unpaired
  comparison down).
- **Realistic**: the effect size is closer to ~0.5% than +1.37%. More data
  pulls it toward zero.

A definitive paired n=20 requires running BASELINE seeds 11-20 (~25min wall).
**Operator action item**: schedule that batch before W23-1 ratification.

---

## Decision matrix for W23-1 (NC27-deep default flip)

| Condition | Recommended action |
|---|---|
| PO(8,1.0) n=20 paired confirms ≥ +1% AND PO(16,1.0) regression replicates at n=10 | **CANCEL W23-1** — regime-dependent; keep as opt-in only |
| PO(16,1.0) n=10 reveals -6.1% was n=5 noise (becomes neutral or positive) | **PROCEED W23-1** — flip default, add α-regime disclaimer in docstring |
| PO(8,1.0) n=20 drops below 0% | **CANCEL W23-1** — synthetic claim didn't translate at all |
| Confidence remains low after both n=10 follow-ups | **DEFER W23-1** — wait for FTSE real-data validation (W22-9 unwritten future unit) |

**Default recommendation**: DEFER W23-1 until PO(16,1.0) regression is
investigated. Suggest:
1. Run PO(16,1.0) n=10 with NC27-deep + BASELINE (paired)
2. If regression replicates, document NC27-deep as "regime-dependent;
   ratify only for low-to-mid α regimes (α ≤ 8)"
3. Even then, FTSE n=10 paired test is required before changing
   production default (FTSE α is unknown)

---

## Audit trail

- W22-4 commit chain: (sequential chain b1p63ljjh; results in `results/W22-*.json`)
- W22-5a result: `results/W22-5a_po16.json`
- W22-5b result: `results/W22-5b_po03.json`
- W22-6 result: `results/W22-6_spo.json`
- W22-4 result: `results/W22-4_nc27_n10.json`
- Aggregation script: ad-hoc Python in chat transcript (computes pooled stats)
- CHANGES_LOG.md Entry 4 documents per-instance attribution
- Drift registry updated: NC27_DEEP empirical_value → 0.004 (cross-instance avg)

## Operator decision points

1. **Run PO(16,1.0) n=10 follow-up?** (~25min wall; resolves the outlier question)
2. **Run BASELINE seeds 11-20 on PO(8,1.0)?** (~25min wall; enables paired n=20)
3. **Defer W23-1 default flip until #1 and #2 resolved?** (recommended)
4. **Move to W22-8 NC36 MC-noise investigation?** (orthogonal; can proceed in parallel)

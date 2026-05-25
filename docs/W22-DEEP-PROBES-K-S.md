# W22 Deep Probes K-S: theory-motivated hypothesis-driven probes

*Designed 2026-05-19 per operator directive: "more sound theory/math/stat/algo motivated fixes and probes."*

## Index

- [K — KF consistency via NIS / NEES (Bar-Shalom)](#probe-k--kf-consistency-via-nis--nees)
- [L — Q estimation from innovation autocorrelation (Mehra 1970)](#probe-l--q-estimation-from-innovation-autocorrelation)
- [M — F-matrix model selection: CV vs RW vs AR(1) vs IMM (Li & Jilkov 2003)](#probe-m--f-matrix-model-selection-cv-vs-rw-vs-ar1)
- [N — ANOVA variance decomposition of Ŝ (Pinheiro & Bates)](#probe-n--anova-variance-decomposition-of-ŝ)
- [O — Per-portfolio dominance probability distribution shape](#probe-o--per-portfolio-dominance-probability-distribution-shape)
- [P — Mutual information KF state ↔ OOS Ŝ (Kraskov 2004)](#probe-p--mutual-information-kf-state--oos-ŝ)
- [Q — HV-contribution bootstrap CI on (ASMS Ŝ − SMS Ŝ)](#probe-q--hv-contribution-bootstrap-ci-on-asms-ŝ--sms-ŝ)
- [R — Pareto front curvature analysis (Auger et al. 2009)](#probe-r--pareto-front-curvature-analysis)
- [S — Intra-period stationarity test (ADF / KPSS / GARCH)](#probe-s--intra-period-stationarity-test)

---

## Probe K — KF consistency via NIS / NEES

**Theory** (Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*, §5.4):

A correctly-specified KF satisfies:
- **NIS** (Normalized Innovation Squared): `ε_z = νᵀ S⁻¹ ν` with `ν = z - H x̂⁻`, `S = H P⁻ Hᵀ + R`. Must be χ²-distributed with dim(z)=2 d.o.f. → mean ≈ 2.
- **NEES** (Normalized Estimation Error Squared): `ε_x = (x̂ - x_true)ᵀ P⁻¹ (x̂ - x_true)`. Must be χ²₄.

NIS is computable online (no ground-truth); NEES requires held-out truth.

**Hypotheses**:
- H0: Mean NIS lies inside 95% χ²₂ acceptance interval `[1.39, 2.78]`.
- H1: Biased high (Q under-estimated → over-confident) OR low (R over-estimated).

**Methodology**:
1. Instrument KF.update to return `(ν, S, ε_z)` per (period, portfolio).
2. Compute time-averaged NIS `ε̄ = (1/N) Σ ε_z` and its 95% CI `[χ²_{2N, 0.025}/N, χ²_{2N, 0.975}/N]`.
3. Stratify by dataset (FTSE, PO(8,1.0), sPO(8,1.0)-cosine) and by within-period regime.

**Decision criteria**:
- NIS high on PO(8,1.0) but inside bounds on FTSE → formalizes "regime-change is the boundary condition"
- NIS biased high on FTSE too → motivates Probe L (Q estimation)
- NIS fine but ASMS wins → win is NOT filter sharpness; pivots to Probe P (decision-level invariance)

**Effort**: New instrumentation (lightweight: return tuple from KF.update); analyzer-only after that.

---

## Probe L — Q estimation from innovation autocorrelation

**Theory** (Mehra, R. K. (1970), "On the identification of variances and adaptive Kalman filtering," IEEE TAC):

If F is correct but Q is misspecified, the innovation sequence `{ν_t}` is COLORED (autocorrelated). White innovations confirm the (F, Q, R) triple. Q can be recovered from innovation autocorrelations `C_k = E[ν_t ν_{t-k}ᵀ]`.

**Hypotheses**:
- H0: Innovation sequence `{ν_t}` per portfolio is white (Ljung-Box p > 0.05 at lag 5).
- H1: Innovations autocorrelated → optimal Q > 0 exists.

**Methodology**:
1. After Probe K instrumentation, log innovation sequence per (algorithm, seed, dataset, portfolio_id) with ≥ 20 KF cycles.
2. Ljung-Box Q-statistic per component, joint test.
3. Estimate Q̂ via Mehra method OR EM (Shumway & Stoffer §6.3).
4. Compare `tr(Q̂)` across regimes — large `Q̂` indicates non-stationarity the filter is absorbing.

**Decision criteria**:
- `Q̂ > 0` significantly AND varies across regimes → ship Q̂-adaptive KF as NC23.
- `Q̂ ≈ 0` everywhere → filter correctly specified BUT F-model is wrong → pivot to Probe M.

**Effort**: Analyzer-only over Probe K's innovation logs. One follow-up smoke with adaptive-Q KF if H1.

---

## Probe M — F-matrix model selection: CV vs RW vs AR(1)

**Theory** (Li & Jilkov (2003), "Survey of maneuvering target tracking. Part I: Dynamic models," IEEE TAES):

Paper Eq 11 picks constant-velocity (CV): `F_CV[0,2] = F_CV[1,3] = 1`. Alternatives:
- **Random walk** `F_RW = I` — ignores velocity (memoryless)
- **AR(1) on velocity** `F_AR[2,2] = F_AR[3,3] = α`, `α ∈ (0,1)` — velocity-damped
- **IMM** (Interacting Multiple Model) — Bayesian mix of CV+RW with regime probability

PO(8,1.0) discontinuities specifically violate CV (velocity undefined at jumps).

**Hypotheses**:
- H0: Per-portfolio log-likelihood `Σ log N(ν_t; 0, S_t)` maximized by `F_CV`.
- H1: Different F (or IMM) beats `F_CV` on (i) PO(8,1.0); (ii) periods following large `|Δ ROI|` returns.

**Methodology**:
1. Offline re-score logged measurement sequences under each F variant (NO algorithm re-run).
2. Compute BIC `= -2 log L + k log N` for model selection (penalizes α parameter).
3. Stratify by dataset + period regime (large-Δ vs small-Δ).
4. Report fraction of (portfolio, period) cells where each F wins.

**Decision criteria**:
- `F_RW` wins on PO(8,1.0) but `F_CV` wins on FTSE → ship dataset-conditional F (NC24).
- IMM wins universally → ship IMM (heavier code but theoretically clean).

**Effort**: Analyzer-only (offline re-scoring). Optional new smoke with chosen F.

---

## Probe N — ANOVA variance decomposition of Ŝ

**Theory** (Pinheiro & Bates, *Mixed-Effects Models in S and S-PLUS*, §1):

Reported gaps sit on top of unknown noise floors. Mixed-effects ANOVA:
```
Var(Ŝ) = σ²_alg + σ²_seed + σ²_period + σ²_data + σ²_alg×data + σ²_resid
```

**Hypotheses**:
- H0: After accounting for seed and period variance, algorithm main-effect is significant on every dataset (F-test p < 0.05).
- H1: `σ²_alg` dominated by `σ²_seed × σ²_period` interaction → reported gap partly seed-lucky.

**Methodology**:
1. Pool per-(alg, seed, period, dataset) Ŝ records from existing smokes.
2. Fit `Ŝ ~ algorithm + (1|seed) + (1|period) + (1|dataset) + algorithm:dataset` via `statsmodels.formula.mixedlm` or scipy linear-mixed-model wrapper.
3. Report variance components, ICC, partial η² per effect.
4. Parametric bootstrap (B=1000) CIs on each variance component.

**Decision criteria**:
- `σ²_alg / σ²_total < 0.05` on any dataset → flag headline gap as inconclusive there; require more seeds.
- Large `algorithm:dataset` interaction → confirms regime-dependent ASMS wins; motivates regime-classifier preconditioner.

**Effort**: Analyzer-only on existing Ŝ logs. **HIGHEST IMMEDIATE LEVERAGE** — uses data we already have.

---

## Probe O — Per-portfolio dominance probability distribution shape

**Theory** (Goldberg & Deb (1991), "A comparative analysis of selection schemes used in genetic algorithms"; current finding: TIP saturated at 0.5 but ASMS still wins):

Even with `λ̄ ≈ 0.5` (saturated TIP), the *variance* `Var(λ_i)` across portfolios may carry signal. Heavy-tailed dominance probability `p_i` means a small subset of "highly anticipated" portfolios drive selection differential.

**Hypotheses**:
- H0: Per-period distribution `{p_i}_{i=1..N_pop}` is symmetric and unimodal (close to uniform under saturation).
- H1: Right-skewed/heavy-tailed (Gini > 0.3, kurtosis > 4) → small subset drives differentiation.

**Methodology**:
1. Per-period log of `{p_i}` (from existing TIP MC logs).
2. Compute: mean, median, Gini coefficient, kurtosis, top-10% mass, KS distance to uniform.
3. Stratify by dataset.
4. Spearman ρ between Gini and (Jaccard distance of selected sets across ASMS vs SMS).

**Decision criteria**:
- `Gini ↑ ⟹ Jaccard ↑` (significant ρ > 0.3) → win mechanism is per-portfolio differentiation under heavy tails; falsifies "TIP saturation benign" framing; motivates Gini-aware anticipative rate.
- No correlation → confirms saturation is benign as currently believed.

**Effort**: Analyzer-only on existing per-portfolio λ/dominance logs.

---

## Probe P — Mutual information KF state ↔ OOS Ŝ

**Theory** (Cover & Thomas, *Elements of Information Theory*, §8; Kraskov et al. (2004), Phys. Rev. E 69:066138):

The paradox: KF predictions are WORSE than persistence (MAE −61/−78%) yet ASMS wins. MI captures whether KF state preserves ORDERING over portfolios even when pointwise prediction is bad. MAE penalizes magnitude error symmetrically; MI captures decision-relevant information.

**Hypotheses**:
- H0: `I(x̂_KF[ROI] ; Ŝ_OOS) ≤ I(ROI_persistence ; Ŝ_OOS)`.
- H1: `I_KF > I_persistence` despite worse MAE → KF compresses noise in decision-useful way.

**Methodology**:
1. Per (dataset, period), build sample `{(x̂_i, Ŝ_OOS,i)}` across portfolios.
2. Estimate MI via k-NN (Kraskov 2004 estimator) — robust to non-Gaussianity.
3. Compare against persistence-baseline MI.
4. Permutation test (B=500) for significance.

**Decision criteria**:
- H1 confirmed → reframe KF as DECISION COMPRESSOR, not predictor; MAE was wrong metric; design future evaluation around rank-correlation / NDCG.
- H1 refuted → KF genuinely uninformative; win must come from non-KF source (NC8b selection-quality alone).

**Effort**: Analyzer-only on existing Probe A logs.

---

## Probe Q — HV-contribution bootstrap CI on (ASMS Ŝ − SMS Ŝ)

**Theory** (Efron & Tibshirani, *An Introduction to the Bootstrap*, §6):

Ŝ aggregates Pareto-front HV across periods. Headline gap could be inflated if HV-contribution `Δᵢ HV` of a few extreme portfolios dominates AND those portfolios are seed-sensitive. Bootstrap over portfolios gives non-parametric CI respecting within-period dependence.

**Hypotheses**:
- H0: 95% bootstrap CI on (ASMS Ŝ − SMS Ŝ) excludes 0 on FTSE.
- H1: CI includes 0 on at least one dataset where point estimate reported as positive → headline gap fragile.

**Methodology**:
1. Stratified block-bootstrap: per period, resample portfolios with replacement keeping period structure.
2. Recompute HV per period; sum; subtract.
3. Repeat B=1000. Report (median, 2.5%, 97.5%) CI.
4. Jackknife-after-bootstrap pseudo-values to identify gap-driving periods.

**Decision criteria**:
- CI excludes 0 → robust; ship as headline.
- Period jackknife reveals 1-2 periods carry gap → flag externally-valid claim conservatively; consider regime-stratified reporting.

**Effort**: Analyzer-only on existing per-period HV logs. **HIGH IMMEDIATE VALUE** for validating headline claim.

---

## Probe R — Pareto front curvature analysis

**Theory** (Auger, Bader, Brockhoff & Zitzler (2009), "Theory of the hypervolume indicator," FOGA):

SMS-EMOA's HV-contribution selection favors knee regions of CONVEX Pareto fronts. If ASMS' KF-perturbed objectives systematically inflate/deflate front curvature, the win could be a GEOMETRIC ARTIFACT of HV's curvature bias, not predictive value.

**Hypotheses**:
- H0: ASMS and SMS produce statistically indistinguishable curvature distributions per period (KS p > 0.05).
- H1: ASMS systematically produces lower-curvature (flatter) fronts → wins via HV's flat-front preference.

**Methodology**:
1. Per period, sort Pareto set by ROI.
2. Compute discrete curvature `κ_i = angle(f_{i-1}, f_i, f_{i+1})` at each interior point.
3. Fit Beta distribution to angle distribution; KS-test ASMS vs SMS distributions.
4. Regress `(ASMS Ŝ − SMS Ŝ)` on `Δ(mean curvature)` across periods.

**Decision criteria**:
- `Δ curvature` explains > 50% of `ΔŜ` variance → most of win is geometric, not predictive; MAJOR REFRAME.
- No relationship → curvature incidental; attribution to anticipation stands.

**Effort**: Analyzer-only on existing Pareto-front logs.

---

## Probe S — Intra-period stationarity test

**Theory** (Phillips & Perron (1988); Dickey-Fuller; KPSS):

KF assumes state evolves Markovianly with stationary return distribution conditional on `x_k`. Real markets show intra-period regime shifts (earnings, central-bank decisions). If returns within a period are non-stationary, R is misspecified — should be heteroscedastic.

**Hypotheses**:
- H0: Within-period return series passes ADF stationarity test (p < 0.05) for ≥ 90% of (asset, period) cells on FTSE.
- H1: Substantial fraction fail → R should be time-varying (GARCH(1,1)) and current R is biased.

**Methodology**:
1. Per (asset, period), run ADF and KPSS on return series.
2. Cross-tabulate stationarity results.
3. For non-stationary cells, fit GARCH(1,1); report mean conditional variance vs fixed R = 0.01.
4. Compute implied effective gain ratio `K_GARCH / K_fixed`.

**Decision criteria**:
- R off by > 2× in many cells → measurement noise misspecified; motivates heteroscedastic R (NC25).
- R approximately correct → measurement model fine; failure elsewhere (F or Q).

**Effort**: Analyzer-only over existing return data (statsmodels.tsa.stattools).

---

## Coverage matrix

| Particular interest area | Covered by |
|---|---|
| KF consistency (NIS/NEES) | K |
| Q estimation from residuals | L |
| F-matrix model selection | M |
| ANOVA decomposition | N |
| Pareto front geometry | R |
| Cross-period MI (KF state ↔ Ŝ) | P |
| HV-contribution bootstrap variance | Q |
| Per-portfolio dominance distribution | O |
| Intra-period stationarity (R misspec) | S |

## Immediate priority (next round)

1. **Probe N (ANOVA)** — uses existing per-seed/period Ŝ data; validates headline claim variance decomposition
2. **Probe Q (bootstrap CI)** — uses existing per-period HV data; defends headline against artifacts
3. **Probe P (MI)** — uses existing predictions.jsonl; resolves the "KF worse but ASMS wins" paradox
4. **Probe K (NIS) + L (Q estimation)** — requires light instrumentation; enables NC23 process noise tuning
5. **Probe M (F-matrix)** — offline re-scoring; identifies whether F_CV is appropriate per dataset

Implementing K-M motivates the **NC23 (process noise Q) ship**: theory says Q > 0 stabilizes KF against unmodeled dynamics; current Q=0 is the strongest possible assumption and likely wrong.

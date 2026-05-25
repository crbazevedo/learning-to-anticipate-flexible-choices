# W22 Research Program — ASMOO / ASMS investigation matrix

*Generated 2026-05-19 per operator directive: expand investigations to
analytical/algebraic, not just statistical; propose sub-areas, approaches,
assumptions, contrarian/counterfactual assumptions across:
Dirichlet filter, anticipative distributions, TIP, historical Kalman
errors (and why not Dirichlet?), expected hypervolume theory, correspondence
mapping, multi-period discount, AMFC soundness; alternative predictive
signals (raw, causal, network); inspection of chain-of-computations.*

## Table of contents

1. [Area I — KF & Anticipation Filter Math (algebraic)](#area-i--kf--anticipation-filter-math)
2. [Area II — Dirichlet Filter (decision-space tracking)](#area-ii--dirichlet-filter)
3. [Area III — Anticipative Distributions](#area-iii--anticipative-distributions)
4. [Area IV — TIP Soundness & Numerical Stability](#area-iv--tip-soundness)
5. [Area V — Expected Hypervolume Theory (Eq 6.41)](#area-v--expected-hypervolume-theory)
6. [Area VI — Correspondence Mapping](#area-vi--correspondence-mapping)
7. [Area VII — Multi-Period Discount Model](#area-vii--multi-period-discount-model)
8. [Area VIII — AMFC Soundness & Signal Quality](#area-viii--amfc-soundness)
9. [Area IX — Alternative Predictive Signals](#area-ix--alternative-predictive-signals)
10. [Inspection Chain Catalog](#inspection-chain-catalog)

## Cross-cutting framing

The current ASMS framework couples THREE filters across spaces:
- **Objective-space** (ROI, risk) → Kalman Filter (Eq 11) → KF prediction
- **Decision-space** (weights) → "Dirichlet" predictor → weight forecast
- **Meta** (KF residuals) → λ^K computation → anticipation rate

Plus THREE decision constructs:
- **TIP** (mutual non-dominance probability) → λ^H per Eq 6.6
- **Anticipative distribution** (current ⊕ predicted) → blended state
- **AMFC** (argmax E[HV] over Pareto front) → implemented portfolio

The breakthrough (NC8c-v2+NC8d, Probe P insight) revealed that the
mechanism works via **per-portfolio differentiation** that the optimizer
exploits via rank-based selection (NDS, HV-contribution, tournament) —
even when point predictions are inaccurate (MAE -61%/-78% vs persistence
but MI 2-6× higher).

Operator directive now expands this to: examine ALL the predictive,
distributional, and decision constructs for ANALYTICAL soundness
(algebraic correctness, numeric stability), CONTRARIAN assumptions
(what if the underlying model is wrong?), COUNTERFACTUALS (what if
we replaced X with Y?), and ALTERNATIVE PREDICTIVE SIGNALS (raw series,
causal inference, asset network centrality).

---

## Area I — KF & Anticipation Filter Math

### Sub-area I.A: Filter structure

**Current** (paper Eq 11): state x = [ROI, risk, ROI_velocity, risk_velocity],
constant-velocity F matrix, F[0,2]=F[1,3]=1, F else identity.

**Algebraic inspection**:
- F is exactly the discrete-time constant-velocity model from target-tracking literature (Li & Jilkov 2003)
- Implicit dt = 1 period; if periods are unequal-length, F should be F_t with dt_t encoded
- Process noise Q = 0 — strongest assumption; means KF asserts zero "model error" → posterior covariance can only shrink (NEVER grow) outside the predict step's F P F^T propagation
- Measurement noise R = diag(0.01, 0.01); time-invariant; means we assume noise structure of OOS observations is constant across periods, regimes

**Contrarian assumption A.1**: F should be IDENTITY (random walk in state). Mathematically simpler; better when velocity is unmeasurable noise. Testable by Probe M (BIC model selection).

**Contrarian assumption A.2**: F should be parameterized AR(1) per velocity component: F[2,2] = α, F[3,3] = β. Damps velocity toward zero — captures mean-reversion. Test α, β via maximum likelihood on innovations.

**Counterfactual A.3**: What if we use a SCALAR KF per asset (per-asset returns, no portfolio aggregation)? 30 KFs running independently might capture asset-level dynamics better than 1 KF on aggregated (portfolio.ROI, portfolio.risk).

**Counterfactual A.4**: What if we use an **Ensemble KF** (EnKF) with N=50 particles, capturing non-Gaussian posterior? Removes Gaussianity assumption; handles heavy-tailed return distributions.

### Sub-area I.B: Q matrix (process noise)

**Theory**: Q > 0 prevents covariance underflow and absorbs unmodeled dynamics (regime shifts, sudden changes). Currently Q = 0 → KF cannot adapt.

**Approach 1 (Mehra 1970)**: estimate Q from innovation autocorrelation. C_k = E[ν_t ν_{t-k}^T]. If sequence is white, Q ≈ 0 is correct; if colored, optimal Q > 0 exists.

**Approach 2 (Sage-Husa adaptive KF)**: online Q estimation using exponentially weighted innovations.

**Approach 3 (operator-set Q)**: heuristic Q = α * R (process noise proportional to measurement noise) for α ∈ [0.01, 1.0].

**Contrarian B.1**: What if Q should be TIME-VARYING? Markets have regime shifts; Q_t could be larger in volatile periods (≈ GARCH-like).

### Sub-area I.C: R matrix (measurement noise)

**Theory**: R encodes observation noise. R = diag(0.01, 0.01) is HEURISTIC; not derived from data.

**Approach (data-driven R)**: estimate R from per-period RESIDUALS of the portfolio against its training MLE. Per asset i:
- σ²_observation_i = Var(daily_return_i - mean_return_i) over training window

Then R = u^T Σ_observation u where u = weights.

**Counterfactual C.1**: What if R should be heteroscedastic? Different assets / regimes have different noise. GARCH(1,1) per asset → R_t per period.

**Counterfactual C.2**: What if the observation model is WRONG? Currently H selects (ROI, risk) from state. But the actual portfolio observation may have additional noise (transaction-cost-adjusted ROI vs gross ROI; estimator vs realized risk).

### Sub-area I.D: Historical Kalman errors (operator-flagged)

**Current code**: `record_kf_residual` accumulates squared residuals; used by λ^K (Eq 6.9) when buffer is full.

**Analytical inspection**:
- What error metric is used? Squared innovation y^T y? Or normalized NIS y^T S^-1 y?
- W17-2 fix added the squared-residual accumulation; need to verify the formula
- Why use Squared Error rather than Mahalanobis (NIS) for the residual signal?

**Probe**: instrument residual logging more granularly. Check λ^K formula matches Eq 6.9.

---

## Area II — Dirichlet Filter

### Sub-area II.A: Is the current implementation a Dirichlet filter?

**Code** (DirichletPredictor.dirichlet_mean_prediction_vec):
```python
rate = 0.5 * anticipative_rate
prediction = prev + rate * (current - prev)
return prediction / sum(prediction)
```

**Algebraic verdict**: this is NOT a Dirichlet filter. It is exponential smoothing on raw weight vectors with re-normalization. A true Dirichlet filter would:

1. Maintain a Dirichlet posterior over the simplex with concentration parameter α_t
2. Update α_{t+1} via Bayesian update: α_{t+1} = α_t + observation_count
3. Predict mean = α / sum(α); predict variance = α(α_total - α) / (α_total² (α_total + 1))

**Counterfactual II.A.1**: Implement a TRUE Dirichlet filter.
- α_0 = uniform prior (Jeffreys' prior: 1/2 per component)
- α_t updated via observations of u*_{t-1} (the AMFC weights count as 1 "observation" each)
- Concentration parameter grows over time → narrower posterior → more confident prediction
- Predict α_{t+1} = α_t (no jump if no observation yet)
- Then mean weights = α_t / sum(α_t)

**Counterfactual II.A.2**: What if we use a Beta-binomial filter per ACTIVE/INACTIVE asset (binary)? Captures the cardinality constraint more naturally than a continuous Dirichlet.

**Counterfactual II.A.3**: What if we use a LOGISTIC-NORMAL filter (Aitchison's compositional data analysis)?
- Transform weights to log-ratio space: y_i = log(u_i / u_d) for i ≠ d (some reference asset)
- Standard KF on y space (since unconstrained)
- Inverse-transform back to simplex

This is the gold-standard for filtering compositional data. Compatible with the existing KF infrastructure.

### Sub-area II.B: Historical Dirichlet errors (operator-flagged)

**Operator question**: Why not historical Dirichlet errors as well?

**Analysis**:
- KF has λ^K = function of cumulative KF residuals (Eq 6.9)
- Dirichlet predictor has no equivalent error accumulator
- Symmetric construction would be: λ^D = function of cumulative Dirichlet residuals
- Combined anticipation rate: λ_combined = (1/3)(λ^H + λ^K + λ^D)

**Probe T (NEW)**: instrument Dirichlet prediction error per period (L1 or KL); accumulate over time; correlate with ASMS performance.

**Counterfactual II.B.1**: If Dirichlet has high errors, weight decision-space anticipation LESS. If KF has high errors, weight objective-space anticipation LESS. Adaptive blending.

---

## Area III — Anticipative Distributions

### Sub-area III.A: Joint vs marginal distributions

**Current implementation**: separate KF for (ROI, risk); single Dirichlet for weights. NO joint modeling of (state, weights) coupling.

**Algebraic question**: portfolio ROI and risk are FUNCTIONS of weights (ROI = w^T μ, risk = w^T Σ w). They are NOT independent. The KF treats them as independent observables; this loses information.

**Approach III.A.1**: Joint distribution over (μ, Σ, w). Wishart-Normal joint prior. Bayesian update. Heavier compute but theoretically clean.

**Approach III.A.2**: Skip joint modeling; instead use the KF prediction of (μ, Σ) at asset level, then COMPUTE portfolio (ROI, risk) from anticipated (μ, Σ) for each candidate weight. This is the "asset-level KF" approach (Counterfactual A.3 above).

**Contrarian III.A.3**: What if the (ROI, risk) of a portfolio is NOT well-summarized by mean and variance? Higher moments (skewness, kurtosis) matter for tail risk. Use moment-matching against historical returns.

### Sub-area III.B: NC12 Eq 15 formula

**Already inspected**: anticipative_covariance = w_c² Σ_c + w_p² Σ_p (per Eq 15) instead of naïve SUM.

**Lingering question**: are the weights w_c = w_p = 0.5 correct? The default assumes EQUAL weight on current and predicted. But if we have high confidence in current (just observed) and lower confidence in predicted (uncertain forecast), w_c > w_p might be optimal.

**Approach III.B.1**: ADAPTIVE weights based on KF prediction error history.
- Recent KF prediction errors small → trust prediction more → w_p ↑
- Recent KF errors large → distrust prediction → w_c ↑

---

## Area IV — TIP Soundness

### Sub-area IV.A: Definition consistency (operator-flagged: critical)

**Paper Definition 6.1**: TIP(t, t+h) = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]

The conditional `| ẑ_t` means: GIVEN we observe ẑ_t (treated as KNOWN), what is the probability ẑ_{t+h} is mutually non-dominated with ẑ_t?

**Code implementation** (`_calculate_tip_with_covariance`):
```python
for _ in range(self.monte_carlo_samples):
    # Sample from current distribution
    current_sample = np.random.multivariate_normal(
        [current_roi, current_risk], current_cov
    )
    # Sample from predicted distribution
    predicted_sample = np.random.multivariate_normal(
        [predicted_roi, predicted_risk], predicted_cov
    )
    # check mutual non-dominance
```

**⚠️ CRITICAL ISSUE**: The code samples BOTH current AND predicted as random variables. But per Definition 6.1, current is FIXED (we condition on it). This violates the conditional!

**Algebraic impact**:
- Joint sampling: estimates Pr[ẑ_t || ẑ_{t+h}] — the JOINT mutual-non-dominance probability
- Conditional sampling (correct): would fix current = observed value (e.g., current_roi, current_risk) and only sample predicted; estimates Pr[ẑ_t || ẑ_{t+h} | ẑ_t]
- The joint version OVER-ESTIMATES TIP because additional variance from current's randomness pushes pairs apart, more likely non-dominated

**Counterfactual IV.A.1**: Fix current = MEAN (no sampling); sample only predicted. This is the conditional version. Compare to current implementation: does TIP drop?

### Sub-area IV.B: Numeric stability

**Code**: `cov_oos = np.cov(arr, rowvar=False, ddof=1)` then `np.random.multivariate_normal(mean, cov_oos)`.

**Issue 1**: cov_oos may not be positive-definite if n_samples (200) is close to dimension (only 2 here, so OK). For larger problems would matter.

**Issue 2**: With NC13a clamp, predicted_cov is capped at 1.0. But for the COSINE sPO mode, predicted_cov may be much smaller. Clamping may HURT in low-cov regimes by upper-bounding when the true cov is fine.

**Issue 3 (numerical underflow)**: TIP MC produces fraction in [0, 1]; clamp_range = (0.05, 0.95) → never exactly 0 or 1 → log(TIP) is well-defined. Good.

**Issue 4**: `linear_entropy(p) = -p log(p) - (1-p)log(1-p)` only defined for p ∈ (0, 1). With clamp this is fine; without it, NaN risk.

### Sub-area IV.C: Theoretical alternative

**Counterfactual IV.C.1**: TIP could be replaced by **dominance probability**: Pr[ẑ_{t+h} dominates ẑ_t | ẑ_t]. This is a one-sided measure (vs TIP's two-sided "neither dominates"). More directly interpretable.

**Counterfactual IV.C.2**: TIP could be replaced by **expected hypervolume change**: E[ΔHV | ẑ_{t+h}]. More aligned with the optimization criterion.

---

## Area V — Expected Hypervolume Theory (Eq 6.41)

### Sub-area V.A: "Conditional over the mean" soundness (operator-flagged)

**Theorem 6.3.1 (thesis p.131)**: E[Δ_S(ẑ_t^(i))] = (m_1^(i) - m_1^(i-1))(m_2^(i+1) - m_2^(i)) + Cov(a, c) - Cov(a, d) - Cov(b, c) + Cov(b, d)

Where:
- a = ẑ_{1,t}^(i) | m̂_{2,t}^(i)  ← self ROI conditioned on self RISK MEAN
- b = ẑ_{1,t}^(i-1) | m̂_{2,t}^(i-1)
- c = ẑ_{2,t}^(i+1) | m̂_{1,t}^(i+1)
- d = ẑ_{2,t}^(i) | m̂_{1,t}^(i)

**Algebraic question**: For bivariate Gaussian, E[X | Y = μ_Y] = μ_X + (σ_XY/σ²_Y)(μ_Y - μ_Y) = μ_X. So conditioning on the MEAN doesn't change the mean!

So **the conditional framing is a NO-OP for the mean term**. The first term (m_1^(i) - m_1^(i-1))(m_2^(i+1) - m_2^(i)) is just the unconditional mean product of differences.

**The conditional matters only for the covariance terms** Cov(a, c) etc. For independent solutions i, i-1, i+1, all cross-Cov terms = 0. The only non-zero one is Cov(a, d) = Cov(ẑ_{1,t}^(i), ẑ_{2,t}^(i)) = KF P[0, 1] for that solution.

**Counterfactual V.A.1**: Is the conditioning even theoretically justified?
- The derivation assumes (a, b, c, d) are conditional Gaussian random variables
- But they're actually the OBJECTIVES of different Pareto-front portfolios
- Different portfolios have different (ROI, risk) — they're not draws from the same distribution
- The variance/covariance interpretation is muddled

**Probe U (NEW)**: re-derive Eq 6.41 from FIRST PRINCIPLES. Is the formula correct? Are there missing terms?

### Sub-area V.B: Hypervolume vs single-point HV vs ε-indicator

**Current**: AMFC uses single-point HV contribution to z_ref. Alternative quality indicators:
- Multiplicative ε-indicator (Zitzler & Künzli 2004)
- IGD (inverted generational distance)
- Coverage metric C(A, B)

**Counterfactual V.B.1**: Use ε-indicator for AMFC: argmax over front of (1 - ε_to_reference_set). Less sensitive to z_ref choice.

### Sub-area V.C: Truncated formula scar (NC EQ1)

Code comment: "Python lacks per-solution MC samples; cross-pair Cov(a,c), Cov(b,c), Cov(b,d) are 0; only within-solution Cov(a,d) = KF P[0,1] retained."

**Algebraic question**: If we IMPLEMENT per-solution MC sampling (matching C++ legacy-cpp-v2), we'd get non-zero cross-pair Covs. Would this change the AMFC ranking? Could be the missing signal.

**Probe V (NEW)**: instrument per-solution MC sample buffer (size 100); compute the full Eq 6.41 with all 4 Cov terms; compare AMFC ranking against truncated version.

---

## Area VI — Correspondence Mapping

### Code inspection needed.

Per `multi_horizon_anticipatory.py` docstrings, "correspondence_mapping" relates Pareto-front portfolios across generations / periods. Used by TIPIntegratedAnticipatoryLearning.

**Question**: How does it work? Is it sound? Does it preserve identity of portfolios across times?

**Probe W (NEW)**: instrument correspondence_mapping; verify it produces a valid bipartite matching (1:1) and that the matching scores correlate with portfolio similarity.

---

## Area VII — Multi-Period Discount Model

### Sub-area VII.A: Horizon weighting

**Current** (multi_horizon learn_population): w_0 = 1 - Σλ, w_h = λ_h per horizon h. λ_h from Eq 7.16 (or 1-TIP under v2 mode).

**Algebraic question**: Should distant horizons be DISCOUNTED more? Standard time-value of information: I(prediction at h) ∝ e^{-βh}.

**Counterfactual VII.A.1**: discount factor γ ∈ (0, 1):
- λ_h_eff = γ^h * λ_h
- More weight on near-term predictions
- Test γ ∈ {1.0 (no discount), 0.9, 0.7, 0.5}

**Counterfactual VII.A.2**: SOFT horizon — instead of discrete h ∈ {1, 2, ..., H-1}, integrate over continuous time with exponential kernel.

### Sub-area VII.B: Horizon selection

**Current**: H = 3 (in scenario name K=3 → 3 horizons including t).

**Probe X (NEW)**: sweep H ∈ {1, 2, 3, 5, 8, 16}; measure ASMS-SMS Δ; find optimal H per dataset.

---

## Area VIII — AMFC Soundness

### Sub-area VIII.A: z_ref choice

**Current**: z_ref = (0.2, 0.0) — risk_max=0.2, return_min=0.0.

**Algebraic question**: AMFC = argmax (ret - return_min) * (risk_max - var). The 0.2 and 0.0 are ARBITRARY constants. Different z_ref would change the ranking.

**Counterfactual VIII.A.1**: Use ADAPTIVE z_ref based on per-period Pareto front extrema:
- return_min_t = min(front.ROI) - margin
- risk_max_t = max(front.risk) + margin

This makes the z_ref scale-aware.

**Counterfactual VIII.A.2**: Use HYPERVOLUME EXCESS over multiple z_refs (robust to z_ref choice).

### Sub-area VIII.B: Increasing AMFC signal (operator's central question)

**Diagnostic from Probe C**: AMFC > Random (p=0.0002) but Spearman ρ(predicted HV, realized HV) = 0.24 only. Mildly informative.

**Approaches to increase signal**:

1. **Better predictor**: replace OOS bootstrap (cheating) with KF-prediction-based EFHV. We've discussed but not shipped.
2. **Ensemble DM**: median of {AMFC, HighROI, Sharpe} — Probe C showed these are comparable; ensemble may be robust.
3. **Per-portfolio uncertainty-aware AMFC**: argmax of E[HV] / Var[HV] (Sharpe-like for HV) — penalizes high-uncertainty picks.
4. **Bayesian model averaging**: maintain posterior over z_ref; integrate AMFC over posterior.

**Probe Y (NEW)**: test each of (1)-(4); compare realized OOS Ŝ.

---

## Area IX — Alternative Predictive Signals

### Sub-area IX.A: Raw signal prediction

**Current**: KF on (ROI, risk) at portfolio level.

**Counterfactual IX.A.1**: Predict ASSET-LEVEL returns r_i,t+1 (30 assets, 30 separate KFs or 1 VAR model). Then compute portfolio (ROI, risk) as functions of asset-level predictions.

Advantage: more data per estimator (n × T returns) vs portfolio level (T portfolio observations).

**Counterfactual IX.A.2**: Predict CORRELATIONS (off-diagonal of Σ_t+1).
- Standard approach: DCC-GARCH (Dynamic Conditional Correlation)
- Correlations are slower-moving than means → more predictable
- Improved Σ prediction → improved portfolio risk prediction

**Counterfactual IX.A.3**: Predict DELAYED CHANGES in correlations.
- Lead-lag relationships: ΔΣ_{i,j,t+1} = f(ΔΣ_{k,l,t-d}, ...) for some lag d
- VAR(p) on flattened correlation tensor

### Sub-area IX.B: Causal inference

**Counterfactual IX.B.1**: GRANGER CAUSALITY between asset returns.
- Test: does asset i's past returns Granger-cause asset j's future returns?
- Build directed causality graph G(assets, edges)
- Use G to weight portfolio selection: prefer portfolios that include "causal hub" assets

**Counterfactual IX.B.2**: TRANSFER ENTROPY (non-linear generalization of Granger).
- Information-theoretic causality
- Captures non-linear dependencies (whereas Granger assumes linear)

**Counterfactual IX.B.3**: CAUSAL INTERVENTION via Pearl's do-calculus.
- "If we change weight on asset i by Δw, what is the predicted ΔROI?"
- Treats the portfolio as an INTERVENTION rather than passive observation

### Sub-area IX.C: Network/Graph-theoretic measures

**Counterfactual IX.C.1**: Build COMPLEX NETWORK from asset return correlations.
- Nodes = assets
- Edges = correlation strength (or transfer entropy, or Granger F-stat)
- Threshold to sparse network

**Counterfactual IX.C.2**: Compute NODE CENTRALITY measures:
- Degree centrality (number of strong correlations)
- Betweenness centrality (asset is on shortest paths between others)
- Eigenvector centrality (recursive importance)
- PageRank (random-walk-based importance)

**Counterfactual IX.C.3**: Use centrality as a FEATURE in portfolio scoring.
- High-centrality assets: capture market-wide moves → diversification benefit
- Low-centrality assets: idiosyncratic returns → alpha potential
- Mix per Markowitz + centrality penalty/bonus

**Counterfactual IX.C.4**: Time-varying network analysis.
- Build network per training window
- Track centrality dynamics → predict regime shifts
- Use regime predictions to gate ASMS vs SMS

---

## Inspection Chain Catalog

For each computation, walk the chain end-to-end:

### Chain 1: TIP computation
**Inputs**: solution.P.kalman_state.x[0:2] (current ROI/risk), .P[:2,:2] (current cov); predicted_solution.P.kalman_state.x_next[0:2], .P[:2,:2] (predicted cov)

**Steps**:
1. MC sample current ~ Normal(mean_curr, cov_curr) [WRONG per Defn 6.1]
2. MC sample predicted ~ Normal(mean_pred, cov_pred)
3. Check mutual_non_dominance(current_sample, predicted_sample)
4. TIP = mean(non_dominance) over MC samples
5. Clamp to (0.05, 0.95)
6. lambda_h = (1/(H-1)) * (1 - H(TIP)) where H = binary entropy

**Issues identified**:
- Step 1 violates conditional definition (sample current instead of fixing)
- Step 4 has no rank-based statistic (could use bootstrap CI for TIP)
- Step 5 clamping prevents log(0) but BIASES estimates near extremes
- Step 6 uses binary entropy H(p) = -p log(p) - (1-p) log(1-p); soundness: yes for Bernoulli, but TIP is a frequentist Probability not a posterior belief

### Chain 2: AMFC selection
(See `compute_per_portfolio_efhv` in oos_evaluator.py)

**Inputs**: pareto_weights (list of vectors), oos_returns (FUTURE DATA — methodological concern)

**Steps**:
1. For each MC scenario e=1..n_mc:
   a. Resample oos_returns with replacement
   b. Compute mu_e = mean, cov_e = covariance
   c. For each portfolio i: ret = mu_e @ w_i, var = w_i @ cov_e @ w_i; HV_e_i = max(0, ret-R1) * max(0, R2-var)
2. per_portfolio_efhv[i] = mean over e of HV_e_i
3. AMFC = argmax per_portfolio_efhv

**Issues**:
- Step 1 uses OOS returns DIRECTLY (oracle-like access; methodological concern but consistent with paper)
- Step 1c computes HV against FIXED z_ref = (R1=0.0, R2=0.2) — arbitrary
- Step 2 averages over MC scenarios: standard Monte Carlo
- Step 3 ties broken by smallest index

### Chain 3: anticipative_mean blend (Eq 14)
(See multi_horizon learn_population)

**Inputs**: solution's current_state [ROI, risk], predicted_states for each horizon h, lambda_rates

**Steps**:
1. anticipatory_state = w_0 * current_state + Σ w_h * predicted_state_h
   where w_0 = 1 - Σ λ_h
2. solution.P.ROI = anticipatory_state[0]
3. solution.P.risk = anticipatory_state[1]

**Issues**:
- Step 1 is a convex combination ON SIMPLEX of weights — correct
- Step 2-3 OVERWRITES solution.P (state mutation) — must be careful about downstream consumers seeing old vs new values
- Computational order: learn_population is called at START of each generation; selection runs immediately after

### Chain 4: Kalman F·x prediction
Already inspected; constant-velocity model assumption is the source of PO(8,1.0) failure.

---

## Priority order for next investigations

1. **TIP IV.A** (conditional vs joint sampling) — operator-flagged critical; algebra is wrong
2. **Eq 6.41 V.C** (re-implement with per-solution MC; restore truncated Cov terms)
3. **Dirichlet II.A** (replace exponential smoothing with TRUE Dirichlet posterior)
4. **AMFC VIII.B** (uncertainty-aware AMFC; potentially Sharpe-like HV)
5. **Multi-horizon VII.B** (sweep H; find optimal)
6. **Causal IX.B.1** (Granger causality network — separate stream, could be its own paper)
7. **Q-noise NC23** (Mehra estimation from innovation autocorrelation)
8. **Asset-level KF IX.A.1** (per-asset KFs instead of portfolio-aggregated)

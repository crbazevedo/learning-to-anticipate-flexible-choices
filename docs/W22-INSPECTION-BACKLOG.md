# W22 Inspection Backlog — controlled probes for ASMS module diagnosis

*Generated 2026-05-18 after the NC1-NC6+EQ1 bundle empirically REGRESSED
(PR #140 finding). Pivoting from smoke-variant chasing to controlled,
first-principles module probes.*

## First-principles framework

For ASMS to beat SMS on **any** dataset, ALL of the following must hold:

| Condition | What it means | Why required |
|---|---|---|
| C1 | KF predicts t+1 (ROI, risk) BETTER than naive persistence | Otherwise anticipation arm has no signal to leverage |
| C2 | TIP varies meaningfully across periods (not stuck at 0.5) | TIP-saturated state mathematically kills λ^H |
| C3 | λ (anticipation rate) varies per-portfolio per-period | Constant λ across population erases differentiation |
| C4 | Anticipative distribution parameters are non-degenerate | Zero-eigenvalue cov → numerical pathology |
| C5 | AMFC picks Pareto-front portfolios whose realized OOS Ŝ correlates with predicted E[HV] | Otherwise DM is uninformative |
| C6 | Pareto front has enough portfolios for the DM to meaningfully choose | pop=20 may produce front of 1-3 → trivial choice |
| C7 | Dirichlet predictor for portfolio composition is informative | Decision-space anticipation analog of C1 |

**Currently we have ZERO evidence on any of C1-C7.** All prior W17-W22 work tested OUTPUTS (OOS Ŝ); none tested whether the INPUTS to those outputs are functioning as designed.

This backlog rectifies that.

---

## Probe spec template

Each probe follows this template:

```
### Probe X: <Name>

**Motivation** (first-principles): why this matters
**Code paths inspected**: file:line refs
**Methodology**: inspection approach
**Tools**: specific analytical / statistical / mathematical instruments
**Hypothesis (H0/H1)**: falsifiable null + alternative
**Tests**: statistical tests with significance levels
**Metrics / KPIs**: what we measure
**Visualizations**: specific plots
**Decision criteria**: if confirmed/refuted, what we do
**Wall-clock budget**: estimated time
**Output artifact**: where the analysis lands
```

---

## Probe A: KF predictive accuracy vs naive persistence

**Motivation** (C1): the entire anticipation arm is predicated on KF being a better predictor than simply assuming `(ROI_{t+1}, risk_{t+1}) = (ROI_t, risk_t)`. If our constant-velocity KF model is wrong for FTSE 2006-2012 (which includes 2008 crash), KF predictions may be NOISIER than persistence — making anticipation a negative-information signal.

**Code paths inspected**:
- `python_refactor/src/algorithms/kalman_filter.py:73-141` (predict + update)
- `python_refactor/src/algorithms/kalman_filter.py:144-178` (paper Eq 11 F matrix, constant-velocity)
- `python_refactor/src/algorithms/sms_emoa.py:_evaluate_solution:345-360` (per-solution KF update site)

**Methodology**:
1. Add per-period logging to `_evaluate_solution`: capture `(period_t, portfolio_id, predicted_t+1_state, observed_t+1_state, persistence_state)` tuples.
2. Run a single ASMS_mHDM_K3 scenario at n=3 seeds, dump JSON log of all tuples.
3. Post-hoc: compute prediction-error MAE/RMSE for KF vs persistence on each (ROI, risk) component.
4. Per-portfolio time-series: KF prediction trajectory vs actual observation trajectory.

**Tools**:
- pandas for tuple aggregation
- scipy.stats.wilcoxon for paired non-parametric test
- scipy.stats.ttest_rel for paired t-test (parametric counterpart)
- numpy.corrcoef for prediction-actual correlation
- matplotlib for trajectory + residual plots

**Hypothesis**:
- **H0**: `|KF_pred − actual|` ≥ `|persistence − actual|` (KF is NO BETTER than persistence)
- **H1**: `|KF_pred − actual|` < `|persistence − actual|` (KF is BETTER)

**Tests**:
- Paired Wilcoxon signed-rank test (one-sided), separately for ROI and risk
- Significance level: α = 0.05 (Bonferroni: α/2 = 0.025 per objective)
- Effect size: relative reduction in MAE (KF MAE / persistence MAE)

**Metrics / KPIs**:
- MAE_ROI(KF), MAE_ROI(persistence), MAE_risk(KF), MAE_risk(persistence)
- Bias_ROI(KF) = mean(KF_pred_ROI − actual_ROI); similar for risk (should be ≈ 0)
- corr(KF_pred, actual) per component; should be > 0.5 for useful prediction
- Wilcoxon p-values; effect-size ratios

**Visualizations**:
- 2×2 residual histogram: rows = ROI/risk, cols = KF/persistence
- Time-series of KF prediction trajectory vs actual (per portfolio, sampled)
- Scatter: KF_pred vs actual with 45° line (perfect prediction)

**Decision criteria**:
- If H0 rejected for both ROI AND risk → KF is functional → anticipation has signal → focus on downstream conditions C2-C7
- If H0 NOT rejected for either → **KF model is wrong** → either replace constant-velocity F matrix or remove KF entirely → ASMS cannot beat SMS until KF is fixed
- If H0 rejected for ROI only → KF helps for return prediction but not risk → consider asymmetric anticipation (only use ROI anticipation)
- If KF has significant bias → systematic prediction error → likely from R matrix issue (NC2 already addressed but may need re-verification)

**Wall-clock budget**: ~2h (45min logging instrumentation + 30min smoke run + 45min analysis)

**Output artifact**: `docs/W22-PROBE-A-KF-PREDICTIVE-ACCURACY.md` + raw JSON at `python_refactor/experiments/results/w22-probe-a-kf-accuracy/per_period_predictions.json`

---

## Probe B: TIP + λ + anticipation_rate signal distributions

**Motivation** (C2 + C3): TIP saturation (Reading C from W18) was the original W17-5 saturation chain. If TIP is stuck at 0.5 most periods, λ^H collapses to 0. If λ_combined is uniform across portfolios, no differentiation.

**Code paths inspected**:
- `python_refactor/src/algorithms/temporal_incomparability_probability.py` (TIP MC + binary entropy)
- `python_refactor/src/algorithms/anticipatory_learning.py:1090` (anticipation_rate = 1 − nd_probability)
- `python_refactor/src/algorithms/multi_horizon_anticipatory.py:165-244` (λ_combined per Eq 7.16)

**Methodology**:
1. Instrument logging in `compute_anticipatory_learning_rate` to dump `(period, portfolio_id, TIP, λ_h, λ_k, λ_combined, anticipation_rate)`.
2. Run ASMS_mHDM_K3 at n=3 seeds.
3. Post-hoc histograms per period, per-population.

**Tools**:
- scipy.stats.entropy (Shannon) for distribution entropy
- scipy.stats.kstest for distribution comparison (uniform / Beta hypothesis)
- numpy.histogram + matplotlib for KDE plots
- Cluster-analysis: scipy.cluster.hierarchy for bimodality detection

**Hypothesis**:
- **H0 (TIP saturation)**: TIP distribution across periods × portfolios has > 50% of mass within (0.45, 0.55) → TIP is functionally constant
- **H1 (TIP varies)**: TIP distribution has Shannon entropy > 1.5 nats (substantial spread)

For λ:
- **H0**: λ_combined coefficient of variation < 0.2 across portfolios within a period → no per-portfolio differentiation
- **H1**: CoV > 0.2 → meaningful per-portfolio signal

**Tests**:
- Kolmogorov-Smirnov test of TIP histogram vs Uniform[0,1]
- Coefficient of variation of λ within each period
- Shannon entropy of TIP marginal

**Metrics / KPIs**:
- Fraction of TIP samples in (0.45, 0.55) — target: < 30% for non-saturated
- Shannon entropy of TIP marginal (bits) — target: ≥ 1.5
- λ_combined CoV per period — target: ≥ 0.2 (meaningful spread)
- λ_combined min, median, max per period

**Visualizations**:
- KDE of TIP per period (small multiples)
- Stacked violin: λ_h vs λ_k vs λ_combined across periods
- Heatmap of anticipation_rate (period × portfolio_id)

**Decision criteria**:
- If TIP is saturated (H1 rejected) → Reading C confirmed at code level → need different uncertainty measure (e.g., raw variance ratio, not symmetric TIP)
- If λ has low CoV → anticipation is uniform → selection sees no signal → fix the per-portfolio differentiation (probably λ^K min-max per Eq 6.9, the EQ2 finding)
- If both signals are healthy → bottleneck is downstream (probes D/E/C)

**Wall-clock budget**: ~1.5h

**Output artifact**: `docs/W22-PROBE-B-SIGNAL-DISTRIBUTIONS.md`

---

## Probe C: AMFC vs alternative DM sensitivity analysis

**Motivation** (C5): the DM (decision maker) picks ONE portfolio from the Pareto front. AMFC = argmax E[single-point HV]. If this formula picks portfolios that don't actually maximize realized OOS Ŝ, the algorithm is producing good fronts but the DM is throwing them away.

**Code paths inspected**:
- `python_refactor/experiments/walk_forward.py:226-228` (AMFC selection)
- `python_refactor/experiments/oos_evaluator.py:_select_amfc_index`
- `python_refactor/experiments/oos_evaluator.py:compute_per_portfolio_efhv`

**Methodology**:
1. Modify `walk_forward.run_walk_forward` to log the FULL Pareto front + AMFC index per period.
2. Re-evaluate (offline) each period's Pareto front under multiple DMs:
   - AMFC (current): argmax E[single-point HV via MC]
   - AMFC-CF (closed-form): argmax E[single-point HV via Option A closed-form]
   - HighROI: argmax mean predicted ROI
   - LowRisk: argmin mean predicted risk
   - Sharpe: argmax (ROI − R1) / sqrt(risk − R2 + ε)
   - Median: 50th percentile by single-point HV
   - First: weights[0] (the W17-4 fallback)
   - Random: 100 random picks averaged
   - **Oracle**: argmax of *actual realized* HV (uses post-hoc OOS observations; cheating, upper bound)

**Tools**:
- pandas for per-period DataFrame of front + DM choices + realized Ŝ
- numpy for vectorized DM scoring
- scipy.stats.kendalltau / spearmanr for DM-vs-realized rank correlation
- matplotlib for OOS Ŝ comparison across DMs

**Hypothesis**:
- **H0**: AMFC OOS Ŝ ≤ Random OOS Ŝ (AMFC is no better than random) ← worst case
- **H1**: AMFC > Random; comparable to Oracle within 20%
- **H2**: AMFC ≥ all alternative single-criterion DMs (current is justified)

**Tests**:
- Paired Wilcoxon: AMFC OOS Ŝ vs each alternative
- Rank correlation: AMFC's `per_portfolio_efhv` ranking vs realized OOS Ŝ ranking per period (Spearman)
- Effect size: (AMFC mean Ŝ − Oracle mean Ŝ) / Oracle mean Ŝ (gap-to-perfect)

**Metrics / KPIs**:
- Per-period: AMFC's Ŝ, alternative DMs' Ŝ, Oracle's Ŝ
- Aggregate: mean OOS Ŝ per DM
- Rank correlation: AMFC's E[HV] ordering vs actual Ŝ ordering (Spearman ρ)
- Gap-to-Oracle: (Oracle − AMFC) / Oracle

**Visualizations**:
- Bar chart: mean OOS Ŝ per DM (with confidence intervals)
- Heatmap per period × DM showing rank within front
- Scatter: AMFC's E[HV] vs realized OOS Ŝ (per portfolio); should be along diagonal if AMFC is informative

**Decision criteria**:
- If AMFC ≈ Random → DM is uninformative; **replace DM** (probably HighROI or Sharpe)
- If Oracle >> AMFC + alternatives → DM is hard problem; consider ensemble (median of DMs)
- If AMFC > alternatives → DM is justified; problem is upstream

**Wall-clock budget**: ~2h (more analysis-heavy)

**Output artifact**: `docs/W22-PROBE-C-AMFC-VS-ALTERNATIVES.md`

---

## Probe D: Pareto front diversity per period

**Motivation** (C6): SMS-EMOA with pop=20, gens=30, cardinality 5-15 may produce a Pareto front of only 1-3 portfolios. If the DM has 1 portfolio to "choose" from, all DMs reduce to the same trivial pick → DM analysis (Probe C) becomes meaningless.

**Code paths inspected**:
- `python_refactor/src/algorithms/sms_emoa.py:_update_pareto_front`
- `python_refactor/src/algorithms/sms_emoa.py:_fast_non_dominated_sort`

**Methodology**:
1. Log per-period: front size, Pareto-rank histogram, distinct-portfolio count, ROI/risk extrema.
2. Run ASMS_mHDM_K3 at n=3 seeds.

**Tools**:
- pandas for per-period DataFrame
- numpy.unique for distinct counting
- matplotlib for distribution plots

**Hypothesis**:
- **H0**: typical front size ≥ 5 (enough choices for DM)
- **H1**: typical front size < 5 (front is sparse)

**Metrics / KPIs**:
- Median front size across periods
- Pareto-rank distribution (1st rank only? multiple ranks?)
- ROI range of front (max − min) per period
- Risk range of front per period

**Visualizations**:
- Time-series of front size per period (per seed)
- Histogram of front sizes pooled across periods
- ROI vs risk scatter of all portfolios per period (color = rank)

**Decision criteria**:
- If front size < 5 typically → **increase pop/gens** (the H2 hypothesis from the algo-param agent: thesis pop=200, gens=500)
- If front size > 10 → DM has enough choices; problem isn't this
- If front has all rank-1 (no dominated portfolios visible) → NDS may be too aggressive

**Wall-clock budget**: ~1h

**Output artifact**: `docs/W22-PROBE-D-PARETO-FRONT-DIVERSITY.md`

---

## Probe E: Anticipative distribution parameter sanity

**Motivation** (C4): the anticipative distribution combines current state + KF prediction. If parameters degenerate (zero eigenvalues in cov, runaway mean drift), the anticipation arm is numerically broken regardless of inputs.

**Code paths inspected**:
- `python_refactor/src/algorithms/anticipatory_learning.py:_create_anticipative_distribution`
- `python_refactor/src/algorithms/anticipatory_learning.py:_update_solution_state_anticipative:1374-1405`

**Methodology**:
1. Log per-period × per-portfolio: `(anticipative_mean, anticipative_covariance, observed_state_mean, observed_state_cov, KF predicted mean/cov)`.
2. Compute eigenvalue spectra of anticipative_covariance.
3. Detect degeneracy: near-zero eigenvalues, condition number blow-up.

**Tools**:
- numpy.linalg.eigvalsh for symmetric eigenvalue computation
- numpy.linalg.cond for condition numbers
- scipy.stats.shapiro / normaltest for Gaussianity assumption check

**Hypothesis**:
- **H0**: anticipative_covariance is positive-definite with condition number < 1e6 across all periods × portfolios
- **H1**: at least 5% of (period, portfolio) instances have degenerate covariance

**Metrics / KPIs**:
- Min eigenvalue distribution
- Condition number distribution
- Fraction of degenerate instances
- Trace of anticipative_covariance over time (should not blow up)

**Visualizations**:
- Heatmap of min(eigenvalue) per (period, portfolio)
- Time-series of condition number per portfolio
- Histogram of anticipative_mean drift (delta from observed_state_mean)

**Decision criteria**:
- If degeneracy > 5% → fix the source (likely related to KF R-clobbering NC2, or numerical pathology in StochasticParams)
- If condition numbers stable < 1e3 → distribution is healthy
- If mean drift large → KF state-evolution unstable → revisit constant-velocity F matrix

**Wall-clock budget**: ~1.5h

**Output artifact**: `docs/W22-PROBE-E-ANTICIPATIVE-DISTRIBUTION-SANITY.md`

---

## Probe F: Dirichlet filtering + prediction informativeness

**Motivation** (C7): the Dirichlet predictor tracks portfolio composition (weights as Dirichlet-distributed parameters) as a decision-space analog to the KF objective-space anticipation. If the Dirichlet predictions are uninformative — i.e., the predicted weights don't correlate with the t+1 actually-realized weights — then anticipation in decision space has no signal.

**Code paths inspected**:
- `python_refactor/src/algorithms/anticipatory_learning.py:886` (Dirichlet predictor call site)
- `python_refactor/src/data/dirichlet_predictor.py` or equivalent (Dirichlet update/predict module)
- `legacy-cpp-v2/source/dirichlet.cpp` (paper-companion reference)

**Methodology**:
1. Log per-period × per-portfolio: `(dirichlet_params_t, predicted_weights_t+1, actual_weights_t+1, persistence_weights = current_weights)`.
2. Compute weight-prediction error metrics:
   - L1 distance: sum |predicted − actual|
   - KL divergence: D_KL(actual || predicted) if both are valid Dirichlet
   - Effective sample size ratio: ESS(predicted) / ESS(actual)
3. Compare to persistence baseline.

**Tools**:
- scipy.special.gammaln + Dirichlet PDF for KL computation
- scipy.stats.dirichlet for distribution sampling/fitting
- numpy.linalg.norm for L1/L2 distances
- matplotlib for trajectory + error plots

**Hypothesis**:
- **H0**: Dirichlet predicted weights ≈ persistence weights (no information added)
- **H1**: Dirichlet predicted weights are closer to actual t+1 weights than persistence

**Tests**:
- Paired Wilcoxon: L1(Dirichlet, actual) vs L1(persistence, actual)
- KL divergence comparison
- Per-asset bias: mean(predicted_i − actual_i) per asset

**Metrics / KPIs**:
- Mean L1 distance Dirichlet vs persistence
- KL divergence per period
- Per-asset prediction bias
- Cardinality preservation: does Dirichlet predict 5-15 active weights?

**Visualizations**:
- Per-asset prediction error histograms
- Time-series of L1 distance per portfolio
- Stacked area: predicted weight composition vs actual over time
- KL divergence per period

**Decision criteria**:
- If Dirichlet ≈ persistence → Dirichlet adds no info → simplify (drop Dirichlet, use persistence)
- If Dirichlet significantly better → decision-space anticipation has signal → focus on integrating it with objective-space anticipation
- If Dirichlet has high bias (e.g., always predicts equal weights) → predictor's prior is too strong → tune alpha-scaling

**Wall-clock budget**: ~2h

**Output artifact**: `docs/W22-PROBE-F-DIRICHLET-INFORMATIVENESS.md`

---

## Sequencing + budget summary

| Probe | Module | Budget | Decision impact |
|---|---|---|---|
| **A** | KF predictive accuracy | ~2h | **First**: if KF is broken, everything else is moot |
| **B** | TIP + λ distributions | ~1.5h | Second: if signals are dead, anticipation is mathematically inactive |
| **D** | Pareto front diversity | ~1h | Cheap; rules out trivial DM situation |
| **C** | AMFC vs alt DMs | ~2h | Higher value once D rules out trivial-front |
| **E** | Anticipative distribution sanity | ~1.5h | After C: if AMFC is OK, dist may be the issue |
| **F** | Dirichlet informativeness | ~2h | Decision-space sibling of A |

**Total: ~10h sequenced**, but A+B+D can run in parallel (different logging surfaces) → ~4-5h wall-clock for the first triage.

## Mitigation decision tree (after all probes)

```
                ┌─ if KF broken → fix F matrix or remove KF
Probe A ─┬──────┤
         │      └─ if KF OK → continue
         │
Probe B ─┼─ if TIP saturated → replace uncertainty measure
         │      └─ if λ uniform → fix per-portfolio differentiation (EQ2)
         │
Probe D ─┼─ if front sparse → increase pop/gens (H2)
         │      └─ if front diverse → continue
         │
Probe C ─┼─ if AMFC ≈ random → replace DM (HighROI? Sharpe?)
         │      └─ if AMFC > alt → continue
         │
Probe E ─┼─ if dist degenerate → fix numerical pathology
         │      └─ if dist healthy → continue
         │
Probe F ─┴─ if Dirichlet uninformative → simplify or fix predictor
                └─ if informative → confirmed signal in decision space
```

## Discipline (per W22 lessons)

- All probes are READ-ONLY logging additions; no algorithm changes
- All hypotheses are FALSIFIABLE with pre-specified thresholds
- All tests are STANDARD (Wilcoxon, KS, Shapiro, scipy) — no custom statistical claims
- All probes have a CLEAR mitigation criterion (if X then Y)
- No bundled changes; one probe → one diagnosis → one targeted fix
- If a probe is INCONCLUSIVE at n=3, escalate to n=10 before drawing conclusions

## Update protocol

This document is updated whenever:
- A probe completes (move 🔴 → 🟢 confirmed / ⚫ refuted)
- A new module-level hypothesis is raised
- A mitigation lands and changes the empirical picture

## Current state (2026-05-18 / updated after NC8b breakthrough + NC13a ship)

| Probe | Status | Output |
|---|---|---|
| A | 🟢 COMPLETE (pre/post-NC7) → ⚫ H0 NOT rejected; NC8b shipped; **post-NC8b+NC13a re-run in flight** | `docs/W22-PROBE-A-KF-PREDICTIVE-ACCURACY.md` + `docs/W22-PROBE-A-KF-PREDICTIVE-ACCURACY-POST-NC7.md` |
| B | 🟢 COMPLETE (pre/post-NC12) → 🔴 99.86%/99.87% TIP saturation both; multi-horizon root identified (NC13) | `docs/W22-PROBE-B-SIGNAL-DISTRIBUTIONS-PRE-NC12.md` + `docs/W22-PROBE-B-SIGNAL-DISTRIBUTIONS-POST-NC8b-NC12.md` |
| C | 🔴 PENDING (analyzer-ready; can run on existing Pareto fronts no new run needed) | TBD |
| D | 🟢 COMPLETE → ✅ PASS (median front size 7 ≥ 5) | `docs/W22-PROBE-D-PARETO-FRONT-DIVERSITY.md` |
| E | 🟢 COMPLETE → 🔴 **ASMS P[ROI] median = 486 vs SMS 0.1** (4860× drift via positive-feedback loop) | `docs/W22-PROBE-E-ANTICIPATIVE-DISTRIBUTION-SANITY.md` |
| F | 🔴 PENDING | TBD |

## 🎯 HILL-CLIMB BREAKTHROUGH (2026-05-18)

**NC8b shipped (commit `27cbcd2`) → ASMS BEATS SMS for the first time** in the W17–W22 chain:

| Run | ASMS Ŝ | SMS Ŝ | Δ % |
|---|---|---|---|
| Baseline (post-NC7) | 0.000381 | 0.000405 | −5.92% |
| **NC8b only** | 0.000422 | 0.000415 | **+1.70%** ✅ |
| NC8b + NC12 | 0.000401 | 0.000414 | −3.09% |

(Caveat: n=2 seeds; needs wider validation.)

NC8b mechanism: `_finalize_offspring_objectives` recomputes Portfolio.compute_efficiency + re-initializes KF state on the ACTUAL crossover/mutation output weights instead of leaving stale random-init values. Both SMS and ASMS gain; ASMS gains more because TIP/λ machinery was amplifying stale noise.

NC12 mechanism: mathematically correct (Eq 15 vs naïve SUM) but ZERO production effect for multi-horizon ASMS scenarios (the path `learn_population` uses) — multi-horizon already uses correct Eq 15 at `multi_horizon_anticipatory.py:666-668`. NC12 may help future single-horizon configurations.

## Positive-feedback loop diagnosis (NC14, 2026-05-18)

Probe E revealed a self-reinforcing loop in ASMS:

```
Large P[:2,:2] → TIP MC samples are pure noise → TIP ≈ 0.5 → λ ≈ 0.5 →
combined_cov = w_h² · predicted_cov puts 0.25 weight on each large
predicted_cov → kalman_state.P[:2,:2] stays large → next iteration's
TIP also noise → loop continues
```

Empirical receipt (Probe E on post-NC7 pre-NC8b data):
- ASMS_mHDM_K3_v2both P[ROI] median = **486** (300 records)
- SMS_RDM_K0 P[ROI] median = **0.1** (344 records)
- 4860× factor difference — entirely attributable to anticipation arm

**NC13a (shipped, commit `3d41e91`)** breaks the loop by clamping
`predicted_cov` ≤ 1.0 in the n-step predictor. Predicted post-NC13a effect:
- combined_cov upper bound becomes `w_0² · current + Σ w_h² · 1.0`
- Steady state: P[:2,:2] ≤ 1 / (1 - w_0²) ≈ 2-100 (far below 486)
- TIP MC samples should concentrate near means → TIP escapes (0.45, 0.55) → λ becomes informative

## Refined priority order (post-breakthrough)

| P | Item | Status | Justification |
|---|---|---|---|
| P0 | NC8b+NC13a 3-seed combined smoke | IN FLIGHT (PID 66499, ~37min) | Validates NC13a effect on multi-horizon TIP |
| P1 | Probe A post-NC8b+NC13a (1 scenario × 1 seed) | IN FLIGHT (PID 67xxx) | Does NC8b unblock KF velocity learning? |
| P2 | NC8c — cross-period KF state persistence | PENDING | Highest remaining structural change; biggest potential gain on KF velocity learning |
| P3 | Probe C (AMFC vs alternative DMs) | analyzer-ready | Can use existing data; tests if DM is informative |
| P4 | Probe F (Dirichlet predictor) | PENDING | Decision-space sibling of A; lower priority since A's mechanism is well-understood |
| P5 | Wider 5-seed NC8b validation | PENDING (CPU-constrained) | Statistical confidence on +1.70% |
| P6 | PO(8,1.0) loader (synthetic data) | PENDING | Separate stream; tests strongest-signal dataset |
| P7 | W21-6 final synthesis | PENDING | Post-probes consolidation

### Probe A summary (2026-05-18)

**Pre-NC7 finding** (PR #141, commit `6d3530b`): KF predictions were bit-identical to persistence (Wilcoxon p=0.28 ROI / p=1.0 risk over 583 records across ASMS_mHDM_K3_v2both + SMS_RDM_K0 × 2 seeds × 23 periods). Logged `kf_P_diag = [0.009, 0.009, 0.1, 0.1]` revealed velocity-component prior uncertainty was **0.1**, not the paper-canonical **1000.0**.

**Root cause (NC7)**: P-matrix init divergence between two code paths:
- `sms_emoa._initialize_kalman_state` (initial-population path): `diag([0.1, 0.1, 1000, 1000])` ✓ paper-canonical
- `kalman_filter.create_kalman_params` (offspring path): `eye(4) * 0.1` ✗ velocity priors clobbered

Offspring (generated via `Solution.__init__` → `Portfolio.initialize_kalman_filter` → `create_kalman_params`) inherited the small velocity prior, making the Kalman gain on velocity ~0 during update → KF could not learn velocity → predictions = persistence.

**NC7 fix** (this branch, post-`6d3530b`):
- `kalman_filter.KalmanParams.__post_init__` → `P = diag([0.1, 0.1, 1000, 1000])`
- `kalman_filter.create_kalman_params` → same
- `sms_emoa._initialize_kalman_state` → already correct; annotated as load-bearing
- `tests/test_kalman_filter.py::TestW22NC7PInitHarmonization` → 4 regression tests pinning the parity invariant (all PASS)

**Post-NC7 re-run** (644 records): velocity prior now active (`kf_P_diag = [542.7, 542.7, 1000, 1000]`) but KF predictions are STILL bit-identical to persistence for ASMS (ratio = 1.000000 exactly per seed). SMS marginal advantage 0.2–1.1% per seed, aggregate Wilcoxon p=0.046 ROI (below Bonferroni α/2=0.025), p=0.886 risk.

**NC8 candidates (deeper structural diagnosis)**:
- **NC8a** — KF measurement = KF init state by construction (innovation always 0 at first update). Init `x = [portfolio.ROI, portfolio.risk, 0, 0]`; first measurement `Z = [portfolio.ROI, portfolio.risk]`; `Z == x_init` → `y = 0` → no learning.
- **NC8b** — Offspring never receive `kalman_update` during evolution (`_evaluate_solution` only called from `_initialize_population`); also `compute_efficiency` is called on RANDOM weights at Solution.__init__ before crossover overwrites the weights → ROI/risk perpetually stale on offspring. **SHIPPED (commit 27cbcd2)**: NC8b-minimal added `_finalize_offspring_objectives` helper called from 4 thesis/v2 operators.
- **NC8c** — KF state does not persist across walk-forward periods; born and dies within each period; intra-period evolution noise is the only signal the KF could fit, not actual t→t+1 dynamics.

**NC12 SMOKING-GUN (SHIPPED, commit 2154270)**:
`AnticipativeDistribution.__init__` was setting
`anticipative_covariance = current_covariance + predicted_covariance` —
a naïve SUM violating paper Eq (15) which specifies weighted-sum-of-squared-weights
`Σ = w_c² · Σ_current + w_p² · Σ_predicted`. The SUM caused
`_update_solution_state_anticipative` (line 1412) to blend kalman_state.P
toward `≥ 2× current_P` each generation, growing P by factor (1+α) per gen.
After ~30 generations: P[0,0] = 542 (from 0.0091 baseline, ~60,000× growth).
With P[:2,:2] large, TIP MC sampling collapses to noise → TIP ≈ 0.5 →
λ uniform → no per-portfolio differentiation. This is the smoking-gun
mechanism for Probe B's verdict.

Note: NC12 affects the SINGLE-HORIZON path (`learn_single_solution`).
The MULTI-HORIZON path (`learn_population`) already correctly uses Eq (15)
formula (`combined_cov = w_0² · current + Σ w_h² · predicted_h` at
multi_horizon_anticipatory.py:666-668). So NC12 fix may not fully resolve
TIP saturation for `ASMS_mHDM_K3*` scenarios — those use multi-horizon.

**NC13 candidate (DEFERRED, hypothesis only)**:
N-step predictor (`n_step_prediction.py:42-55`) compounds covariance via
`predicted_cov = F @ current_cov @ F.T` without process-noise cap or
clamping. With NC7's high velocity prior (P[2,2]=1000), after each step
the position component gains the full velocity uncertainty:
`P_next[0,0] = P[0,0] + 2·P[0,2] + P[2,2] ≈ 1000.1`. For h=2:
`P_step2[0,0] ≈ 2000.2`. With h=3: `~3000`.
TIP MC sampling uses `predicted_cov` directly — samples with std `√3000 ≈ 55`
around means ≈ 0.001 are essentially independent random noise → mutual
non-dominance ≈ 0.5 → TIP saturates independently of NC12 fix in
multi-horizon path. Predicted mechanism for residual TIP saturation if
post-NC12 Probe B re-run still shows ≈ 0.5 saturation.

Mitigation candidates if NC13 confirmed:
- NC13a: clamp `predicted_cov` to bounded value (e.g., `min(F @ cov @ F^T, max_cov_clamp)`)
- NC13b: add process-noise Q < 1 with damping per Q paper convention
- NC13c: use FRESH measurement-noise R for TIP (not the propagated KF P)

**Mitigation path priorities**:
1. NC8b shipped (commit 27cbcd2) — being measured
2. NC12 shipped (commit 2154270) — being measured
3. If post-NC12 TIP still saturated → NC13 (likely the cause for multi-horizon)
4. If KF predictions still = persistence → NC8a, NC8b-extended (offspring `kalman_update`), or NC8c (cross-period persistence)
5. If even all NC8/NC12/NC13 insufficient → F-matrix replacement (random walk / AR(1) / regime-switching)

### Probe A summary (2026-05-18)

**Pre-NC7 finding** (PR #141, commit `6d3530b`): KF predictions were bit-identical to persistence (Wilcoxon p=0.28 ROI / p=1.0 risk over 583 records across ASMS_mHDM_K3_v2both + SMS_RDM_K0 × 2 seeds × 23 periods). Logged `kf_P_diag = [0.009, 0.009, 0.1, 0.1]` revealed velocity-component prior uncertainty was **0.1**, not the paper-canonical **1000.0**.

**Root cause (NC7)**: P-matrix init divergence between two code paths:
- `sms_emoa._initialize_kalman_state` (initial-population path): `diag([0.1, 0.1, 1000, 1000])` ✓ paper-canonical
- `kalman_filter.create_kalman_params` (offspring path): `eye(4) * 0.1` ✗ velocity priors clobbered

Offspring (generated via `Solution.__init__` → `Portfolio.initialize_kalman_filter` → `create_kalman_params`) inherited the small velocity prior, making the Kalman gain on velocity ~0 during update → KF could not learn velocity → predictions = persistence.

**NC7 fix** (this branch, post-`6d3530b`):
- `kalman_filter.KalmanParams.__post_init__` → `P = diag([0.1, 0.1, 1000, 1000])`
- `kalman_filter.create_kalman_params` → same
- `sms_emoa._initialize_kalman_state` → already correct; annotated as load-bearing
- `tests/test_kalman_filter.py::TestW22NC7PInitHarmonization` → 4 regression tests pinning the parity invariant (all PASS)

**Post-NC7 re-run**: launched against same scenario set; verifies the fix yields informative KF predictions (KF MAE < persistence MAE, Wilcoxon p < 0.025 for at least one of ROI/risk).

# BACKLOG — full catalog of gaps, errors, and inconsistencies

*Authored 2026-05-17 after W14-2 walk-forward smoke confirmed S2 ≤ S0
direction is NOT a methodology artifact. Triggered by operator request
for comprehensive cataloging vs the Azevedo PhD thesis (UNICAMP 2014;
see `docs/THESIS-INDEX.md`) and the IEEE TCYB 2015 paper.*

**Scope**: every open finding from W7→W14 retros + the diagnostic Q&A
on 2026-05-17. Items are prioritized into waves W15–W19 (revised from
the prior "W15 = Option B Fig 7.15 grid" framing, which is now W18's
deliverable instead of the next wave).

**Grounding discipline (added 2026-05-17 per operator):** every item
has a **Grounding** block citing the source (thesis section / page /
equation OR paper / dataset / implementation pattern), with verbatim
excerpts when load-bearing. W15+ unit contracts MUST inherit these
grounding pointers verbatim into `read_contract.must_read` per the
canonical template in §6.

## Severity ladder

| Sev | Meaning |
|---|---|
| **BLOCKER** | Paper claim direction CANNOT replicate without it |
| **HIGH** | Likely contributing to current S2 ≤ S0 direction reversal |
| **MEDIUM** | Quality / interpretability / methodological soundness |
| **LOW** | Nice-to-have; doesn't block science but does block presentation |

## Category labels

- **CORR** — algorithm / math correctness
- **FID** — thesis fidelity (parameter values mismatch)
- **INST** — instrumentation / traceability
- **METH** — statistical / protocol methodology
- **REPT** — reporting deliverables
- **INFRA** — CI / hygiene / repo state

## Grounding source codes

- **TH** = Azevedo PhD thesis (`docs/Azevedo_CarlosRenatoBelo_D.pdf`). Page numbers refer to printed thesis pagination (PDF page = printed + 20 frontmatter offset).
- **PAP** = Azevedo & Von Zuben IEEE TCYB 2015 paper (`docs/paper.pdf`).
- **CPP2** = 2015-era paper-companion C++ reference at `legacy-cpp-v2/` (GitHub `crbazevedo/anticipatory-learning-asmoo` @ `6643c92`). **Cross-validation oracle for the Python port.** Has `dirichlet.cpp`, `asms_emoa.cpp`, English filenames. Use this for any CPP cross-reference.
- **CPP1** = 2013-era thesis-companion C++ at `legacy-cpp/`. Historical provenance only — pre-dates the post-thesis refinement that added Dirichlet + ASMS-EMOA naming. **NOT a cross-validation oracle.**
- **REF** = third-party reference (cited via thesis bibliography pp. 198+).
- **STD** = standard implementation pattern (no direct paper grounding; cite community practice).

---

# §1 — Full catalog (grounded)

## §1.1 BLOCKER items (must land before any meaningful paper-replication claim)

### B1 — Internal HV reference point wrong scale [CORR + FID]

- **File**: `python_refactor/src/algorithms/sms_emoa.py:55-75` (defaults); `:384, :409` (call sites)
- **Current**: `R1=0.0` (ROI), `R2=1.0` (risk) defaults, used in `_compute_hypervolume_contributions_class` and `_compute_stochastic_hypervolume_contributions_class`
- **Thesis intent**: `z_ref = (0.2, 0.0)^T` — risk_max=0.2, return_min=0.0
- **Effect**: SMS-EMOA's INTERNAL Pareto sorting + Δ_S contributions are computed against the wrong reference rectangle. Selection pressure points at the wrong corner of objective space. EFHV exported in metrics is also against the wrong reference (W13-1 threading kwarg makes it visible but doesn't fix the reference).
- **Fix**: pass `z_ref=(0.2, 0.0)` from validation_matrix's per-scenario config; thread through to SMSEMOA constructor; verify all 3 HV computation sites use it.
- **Wave**: W15
- **Grounding**:
  - **TH §7.2.3 "ASMS Parameters", p. 147**:
    > "Finally, the reference point for computing Hypv was set to `z^ref = (0.2, 0.0)^T` in terms of risk and return, coinciding with the objective space feasibility boundaries (maximum risk of 20% and minimum mean return of 0%)."
  - **TH §3.1.1 "The Hypervolume (S-Metric) Indicator", p. 57** — definition of S-metric and reference-point role
  - **PAP §IV-B, Eq (16)** — corresponds to thesis Eq 6.35 (expected Δ_S contribution) — same reference-point dependence
  - **REF [Auger, Bader, Brockhoff, Zitzler 2009] "Theory of the hypervolume indicator: Optimal μ-distributions and the choice of the reference point" (FOGA)** — choice of z_ref is load-bearing for optimal μ-distributions

### B2 — K (OAL window-size) not threaded [CORR + FID]

- **File**: `python_refactor/experiments/validation_matrix.py:60-95` (SCENARIOS dict)
- **Current**: SCENARIOS map `max_horizon` (= paper Eq 14 **H**), NOT K. S0..S4 effectively run K=0.
- **Thesis intent**: K ∈ {0, 1, 2, 3} is the OAL historical-window size that drives `λ_{t+h}^K` in Eq 7.16. Fig 7.15 shows K=0 vs K>0 IS the anticipation factor.
- **Effect**: S2 "anticipatory" config is currently K=0 ≈ S0 baseline at the OAL-rate level. Anticipation cannot fire as the thesis intends.
- **Fix**: re-key SCENARIOS to expose K explicitly: `{ASMS,SMS} × {mHDM,RDM} × K ∈ {0,1,2,3}` factorial (matches thesis Fig 7.15 framing). Pass K through to AnticipatoryLearning constructor; verify it's consumed in `compute_anticipatory_learning_rate`.
- **Was**: W13-CARRY-1
- **Wave**: W15
- **Grounding**:
  - **TH §7.1.1 "Investigated Algorithmic Variants", p. 140**:
    > "The anticipation factor is controlled by four levels of window size (K): K ∈ {0, 1, 2, 3}. For K = 0, we have the myopic baseline SMS-EMOA (SMS, in short) for which constant predictions are made, i.e., `û_{t+h} ~ û_t` and `ẑ_{t+h} ~ ẑ_t`. The stationarity assumption implies the estimated Temporal Incomparability Probabilities (TIPs, see chapter 6, Eq. (6.4)) equal 1/2 and, hence, all anticipation rates are self-adjusted to λ_{t+h}^{(i)} = 1 (i = 1, …, N). This case is equivalent to a DM betting on a static market. For K > 0, the past solutions and objective distributions obtained in the latest K periods serve as input to the KF and DD tracking methods, in which case anticipation is enabled."
  - **TH §7.2.3 "Bayesian Tracking Parameters", p. 146**:
    > "Online Anticipatory Learning (OAL) was used to self-adjust the anticipation rates using Eqs. (6.6) and (6.9) at chapter 6. Hence, we have set `λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K))`, for which the anticipation horizon is H = 2 (one-step-ahead prediction)."
  - **TH Fig 7.15, p. 157** — Boxplots of Out-Of-Sample Future Average Hypervolume per (K, DM) for all 10 instances; the K-axis IS the dial we're not currently turning
  - **TH §6.1.1-6.1.4 "Online Anticipatory Learning in the Objective Space", pp. 116-120** — the formal derivation of λ from KF prediction error + TIP

### B3 — Cardinality constraint not enforced [CORR + FID]

- **File**: `python_refactor/src/portfolio/portfolio.py:57` declares `max_cardinality=10`; nowhere enforced
- **Current**: `compute_efficiency` measures cardinality post-hoc; no projection / no penalty / no repair operator
- **Thesis intent**: c_l=5, c_u=15 (5–15 assets active per portfolio)
- **Effect**: 98-asset over-diversification → portfolios converge to ~uniform-ish weights → all scenarios produce ~the index → no anticipation signal because everyone holds the same basket
- **Fix**: post-operator cardinality projection (zero out smallest weights until cardinality ≤ c_u; if cardinality < c_l, add random small weights to inactive assets). Or simpler: penalty term in fitness.
- **Wave**: W15
- **Grounding**:
  - **TH §7.2.3 "Constraint Handling", p. 146**:
    > "We considered minimum and maximum cardinality of `c_l = 5` and `c_u = 15` assets."
  - **TH §7.2 Eq (7.3), p. 142**:
    > "s.t. `c_l ≤ c(u_t) ≤ c_u`,"
    > "where c(u_t) computes the number of assets in u_t with non-zero weight (u_t > 0)."
  - **TH §7.2.3 "Search Operators", p. 147** (the add/remove-asset mutation is THE thesis mechanism for cardinality control — see B4 grounding for full text)

### B4 — Non-simplex weights AND wrong crossover operator [CORR + FID]

- **File**: `python_refactor/src/algorithms/operators.py:14-44` (`sbx_crossover`); `:126-168` (`mutation`)
- **Current**:
  - SBX (Simulated Binary Crossover) used; can produce negative weights
  - Normalize-by-sum: `weights / np.sum(weights)` — doesn't clip to non-negative
  - If sum > 0 but some weights < 0, the "normalized" result still has negative weights → breaks Markowitz `w^T Σ w`
- **Thesis intent**: **uniform crossover** (NOT SBX), with explicit add/remove-asset mutation; portfolio weights lie in the (d-1)-simplex (≥ 0, sum = 1)
- **Effect**: (a) negative weights break risk computation; (b) SBX is the wrong operator family entirely
- **Fix**: replace SBX with uniform crossover per thesis spec; add the dual-mode mutation (50/50 modify-existing vs add/remove); clip to non-negative + renormalize as final step
- **Wave**: W15
- **Grounding**:
  - **TH §7.2 "Solving Portfolio Selection with AS-MOO", p. 141**:
    > "`u ∈ S^{N-1}` denote the proportions of wealth to be invested"
    > (where `S^{N-1}` is the (N-1)-simplex — `u_i ≥ 0`, `Σu_i = 1`)
  - **TH §7.2.3 "Search Operators", p. 147**:
    > "We utilized **uniform crossover** over the mean DD vectors. For mutation, we randomly choose between (1) modifying the non-zero weights; or (2) adding/removing assets. If operator (1) is selected, then, with probability 1/2, we either increase or decrease the investment on a randomly chosen asset by a uniformly drawn factor from 10 to 50%. If (2) is selected, then, with probability 1/2, we either add or remove a randomly chosen asset. If it is removed, we simply set its weight to zero. If it is added, we randomly set its weight within a ±10% range from an equally-balanced allocation. **All modified DD vectors are renormalized.**"
  - **TH §6.2.5 "Maximum A Posteriori Correction for DD Mean Decision Vectors", p. 128** — the renormalization + MAP-correction step that keeps DD vectors in the simplex after Dirichlet update

---

## §1.2 HIGH items (likely contributing to direction reversal)

### H1 — Transaction costs NOT integrated into anticipatory HV objective [CORR + FID]

- **File**: `python_refactor/src/config/thesis_parameters.py:107` declares `TRANSACTION_COST_RATE=0.001`; only used downstream in `portfolio_evaluator` (final-wealth path)
- **Thesis intent**: objective vector includes h-cost component; AS-MOO solver sees this in `z_t = g(u_t, χ_t) + h(u_t, u*_{t-1})`
- **Effect**: optimizer can't avoid high-churn EMFCs because it doesn't see the cost; S2 might be implementing EMFCs whose realized future HV is eaten by trading fees
- **Fix**: subtract `h(u_t, u*_{t-1})` from anticipatory ROI estimate before computing HV; track previous-period implemented portfolio per seed for `u*_{t-1}` reference
- **Wave**: W16
- **Grounding**:
  - **TH §7.2 Eqs (7.4)-(7.5), p. 142**:
    > "Following the AS-MOO notation, `z_t|u*_{t-1} = g(u_t, χ_t) + h(u_t, u*_{t-1})`, where: `g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T` (7.4) and `h(u_t, u*_{t-1}) = (0, h(u_t, u*_{t-1}))^T` (7.5), in which `h` is a cost function representing all incurring transaction fees and commissions."
  - **TH §7.2.3 "Brokerage Fees", Table 7.1, p. 144** — Brazilian Securities Commission fee schedule used by the thesis:
    | Traded value | Proportional Fee | Fixed Fee |
    |---|---|---|
    | < 135.07 | 0.0% | 2.70 |
    | ≥ 135.08 and < 498.62 | 2.0% | 0.00 |
    | ≥ 498.63 and < 1,514.69 | 1.5% | 2.49 |
    | ≥ 1,514.70 and < 3,029.38 | 1.0% | 10.06 |
    | ≥ 3,029.39 | 0.5% | 25.21 |
  - **TH §5.2.4 "Time-Linkage Formulation", p. 105** — formal derivation of why `h` enters the objective at the OPTIMIZATION step, not just the wealth-evaluation step

### H2 — λ formula completeness (Eq 7.16) uncertain [CORR + INST]

- **File**: `python_refactor/src/algorithms/anticipatory_learning.py:249` `compute_anticipatory_learning_rate`
- **Current**: single-component computation (verify by trace logging); thesis says `λ_{t+h} = ½(λ_{t+h}^H + λ_{t+h}^K)`
- **Effect**: if only λ^H fires (TIP-driven), the K-historical-window self-adjustment doesn't kick in → tracking is sub-optimal → anticipatory rate mis-calibrated
- **Fix**: verify both arms fire; if not, implement λ^K (squared-residual sum from historical KF predictions per Eq 7.16). Add per-portfolio per-generation trace export.
- **Wave**: W15 (verify) + W16 (fix if needed)
- **Grounding**:
  - **TH §7.2.3, Eq (7.16), p. 146**:
    > "`λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K))`, for which the anticipation horizon is H = 2 (one-step-ahead prediction). The anticipation rate of each portfolio is thus determined not only by the estimated temporal incomparability probability between the current and the predictive objective distribution (`λ_{t+h}^(H)`), but also by the observed normalized sum of KF squared residuals (`λ_{t+h}^(K)`)."
    > 
    > "The motivation for taking the average (the 1/2 factor) of the two aforementioned self-adjustment strategies for setting `λ_{t+h}` can be explained by the intuition of providing a *balanced tension* in the OAL rule between:
    > - The desire of *trusting* in decision paths that *were shown* to lead to higher predictable consequences *in the past* (`λ^{K}_{t+h}` in Eq (6.9)); and
    > - The desire of *trusting* in decision paths that *are estimated* to lead to higher predictable consequences *in the future* (`λ^{H}_{t+h}` in Eq (6.6))."
  - **TH Eq (6.6), p. 117** — defines `λ_{t+h}^(H)` (TIP-based, future-looking)
  - **TH Eq (6.9), p. 119** — defines `λ_{t+h}^(K)` (squared-KF-residual, past-looking)

### H3 — Date range mismatch [FID]

- **File**: `python_refactor/experiments/validation_matrix.py:105-118` WINDOWS dict
- **Current**: `paper` window = 2003-01-01 → 2012-11-20 (~2600 days; superset of thesis range)
- **Thesis intent**: FTSE-100 daily 2006-11-20 → 2012-12-31 (~1500 days, T=24 rolling 50-day-shift periods)
- **Effect**: training data span is too long; thesis aggregates over T=24 periods, ours implies T≈44 — different power, different regime mix
- **Fix**: slice paper window to 2006-2012 to match thesis
- **Wave**: W15
- **Grounding**:
  - **TH §7.2.3 "Artificial and Real-World Datasets", p. 145**:
    > "The total number of periods for all instances is T = 25. The real-world scenarios are composed of daily adjusted close prices collected between 20/11/2006 – 31/12/2012, from which 50 days lagged returns were computed: `r_{n,k} = (V_{n,k+50} - V_{n,k}) / V_{n,k+50}`, where `V_{n,k+50}` is the value of the n-th asset at day k+50. Each investment period comprises one and a half year worth of daily returns data."
    > 
    > Footnote 4 (p. 145): "Except for FTSE with T = 24 due to data availability issues at the time the experiments were executed."
  - **TH §7.2.3 p. 146**:
    > "Between each period, the 50 oldest lagged returns are discarded and the 50 latest ones are included in the sample from which the parameters `{μ̂_t, Σ̂_t}_{t=1}^T` are estimated, where 50 business days roughly corresponds to two and a half months. Thus, the period t = 1 in FTSE comprises data from 20/11/2006 – 20/05/2008, t = 2 to 01/02/2007 – 30/07/2008, and so forth."

### H4 — Asset universe mismatch (98 vs 87) [FID]

- **File**: `legacy-cpp/executable/data/ftse-original/` has 98 CSVs
- **Thesis intent**: d=87 for FTSE
- **Effect**: 11 extra assets may include delisted / illiquid ones that pollute optimization
- **Fix**: identify the 87 thesis-faithful subset (cross-ref the thesis-cited dataset URL if available; otherwise drop the 11 with most missing/quirky data); document the asset list explicitly
- **Was**: noted in `docs/THESIS-INDEX.md` §3
- **Wave**: W16
- **Grounding**:
  - **TH §7.2.3 "Artificial and Real-World Datasets", p. 145**:
    > "All benchmarks provide d = 30 simulated assets for composing the portfolios, whereas for the real-world instances we have **d = 87 for FTSE**; d = 30 for DJI; and d = 49 for HSI, which are represented in a (d − 1)–Simplex space."
    > 
    > Footnote 3 (p. 145): "The data used in our experiments is publicly available at ..." [URL truncated in thesis PDF; needs follow-up]

### H5 — H (anticipation horizon) varies per scenario [FID]

- **Current**: S1 H=2, S2 H=3, S3 H=2, S4 H=3
- **Thesis intent**: H=2 fixed (one-step-ahead prediction with horizon length 2) across all anticipatory variants
- **Effect**: scenarios aren't ablating cleanly — the horizon ablation we INTENDED with S2 vs S3 isn't isolating K (it's varying H, which isn't what the paper varies)
- **Fix**: fix H=2 across all anticipatory variants; vary K instead (resolves B2 + H5 simultaneously)
- **Wave**: W15
- **Grounding**:
  - **TH §7.2.3, Eq (7.16) and surrounding text, p. 146** (same as H2 above):
    > "for which the anticipation horizon is **H = 2** (one-step-ahead prediction)."
  - **TH §5.2.4 "Time-Linkage Formulation", p. 105** — derivation of H as the "anticipation horizon" in the TL-AS-MOO framework
  - **TH §6.3 "Computing the Expected Anticipatory Hypervolume Contributions", p. 128** — H appears in the convex-combination weights for the multi-horizon expected HV

### H6 — Extrema (anchor) points not preserved [CORR]

- **File**: `python_refactor/src/algorithms/sms_emoa.py` evolution loop
- **Current**: no explicit preservation of best-ROI / best-risk anchor solutions; SMS-EMOA's reduce step can drop them
- **Effect**: HV oscillates more than it should; the Pareto front extent shrinks over generations because extreme solutions get pruned
- **Fix**: keep ranked-1 anchor points immune to reduce; verify via per-gen ROI / risk range trace
- **Wave**: W15 (or W17 if low priority)
- **Grounding**:
  - **TH §3.4.2 "S-Metric Selection Evolutionary Algorithm (SMS-EMOA)", pp. 69-70** — describes the base SMS-EMOA reduce step (Beume et al. 2007 [REF 32])
  - **REF [Beume, Naujoks, Emmerich 2007] EJOR 181(3):1653-1669** — original SMS-EMOA paper; standard practice preserves extrema (HV indicator is most sensitive to anchor solutions because they define the bounding box)
  - **STD** — standard SMS-EMOA / NSGA-II implementations explicitly preserve rank-1 anchors

### H7 — Correspondence mapping NOT wired into experiment path [CORR + FID]

- **File**: `python_refactor/src/algorithms/correspondence_mapping.py` exists; not invoked from `experiment_manager._run_algorithm` or `sms_emoa.run`
- **Thesis intent**: correspondence mapping aligns current vs. future projected Pareto sets across rolling periods
- **Effect**: rolling-period anticipation can't compare like-for-like portfolios across periods → λ self-adjustment uses noisy alignment → predictions degraded
- **Fix**: wire into anticipatory_learning rolling path; verify alignment via trace export
- **Wave**: W17
- **Grounding**:
  - **TH §6.2.2 "Correspondence Mapping for Multiple Decision Vectors Tracking", pp. 122-123**:
    > "Tracking and predicting time-varying populations of solutions requires establishing correspondences between solutions across consecutive periods. We use a correspondence mapping based on minimum-cost bipartite matching over the simplex distance between candidate solutions."
  - **TH Fig 6.4, p. 132** — visualization of the N possible predictive trajectories for the sequence of future (H-1)N Estimated Maximal Flexible Choices (EMFCs) over time, conditioned on each available candidate Pareto-efficient decision in the estimated essential set

---

## §1.3 MEDIUM items (quality / methodology / interpretability)

### M1 — No per-generation HV trajectory exported [INST]

- **File**: `python_refactor/src/algorithms/sms_emoa.py:83` `hypervolume_history` tracked internally; only `final_hypervolume` exported in `metrics_collector.optimization_metrics`
- **Effect**: can't see if HV is improving over generations; can't diagnose convergence vs early stopping
- **Fix**: export per-gen HV list in `algorithm_metrics`; render as line plot per scenario
- **Wave**: W16
- **Grounding**:
  - **STD** — interpretability standard for EAs; no direct thesis requirement
  - **TH Fig 7.16(a), p. 160** — paper plots `Ŝ_{t+1}` over PERIODS (not generations), but the operator question explicitly asks about per-generation; same plumbing supports both

### M2 — No per-generation EFHV trajectory exported [INST]

- **Same shape as M1** but for `stochastic_hypervolume_history`
- **Wave**: W16
- **Grounding**: Same as M1 (STD + TH Fig 7.16)

### M3 — Per-portfolio λ traces not logged [INST]

- **File**: `compute_anticipatory_learning_rate` computes λ per call, no record
- **Operator request**: "how are the traces of lambda across time steps for each ordered portfolio in the population? Both components (kalman error and confidence)?"
- **Fix**: per-call λ logging keyed by (period, generation, solution_rank); export as CSV alongside metrics.csv
- **Wave**: W16
- **Grounding**:
  - **TH §7.3.1 "Estimated Confidence Over the Stochastic Pareto Frontiers (SPFs)", p. 148**:
    > "We recall that, according to the OAL rules proposed in chapter 6, the available KF predictive objective distributions of portfolios associated with higher predictive confidence are more intensely incorporated into the resulting anticipatory distributions... Moreover, recall that rank 1 portfolios correspond to the highest risk/return, whereas a rank 20 one corresponds to the lowest risk/return in the population."
  - **TH Figs 7.4-7.13, pp. 150-155** — show `1 - E[TIP]` per portfolio rank, averaged over all periods, for each (instance, K, DM) combination. THIS IS exactly the trace the operator is asking for.

### M4 — Per-portfolio TIP traces not logged [INST]

- **Same shape as M3** but for `temporal_incomparability_probability`
- **Wave**: W16
- **Grounding**:
  - **TH Eq (6.4), p. 116** — TIP definition
  - **TH §7.3.1 + Figs 7.4-7.13** — same provenance as M3 (the figures show `1-E[TIP]` which is the complement of TIP)

### M5 — Anticipated EMFC not compared vs alternatives [INST + METH]

- **Operator request**: "Is the anticipated maximal flexible choice 'better' in terms of predicted future hypv when compared to other portfolio available?"
- **Fix**: per period, compute predicted future HV for ALL Pareto-flexible portfolios; rank EMFC; export rank
- **Wave**: W16
- **Grounding**:
  - **TH §7.3.6 "Average Predicted Future Hypv Along the Evolved SPFs", p. 171**:
    > "We also assess whether the EMFCs predicted to lead to maximal future Hypv are indeed those associated with the highest projected future Hypv among the population members."
  - **TH §6.4 Eq (6.42), p. 133** — formal AMFC selection that requires comparing all candidates

### M6 — Coherence per Eq 7.14 not computed/logged [METH + INST]

- **Thesis intent**: coherence = mean cosine similarity of each portfolio to population centroid in simplex
- **Current**: `algorithm.diversity_metric` is a different formula; coherence per Eq 7.14 not implemented
- **Fix**: implement `coherence(pareto_set)` function; export per-period
- **Wave**: W16
- **Grounding**:
  - **TH §7.2.2 "Similarity in the Decision Space", Eq (7.14)-(7.15), p. 145**:
    > "we propose a coherence measure over the Pareto-flexible sets inspired in [106]. It consists of summing over the cosine similarities of each portfolio to the population centroid, `u_t^C`: `Coherence(Û_t^{N*}) = Σ_{i=1}^N (m̂_{u_t^{(i)*}} · u_t^C) / (||m̂_{u_t^{(i)*}}||_2 · ||u_t^C||_2)` (7.14), where the operator `·` denotes the dot product, and `u_t^C = (1/N) Σ_{i=1}^N m̂_{u_t^{(i)*}}` (7.15)."
  - **REF [106]** in thesis bibliography (the paper coherence is "inspired by")

### M7 — POCID per Eq 7.13 not computed [METH]

- **Thesis intent**: percentage of correct direction predictions over T-1 transitions
- **Current**: not implemented
- **Fix**: implement; export per-scenario in report
- **Wave**: W16
- **Grounding**:
  - **TH §7.2.2 "Percentage Of Change In Direction, POCID", Eq (7.13), p. 144**:
    > "The POCID is a commonly used measure in time series forecasting applications [155] that captures the proportion of correct predictions w.r.t. the directions toward which the j-th objective function change between subsequent periods: `POCID(j) = 100 × (1 / (N × (T - 1))) Σ_{i=1}^N Σ_{t=1}^{T-1} D(j, i, t)` (7.13), `D(j, i, t) = { 1, if (z_{j,t+1}^{(i)*} - z_{j,t}^{(i)*})(m̂_{ẑ_{j,t+1}^{(i)*}} - m̂_{ẑ_{j,t}^{(i)*}}) ≥ 0; 0, otherwise }`"
  - **TH §7.3.2, p. 155-156** — POCID interpretation + Figs 7.14 boxplots
  - **REF [155]** in thesis bibliography

### M8 — No wealth time-series exported [INST]

- **Current**: `summary.final_portfolio_value` scalar only
- **Operator request**: "Wealth evolution?"
- **Fix**: export per-period wealth `W_t` (thesis Eq 7.12) including transaction costs
- **Wave**: W16
- **Grounding**:
  - **TH §7.2.2 "Step 5" + Eq (7.12), p. 144**:
    > "Implement the selected mean portfolio `m̂_{û*_t}` from the evolved SPF, advance to t ← t+1, and update the current investor wealth as: `W_t ← W_{t-1} + mean observed return for û*_t − transaction costs of transforming m̂_{û*_{t-1}} into m̂_{û*_t}` (7.12)."
  - **TH §7.3.5 "Dynamics of the Investor Wealth", p. 171** + **Figs 7.16-7.25 part (d)** — wealth evolution plots per investment period

### M9 — No bootstrap CI on Ŝ_{t+1} per scenario [METH]

- **Current**: report raw mean ± std
- **Standard practice** for OOS reporting
- **Fix**: bootstrap-resample seed-level Ŝ values; 95% CI on grand mean
- **Wave**: W17
- **Grounding**:
  - **STD** — bootstrap CI standard for non-parametric inference; no direct thesis requirement (thesis uses ANOVA + Mann-Whitney)
  - **W8-2 (already implemented)** `python_refactor/experiments/stats.py:139` `bootstrap_ci` already exists; just need wiring

### M10 — No statistical testing in populated report [METH + REPT]

- **Operator request implicit in "data viz and tables"**
- **Thesis intent**: 2-way ANOVA WS × DM with significance markers
- **Current**: only Welch t reported in OOS reports; no ANOVA
- **Fix**: implement 2-way ANOVA per Table 7.2; add Mann-Whitney pairwise post-hoc (already have stats helpers in W8-2 — just need wiring)
- **Wave**: W18
- **Grounding**:
  - **TH §7.2.2, p. 144**:
    > "In order to assess the statistical significance of the anticipation and DM factors, we use a two-way Analysis of Variance (ANOVA). Pairwise significance is assessed with the non-parametric Mann-Whitney U test, which does not require a normality assumption."
  - **TH Table 7.2, p. 159** — ANOVA for WS and DM factors on `Ŝ_{t+1}` with significance markers (**p<0.01; *p<0.05)
  - **TH §7.3.3, p. 158** — interpretive narrative around the ANOVA results

### M11 — No constraint violation penalty in fitness [CORR]

- **Related to B3**: even if cardinality is post-projected, other constraints (min return, max risk) have no penalty path
- **Wave**: W17 (lower priority; B3 covers most of the harm)
- **Grounding**:
  - **TH §7.2 Eq (7.1)-(7.3), p. 142**:
    > "`min_{u_t} u_t^T Σ̂_{r,t} u_t` (7.1), `max_{u_t} μ̂_{r,t}^T u_t - h(u_t, u*_{t-1})` (7.2), s.t. `c_l ≤ c(u_t) ≤ c_u`" (7.3)
  - **TH §7.2.3 "Constraint Handling" p. 146-147** — describes ε-feasibility for stochastic objective vectors:
    > "The feasibility probability was set to 99% (ε = 0.99, see Def. 5.5). For two anticipatory distributions ẑ_t^(a) and ẑ_t^(b), constraint handling in ASMS is done similarly as in [69]: If both ẑ_t^(a) and ẑ_t^(b) are ε-feasible, then PD is applied over the mean vectors as in Def. 5.3; Else if `p(ẑ_t^(a))_ε = Pr{ẑ_t^(a) ∈ Z_t} ≥ ε` but `p(ẑ_t^(b))_ε < ε`, then `ẑ_t^(a) ≼ ẑ_t^(b)` (and vice-versa); Else, if both are not ε-feasible, then PD is applied to compare the vectors..."
  - **REF [69]** in thesis bibliography — constraint-handling precedent

### M12 — AMFC identification (Eq 6.42) not visible in code [CORR]

- **Thesis intent**: `û*_t = arg max_u in U_t (1/E) Σ S(...)` — explicit MC-driven AMFC selection
- **Current**: SMS-EMOA returns the trained population; not clear where AMFC is identified vs random selection
- **Fix**: implement explicit AMFC identification at end of each period; log which Pareto rank was chosen
- **Wave**: W17
- **Grounding**:
  - **TH §6.4 "Estimating the Anticipated Maximal Flexible Choice", Eq (6.42), p. 133**:
    > "`û*_t = arg max_{û_t ∈ Û_t^{N*}} (1/E) Σ_{e=1}^E S( Σ_{h=1}^{H-1} λ_{t+h} { ẑ_{e,t+h}^{(i)} | û_t }_{i=1}^N )`" (6.42)
  - **TH §6.5.2 "The Anticipatory S-Metric Selection EMOA", Pseudocode 8 line 16, p. 135**:
    > "**Identify the Anticipated Maximal Flexible Choice `û*_t` using Eq. (6.42)** ▷ Section 6.4"
  - **TH Pseudocode 9 "Anticipatory Reduce", p. 135** — also relies on AMFC

### M13 — Bayesian tracking confidence per portfolio not exported [INST]

- **Operator request**: "how is tip across portfolios and time steps?" and confidence in §7.3.1
- **Current**: KF covariance P_t computed per portfolio; not exported
- **Fix**: export per-portfolio confidence (1 - E[TIP]) traces
- **Wave**: W16
- **Grounding**:
  - **TH §7.3.1, p. 148** + **Figs 7.4-7.13, pp. 150-155** — same provenance as M3 (confidence = `1 - E[TIP]`)
  - **TH Eq (6.6), p. 117** — `λ^H` depends on TIP; therefore TIP traces ARE the confidence traces

### M14 — No early-stopping / convergence check [METH]

- **Current**: fixed 30 generations regardless of convergence
- **Effect**: wasted compute if HV plateaus early; missed opportunity if 30 isn't enough
- **Fix**: optional early-stop on HV plateau (window of 5 gens with < 1% relative change)
- **Wave**: W17 (low impact for this dataset; nice-to-have)
- **Grounding**:
  - **STD** — standard EA early-stopping; no direct thesis requirement (thesis fixes 30 gens, so this is OPTIONAL convenience)
  - **TH §7.2.3 "ASMS Parameters", p. 147**:
    > "30 generations were executed for each investment period."

### M15 — Non-dominated sorting basis verification [CORR]

- **Operator question**: "How are you applying non-dominated sorting? Over the expected (mean) return/risk?"
- **Current**: `solution.objectives = [P.ROI, P.risk]` deterministic means — thesis §6.5.1 says this is correct ✅
- **Action**: verify and document; no code change needed
- **Wave**: W15 (verify only)
- **Grounding**:
  - **TH §6.5.1 "Sorting the Population of Random Objective Vectors", p. 134**:
    > "we argue that, since we are already able to handle the estimated uncertainty (see section 6.1) and to combine the learned predictive correlation into the computation of E[Δ_S] (see section 6.3), there is little need to incorporate uncertainty awareness directly into the dominance relation. Therefore, our proposed non-dominated sorting procedures are executed in terms of the deterministic Pareto Dominance **over the estimated means of the random objective vectors**"
  - **TH §6.6 contribution #4, p. 136** — "A procedure for identifying the candidate anticipatory decision maximizing the future S-Metric in the time-linkage regime has been proposed"
  - **TH Theorem 6.5.1, p. 134** (stochastic dominance theorem) — explains why deterministic NDS over means is justified

---

## §1.4 LOW items (presentation / repo hygiene)

### L1 — Tables C/E/F/G/H builders stubbed [REPT]

- **Was**: W8-CARRY-3
- **File**: `python_refactor/experiments/report.py` — builders for tables C-H return `[]`
- **Wave**: W18
- **Grounding**:
  - **`docs/ANALYTICS-PLAN.md` §7 (W7-2)** — table catalog
  - **`docs/VALIDATION-RESULTS.md` §3-8** — template structure
  - Each table maps to thesis figures/tables as per `ANALYTICS-PLAN.md`

### L2 — F1-F12 figures not generated [REPT]

- **File**: `python_refactor/experiments/figures.py` exists from W7-1; never invoked end-to-end
- **Wave**: W18
- **Grounding**:
  - **TH Figs 7.4-7.13 (confidence per rank)** → F1-class
  - **TH Fig 7.15 (boxplots OOS Future HV per K, DM)** → F2-class
  - **TH Figs 7.16-7.25 (per-instance dynamics — Future Hypv / Cardinality / Coherence / Wealth)** → F3-class
  - **`docs/ANALYTICS-PLAN.md` §6** — figure catalog with F1-F12 mapping to thesis figures

### L3 — VALIDATION-RESULTS.populated.md mostly 🚧 placeholders [REPT]

- **Status**: §0 receipt populated; Table A populated for S0+S2; §1 has W13-3 + W14 walk-forward findings; rest are placeholders
- **Wave**: W18
- **Grounding**: composite — derives from L1 + L2 + all the previous instrumentation items

### L4 — W1-W6 retros use non-canonical ADR-004 keys [INFRA]

- **Was**: W9-3-CARRY-1 (optional)
- **Effect**: `scripts/pre-pr-reflect-validate.sh --all` shows 22+ stale retros
- **Fix**: mechanical re-key of W1-W6 retros (same pattern as W9-3 did for W7+W8)
- **Wave**: W19
- **Grounding**:
  - **dfg-harness ADR-004** (per `scripts/pre-pr-reflect-validate.sh` header) — canonical retro-frontmatter keys

### L5 — Pre-pr battery has 4 carried failures [INFRA]

- **Items**: ruff (lint), pytest (unrelated failures), branch-shape (`main` vs `master` reference), skill-assessment-gate (kit/scripts path)
- **Carried since**: W7-era
- **Effect**: dfg pre-pr exits 1 even on green wave-close commits
- **Wave**: W19
- **Grounding**:
  - **dfg-harness ADR-001** (pre-PR battery spec)

### L6 — Algorithm log lines pollute stderr in batch mode [INFRA]

- **Current**: `print("Generation X: ...")` from sms_emoa.py:124 fires for every gen of every run
- **Effect**: noise in walk-forward sweep logs (44 periods × 60 seeds × 30 gens = lots of lines)
- **Wave**: W19
- **Grounding**: **STD** — Python `logging` module discipline

### L7 — Stochastic seed handling under ProcessPoolExecutor [CORR]

- **Current**: `np.random.seed(seed)` set inside subprocess; should isolate but `Portfolio.mean_ROI` is class-level
- **Wave**: W19
- **Grounding**: **STD** — Python multiprocessing + module-level mutable state hazard

### L8 — `kit/` symlink trap [INFRA]

- **Was**: W9-3 scar
- **Wave**: W19 (optional defensive)
- **Grounding**: internal — `kit/` is a symlink into the dfg-harness install; documented in W9-3 retro

---

# §2 — Wave plan

| Wave | Theme | Items | Sizing | Estimated wall-clock |
|---|---|---|---|---|
| **W15** | Algorithm correctness (BLOCKERS) | B1 (ref point) + B2 (K threading) + B3 (cardinality) + B4 (uniform crossover + simplex projection) + H3 (date range) + H5 (H=2 fixed) + M15 (verify NDS) | M-L | 2-4h engineering + 1-2h re-runs |
| **W16** | Instrumentation + λ-completion + secondary fidelity | H2 (λ Eq 7.16) + H4 (87 assets) + M1-M8 + M13 (per-gen / per-portfolio traces) | M-L | 3-5h engineering + 2-4h compute |
| **W17** | Advanced (txn costs + correspondence + extras) | H1 (txn costs in HV) + H6 (extrema) + H7 (corr mapping wiring) + M9-M12 + M14 | M | 3-5h |
| **W18** | Comprehensive report + viz (the operator-asked report) | L1 (tables C-H) + L2 (figures F1-F12) + L3 (populate full results doc) + M10 (ANOVA per Table 7.2) | M | 2-4h engineering + 4-8h compute |
| **W19** | Repo hygiene (low-priority cleanup) | L4-L8 | S | 1-2h |

**Critical path to paper-replication answer:** W15 → W16 (λ verification only — defer instrumentation if compute-bound) → quick walk-forward re-run.

**Critical path to "the report":** W15 + W18.

---

# §3 — Cross-reference: thesis sections informing each backlog item

| Thesis section | Backlog items it grounds |
|---|---|
| §6.1.1-6.1.4 OAL in Objective Space | B2, H2 |
| §6.2.2 Correspondence Mapping | H7 |
| §6.2.5 MAP Correction for DD | B4 (simplex preservation post-Dirichlet) |
| §6.3 Computing Expected Anticipatory HV | H5 (H in convex combo weights) |
| §6.4 Estimating AMFC + Eq (6.42) | M5, M12 |
| §6.5.1 NDS over mean vectors | M15 (verify) |
| §6.5.2 ASMS-EMOA + Pseudocode 8 | M12 |
| §7.1.1 Algorithmic Variants (K=0,1,2,3) | B2 |
| §7.2 + Eqs (7.1)-(7.5) | B3, B4, H1, M11 |
| §7.2.2 Step 4 + Eqs (7.10)-(7.11) | (W13-2 already) |
| §7.2.2 Step 5 + Eq (7.12) | M8 |
| §7.2.2 + Eq (7.13) POCID | M7 |
| §7.2.2 + Eqs (7.14)-(7.15) Coherence | M6 |
| §7.2.2 ANOVA + Mann-Whitney | M10 |
| §7.2.3 ASMS Parameters | B1, B2, B3, H3, H4, H5, M14 |
| §7.2.3 Constraint Handling | B3, M11 |
| §7.2.3 Search Operators | B4 |
| §7.2.3 Eq (7.16) λ formula | B2, H2 |
| §7.3.1 + Figs 7.4-7.13 | M3, M4, M13 |
| §7.3.2 + Fig 7.14 POCID | M7 |
| §7.3.3 + Fig 7.15 + Table 7.2 | M10, W18 |
| §7.3.4 + Figs 7.16-7.25 | M6, M8 |
| §7.3.5 Dynamics of Wealth | M8 |
| §7.3.6 Predicted Future Hypv Along SPFs | M5 |
| Table 7.1 Brokerage Fees | H1 |

---

# §4 — How to use this backlog

1. **Before each new wave**: open this file; pick items per the wave plan in §2.
2. **When new findings surface**: append to the appropriate §1.x sub-section; bump severity if needed.
3. **When an item closes**: mark it ✅ inline and reference the closing PR.
4. **Wave-close ceremonies**: reference §1 item IDs (B1, H2, etc.) in `carry_forward.next_action` fields.
5. **PR descriptions**: cite the backlog item ID (e.g., "closes BACKLOG B1, B4") for traceability.
6. **Unit contracts**: MUST inherit grounding pointers verbatim per §6 template.

---

# §5 — Open carries snapshot (cross-ref to current `.dfg/checkpoints/`)

| Carry ID | Item | Wave | Closes via |
|---|---|---|---|
| W8-CARRY-3 | L1 (Tables C-H) | W18 | builders implementation |
| W9-3-CARRY-1 | L4 (W1-W6 retros) | W19 | mechanical re-key |
| W12-CARRY-1 | (subsumed by B2) | W15 | K threading + ref point |
| W13-CARRY-1 | B2 (K threading) | W15 | SCENARIOS re-key |
| W13-CARRY-2 | RESOLVED 2026-05-17 by W14-2 smoke | — | walk-forward direction unchanged → root cause is NOT methodology |

---

# §6 — Canonical wave-unit contract template (grounding discipline)

Every W15+ unit's `.dfg/agents/W<N>-<n>-<slug>.md` MUST follow this
template. **The grounding pointers come verbatim from the BACKLOG.md
item; no paraphrasing.**

## Schema constraint (W15-gate receipt)

The dfg substrate schema for `read_contract.must_read` requires a
**plain-string list of relative paths** — NOT a list of dicts with
`path:`, `pages:`, `sections:`, `excerpt:`, `reason:` fields. Attempting
the dict form fails `dfg validate` with errors of the form
"<dict> is not of type 'string'".

**The rich grounding (page numbers, sections, verbatim excerpts,
per-file reasons) belongs in the contract body markdown below the
YAML frontmatter**, not in the YAML field itself. The body is human-
readable, survives schema evolution, and is echoed verbatim into the
PR description per Rule 7.

## Template

```yaml
---
id: W<N>-<n>
role: code-fixer | tooling-author | experimenter | verifier
name: <Short title>
purpose: "<One sentence stating which BACKLOG.md items this unit closes>"
wave: W<N>
unit: W<N>-<n>
depends_on: [<other unit ids>]
blocks: [<later unit ids>]
governance_tier: VT1 | VT2
sized: S | M | L
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/operators.py
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - <files this unit produces or modifies>
  branch_name: feat/w<N>-<n>-<slug>
  acceptance: >
    <Specific testable criteria>
dispatch_instructions: |
  <Implementation guidance referencing the grounding>
  Closes BACKLOG: <B1, B4, etc.>

  What NOT to do:
    - <constraints, e.g., don't modify other modules>
---

# W<N>-<n> — <Title>

Closes BACKLOG.md items: **<B1>**, **<B4>**.

## Thesis grounding

### Item B1 — Internal HV reference point wrong scale

**docs/BACKLOG.md §1.1 B1** — read FIRST for full diagnostic context.

**Thesis §7.2.3 ASMS Parameters (p. 147)** — verbatim:
> "the reference point for computing Hypv was set to z^ref = (0.2, 0.0)^T..."

**Thesis §3.1.1 The Hypervolume (S-Metric) Indicator (p. 57)** —
formal HV definition (reference-point role).

### Item B4 — Non-simplex weights AND wrong crossover operator

**docs/BACKLOG.md §1.1 B4** — read FIRST for full diagnostic context.

**Thesis §7.2 + §7.2.3 Search Operators (p. 141, 147)** — verbatim:
> "We utilized uniform crossover over the mean DD vectors. For mutation,
>  we randomly choose between (1) modifying the non-zero weights; or
>  (2) adding/removing assets..."

This defines the EXACT operators to implement; uniform crossover, NOT SBX.

## Files to touch (existing code context)

- `python_refactor/src/algorithms/operators.py` — module being edited
- `python_refactor/src/algorithms/sms_emoa.py` — caller of crossover/mutation

## Acceptance

<Specific testable criteria>
```

## Rules

1. **Catalog entry first**: every contract body leads with the
   `docs/BACKLOG.md` section pointer for that item. This forces
   readers to absorb the grounding before reading source files.

2. **Verbatim excerpts**: thesis excerpts in the body markdown must
   be verbatim quotes (no paraphrasing). Use thesis printed page
   numbers (not PDF page numbers).

3. **Multiple grounding sources per item**: when an item is grounded
   in both thesis AND paper AND a third-party reference, list all
   three in the body's grounding section.

4. **Per-file reasons in body**: the body section "Files to touch"
   explains WHY each file in `must_read` is required. Vague reasons
   like "background context" are not acceptable.

5. **Implementation patterns (STD)**: when grounding is STD (no
   direct paper requirement), cite the community-standard practice
   explicitly (e.g., "STD: Python logging module discipline; see
   PEP 282").

6. **Multi-item units**: when a unit closes ≥ 2 BACKLOG items, the
   `purpose` field lists ALL closures, and each item gets its own
   ### subheading in the body's grounding section. No collapsing.

7. **PR description echo**: when shipping the PR for a unit, the PR
   body must reproduce the grounding excerpts in a "Grounding"
   section. This ensures the grounding survives squash-merge into
   master and is searchable via `gh pr view`.

8. **YAML must_read = plain strings only**: per W15-gate schema
   receipt — the YAML field cannot use dicts. Rich grounding lives
   in the body. `dfg validate` enforces this.

## Anti-patterns to avoid

- ❌ "see BACKLOG.md" — too vague; specify the item ID
- ❌ "thesis §7" — too broad; specify section + page + equation
- ❌ Paraphrasing the thesis — quote verbatim
- ❌ "this is well known in EA literature" — cite the specific
  reference and quote
- ❌ Reading code without reading the catalog first — every grounded
  item LEADS with the BACKLOG.md pointer in the body's grounding section
- ❌ Dict-form `must_read` entries in YAML — schema rejects them
  ("is not of type 'string'"); use plain-string list + body markdown

---

# §7 — Glossary cross-reference

| Symbol | Meaning | Where defined |
|---|---|---|
| `Û_t^{N*}` | Trained Pareto-flexible set at period t (N=20 portfolios) | TH §6.4 |
| `χ_{t+1}` | Future state parameters (μ_{t+1}, Σ_{t+1}) | TH §7.2.2 Step 4 |
| `m̂_{u*_t}` | Mean implemented portfolio (EMFC for the mHDM strategy) | TH §6.4 |
| `Ŝ_{t+1}` | Sample-average future hypervolume | TH Eq (7.11) |
| `λ_{t+h}` | Anticipation rate at horizon h | TH Eq (7.16) |
| `K` | OAL historical-window size | TH §7.1.1 (K ∈ {0,1,2,3}) |
| `H` | Anticipation horizon | TH §7.2.3 (H=2 fixed) |
| `T` | Total number of decision periods | TH §7.2.3 (T=25 default, T=24 FTSE) |
| `E` | Number of MC scenarios per period | TH §7.2.3 (E=1000) |
| `c_l, c_u` | Cardinality bounds | TH §7.2 Eq (7.3) |
| `z_ref` | HV reference point | TH §7.2.3 (z_ref=(0.2, 0.0)) |
| `N` | Population size | TH §7.2.3 (N=20) |
| `d` | Number of assets | TH §7.2.3 (d=87 FTSE) |
| `TIP` | Temporal Incomparability Probability | TH Eq (6.4) |
| `EMFC` | Estimated Maximal Flexible Choice | TH §5.2.10 + §6.4 |
| `AMFC` | Anticipated Maximal Flexible Choice (= EMFC selected via Eq 6.42) | TH §6.4 Eq (6.42) |
| `SPF` | Stochastic Pareto Frontier | TH §5.2 |
| `OAL` | Online Anticipatory Learning | TH §6 (full chapter) |
| `mHDM` | maximal-Hypv Decision Maker (implements EMFC) | TH §7.1 |
| `RDM` | Random Decision Maker (samples Pareto-flexible set) | TH §7.1 |
| `ASMS-EMOA` | Anticipatory S-Metric Selection EMOA | TH §6.5.2 + Pseudocode 8 |

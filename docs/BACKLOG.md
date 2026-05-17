# BACKLOG — full catalog of gaps, errors, and inconsistencies

*Authored 2026-05-17 after W14-2 walk-forward smoke confirmed S2 ≤ S0
direction is NOT a methodology artifact. Triggered by operator request
for comprehensive cataloging vs the Azevedo PhD thesis (UNICAMP 2014;
see `docs/THESIS-INDEX.md`) and the IEEE TCYB 2015 paper.*

**Scope**: every open finding from W7→W14 retros + the diagnostic Q&A
on 2026-05-17. Items are prioritized into waves W15–W19 (revised from
the prior "W15 = Option B Fig 7.15 grid" framing, which is now W18's
deliverable instead of the next wave).

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

---

# §1 — Full catalog

## §1.1 BLOCKER items (must land before any meaningful paper-replication claim)

### B1 — Internal HV reference point wrong scale [CORR + FID]

- **File**: `python_refactor/src/algorithms/sms_emoa.py:55-75`
- **Current**: `R1=0.0` (ROI), `R2=1.0` (risk) defaults, used in `_compute_hypervolume_contributions_class` (line 384) and `_compute_stochastic_hypervolume_contributions_class` (line 409)
- **Thesis (§7.2.3)**: `z_ref = (0.2, 0.0)^T` — risk_max=0.2, return_min=0.0
- **Effect**: SMS-EMOA's INTERNAL Pareto sorting + Δ_S contributions are computed against the wrong reference rectangle. Selection pressure points at the wrong corner of objective space. EFHV exported in metrics is also against the wrong reference (W13-1 threading kwarg makes it visible but doesn't fix the reference).
- **Fix**: pass `z_ref=(0.2, 0.0)` from validation_matrix's per-scenario config; thread through to SMSEMOA constructor; verify all 3 HV computation sites use it.
- **Wave**: W15

### B2 — K (OAL window-size) not threaded [CORR + FID]

- **File**: `python_refactor/experiments/validation_matrix.py:60-95` (SCENARIOS dict)
- **Current**: SCENARIOS map `max_horizon` (= paper Eq 14 **H**), NOT K. S0..S4 effectively run K=0.
- **Thesis (§7.2.3)**: K ∈ {0, 1, 2, 3} is the OAL historical-window size that drives `λ_{t+h}^K` in Eq 7.16. Fig 7.15 shows K=0 vs K>0 IS the anticipation factor.
- **Effect**: S2 "anticipatory" config is currently K=0 ≈ S0 baseline at the OAL-rate level. Anticipation cannot fire as the thesis intends.
- **Fix**: re-key SCENARIOS to expose K explicitly: `{ASMS,SMS} × {mHDM,RDM} × K ∈ {0,1,2,3}` factorial (matches thesis Fig 7.15 framing). Pass K through to AnticipatoryLearning constructor; verify it's consumed in `compute_anticipatory_learning_rate`.
- **Was**: W13-CARRY-1
- **Wave**: W15

### B3 — Cardinality constraint not enforced [CORR + FID]

- **File**: `python_refactor/src/portfolio/portfolio.py:57` declares `max_cardinality=10`; nowhere enforced
- **Current**: `compute_efficiency` measures cardinality post-hoc; no projection / no penalty / no repair operator
- **Thesis (§7.2.3)**: c_l=5, c_u=15 (5–15 assets active per portfolio)
- **Effect**: 98-asset over-diversification → portfolios converge to ~uniform-ish weights → all scenarios produce ~the index → no anticipation signal because everyone holds the same basket
- **Fix**: post-operator cardinality projection (zero out smallest weights until cardinality ≤ c_u; if cardinality < c_l, add random small weights to inactive assets). Or simpler: penalty term in fitness.
- **Wave**: W15

### B4 — Non-simplex weights possible from SBX crossover [CORR]

- **File**: `python_refactor/src/algorithms/operators.py:14-44` (SBX crossover + normalize)
- **Current**: SBX can produce negative weights; `crossover` only renormalizes by sum (`weights / np.sum(weights)`) — doesn't clip to [0, 1]. If sum > 0 and some weights < 0, the "normalized" result still has negative weights.
- **Thesis**: portfolio weights lie in the (d-1)-simplex (≥ 0, sum = 1). No short positions.
- **Effect**: negative weights break Markowitz variance formula (`w^T Σ w` can become negative or weird); selection pressure misdirected.
- **Fix**: clip to non-negative before renormalizing: `weights = np.maximum(weights, 0); weights = weights / weights.sum() if weights.sum() > 0 else uniform`. Likely affects `mutation` too — audit both.
- **Wave**: W15

---

## §1.2 HIGH items (likely contributing to direction reversal)

### H1 — Transaction costs NOT integrated into anticipatory HV objective [CORR + FID]

- **File**: `python_refactor/src/config/thesis_parameters.py:107` declares `TRANSACTION_COST_RATE=0.001`; only used downstream in `portfolio_evaluator` (final-wealth path)
- **Thesis Eq 7.4-7.5**: objective vector includes h-cost component reflecting transaction fees; AS-MOO solver SHOULD see this in `z_t = g(u_t, χ_t) + h(u_t, u*_{t-1})`
- **Effect**: optimizer can't avoid high-churn EMFCs because it doesn't see the cost; S2 might be implementing EMFCs whose realized future HV is eaten by trading fees
- **Fix**: subtract `h(u_t, u*_{t-1})` from anticipatory ROI estimate before computing HV; track previous-period implemented portfolio per seed for `u*_{t-1}` reference
- **Wave**: W16

### H2 — λ formula completeness uncertain (Eq 7.16) [CORR + INST]

- **File**: `python_refactor/src/algorithms/anticipatory_learning.py:249` `compute_anticipatory_learning_rate`
- **Current**: single-component computation (verify by trace logging); thesis says `λ_{t+h} = ½(λ_{t+h}^H + λ_{t+h}^K)`
- **Effect**: if only λ^H fires (TIP-driven), the K-historical-window self-adjustment doesn't kick in → tracking is sub-optimal → anticipatory rate mis-calibrated
- **Fix**: verify both arms fire; if not, implement λ^K (squared-residual sum from historical KF predictions per Eq 7.16). Add per-portfolio per-generation trace export.
- **Wave**: W15 (verify) + W16 (fix if needed)

### H3 — Date range mismatch [FID]

- **File**: `python_refactor/experiments/validation_matrix.py:105-118` WINDOWS dict
- **Current**: `paper` window = 2003-01-01 → 2012-11-20 (~2600 days; superset of thesis range)
- **Thesis (§7.2.3)**: FTSE-100 daily 2006-11-20 → 2012-12-31 (~1500 days, T=24 rolling 50-day-shift periods)
- **Effect**: training data span is too long; thesis aggregates over T=24 periods, ours implies T≈44 — different power, different regime mix
- **Fix**: slice paper window to 2006-2012 to match thesis
- **Wave**: W15

### H4 — Asset universe mismatch (98 vs 87) [FID]

- **File**: `legacy-cpp/executable/data/ftse-original/` has 98 CSVs
- **Thesis §7.2.3**: d=87 for FTSE
- **Effect**: 11 extra assets may include delisted / illiquid ones that pollute optimization
- **Fix**: identify the 87 thesis-faithful subset (cross-ref the thesis-cited dataset URL if available; otherwise drop the 11 with most missing/quirky data); document the asset list explicitly
- **Was**: noted in `docs/THESIS-INDEX.md` §3
- **Wave**: W16

### H5 — H (anticipation horizon) varies per scenario [FID]

- **Current**: S1 H=2, S2 H=3, S3 H=2, S4 H=3
- **Thesis (§7.2.3)**: H=2 fixed (one-step-ahead prediction with horizon length 2) across all anticipatory variants
- **Effect**: scenarios aren't ablating cleanly — the horizon ablation we INTENDED with S2 vs S3 isn't isolating K (it's varying H, which isn't what the paper varies)
- **Fix**: fix H=2 across all anticipatory variants; vary K instead (resolves B2 + H5 simultaneously)
- **Wave**: W15

### H6 — Extrema (anchor) points not preserved [CORR]

- **File**: `python_refactor/src/algorithms/sms_emoa.py` evolution loop
- **Current**: no explicit preservation of best-ROI / best-risk anchor solutions; SMS-EMOA's reduce step can drop them
- **Thesis**: anchors typically preserved in NSGA-II / SMS-EMOA implementations; affects HV computation (anchors define the front extent)
- **Effect**: HV oscillates more than it should; the Pareto front extent shrinks over generations because extreme solutions get pruned
- **Fix**: keep ranked-1 anchor points immune to reduce; verify via per-gen ROI / risk range trace
- **Wave**: W15 (or W17 if low priority)

### H7 — Correspondence mapping NOT wired into experiment path [CORR + FID]

- **File**: `python_refactor/src/algorithms/correspondence_mapping.py` exists; not invoked from `experiment_manager._run_algorithm` or `sms_emoa.run`
- **Thesis §6.2.2**: correspondence mapping aligns current vs. future projected Pareto sets across rolling periods
- **Effect**: rolling-period anticipation can't compare like-for-like portfolios across periods → λ self-adjustment uses noisy alignment → predictions degraded
- **Fix**: wire into anticipatory_learning rolling path; verify alignment via trace export
- **Wave**: W17

---

## §1.3 MEDIUM items (quality / methodology / interpretability)

### M1 — No per-generation HV trajectory exported [INST]

- **File**: `python_refactor/src/algorithms/sms_emoa.py:83` `hypervolume_history` tracked internally; only `final_hypervolume` exported in `metrics_collector.optimization_metrics`
- **Effect**: can't see if HV is improving over generations; can't diagnose convergence vs early stopping
- **Fix**: export per-gen HV list in `algorithm_metrics`; render as line plot per scenario
- **Wave**: W16

### M2 — No per-generation EFHV trajectory exported [INST]

- **Same shape as M1** but for `stochastic_hypervolume_history`
- **Wave**: W16

### M3 — Per-portfolio λ traces not logged [INST]

- **File**: `compute_anticipatory_learning_rate` computes λ per call, no record
- **Operator request**: "how are the traces of lambda across time steps for each ordered portfolio in the population? Both components (kalman error and confidence)?"
- **Fix**: per-call λ logging keyed by (period, generation, solution_rank); export as CSV alongside metrics.csv
- **Wave**: W16

### M4 — Per-portfolio TIP traces not logged [INST]

- **Same shape as M3** but for `temporal_incomparability_probability`
- **Wave**: W16

### M5 — Anticipated EMFC not compared vs alternatives [INST + METH]

- **Operator request**: "Is the anticipated maximal flexible choice 'better' in terms of predicted future hypv when compared to other portfolio available?"
- **Fix**: per period, compute predicted future HV for ALL Pareto-flexible portfolios; rank EMFC; export rank
- **Wave**: W16

### M6 — Coherence per Eq 7.14 not computed/logged [METH + INST]

- **Thesis Eq 7.14**: coherence = mean cosine similarity of each portfolio to population centroid in simplex
- **Current**: `algorithm.diversity_metric` is a different formula; coherence per Eq 7.14 not implemented
- **Fix**: implement `coherence(pareto_set)` function; export per-period
- **Wave**: W16

### M7 — POCID per Eq 7.13 not computed [METH]

- **Thesis Eq 7.13**: percentage of correct direction predictions over T-1 transitions
- **Current**: not implemented
- **Fix**: implement; export per-scenario in report
- **Wave**: W16

### M8 — No wealth time-series exported [INST]

- **Current**: `summary.final_portfolio_value` scalar only
- **Operator request**: "Wealth evolution?"
- **Fix**: export per-period wealth `W_t` (thesis Eq 7.12) including transaction costs
- **Wave**: W16

### M9 — No bootstrap CI on Ŝ_{t+1} per scenario [METH]

- **Current**: report raw mean ± std
- **Thesis-fidelity not strictly required** but standard practice for OOS reporting
- **Fix**: bootstrap-resample seed-level Ŝ values; 95% CI on grand mean
- **Wave**: W17

### M10 — No statistical testing in populated report [METH + REPT]

- **Operator request implicit in "data viz and tables"**
- **Thesis Table 7.2**: 2-way ANOVA WS × DM with significance markers
- **Current**: only Welch t reported in OOS reports; no ANOVA
- **Fix**: implement 2-way ANOVA per Table 7.2; add Mann-Whitney pairwise post-hoc (already have stats helpers in W8-2 — just need wiring)
- **Wave**: W18

### M11 — No constraint violation penalty in fitness [CORR]

- **Related to B3**: even if cardinality is post-projected, other constraints (min return, max risk) have no penalty path
- **Wave**: W17 (lower priority; B3 covers most of the harm)

### M12 — AMFC identification (Eq 6.42) not visible in code [CORR]

- **Thesis Eq 6.42**: `û*_t = arg max_u in U_t (1/E) Σ S(...)` — explicit MC-driven AMFC selection
- **Current**: SMS-EMOA returns the trained population; not clear where AMFC is identified vs random selection
- **Fix**: implement explicit AMFC identification at end of each period; log which Pareto rank was chosen
- **Wave**: W17

### M13 — Bayesian tracking confidence per portfolio not exported [INST]

- **Operator request**: "how is tip across portfolios and time steps?" and confidence in §7.3.1
- **Current**: KF covariance P_t computed per portfolio; not exported
- **Fix**: export per-portfolio confidence (1 - E[TIP]) traces
- **Wave**: W16

### M14 — No early-stopping / convergence check [METH]

- **Current**: fixed 30 generations regardless of convergence
- **Effect**: wasted compute if HV plateaus early; missed opportunity if 30 isn't enough
- **Fix**: optional early-stop on HV plateau (window of 5 gens with < 1% relative change)
- **Wave**: W17 (low impact for this dataset; nice-to-have)

### M15 — Non-dominated sorting basis verification [CORR]

- **Operator question**: "How are you applying non-dominated sorting? Over the expected (mean) return/risk?"
- **Current**: `solution.objectives = [P.ROI, P.risk]` deterministic means — thesis §6.5.1 says this is correct ✅
- **Action**: verify and document; no code change needed
- **Wave**: W15 (verify only)

---

## §1.4 LOW items (presentation / repo hygiene)

### L1 — Tables C/E/F/G/H builders stubbed [REPT]

- **Was**: W8-CARRY-3
- **File**: `python_refactor/experiments/report.py` — builders for tables C-H return `[]`
- **Wave**: W18

### L2 — F1-F12 figures not generated [REPT]

- **File**: `python_refactor/experiments/figures.py` exists from W7-1; never invoked end-to-end
- **Wave**: W18

### L3 — VALIDATION-RESULTS.populated.md mostly 🚧 placeholders [REPT]

- **Status**: §0 receipt populated; Table A populated for S0+S2; §1 has W13-3 + W14 walk-forward findings; rest are placeholders
- **Wave**: W18

### L4 — W1-W6 retros use non-canonical ADR-004 keys [INFRA]

- **Was**: W9-3-CARRY-1 (optional)
- **Effect**: `scripts/pre-pr-reflect-validate.sh --all` shows 22+ stale retros
- **Fix**: mechanical re-key of W1-W6 retros (same pattern as W9-3 did for W7+W8)
- **Wave**: W19

### L5 — Pre-pr battery has 4 carried failures [INFRA]

- **Items**: ruff (lint), pytest (unrelated failures), branch-shape (`main` vs `master` reference), skill-assessment-gate (kit/scripts path)
- **Carried since**: W7-era
- **Effect**: dfg pre-pr exits 1 even on green wave-close commits
- **Fix**: audit each: (a) `make fix` for ruff; (b) inventory pytest failures unrelated to the algorithm code; (c) configure branch-name expectation to `master`; (d) `kit/scripts/skill-assessment-gate-check.py` git diff target
- **Wave**: W19

### L6 — Algorithm log lines pollute stderr in batch mode [INFRA]

- **Current**: `print("Generation X: ...")` from sms_emoa.py:124 fires for every gen of every run
- **Effect**: noise in walk-forward sweep logs (44 periods × 60 seeds × 30 gens = lots of lines)
- **Fix**: switch to `logging` with WARNING-level default; let pytest see INFO; sweep sees nothing
- **Wave**: W19

### L7 — Stochastic seed handling under ProcessPoolExecutor [CORR]

- **Current**: `np.random.seed(seed)` set inside subprocess; should isolate but `Portfolio.mean_ROI` is class-level so a stale class state could leak if multiple seeds run in the same process (ProcessPoolExecutor's `chunksize=1` mitigates but isn't guaranteed)
- **Fix**: instance-level Portfolio state (related to a future refactor noted in W13-1 retro)
- **Wave**: W19

### L8 — `kit/` symlink trap [INFRA]

- **Was**: W9-3 scar (wrote tooling into `kit/` which is a symlink to dfg-harness)
- **Current state**: scripts/ used as the repo-owned tooling dir; no recurrence
- **Fix**: add `git check-ignore` reminder in pre-pr; potentially add `.gitignore` for `kit/scripts/*` to prevent accidental writes
- **Wave**: W19 (optional defensive)

---

# §2 — Wave plan

| Wave | Theme | Items | Sizing | Estimated wall-clock |
|---|---|---|---|---|
| **W15** | Algorithm correctness (BLOCKERS) | B1 (ref point) + B2 (K threading) + B3 (cardinality) + B4 (simplex projection) + H3 (date range) + H5 (H=2 fixed) + M15 (verify NDS) | M-L | 2-4h engineering + 1-2h re-runs |
| **W16** | Instrumentation + λ-completion + secondary fidelity | H2 (λ completeness) + H4 (87 assets) + M1-M8 + M13 (per-gen / per-portfolio traces) | M-L | 3-5h engineering + 2-4h compute |
| **W17** | Advanced (txn costs + correspondence + extras) | H1 (txn costs in HV) + H6 (extrema) + H7 (corr mapping wiring) + M9-M12 + M14 | M | 3-5h |
| **W18** | Comprehensive report + viz (the operator-asked report) | L1 (tables C-H) + L2 (figures F1-F12) + L3 (populate full results doc) + M10 (ANOVA per Table 7.2) | M | 2-4h engineering + 4-8h compute (full 30-seed × 24-period × all-fixes run) |
| **W19** | Repo hygiene (low-priority cleanup) | L4-L8 | S | 1-2h |

**Critical path to paper-replication answer:**

W15 → W16 (λ verification only — defer instrumentation if compute-bound) →
quick walk-forward re-run → if S2 > S0, success; if not, W17 → re-run.

**Critical path to "the report":** W15 + W18 (skipping W16 instrumentation
and W17 advanced features if necessary; the report shows the FINAL state,
not the per-gen traces).

---

# §3 — Cross-reference to thesis sections

| Thesis section | Backlog items it informs |
|---|---|
| §6.5.1 NDS over mean vectors | M15 (verify) |
| §6.4 Estimating AMFC | M12 |
| Eq 6.42 AMFC identification | M12 |
| §7.2.3 ASMS Parameters | B1, B2, B3, B4, H3, H4, H5 |
| Eq 7.4-7.5 Objective with h-cost | H1 |
| Eq 7.13 POCID | M7 |
| Eq 7.14 Coherence | M6 |
| Eq 7.16 λ_{t+h} = ½(λ^H + λ^K) | H2 |
| §7.3.1 Bayesian tracking confidence | M13 |
| §7.3.2 POCID effects | M7 |
| §7.3.3 OOS Future HV (Eqs 7.10+7.11) | already in W13-2 ✓ |
| §7.3.4 Coherence / Cardinality / Wealth | M6, M8 (B3 for cardinality) |
| §7.3.5 Wealth evolution | M8 |
| Table 7.2 ANOVA (WS, DM, WS:DM) | M10 |
| Fig 7.15 boxplots OOS Future HV per (K, DM) | W18 deliverable |
| Fig 7.16-7.25 per-instance dynamics | W18 deliverable |

---

# §4 — How to use this backlog

1. **Before each new wave**: open this file; pick items per the wave plan in §2.
2. **When new findings surface**: append to the appropriate §1.x sub-section; bump severity if needed.
3. **When an item closes**: mark it ✅ inline and reference the closing PR.
4. **Wave-close ceremonies**: reference §1 item IDs (B1, H2, etc.) in `carry_forward.next_action` fields.
5. **PR descriptions**: cite the backlog item ID (e.g., "closes BACKLOG B1, B4") for traceability.

---

## §5 — Open carries snapshot (cross-ref to current `.dfg/checkpoints/`)

| Carry ID | Item | Wave | Closes via |
|---|---|---|---|
| W8-CARRY-3 | L1 (Tables C-H) | W18 | builders implementation |
| W9-3-CARRY-1 | L4 (W1-W6 retros) | W19 | mechanical re-key |
| W12-CARRY-1 | (subsumed by B2) | W15 | K threading + ref point |
| W13-CARRY-1 | B2 (K threading) | W15 | SCENARIOS re-key |
| W13-CARRY-2 | RESOLVED 2026-05-17 by W14-2 smoke | — | walk-forward direction unchanged → root cause is NOT methodology |

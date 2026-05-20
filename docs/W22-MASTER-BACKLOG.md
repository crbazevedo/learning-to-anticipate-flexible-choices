# W22 Master Backlog — full visibility

**Status:** active push-through autonomous mode
**Branch:** `feat/w22-inspection-backlog` (20+ commits ahead of base)
**Updated:** 2026-05-20

This doc is the SINGLE SOURCE OF TRUTH for outstanding work. All other
W22-*.md docs roll up to entries here. When a deliverable ships, update
the Status column with the commit hash.

---

## Section A — Inspections (analytical findings)

| # | Inspection | Status | Linked NCs/Probes |
|---|---|---|---|
| 1 | TIP joint vs conditional sampling | SHIPPED `7689ab9` | NC31 conditional mode (`7940604`) |
| 2 | Eq 6.41 truncation (75% under-estimate) | SHIPPED `b67601d` | NC26-deep DEFERRED (needs market-Σ) |
| 3 | Dirichlet predictor = exponential smoothing | SHIPPED `72dfea0` | NC27 (`52a1522`), NC27-deep (`b9ccaad` + `9c51faf`) |
| 4 | Correspondence mapping = dead infrastructure | SHIPPED `33c533d` | NC28 deferred (operator decision: remove or wire?) |
| 5 | Multi-horizon discount is not really a discount | SHIPPED `14f06fe` | NC29 w_0 floor + NC29a γ^h |
| 6 | AMFC implementation uses wrong objective | SHIPPED `8b0f983` + `bccfa40` (correction) | NC30 family complete |

## Section B — Structural fixes shipped (operator directive 2026-05-19)

| # | NC | Commit | Default | Status |
|---|---|---|---|---|
| 1 | NC30-v2 CRN + analytical + tied-mean | `03a3956` | ON | SHIPPED |
| 2 | NC30 b data-derived z_ref | `03a3956` | ON | SHIPPED |
| 3 | NC29 w_0 floor (≥0.2) | `03a3956` | ON | SHIPPED |
| 4 | NC29a γ^h geometric discount | `f42cc9d` | ON | SHIPPED |
| 5 | NC13b smooth TIP clamp | `308de50` | OFF | SHIPPED (opt-in) |
| 6 | NC27-deep DirichletPosteriorWrapper | `9c51faf` | OFF | SHIPPED (opt-in) |
| 7 | NC30 c continuous variance penalty | `300fedc` | OFF | SHIPPED (opt-in) |
| 8 | NC31 TIP Defn 6.1 conditional | `7940604` | OFF | SHIPPED (opt-in) |

## Section C — Deferred structural fixes (need plumbing)

| NC | Blocker | Effort |
|---|---|---|
| NC26-deep | Market-Σ piped into SMSEMOA layer | L (~500 LOC + tests) |
| NC30 Option B (path-dependent forecast) | Market-Σ same dependency | L (~400 LOC + tests) |
| NC28 (correspondence rewire OR removal) | Operator decision: kill or wire? | S/M depending |

---

## Section D — Probes designed but not implemented

### Probes Q/R/S (alt predictive signals — docs/W22-ALT-SIGNAL-PROBES.md)

| Probe | Hypothesis | Status | LOC est. |
|---|---|---|---|
| Q-v1 (AR(1)) | Per-asset AR(1) beats no-change baseline downstream | DESIGNED | 200-300 |
| Q-v2 (GARCH) | GARCH(1,1) adds conditional-variance signal | DESIGNED | 600 |
| Q-v3 (VAR(1)) | Cross-asset lead-lag at lag 1-5 | DESIGNED | 300 |
| R (Granger) | Causal graph propagation adds wealth | DESIGNED | 400-500 |
| S-v1 (eigenvector centrality) | Hub-tilt improves wealth | DESIGNED | 300-400 |
| S-v2 (centrality-KF) | Augmenting KF state with centrality helps | DESIGNED | 500-700 |

### New probes T-AE (designed this turn)

| Probe | Hypothesis | LOC est. |
|---|---|---|
| T | NC29a γ sensitivity sweep ({0.5, 0.7, 0.9, 0.99}) | 150 |
| U | NC30 c variance_penalty α sensitivity ({0, 0.1, 1, 10}) | 150 |
| V | AMFC opt-in fix combination ablation (5 fixes × pairwise) | 300 |
| W | NC27-deep production impact on FTSE | 200 |
| X | NC31 conditional TIP impact on FTSE | 200 |
| Y | NC13b smooth clamp impact on TIP-saturation regimes | 250 |
| Z | Stability factor restoration (Reading-F revisited) | 200 |
| AA | AMFC-vs-Hv-DM telemetry harness on FTSE | 250 |
| AB | λ^K per-portfolio differentiation (Inspection 6) | 400 |
| AC | KF NIS/NEES consistency diagnostics | 350 |
| AD | Stochastic vs deterministic Δ_S accuracy comparison | 200 |
| AE | Asset universe scaling — ASMS uplift vs K | 300 |
| AF | Mehra (1970) Q-noise estimation from KF residuals | 500 |
| AG | Constant-velocity F vs random-walk F selection | 300 |
| AH | Aggregated AMFC vs Hv-DM weighting on FTSE | 250 |

---

## Section E — Hypotheses (testable claims awaiting empirical validation)

### Validated
- **H-NC8c-v2**: cross-period KF position carry-forward gives +7.50% on FTSE n=10 p=0.003 (BREAKTHROUGH — W22 prior)
- **H-Inspection-3**: TRUE Dirichlet posterior is 2.8× more accurate than exponential smoothing on synthetic Dirichlet data (locked in `test_inspection_3_accuracy_gain_vs_dirichlet_predictor`)
- **H-Inspection-6**: AMFC current-Δ_S disagrees with TRUE-AMFC (forward-HV-given-choice) on 82% of synthetic Pareto fronts (locked in `inspect_amfc.py TEST 6`)
- **H-Inspection-2**: Eq 6.41 truncation under-estimates E[Δ_S] by 75% under shared-market-noise (locked in `inspect_efhv_eq641.py`)

### Open (need empirical work)
- **H-NC30-v1-production**: Hv-DM-AMFC beats Hv-DM by ≥3.0% on FTSE 2015 n≥10 (PASS criterion in NC30 contract)
- **H-NC27-deep-prod**: TRUE Dirichlet posterior in production beats exponential smoothing on FTSE n≥10
- **H-NC29a-γ-optimal**: optimal γ ∈ {0.5, 0.7, 0.9, 0.99} for FTSE / NASDAQ
- **H-NC30c-α-optimal**: optimal variance_penalty α for FTSE
- **H-NC13b-tail-signal-matters**: smooth clamp on saturated-TIP periods produces non-trivial λ^H signal that doesn't exist with hard clip
- **H-NC31-TIP-conditional-impact**: Defn 6.1 conditional mode changes λ^H by < 5% (per Inspection 1 empirical equivalence finding); changes final wealth by < 2%
- **H-Probe-Q-AR1**: AR(1) per-asset predictions add downstream wealth beyond no-change baseline
- **H-Probe-S-hub-tilt**: positive γ tilt toward eigenvector-central assets improves Sharpe
- **H-Probe-R-Granger**: ≥5% of asset pairs are robustly Granger-causal under FDR<0.05
- **H-stability-factor-load-bearing**: Reading-F's flip from per-solution to 1.0 stability cost or gained X% wealth

### Contrarian / what-if hypotheses
- **C-H1**: ASMS uplift is REGIME-dependent, not universal. Test by running across FTSE / NASDAQ / HangSeng / Sao Paulo / EuroStoxx in parallel.
- **C-H2**: NC30-v1 (operator-correct AMFC) UNDERPERFORMS Hv-DM on FTSE because Hv-DM benefits from the implicit anticipation in mutated ẑ_t. Possible PR loss.
- **C-H3**: NC27-deep posterior is TOO aggressive — over-fits to recent observations in non-stationary regimes. Could lose on regime shifts.
- **C-H4**: TIP clamp at [0.05, 0.95] IS load-bearing (its boundedness stabilizes downstream λ computations). NC13b's smooth squash could destabilize.
- **C-H5**: ASMS's edge is entirely from the (ROI, risk) KF predictions and the "AMFC", "TIP", "Dirichlet" pipeline are scaffold. Test by ablating each layer.

---

## Section F — Counterfactual experiments (probes by exclusion)

| Name | Question |
|---|---|
| CF-1 | What if we run SMS-EMOA (no anticipation) on the same FTSE data — what's the baseline? |
| CF-2 | What if we run ASMS with all structural fixes OFF — does it still get +7.50%? |
| CF-3 | What if we run ASMS with all 8 structural fixes ON — does it beat the baseline by more? |
| CF-4 | What if we ablate NC8c-v2 (the breakthrough) and keep only NC30 family — what's lost? |
| CF-5 | What if AMFC uses argmax(forecast E[HV]) but with z_ref = sliding window — does NC30 b dominate? |
| CF-6 | What if we replace KF state evolution F=constant-velocity with F=random-walk — what changes? |
| CF-7 | What if we increase MC sampling n_mc from 200 to 1000 in AMFC — does signal/noise improve? |
| CF-8 | What if we ablate Eq 14 anticipation entirely but keep NC30-v1 AMFC selection — separation? |

---

## Section G — Cross-cutting research questions

### Q1: What is the marginal contribution of each W22 NC?
Each NC has been justified algebraically + tested in isolation. But the JOINT effect across NCs on production wealth is unknown. **Probe V** (combination ablation) addresses this.

### Q2: Are the opt-in structural fixes safe to ratify as production defaults?
The 4 OFF-by-default fixes (NC13b, NC27-deep, NC30 c, NC31) need empirical FTSE validation before flipping defaults. **Probes W, X, Y** address.

### Q3: Is the (ROI, risk) state-space adequate, or do we lose signal by collapsing per-asset returns?
**Probes Q, R, S** test alternative signal extractions that preserve per-asset / per-pair information.

### Q4: Does ASMS uplift scale with universe size K?
FTSE K=30 shows +7.50%. NASDAQ K=81 untested. HangSeng K=31 untested. **Probe AE** tests.

### Q5: Is the KF properly tuned?
Q-noise estimation (Mehra 1970), NIS/NEES consistency, F-matrix selection — these are KF-tuning questions that determine whether the predictor is well-calibrated. **Probes AC, AF, AG**.

---

## Section H — Empirical work order (priority-ranked)

For an operator-driven session with compute budget:

| Priority | Action | Compute estimate |
|---|---|---|
| P0 | **Probe AA**: AMFC-vs-Hv-DM telemetry on FTSE n=10 with NC30-v1 (Hv-DM-AMFC dm_type) | 100 min |
| P0 | **Probe V**: NC opt-in combination ablation (smoke first 4 combos) | 60 min |
| P1 | **Probe T**: NC29a γ sensitivity on FTSE n=5 | 75 min |
| P1 | **Probe W**: NC27-deep production smoke on FTSE n=5 | 60 min |
| P1 | **Probe X**: NC31 conditional TIP smoke on FTSE n=5 | 60 min |
| P2 | **Probe Q-v1**: AR(1) per-asset prediction smoke | 90 min |
| P2 | **Probe S-v1**: eigenvector centrality tilt smoke | 75 min |
| P3 | **Probe AE**: scaling NASDAQ K=81 n=5 | 180 min |
| P4 | **Probes AC/AF/AG**: KF diagnostics + Mehra Q + F selection | 240 min combined |
| P5 | **Probe R**: Granger causality (high risk of FAIL per literature) | 200 min |

## Section I — Dispatch matrix for parallel autonomous work

The following can be done IN PARALLEL by independent agents (no shared state):

| Agent | Task | Independent? | Tools needed |
|---|---|---|---|
| A1 | Implement Probe Q-v1 (AR(1) per-asset baseline) | YES | numpy, statsmodels |
| A2 | Implement Probe S-v1 (eigenvector centrality tilt) | YES | networkx, numpy |
| A3 | Implement Probe T (γ sensitivity sweep harness) | YES | itertools, no external deps |
| A4 | Implement Probe AA (telemetry harness + analysis) | YES | numpy, json |
| A5 | Implement Probe AC (KF NIS/NEES diagnostics) | YES | scipy.stats |
| A6 | Implement Probe AD (Stochastic vs Deterministic Δ_S compare) | YES | numpy |

Each agent ships a contained Python module + tests + a docs/W22-PROBE-{X}-{NAME}.md
design doc. No agent modifies shared code paths — they all build orthogonal probes.

---

## Section J — Risks and known issues

- **Test surface fragility**: 14 pre-existing failures in test_anticipatory_learning, 5 in test_algorithms, 4 errors in test_scenario_differentiation — these block a "green CI" claim. Operator may want a separate fix-pre-existing-tests session.
- **Branch length**: 20+ commits ahead of base. PR review will be a lift. Could break into smaller PRs (e.g., one per NC) but risks losing the synthesis context.
- **Compute budget**: many probes need ASMS runs (5-15 min each). 10-seed × 5-probe smoke is ~10 hrs serial wall-clock. Parallel dispatch helps but isn't free.
- **Operator overload**: 8 new opt-in flags + many probes. Need a "what to flip first" prioritization for ratification.

---

## Section K — Where to look

- **Code**: `python_refactor/src/algorithms/{amfc_selector, anticipatory_learning, multi_horizon_anticipatory, temporal_incomparability_probability, sms_emoa}.py`
- **Tests**: `python_refactor/tests/test_nc{27,27_deep,29,29a,30,31}_*.py` (102/102 passing)
- **Synthesis docs**:
  - `docs/W22-INSPECTIONS-SYNTHESIS.md` — 6-inspection roll-up
  - `docs/W22-STRUCTURAL-FIXES.md` — 8-fix catalog
  - `docs/W22-NC30-CONTRACT.md` — AMFC design
  - `docs/W22-ALT-SIGNAL-PROBES.md` — Q/R/S probe specs
  - `docs/W22-RESEARCH-PROGRAM.md` — original 9 areas
  - `docs/W22-MASTER-BACKLOG.md` — THIS DOC
- **Telemetry**: `from src.algorithms.amfc_selector import get_amfc_telemetry, reset_amfc_telemetry`

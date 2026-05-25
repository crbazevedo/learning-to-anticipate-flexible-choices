# W22 Next-Steps — NC32 / NC34 / NC35 / NC36 + AD-fix

**Status:** active push; full autonomy
**Trigger:** operator approval 2026-05-20 + new directive — incorporate benefits + propose theory/math/stats/algo improvements in Dirichlet tracking, OAL, ASMS-EMOA inner mechanisms (anticipatory operators), TIP, KF, with Dirichlet-KF coupling

This doc catalogs 5 next-step deliverables. Each has: hypothesis, math/theory foundation, implementation plan, falsifiable success criterion, expected scars.

---

## Section A — Incorporate benefits (Phase 1)

### A1. NC-AD-fix: Eq 6.41 rectangle alignment

**Bug** (per Probe AD `881d0f2`): `sms_emoa.py:772-781` middle-branch formula uses
`(ROI_i - ROI_{i+1}) · (risk_{i-1} - risk_i)` but Eq 6.41 specifies
`(ROI_i - ROI_{i-1}) · (risk_{i+1} - risk_i)`. The stochastic `- Cov` correction
is derived against Eq 6.41 → overshoot in legacy rectangle.

**Decision**: realign deterministic to Eq 6.41 (operator's audit cited Eq 6.41).

**Implementation**:
- `sms_emoa.py:_compute_hypervolume_contributions_class` middle-branch: swap formulae
- Update test `test_sms_emoa.py` if it pins the old formula
- Probe AD's SCAR-pinned regression test will FLIP from documenting the bug → confirming fix

**Falsifiable criterion**: Probe AD's `test_stochastic_closer_to_mc_than_deterministic_when_cov_nonzero` PASSES after fix (currently SCAR-pinned).

---

## Section B — Theory next-step (Phase 2)

### B1. NC36: Analytical TIP under bivariate Gaussian

**Hypothesis**: TIP can be computed in CLOSED FORM under the bivariate Gaussian
forecast model used by the KF, eliminating MC noise entirely.

**Math** (per inspect_tip_chain.py `tip_analytical_conditional`):

For TIP_conditional with current = (c_ROI, c_risk) FIXED and predicted ~ N(μ, Σ):
- Standardize: z1 = (c_ROI - μ_ROI) / σ_ROI, z2 = (c_risk - μ_risk) / σ_risk
- Correlation ρ = Σ_{12} / (σ_ROI · σ_risk)
- P[predicted_ROI < c_ROI ∧ predicted_risk > c_risk] = Φ(z1) - Φ_2(z1, z2; ρ)
- P[predicted_ROI > c_ROI ∧ predicted_risk < c_risk] = Φ(z2) - Φ_2(z1, z2; ρ)
- TIP_conditional = 1 - P[c dom p] - P[p dom c]

Where Φ is standard normal CDF, Φ_2 is bivariate normal CDF (via `scipy.stats.multivariate_normal.cdf`).

**Benefit**: eliminates MC noise → deterministic TIP across calls; faster (analytical) than `monte_carlo_samples=1000` MC; better convergence.

**Implementation**:
- `temporal_incomparability_probability.py`: add `_calculate_tip_analytical_conditional()` method
- Env var `W22_NC36_TIP_ANALYTICAL=1` switches `_calculate_tip_with_covariance` to call the analytical version
- Default OFF for backward compat
- Composes with NC31 (also fixes the joint-vs-conditional question definitionally)

**Falsifiable**:
- Analytical TIP matches MC TIP (10000 samples) within ±0.02 across Inspection 1 scenarios
- Determinism: 10 calls with same input return identical value (vs MC variance)

**Scars**: PSD constraint required on Σ (already handled by KF state); ρ ∈ [-1, 1] (analytic formula degenerates at ρ=±1, handled by epsilon).

---

### B2. NC32: Joint Logistic-Normal-KF (LNKF) — Dirichlet ↔ KF coupling

**Hypothesis**: portfolio weights w ∈ Δ^{d-1} can be tracked by a STANDARD KF
operating in unconstrained Aitchison log-ratio coordinates `y = log(w_i / w_d)`.
This couples the (ROI, risk) KF with the weight tracker into a single
mathematical framework, eliminating the artificial separation between the
"Dirichlet predictor" (decision space) and "Kalman filter" (objective space).

**Math**:
- Forward transform (Aitchison): y_i = log(w_i / w_d) for i ∈ {0, …, d-2}
- Inverse: w_d = 1/(1 + Σ exp(y_i)), w_i = exp(y_i) · w_d
- KF state y_t ∈ R^{d-1}; F = I (random walk) or AR(1) per-component
- Observation: y_obs = forward(w_realized)
- Joint state (Phase 2): x_joint = [ROI, risk, y_1, ..., y_{d-1}]^T
  - F_joint = block_diag(F_perf, F_weights)
  - H_joint = block_diag(H_perf, I_{d-1})
  - Off-diagonal Q couples (ROI, risk) trajectory to weight evolution (allowing
    weight changes to PREDICT ROI/risk changes)

**Benefit**:
- Uses standard KF infrastructure (no new filter math needed)
- Closed-form posterior variance per weight (vs current Dirichlet's approximation)
- Direct coupling: realized weights inform predicted (ROI, risk) and vice versa
- Replaces the "two separate filters" architecture flagged by Inspection 3

**Implementation (Phase 1 — independent y-KF)**:
- New module `src/algorithms/logistic_normal_kf.py`:
  - `LogisticNormalKF(d, init_alpha=0.5, process_noise=0.01)` class
  - `observe(w_realized)` — forward-transform + KF update step
  - `predict(h_steps=1)` — KF predict; returns (μ_y, Σ_y) at horizon h
  - `predict_simplex_mean()` — inverse-transform μ_y to simplex mean
  - `predict_simplex_samples(n_mc)` — sample from N(μ_y, Σ_y) and inverse-transform
- Tests (15+): construction, observe-and-predict, multi-step prediction, round-trip identity, comparable accuracy to DirichletPosteriorPredictor

**Falsifiable criterion**: LNKF accuracy is COMPARABLE to NC27-deep `DirichletPosteriorPredictor` on Inspection 3 scenario (within 50% relative). Coupling benefits accrue in Phase 2.

**Scars**: log-ratio is reference-asset dependent (results equivariant); EPS clipping at zero weights; PSD Q matrix tuning needed.

---

### B3. NC34: Anticipatory mutation scorer — operators that anticipate offspring

**Hypothesis**: standard mutation perturbs weights randomly; ASMS would benefit
from offspring SCORED BY THEIR PREDICTED FUTURE Δ_S contribution. Selecting
mutants that ANTICIPATE high future contribution beats blind perturbation.

**Math**:
- For each of K candidate mutants {m_1, ..., m_K}:
  - Forecast each mutant's (ROI, risk) at horizon h via KF (using parent's KF state as prior)
  - Compute predicted_Δ_S(m_k) = front-contribution(m_k forecast, current Pareto frontier forecast)
- Selection: argmax_k predicted_Δ_S, OR weighted random with probability ∝ exp(β · predicted_Δ_S)
- For probability-of-non-dominance scoring: compute P_ND(m_k vs current solutions) via TIP

**Benefit**:
- Genetic search becomes anticipatory (operators anticipate offspring fitness)
- Reduces wasted evaluations on dominated mutants
- Operator-flagged "operators that preserve anticipatory signals"

**Implementation**:
- New module `src/algorithms/anticipatory_mutation_scorer.py`:
  - `score_mutants(parent, mutants, current_front, kf_predictor, h=1) -> np.ndarray`
  - `select_best_mutant(parent, mutants, current_front, kf_predictor, h=1)` — argmax
  - `select_weighted_random(parent, mutants, current_front, kf_predictor, h=1, beta=1.0, rng)` — softmax
- Tests (10+): score sanity (better mutant → higher score), argmax returns valid mutant, softmax invariants, zero-mutants edge

**Falsifiable**: on synthetic, anticipatory scorer's expected Δ_S of selected mutant > expected Δ_S of random mutant by ≥ 20% on average.

**Scars**: cost scales O(K · n_mc); for K=10, h=1, n_mc=50 → 500 forecasts per mutation call; tractable. Forecast accuracy depends on KF tuning (so couples with NC32).

---

### B4. NC35: Accumulated future Δ_S over H periods

**Hypothesis**: AMFC's current implementation (NC30-v1) uses single-period
forward forecast. Multi-period accumulated expectation is the true forward-
looking objective:

    Q^A(s) = Σ_{h=1}^H γ^h · E[Δ_S^(s)_{t+h}]

Operator-flagged "future expected hypervolume contribution w.r.t. whole
accumulated future period."

**Math**:
- For each candidate s on the current Pareto frontier:
  - For each h ∈ {1, ..., H}:
    - Forecast s's (ROI, risk) at t+h via KF
    - Forecast the OTHER solutions' (ROI, risk) at t+h (same KF; or assume static)
    - Compute predicted_Δ_S(s, t+h) using front-contribution formula
  - Q^A(s) = Σ_{h=1}^H γ^h · predicted_Δ_S(s, t+h)
- argmax_s Q^A(s) is the NC35 selection

γ is reused from NC29a (default 0.9).

**Benefit**:
- Aligns AMFC with the operator's intent of "anticipative maximal flexible choice"
  over multiple future periods (not just t+1)
- Solutions whose contribution PERSISTS over multiple periods get higher Q^A
- Compositional with NC30 family (CRN + analytical + variance-penalty + tie-break)

**Implementation**:
- Extend `amfc_selector.py`:
  - Add `horizon_accumulated: int = 1` kwarg to `select_amfc`
  - When > 1, sum γ^h-discounted contributions over h=1..H
  - γ pulled from `W22_NC29A_GAMMA` env var (default 0.9)
- Tests (10+): identity at H=1 reduces to NC30-v1; H=3 produces longer-decay contributions; γ=0 reduces to no-decay sum

**Falsifiable**:
- At H=1: NC35 ≡ NC30-v1 (regression test)
- At H>1: Q^A favors solutions with persistent forecast positions
- Compute cost stays under 200ms for |P|=20, H=3, n_mc=50

**Scars**: KF forecast variance grows with h → noisy at H≥5; mitigated by γ^h decay. Solutions whose forecast diverges quickly (high Σ_h) naturally get lower Q^A weight (since variance penalty composes).

---

## Section C — Cross-cutting integration

### Dirichlet ↔ KF coupling map (full picture)

| Direction | Mechanism | Status |
|---|---|---|
| Dirichlet posterior var → KF Q | High concentration α → tight posterior → low Q | NC33 candidate (deferred) |
| KF residual → Dirichlet update aggressiveness | Innovation magnitude → concentration_increment scaling | NC37 candidate (deferred) |
| LN-Gaussian on simplex → KF on y-space | Standard KF on log-ratio coords | **NC32** (this push) |
| Joint state (ROI, risk, y) | One KF for everything | NC32 Phase 2 (deferred) |
| KF predicts → Dirichlet observes | Sequential KF→Dir | Existing path |
| Dirichlet predicts → KF observes | Sequential Dir→KF | **NC32 enables this** (y_observed → KF predict) |

---

## Section D — Execution plan

**Foreground (this turn)**:
1. NC-AD-fix: rectangle alignment in sms_emoa.py + regression test update
2. NC36: analytical TIP module + env-var dispatch + tests

**Parallel agents (this turn)**:
3. NC32: Logistic-Normal-KF standalone module + tests
4. NC34: anticipatory mutation scorer standalone module + tests
5. NC35: accumulated future Δ_S extension to select_amfc + tests

**Deferred to next push**:
- NC33 (Dirichlet→KF Q coupling)
- NC37 (KF→Dirichlet update coupling)
- NC32 Phase 2 (joint state)

---

## Section E — Hypotheses register update

Adding to W22-MASTER-BACKLOG.md Section E:

- **H-NC36-TIP-analytical-eq-MC**: analytical TIP within ±0.02 of MC(10000) across 5 Inspection-1 scenarios. Deterministic across calls.
- **H-NC32-LNKF-comparable-to-NC27deep**: LN-KF accuracy within 50% relative of DirichletPosteriorPredictor on synthetic Dirichlet data. Both should converge similarly.
- **H-NC34-anticipatory-beats-random**: expected Δ_S of anticipatory-scored mutant ≥ 20% higher than random mutant on synthetic.
- **H-NC35-H1-identity**: at H=1, NC35 ≡ NC30-v1 exactly.
- **H-NC35-multi-period-different**: at H=3, NC35 argmax differs from NC30-v1 argmax in ≥ 15% of synthetic cases (real signal of persistence).
- **H-AD-fix-pinned-test-flips**: Probe AD's currently-SCAR-pinned test `test_stochastic_closer_to_mc_than_deterministic_when_cov_nonzero` PASSES after rectangle alignment.

---

## Section F — Risks

- **Shared-code modifications**: NC-AD-fix and NC35 (via amfc_selector extension) touch shared code. Will need careful test coverage to ensure no regression.
- **Existing tests may pin old rectangle**: a `test_sms_emoa.py` may verify the old formula numerically. Need to update that to the new (Eq 6.41 correct) numbers.
- **LN-KF reference asset choice**: results are equivariant under reference change, but numerical stability may favor specific choices. Implementation should expose this as a parameter.
- **Mutation scorer composition with existing operators**: standalone in this push; integration into the SMSEMOA evolutionary loop is operator's decision.

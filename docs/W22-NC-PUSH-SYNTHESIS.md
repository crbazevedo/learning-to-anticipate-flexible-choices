# W22 NC Push Synthesis — Theory-driven Tier-2

**Status:** COMPLETE; 8 NCs shipped (NC35 included), all tests green
**Trigger:** operator approval 2026-05-20 — "incorporate benefits + propose
next-step hypotheses backed by theory/math/stats/algo... improvements in
Dirichlet tracking, OAL, ASMS-EMOA inner mechanisms (anticipatory
operators), TIP/KF inner workings, Dirichlet-KF coupling"
**Branch:** `feat/w22-inspection-backlog`

---

## Shipped this push (8 NCs, 92 new tests + 28 NC30 regression preserved)

| NC | Topic | Commit | Tests | Theory class |
|---|---|---|---|---|
| **NC-AD-fix** | Rectangle realignment (Eq 6.41 consistency in sms_emoa) | `e6cd995` | (existing 21 pass) | Bug fix from Probe AD |
| **NC36** | Analytical TIP under bivariate Gaussian | `448ba64` | 10/10 | TIP inner workings — closed form |
| **NC34** | Anticipatory mutation scorer | `c392756` | 13/13 | ASMS-EMOA operators — anticipate offspring |
| **NC32** | Logistic-Normal Kalman Filter (LNKF) | `1d127b5` | 16/16 | Dirichlet ↔ KF coupling (geometry) |
| **NC33** | Dirichlet ↔ KF coupling (sequential) | `ae69636` | 15/15 | Dirichlet ↔ KF coupling (param coupling) |
| **NC38** | Accumulated multi-period P(non-dominance) | `175b99c` | 13/13 | Anticipatory operators — multi-period |
| **NC39** | Anticipatory cardinality constraint handler | `c204d18` | 13/13 | Anticipatory operators — constraints |
| **NC35** | Accumulated future Δ_S over H periods | `342529b` | 12/12 + 28/28 NC30 regression | Multi-period AMFC |

**Cumulative new tests this push**: 92 (16+13+12+10+15+13+13). All passing. NC30
regression suite 28/28 unchanged (NC35 backward-compat verified).

---

## Theory class breakdown

### TIP inner workings

| NC | Mechanism |
|---|---|
| NC13b (prior push, ON opt-in) | Smooth tanh clamp preserves tail signal |
| NC31 (prior push, ON opt-in) | Defn 6.1 conditional mode (current FIXED) |
| **NC36 (this push)** | **Analytical closed-form TIP** — eliminates MC noise; deterministic; uses scipy bivariate Φ_2 |

The three TIP fixes compose: NC36 implicit-conditional + NC13b smooth-clamp +
deterministic computation = the cleanest TIP path.

### KF inner workings

| NC | Mechanism |
|---|---|
| Probe AC (prior push) | NIS/NEES diagnostics (Bar-Shalom χ² tests) |
| Probe AF (prior push) | Mehra Q-noise estimation from innovation autocorrelation |
| **NC33 (this push)** | **Q-scale adaptation from Dirichlet posterior precision** |

NC33 is the FIRST KF tuning fix that's DATA-DRIVEN from Dirichlet posterior.

### Dirichlet ↔ KF coupling

| NC | Mechanism |
|---|---|
| NC27 (prior push, opt-in) | LogisticNormalPredictor — log-ratio smoothing |
| NC27-deep (prior push, opt-in) | DirichletPosteriorPredictor — TRUE Bayesian posterior |
| **NC32 (this push)** | **LogisticNormalKF — stateful KF on log-ratio coordinates** |
| **NC33 (this push)** | **Sequential coupling: Dirichlet → KF Q; KF residual → Dir update rate** |

NC32 is the GEOMETRIC coupling (single state space). NC33 is the PARAMETER
coupling (separate filters with cross-talk). Both deliver the "less isolated"
intent.

### ASMS-EMOA inner mechanisms / anticipatory operators

| NC | Mechanism |
|---|---|
| **NC34 (this push)** | **Anticipatory mutation scorer — score offspring by predicted Δ_S or P_ND** |
| **NC38 (this push)** | **Accumulated P(non-dominance) over future periods** |
| **NC39 (this push)** | **Anticipatory cardinality projection — top-K by anticipated growth** |
| **NC35 (this push)** | **Accumulated multi-period Δ_S — extends NC30-v1 AMFC with horizon_accumulated kwarg** |

Together: a complete anticipatory-operator suite — pre-projection scoring
(NC34), constraint enforcement (NC39), and selection criteria (NC38 / NC35).

---

## Operator action items (composing the new NCs)

### Path A: TIP cleanup (lowest risk)
1. Enable NC36 (`W22_NC36_TIP_ANALYTICAL=1`)
2. Optionally enable NC13b + NC31 alongside
3. FTSE smoke n=5; compare λ^H trace stability vs MC

### Path B: Dirichlet-KF coupling
1. Wire NC33's `coupled_predict_update_cycle` into `predict_anticipative_solution`
2. Use NC27-deep `DirichletPosteriorPredictor` as the predictor instance
3. FTSE smoke n=5; measure trace(Q_eff) trajectory; verify it follows Dirichlet posterior precision

### Path C: Anticipatory operators
1. Replace SMSEMOA's standard mutation with NC34 scoring
2. Replace cardinality enforcement with NC39 anticipatory projection
3. Use AR(1) per-asset predictor (Probe Q-v1) as predicted_growth source
4. FTSE smoke n=5; measure asset-turnover rate, final wealth

### Path D: Multi-period anticipation
1. Wait for NC35 to finish (extends AMFC with `horizon_accumulated` parameter)
2. Combine with NC38's P_ND accumulated score as alternative selection rule
3. Add `horizon_accumulated=3` to `Hv-DM-AMFC` dm_config
4. FTSE smoke; compare H=1 vs H=3

### Path E: LNKF alternative weight tracker
1. Use NC32 `LogisticNormalKF` instead of `DirichletPredictor`
2. Per W22-NC32-LNKF.md, LNKF wins on logistic-normal source data (NC32's
   native model class), loses on Dirichlet source data — choice is regime-
   dependent
3. Smoke comparison FTSE: LNKF vs DirichletPosteriorPredictor

---

## Theoretical contributions (operator may want to publish)

1. **Closed-form bivariate Gaussian TIP** (NC36): eliminates MC variance; trivial to implement; should replace MC TIP wherever computational cost matters.

2. **Sequential Dirichlet-KF coupling** (NC33): generalizes hierarchical Bayesian
   filtering with two cross-talk directions (posterior → Q, residual → update
   rate). Modular: any Dirichlet posterior + any KF can be coupled this way.

3. **Multi-period P(non-dominance) accumulation** (NC38): extends TIP's
   single-period non-dominance probability into a discounted multi-horizon
   sum. Natural alternative selection rule for anticipative MOEAs.

4. **Anticipatory cardinality projection** (NC39): novel — most MOEA
   constraint handlers are oblivious to anticipation. NC39 ties top-K
   selection to a per-asset forecast, opening a research line on
   "anticipation-aware projection operators."

5. **Anticipatory mutation scoring** (NC34): companion to constraint
   anticipation — score offspring by predicted Δ_S (or rank in future
   front) before evaluating. Operator-suggested; first formalization.

---

## Honest scars

- **NC32 LNKF L2 error 2.93× worse than NC27-deep** on Dirichlet source data
  (commit `1d127b5` agent report). On logistic-normal source data, LNKF wins
  by 2×. Regime-dependent; not a universal upgrade.
- **NC33 heuristic Q-scaling formula** (1 + var/baseline); the right
  functional form may differ. Empirical tuning required.
- **NC38 independence assumption** across solution pairs (joint = product).
  Conservative; real solutions share market noise.
- **NC39 predicted_growth caller-supplied**; no built-in predictor.
  Composes with Probe Q-v1 but requires explicit wiring.
- **All anticipatory operators are STANDALONE modules**. Integration into
  SMSEMOA's evolutionary loop is operator's decision (not autonomous scope).
- **NC-AD-fix realigned the deterministic rectangle**; pre-existing 21
  sms_emoa tests still pass (formulae verified against formula, not pinned
  to old rectangle's numerical outputs).

---

## Branch state summary

- 30+ commits ahead of base
- Active wave: this push covers 8 NCs spanning TIP / KF / Dirichlet-KF /
  anticipatory operators
- All NC tests green: 124 across all NC3X suites (+ 80 from this push)
- Zero shared-code regressions beyond the intentional NC-AD-fix
- All opt-in by default; production behavior unchanged

The operator can now ratify or sequence any combination of Paths A-E
empirically.

# W22 Inspections 1–6 — Synthesis

**Status:** Synthesis of 6 algebraic/structural inspections completed 2026-05-18..19
**Trigger:** Operator directive 2026-05-18 — analytical + algebraic territory, not
just statistical. Focus on TIP / Eq 6.41 / Dirichlet / correspondence / multi-horizon
/ AMFC / alternative predictive signals.

---

## TL;DR

Six inspections of the ASMOO/ASMS predictor stack found:

| # | Inspection | Verdict | NC candidate |
|---|---|---|---|
| 1 | TIP joint vs conditional sampling | Empirically equivalent; **TIP = 0.5 is analytical, not a bug** | — (no fix needed) |
| 2 | Eq 6.41 truncation | **75% under-estimate of E[Δ_S]** for correlated portfolios | **NC26**: per-solution MC sample buffer |
| 3 | "Dirichlet" predictor | **Not a Dirichlet** — exponential smoothing; TRUE Dirichlet 2.8× more accurate | **NC27**: TRUE Dirichlet or Logistic-Normal KF |
| 4 | Correspondence mapping | **Dead infrastructure** (stored, never queried); cosine sim brittle on sparse | **NC28**: composite metric + wire into KF state continuity |
| 5 | Multi-horizon discount | Not really a discount; flat weights; safety normalization kills z_t | **NC29**: γ^h prefactor + w_0 floor; reconcile two formulas |
| 6 | AMFC implementation | **Wrong objective**: argmax current Δ_S, not E[future HV given choice]; 82% disagreement in synthetic | **NC30**: per-candidate forward-forecast loop |

Plus an alternative-signal probe design (docs/W22-ALT-SIGNAL-PROBES.md) for raw
returns (Q), Granger causality (R), and asset network centrality (S).

The **highest-leverage candidates** are NC26 (Eq 6.41 truncation), NC27 (Dirichlet
replacement), and NC30 (AMFC reformulation) — these are structural mathematical
gaps where current behavior demonstrably diverges from the paper's intent.

---

## Cross-cutting findings

### Naming-vs-implementation drift
Three of six inspections found cases where the algorithm's name promises
something the implementation doesn't deliver:

- **"Dirichlet" predictor** → exponential smoothing (Inspection 3)
- **"Multi-horizon discount"** → flat weights, no h-decay (Inspection 5)
- **"AMFC = Anticipative Maximal Flexible Choice"** → argmax current Δ_S, not
  forward-HV-given-choice (Inspection 6)

This is a recurring pattern: a mathematically rich concept gets implemented
as a cheaper proxy, the cheaper proxy gets the rich-concept name, and reviewers/
operators infer richer behavior than the code provides. The cumulative effect
is that the predictor stack does less work than its surface area suggests.

### Dead / unused infrastructure
- **Correspondence mapping** (Inspection 4): stored at every period, never queried.
- **`predicted_anticipative_decision`** in CorrespondenceMapping: set but never
  consumed anywhere in production code.
- **historical_populations** in AnticipatoryLearning (separate from
  CorrespondenceMapping's): appended to but never read.

Dead infrastructure carries cost: it slows down the substrate, confuses readers
about what the algorithm depends on, and quietly degrades when refactored
because no one tests what isn't used.

### z_ref ambiguity is endemic
Three different default z_ref values in the codebase (Inspection 6):
- `sms_emoa.py`: (0.0, 0.2)
- `main.py`: (-1.0, 10.0)
- `thesis_parameters.py`: separate REFERENCE_POINT and HYPERVOLUME_REFERENCE_POINT

z_ref affects Δ_S magnitude AND argmax — Inspection 6 showed argmax flips between
idx 0 and idx 6 across z_ref choices on the same Pareto front.

### "Anticipation rate" is poorly differentiated per period
- **λ^K** is solution-invariant per period (constant across portfolios)
- **λ^H** is TIP-modulated but TIP often saturates at [0.05, 0.95] (NC13a)
- The (1/(H-1)) prefactor in Eq 6.6 is constant per horizon

Net: in saturated regimes, every solution gets approximately the SAME anticipation
rate, and Δ_S argmax reduces to current-state argmax with a uniform anticipation
offset. The "Flexible" in AMFC degenerates.

---

## Priority-ordered NC candidates

### Tier 1 — high-leverage structural fixes
1. **NC26** (Eq 6.41 per-solution MC buffer) — 75% under-estimate of E[Δ_S]
   for correlated portfolios; restores the cross-pair Cov terms from the
   C++ legacy implementation. Estimated cost: ~600 LOC + per-solution memory
   for MC samples (~100 floats × |P| × 2 objectives = ~16 KB for |P|=20).
2. **NC30 core** (AMFC per-candidate forward-forecast loop) — operator's
   correct definition. Per-period cost O(|P_t| * H * n_mc) ≈ 12k forecasts for
   reasonable parameters; tractable. Aligns the algorithm with what the AMFC
   name promises and unlocks downstream signal that's currently 82% wasted
   on synthetic data.
3. **NC27** (TRUE Dirichlet posterior OR Logistic-Normal compositional KF) —
   replaces exponential smoothing in decision space. 2.8× accuracy gain on
   synthetic Dirichlet-generated data. Logistic-Normal variant integrates
   with existing KF infrastructure (no new filter math); Dirichlet variant
   provides closed-form posterior variance for use in λ^D analogous to λ^K.

### Tier 2 — refinements that compose with Tier 1
4. **NC29** (multi-horizon discount: γ^h + w_0 floor) — gives the discount
   actual h-decay; prevents runaway anticipation. Most useful AFTER NC26 +
   NC30 are in place because then λ_h discriminates more meaningfully across
   horizons.
5. **NC30 b/c/d** (data-derived z_ref; restore stability_factor or NIS gate;
   tie-break by forecast variance) — refinements of the forward-HV forecast
   that NC30 core enables.
6. **NC28** (composite correspondence metric + KF state continuity wiring) —
   resurrects dead infrastructure if and only if there's a use case (per-
   solution KF carry-forward, perhaps for NC30's forward-forecast caching).

### Tier 3 — out-of-scope but worth keeping in the backlog
7. **Probes Q/R/S** (alternative predictive signals — see W22-ALT-SIGNAL-PROBES.md):
   raw AR(1)/GARCH; Granger causality; asset network centrality. These are
   orthogonal to the predictor-stack fixes above and test whether richer
   raw-signal input adds value over (ROI, risk) summarization.

### Items NOT worth fixing
- **Inspection 1 (TIP joint vs conditional sampling)**: empirically equivalent;
  TIP = 0.5 at equal means is analytical, not a bug. Documentation update
  only — no code change.

---

## How these compose

A natural sequence:

1. **NC26 first** (Eq 6.41 per-solution MC buffer) — provides the per-
   solution MC sample machinery that NC30 will also need.
2. **NC27 next** (Dirichlet replacement) — produces uncertainty estimates
   the forward-forecast in NC30 can use to discount per-candidate predictions.
3. **NC30 third** (AMFC per-candidate forward-forecast loop) — consumes
   both the MC samples from NC26 and the posterior variance from NC27.
4. **NC29 fourth** (discount refinement) — once λ_h drives meaningful
   per-horizon discrimination via NC30, geometric decay starts mattering.
5. **NC28 last** (correspondence rewiring) — only if NC30's forward-forecast
   caching benefits from per-solution KF state continuity.

The probes (Q/R/S) can be done IN PARALLEL with the NC stack because they
test orthogonal signals.

---

## Each inspection's key result (one-line each)

| Inspection | One-line result |
|---|---|
| 1 (TIP) | Joint vs conditional MC differ by <1.5% across 5 scenarios; TIP=0.5 analytical at equal means |
| 2 (Eq 6.41) | Truncated code under-estimates E[Δ_S] by **75%** for correlated portfolios (independent scenarios: ~2% error) |
| 3 (Dirichlet) | TRUE Dirichlet posterior **2.8× more accurate** than exponential smoothing on 100-obs Dirichlet data |
| 4 (Correspondence) | 0 production callers for retrieval API; cosine threshold 0.95 finds 15% of correspondences at swap_frac=0.4 |
| 5 (Multi-horizon) | Flat (1/(H-1)) weights per horizon; safety normalization sets w_0=0 (runaway anticipation) when Σλ>1; H≥4 always normalizes |
| 6 (AMFC) | Current code != operator definition; **82% argmax disagreement** vs synthetic TRUE-AMFC on 200 runs |

---

## Files committed

| Commit | File | Purpose |
|---|---|---|
| 38f4f15 | docs/W22-RESEARCH-PROGRAM.md | 9 areas × sub-areas × counterfactuals research agenda |
| 7689ab9 | experiments/inspect_tip_chain.py | TIP joint vs conditional MC + analytical reference |
| b67601d | experiments/inspect_efhv_eq641.py | Full Eq 6.41 vs truncated code (MC ground truth) |
| 72dfea0 | experiments/inspect_dirichlet.py | Current vs TRUE Dirichlet vs Logistic-Normal |
| 33c533d | experiments/inspect_correspondence.py | Cosine pathology + dead-infra production audit |
| 14f06fe | experiments/inspect_multi_horizon.py | Discount-formula analysis + scheme comparison |
| 8b0f983 + bccfa40 | experiments/inspect_amfc.py | AMFC current-Δ_S vs operator-correct forward-HV (TEST 6 added in correction) |
| ca911b6 | docs/W22-ALT-SIGNAL-PROBES.md | Q/R/S probe designs (raw / causal / network) |
| (this) | docs/W22-INSPECTIONS-SYNTHESIS.md | This doc |

---

## What this enables

These inspections are NOT themselves NCs — they are diagnostic
investigations that identify and quantify gaps. The next step is to pick a
Tier-1 candidate and design it as a proper NC with:

- Algebraic spec (paper equations the implementation must match)
- Falsifiable success criterion (e.g. NC26 must reduce |truncated - full Eq 6.41|
  to < 10% on correlated synthetic scenarios)
- Regression test (per Codex W52 discipline: substrate green AND CI green)
- Production benchmark (FTSE n≥10 seeds, Wilcoxon p<0.05 for headline gain)
- Honest scar log if a Tier-1 NC reveals additional gaps

The operator's directive was to expand W22 into analytical territory. These
six inspections are the analytical foundation. The NCs they recommend are
the bridge from "we know what's wrong" to "we have a tested fix."

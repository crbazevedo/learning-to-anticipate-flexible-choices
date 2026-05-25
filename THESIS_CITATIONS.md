# W22 THESIS_CITATIONS — every NC tied to thesis/paper

**Purpose**: maintain an audit-trail linking every W22 NC to its source in:
- **Thesis**: Azevedo (2015) PhD Dissertation, FEEC/Unicamp — uses 6.x equation numbering
- **Paper**: Azevedo & Von Zuben (2015) IEEE TCYB — uses (11)..(18) equation numbering

Per W1-5 (existing plan): paper-equation citations are CANON; thesis citations
are aliased for cross-reference.

**Operator directive 2026-05-20**: "Always reference the thesis."

---

## Index — per-NC thesis anchor

| NC | Paper Eq | Thesis Eq/§ | Mechanism | Status |
|---|---|---|---|---|
| **NC8c-v2** | Eq (11) | §6.1.1, Eq 6.1 | Cross-period KF position carry-forward | IN PRODUCTION (pre-W22) |
| **NC8d** | Eq (14) | §6.1.5 / 7.1, Eq 6.10 | Predict-before-update sequence | IN PRODUCTION (pre-W22) |
| **NC-AD-fix** | Eq (15) | §6.1.7, Eq 6.41 | Rectangle alignment in deterministic Δ_S | IN PRODUCTION (W22) |
| **NC13b** | Eq (12) | §6.1.5, Eq 6.5 | Smooth tanh TIP clamp (replaces hard 0.05/0.95 clip) | OPT-IN env var; -0.41% neutral |
| **NC15** | Eq (12) | §6.1.5 | Per-portfolio λ shrinkage by KF position uncertainty | OPT-IN env var; UNTESTED |
| **NC27** | Eq (16) | §6.2.1, Eq 6.7 | Logistic-Normal weight predictor (alt to Dirichlet exp-smoothing) | OPT-IN env var; UNTESTED |
| **NC27-deep** | Eq (16)/(17) | §6.2.1, Eq 6.7 | TRUE Dirichlet posterior wrapper (Bayesian α update) | OPT-IN env var; +1.37% paired n=10 |
| **NC29** | Eq (14) | §6.1.5, Eq 6.10 | w_0 floor in multi-horizon blend (prevents runaway anticipation) | IN PRODUCTION (W22) |
| **NC29a** | Eq (13) | §6.1.5, Eq 6.6 | γ^h geometric discount in λ^H (replaces flat 1/(H-1)) | IN PRODUCTION (W22) |
| **NC30 family** | Eq (15) + AMFC | §6.1.7, Eq 6.41 + AMFC concept | AMFC: argmax E[future HV-contrib given choice]; CRN + analytical + tied-mean averaging + data-derived z_ref | IN PRODUCTION (W22), Hv-DM-AMFC dm_type |
| **NC30 c** | — | extension of §6.1.7 | Continuous variance penalty in AMFC scoring | KWARG OPT-IN |
| **NC30 d** | — | extension | Tie-break by lowest forecast variance | KWARG OPT-IN |
| **NC31** | Eq (12) | §6.1.5, Defn 6.1 | TIP conditional sampling (current FIXED, predicted sampled) — Defn-6.1 correct | OPT-IN env var; -4.44% |
| **NC32 (LNKF)** | Eq (16)+(11) | combination of §6.1.1 KF + §6.2.1 Dirichlet | Logistic-Normal KF — KF in Aitchison log-ratio coords | STANDALONE (NOT WIRED) |
| **NC33** | Eqs (11)+(16) | combination | Sequential Dirichlet ↔ KF param coupling (Q-scale, residual→increment) | STANDALONE |
| **NC34** | — | extension of §6.1.7 | Anticipatory mutation scorer — score offspring by predicted Δ_S | STANDALONE |
| **NC35** | Eq (15) | extension of §6.1.7 | Accumulated future Δ_S over H periods (multi-period AMFC) | KWARG OPT-IN |
| **NC36** | Eq (12) | §6.1.5 closed-form | Analytical bivariate Gaussian TIP (no MC) | OPT-IN env var; -6.13% |
| **NC38** | Eq (12) extension | extension of §6.1.5 | Accumulated multi-period P(non-dominance) selection rule | STANDALONE |
| **NC39** | (no direct paper eq) | thesis §7.2.3 cardinality constraint | Anticipatory top-K cardinality projection (vs naive by weight) | STANDALONE |
| **NC26-deep** | Eq (15) | §6.1.7, Eq 6.41 with cross-Cov terms | Eq 6.41 75% under-estimate fix via market-Σ + per-solution MC samples | DEFERRED (needs market-Σ plumbing) |

---

## Reverse index — thesis/paper section → NCs

### Paper §IV-A (KF infrastructure) — Eq (11) state evolution

- NC8c-v2 — cross-period KF position carry-forward
- NC32 — alternative KF on log-ratio coords
- NC33 — Dirichlet-posterior-conditioned Q scaling

### Paper §IV-A.2 (TIP definition) — Eq (12)

- NC13b — smooth clamp
- NC15 — λ shrinkage
- NC31 — Defn 6.1 conditional sampling
- NC36 — analytical closed form
- NC38 — multi-period P(ND) extension

### Paper §IV-A.3 (λ^H computation) — Eq (13)

- NC29a — γ^h geometric discount

### Paper §IV-A.4 / §6.1.5 (multi-horizon blend) — Eq (14)

- NC8d — predict-before-update
- NC29 — w_0 floor

### Paper §IV-B (Hv-DM / EFHV / AMFC) — Eq (15) / §6.1.7 / Eq 6.41

- NC-AD-fix — rectangle alignment
- NC30 family — operator-correct AMFC (per-candidate forward forecast)
- NC34 — anticipatory mutation
- NC35 — accumulated future Δ_S
- NC26-deep (deferred) — Eq 6.41 75% under-estimate fix

### Paper §IV-C (Dirichlet weight prediction) — Eq (16)/(17) / §6.2.1

- NC27 — Logistic-Normal alt
- NC27-deep — TRUE Dirichlet posterior

### Thesis §7.2.3 (cardinality / portfolio constraints)

- NC39 — anticipatory top-K projection

---

## Provenance: ad-hoc inspections + probes

### Inspections (algebraic audits) → NCs

| Inspection | Finding | Resulting NCs |
|---|---|---|
| Inspection 1 (TIP joint vs conditional) | <1.5% empirical delta; code violates Defn 6.1 | NC31 (formal fix), NC36 (closed-form variant) |
| Inspection 2 (Eq 6.41 truncation) | 75% under-estimate for correlated portfolios | NC-AD-fix (rectangle align); NC26-deep (deferred) |
| Inspection 3 (Dirichlet = exponential smoothing) | 2.8× accuracy gap on synthetic Dirichlet data | NC27 (LN), NC27-deep (TRUE posterior) |
| Inspection 4 (Correspondence mapping dead) | Stored never queried | (no NC; documented as dead infra) |
| Inspection 5 (Multi-horizon not really a discount) | Flat (1/(H-1)); runaway when Σλ>1 | NC29 (w_0 floor), NC29a (γ^h discount) |
| Inspection 6 (AMFC wrong objective) | argmax current Δ_S, not E[future HV given choice] | NC30 family (CRN+analytical+tied-mean+derive_zref+penalty+tie-break) |

### Probes (diagnostic tools, not paper-grounded NCs)

| Probe | Topic | Paper anchor (if any) |
|---|---|---|
| Probe AC | KF NIS/NEES diagnostics | Eq (11) (KF consistency) |
| Probe AF | Mehra Q-noise estimation | Eq (11) (KF tuning theory) |
| Probe Q | AR(1) per-asset prediction | extension (paper has no raw-asset prediction) |
| Probe S | Asset network centrality | extension (paper has no network-graph reasoning) |
| Probe AD | Stochastic vs deterministic Δ_S | Eq (15) / Eq 6.41 |
| Probe AI / AI-2 | Pareto front size boundary bias | extension of §6.1.7 |
| Probe T | NC29a γ sensitivity | Eq (13) |
| Probe U | NC30 c α sensitivity | extension of §6.1.7 |
| Probe Y | NC13b smooth clamp impact | Eq (12) |
| Probe Z | Stability factor (Reading-F) | §IV-B Δ_S stability multiplier |
| Probe AA | AMFC telemetry harness | meta tool |
| Probe AB | Per-portfolio λ^K differentiation | Eq (13) extension |
| Probe V | Combination ablation framework | meta tool |
| Probe DRIFT | Synthetic-vs-empirical drift registry | meta tool |

---

## Update discipline

- When a new NC is created: add row to "Index" table + appropriate
  reverse-index sections
- When an NC's status changes: update Status column
- When an NC's thesis anchor is re-validated (e.g., found a more precise
  equation reference): update both columns
- Cross-reference from CHANGES_LOG.md entries via `thesis_anchor` field

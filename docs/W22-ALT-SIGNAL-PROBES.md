# W22 Alternative Predictive Signal Probes

**Status:** Design doc for exploration probes Q, R, S
**Date:** 2026-05-19
**Linked:** W22-RESEARCH-PROGRAM.md Area IX (alternative predictive signals)
**Trigger:** operator directive 2026-05-18 — "how prediction is incorporated. We are
assuming separate filters in decision and object spaces for tracking. But what if
we consider other predictive signals such as: prediction over raw signal (i.e.
series or returns and/or delayed changes in correlations between assets, etc.)?
Or what if we do causal inference or identify node centrality and other
graph-theoretic measures over a complex network composed of assets return and
volatility temporal relationships?"

---

## Why this matters

The current ASMOO predictor stack:

| Layer | Object | Filter |
|---|---|---|
| Objective space | (ROI, risk) per solution | Kalman filter (Eq 11) |
| Decision space | portfolio weight vector | "Dirichlet" predictor (exponential smoothing — see Inspection 3) |

Both filters operate on **derived, low-dimensional summaries** of the asset
universe. The raw signal — per-asset returns, return correlations, volatility
clustering — is COLLAPSED into (ROI, risk) before any prediction happens.

The hypothesis these probes test: **a richer predictive signal extracted from
the raw return / volatility series outperforms (ROI, risk) state tracking.**
If true, we have a structural path to higher anticipation gain that's
orthogonal to the NC8/NC13/NC23 KF-tuning work.

---

## Probe Q: Per-Asset Return / Volatility Prediction

### Hypothesis
**Q-H1:** Per-asset AR(1) or GARCH(1,1) predictions of next-period return
beat the implicit "no change" baseline that ROI-level KF makes.

**Q-H2:** Knowing per-asset NEXT-period predicted return changes which assets
the portfolio optimizer selects (downstream HV gain).

### Mechanism
For each of the K assets (K ∈ {30, 31, 81, 87}):
1. Fit AR(1) on past 60 daily returns
2. Predict r_{k,t+1} = ρ_k * r_{k,t} + (1-ρ_k) * μ_k
3. Use predicted returns as INPUT to portfolio mean-variance optimizer
   instead of historical mean returns
4. Run ASMOO with predicted-return inputs

### Measurements
- **Q-M1:** RMSE of per-asset AR(1) predictions vs realized returns; compare
  to a no-change baseline (predict r_{t+1} = r_t).
- **Q-M2:** Q-H1 win rate (fraction of assets where AR(1) RMSE < no-change RMSE).
- **Q-M3:** Final-wealth (or HV) gain of ASMOO with AR(1)-predicted inputs
  vs ASMOO with historical-mean inputs, n ≥ 10 seeds per regime.
- **Q-M4:** Wilcoxon signed-rank p-value (gain > 0 over seeds).

### Success criterion
**Q-PASS:** Q-M3 mean ≥ +5%, Q-M4 p < 0.05 on at least one regime (FTSE 2015
or NASDAQ 2010).

### Falsifier
**Q-FAIL:** Q-M2 ≤ 50% AND Q-M3 mean ≤ 0% on all regimes. Means AR(1)
prediction is no better than persistence at the per-asset level, AND the
downstream optimizer doesn't extract a HV signal even if individual
predictions are slightly better.

### Variants
- **Q-v1**: AR(1) baseline (as above)
- **Q-v2**: GARCH(1,1) on volatility, ARMA(1,1) on mean — full conditional
  variance prediction; feeds risk model with PREDICTED conditional covariance
- **Q-v3**: VAR(1) on a SUBSET of assets (top-K by past trading volume) — captures
  cross-asset lead-lag effects

### Estimated cost
- Implementation: 200–300 LOC for AR(1) variant; ~600 LOC for GARCH variant
- Compute: 50 seeds × 2 regimes × 3 variants = 300 runs × ~5 min = 25 hrs

---

## Probe R: Causal Inference Between Asset Returns

### Hypothesis
**R-H1:** Some asset pairs exhibit Granger-causality (or transfer entropy)
at lag 1–5 days, with non-zero coefficient significant after Bonferroni
correction.

**R-H2:** Conditioning portfolio selection on identified causal parents
(e.g., "if XOM ↑ today, predict CVX ↑ tomorrow") changes the realized
portfolio return distribution.

**R-H3:** A causal-graph-informed weight perturbation rule outperforms the
current Dirichlet exponential smoothing in decision space.

### Mechanism
**Causal discovery:**
1. For each asset pair (i, j), test H0: returns of i don't Granger-cause j
2. Use Toda-Yamamoto or block-bootstrap to handle non-stationarity
3. Build directed graph G with edges (i → j) for significant Granger pairs
4. Compute transfer entropy (Schreiber 2000) as continuous alternative

**Use in ASMOO:**
1. At each rebalance, propagate today's per-asset returns through G to get
   PREDICTED next-period contributions
2. Adjust the predicted mean return vector by the propagated signal
3. Pass to portfolio optimizer as in Probe Q

### Measurements
- **R-M1:** Fraction of asset pairs with significant Granger-causality (FDR < 0.05)
- **R-M2:** Mean transfer entropy across edges in graph
- **R-M3:** Final-wealth gain of causal-graph-augmented ASMOO vs vanilla ASMOO
- **R-M4:** Stability of identified causal edges across train/test splits
  (Jaccard similarity of edge sets ≥ 0.5)

### Success criterion
**R-PASS:** R-M1 ≥ 5% (meaningful causal structure exists); R-M3 mean ≥ +5%
with Wilcoxon p < 0.05; R-M4 ≥ 0.5 (stable edges, not data dredging).

### Falsifier
**R-FAIL:** R-M1 < 1% (no real causal structure beyond chance) OR R-M4 < 0.3
(edges are sample-specific noise). In either case, causal inference is not
extracting reusable structure.

### Risk
Granger-causality in financial returns is notoriously fragile (Hong &
Sun, 2009 — fewer than 5% of pairs are robustly causal; most "significance"
is data mining). Probe R needs aggressive multiple-testing correction or
will produce false positives. Recommend running on hold-out periods only.

### Estimated cost
- Implementation: 400–500 LOC (statsmodels Granger + custom transfer entropy)
- Compute: 60 seeds × 2 regimes × ~10 min causal-discovery = 20 hrs

---

## Probe S: Asset Network Centrality

### Hypothesis
**S-H1:** Asset-return correlation networks exhibit non-trivial centrality
distribution (Gini > 0.5 for eigenvector centrality), indicating that some
assets are HUBS that lead/lag others.

**S-H2:** Tilting portfolio weights toward HIGH-CENTRALITY assets captures
their hub effect.

**S-H3:** Tilting AWAY from high-centrality assets reduces systemic risk
(less exposure to common factor shocks).

(Note: S-H2 and S-H3 are OPPOSING predictions — both worth testing because
the optimal direction is regime-dependent.)

### Mechanism
At each rebalance, construct the K×K rolling-window correlation matrix:
1. **Pearson** for linear co-movement
2. **DCCA** (detrended cross-correlation, Peng et al 1994) for non-stationary
3. **Mutual information** (Kraskov k-NN) for non-linear dependencies

Build a network:
- Nodes = assets
- Edges = correlation > threshold (or top-K by absolute correlation)
- Edge weights = correlation magnitude

Compute centralities:
- **Eigenvector centrality** (PageRank-style): high if connected to many
  other central assets
- **Betweenness centrality**: high if lies on many shortest paths
- **Closeness centrality**: high if reachable from all others
- **Clustering coefficient**: high if neighborhood is densely connected

Use in ASMOO:
- **S-v1 (hub-tilt):** Multiply each asset's predicted return by (1 + γ·cent_k)
  where γ ∈ {-0.5, 0, +0.5} (negative = anti-hub, positive = hub-following)
- **S-v2 (centrality-conditioned KF):** Augment KF state with predicted
  network centrality of the asset; let the KF velocity component learn from
  hub-effect lead-lag

### Measurements
- **S-M1:** Centrality distribution Gini coefficient (test S-H1)
- **S-M2:** Centrality persistence across periods (correlation between
  centrality_t and centrality_{t+1})
- **S-M3:** Final-wealth gain of S-v1 (best γ) vs vanilla ASMOO
- **S-M4:** Sharpe ratio change of S-v2 vs vanilla ASMOO
- **S-M5:** Tail risk (CVaR_5%) change for both variants

### Success criterion
**S-PASS:** S-M1 ≥ 0.5; S-M2 ≥ 0.3 (centrality is not random); S-M3 mean
≥ +5% Wilcoxon p < 0.05.

### Falsifier
**S-FAIL:** S-M1 < 0.3 OR S-M2 < 0.1. Means correlation networks don't have
stable hub structure that ASMOO can exploit.

### Estimated cost
- Implementation: 500–700 LOC (networkx + DCCA + Kraskov MI from probe-P toolkit)
- Compute: 60 seeds × 2 regimes × 3 centrality variants × ~7 min = 21 hrs

---

## Common methodology (probes Q, R, S)

### Regimes
1. **FTSE 2015** (n=30 assets, 252 trading days) — primary benchmark, where
   ASMS was validated +7.50% n=10 p=0.003
2. **NASDAQ 2010** (n=81 assets) — high-correlation tech-heavy regime
3. **HangSeng 2015** (n=31 assets) — Asia regime sanity check (optional)

### Seeds
n=10 minimum for hypothesis testing; n=30 for confirmatory probes that PASS
initial gate.

### Reporting template
```
Probe X (variant Y), regime Z, n=N seeds
  Hypothesis: <one-liner>
  Mean Δ-wealth: +X.XX%
  Median Δ-wealth: +X.XX%
  Wilcoxon signed-rank p-value: 0.XXX
  Win rate (seeds where Δ > 0): XX/N (XX%)
  Honest scars: <gotchas, caveats, regime-specific weirdness>
```

### Negative-result discipline
A probe FAIL is as valuable as a probe PASS. Failed probes get committed with
the same rigor (full log, summary doc, citation in INSPECTION-BACKLOG.md).
Per Codex W52 discipline: never paper over a fail to make a probe look like a
pass. The honest-scar list is load-bearing for the next probe design.

---

## Priority ordering and decision rationale

| Order | Probe | Why |
|---|---|---|
| 1 | **Q-v1 (AR(1))** | Cheapest to implement; clean signal-vs-noise question; if fails, kills the whole alt-signal direction at low cost |
| 2 | **S-v1 (eigenvector centrality, γ=+0.5)** | Centrality is a well-studied financial signal; literature shows ~5–10% Sharpe gain possible; medium implementation cost |
| 3 | **Q-v2 (GARCH)** | If Q-v1 passes, GARCH adds conditional-variance prediction with ~3× LOC. If Q-v1 fails, skip |
| 4 | **R (Granger)** | High variance, high risk; defer until Q and S are done so we know whether ANY raw-signal predictor works |

### What we are NOT doing
- **Deep learning predictors** (LSTM, Transformer) — out of scope; we are
  testing whether classical predictive signals add value, not whether ML
  hyper-parameter tuning produces SOTA
- **Cross-asset trade execution modeling** — orthogonal to W22's anticipation focus
- **Order book / microstructure signals** — daily data only

### Decision branches
- **All 3 PASS:** strong case for raw-signal layer; design NC31+ to integrate
  as parallel predictor stack (raw-signal → joint with (ROI, risk) KF)
- **Q PASS, others FAIL:** AR(1) is the only useful raw signal; implement
  as cheapest possible production NC
- **Q FAIL:** raw per-asset prediction doesn't help; move to alternative
  decision-space filters (logistic-normal compositional KF per Inspection 3 NC27)

---

## Linkage to other W22 work

- **Inspection 3 (Dirichlet):** if true Dirichlet or logistic-normal beats
  exponential smoothing in decision space, that's COMPLEMENTARY to raw-signal
  alternatives — different layer of the predictor stack
- **Inspection 5 (multi-horizon discount):** raw-signal prediction is naturally
  per-period; multi-horizon discount applies to whichever signal is chosen
- **NC23 (Q-noise estimation, Mehra):** raw-signal probes will INFORM how
  much process noise the (ROI, risk) KF should expect; if raw signals
  capture the structure, KF noise can be tightened
- **Probe P (mutual information):** Kraskov k-NN MI from Probe P is REUSABLE
  for Probe S (non-linear correlation network) — share the toolkit

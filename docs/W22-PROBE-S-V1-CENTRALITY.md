# W22 Probe S-v1 — Asset Network Eigenvector Centrality

**Status:** v1 module landed (`python_refactor/src/probes/probe_s_centrality.py`).
Not yet wired into ASMOO / `thesis_aligned_experiment.py`.
**Date:** 2026-05-20
**Linked:** `docs/W22-ALT-SIGNAL-PROBES.md` § Probe S, `docs/W22-MASTER-BACKLOG.md`
**Branch:** `feat/w22-inspection-backlog`

---

## Hypothesis

Restating from `W22-ALT-SIGNAL-PROBES.md` for self-containment:

- **S-H1** — Asset-return correlation networks exhibit a non-trivial
  centrality distribution: Gini coefficient of eigenvector centrality
  ≥ 0.5 across realistic asset universes (K ∈ {30, 31, 81, 87}). I.e.
  some assets are HUBS that lead/lag others; the network is not flat.

- **S-H2** — Tilting portfolio weights TOWARD high-centrality assets
  (γ > 0 in the multiplicative tilt rule below) captures their hub
  effect and improves downstream final wealth / HV.

- **S-H3** — Tilting AWAY from high-centrality assets (γ < 0)
  reduces systemic-risk exposure (lower CVaR_5%) at the cost of
  expected return. S-H2 and S-H3 are intentionally OPPOSING and both
  worth testing; which direction wins is regime-dependent.

The v1 probe surfaces the *mechanism* — building the network, computing
centrality, and the hub-tilt operator — without yet wiring it into an
ASMOO experiment.

---

## Mechanism

### 1. Correlation network
At each rebalance:
1. Take the last `W` daily returns (default `W = 60`, matching Probe Q).
2. Compute the K×K Pearson correlation matrix `C` over the window.
3. Optionally sparsify to the top-K strongest absolute-correlation edges
   per node (symmetrized via logical OR — an edge survives if it is in
   the top-K of EITHER endpoint).

The probe API:
```python
adj = build_correlation_network(returns_matrix, top_k_edges_per_node=None)
```
where `returns_matrix` is a `T × K` ndarray.

### 2. Eigenvector centrality
Pure NumPy power iteration on `|adj|` (we take absolute value so that
anti-correlated hubs also register; the dominant eigenvector of a non-
negative irreducible matrix is non-negative by Perron-Frobenius).
L1-normalized output sums to 1, giving each entry a direct "fraction
of hub-mass" reading that flows straight into the Gini computation.

```python
cent = eigenvector_centrality(adj, max_iter=100, tol=1e-6)
```

### 3. Betweenness centrality
Auxiliary measure for S-M2 (centrality persistence) and as a fallback
ordering check. Uses `scipy.sparse.csgraph.shortest_path` for
performance; edge "distances" are `1 - |corr_ij|` (strong correlations
become short paths). Pure-NumPy Brandes BFS is the fallback when SciPy
is absent.

```python
bw = betweenness_centrality(adj)
```

### 4. Gini coefficient of centrality
Standard Sen / Brown formula on the L1-normalized centrality vector.
Returns 0 for uniform, → 1 as mass concentrates on one node:

```python
gini = centrality_gini(cent)
```

### 5. Hub-tilt rule (S-v1 weight perturbation)
```
w_k <- max(0, w_k * (1 + γ · cent_k))
w   <- w / Σ_k w_k                    # renormalize to simplex
```
γ ∈ {-0.5, 0, +0.5} per the spec; the API accepts any real γ and clips
to the non-negative orthant before renormalization:

```python
w_tilt = hub_tilt_weights(base_weights, centrality, gamma)
```

---

## Success criteria

Per `W22-ALT-SIGNAL-PROBES.md`:

| Measure | Threshold |
|---|---|
| **S-M1** Gini of eigenvector centrality | ≥ 0.5 (a priori for S-H1) |
| **S-M2** Persistence corr(cent_t, cent_{t+1}) | ≥ 0.3 |
| **S-M3** Final-wealth gain S-v1 (best γ) vs vanilla ASMOO | ≥ +5% mean, Wilcoxon p < 0.05 |
| **S-M4** Sharpe ratio change | meaningful only if S-M3 passes |
| **S-M5** Tail risk (CVaR_5%) change | sign-of-effect for both γ > 0 and γ < 0 |

**S-PASS** requires S-M1, S-M2, S-M3 all green. The v1 module makes
S-M1 directly measurable today; S-M2 needs a rolling-window driver
script; S-M3 needs the wiring step.

**S-FAIL** if S-M1 < 0.3 OR S-M2 < 0.1 — would mean correlation
networks lack stable hub structure that ASMOO can exploit, and the
probe is properly falsified.

---

## Expected scars

Documented in advance so they don't surface as surprises during S-M3:

1. **Hub effect is regime-dependent.** Crisis periods (2008, 2020-03)
   compress correlations toward 1; the entire network becomes one
   giant cluster and Gini collapses. Hub-tilt loses signal exactly
   when one most wants it.

2. **Correlation ≠ causation.** Probe S identifies co-movement hubs,
   not lead-lag hubs. A high-eigenvector-centrality asset is one
   correlated with many other central assets — it might be a passive
   recipient of a common factor (e.g., SPY-tracking stock) rather than
   an active leader. Probe R (Granger causality) is the partner that
   addresses lead-lag.

3. **Pearson is fragile under heavy tails.** Daily returns are not
   Gaussian; |Pearson| can over-state co-movement during quiet periods
   and miss tail co-movement. A v2 may want Spearman or DCCA (already
   noted in the parent doc as Probe S-v2).

4. **Hub-tilt collides with the simplex constraint.** Strong positive
   γ pushes the hub weight toward 1 and squeezes the rest toward 0;
   the optimizer's other objectives (diversification, risk) react
   non-linearly. We expect a hump-shaped γ → final-wealth curve, not
   a monotone one. The CONTAINED v1 reports the raw tilt; finding the
   optimum γ per regime is a downstream sweep.

5. **The hub may be cash / a quasi-cash asset.** If the asset universe
   includes a low-vol asset, it can score high on centrality because
   its returns correlate weakly with everything (and thus rank high
   on certain centrality definitions). For S-v1 we use |corr|, which
   mitigates this — but it's worth checking which asset wins the hub
   ranking before celebrating S-M1.

6. **Power iteration can stall on near-degenerate spectra.** When the
   two largest |eigenvalues| are very close (e.g., a disconnected-
   components network), power iteration converges slowly or oscillates
   between subspaces. The `tol=1e-6` default catches typical cases;
   a future scar would be a regime where it doesn't.

---

## Integration sketch (NOT wired in v1)

The probe is intentionally *not* imported anywhere in
`thesis_aligned_experiment.py`, `sms_emoa.py`, or any optimizer code
path. To wire it later, a new experiment script (separate file) would:

```python
# pseudocode for a future thesis_aligned_experiment_probe_s.py
from src.probes.probe_s_centrality import (
    build_correlation_network,
    eigenvector_centrality,
    centrality_gini,
    hub_tilt_weights,
)

# After each rebalance, BEFORE the optimizer commits the weights:
window = returns_matrix[t - W : t]                # T x K
adj = build_correlation_network(window, top_k_edges_per_node=10)
cent = eigenvector_centrality(adj)
log["gini"].append(centrality_gini(cent))         # S-M1

base_w = optimizer.solve(...)                     # vanilla ASMOO output
tilted_w = hub_tilt_weights(base_w, cent, gamma=GAMMA_SWEEP[regime])
portfolio.execute(tilted_w)                       # use tilted instead
```

Two integration paths are possible and SHOULD be A/B tested:

- **Path A (post-optimizer tilt)** — solve the vanilla ASMOO problem,
  then apply the hub-tilt to the resulting weights. Simplest; preserves
  the optimizer's risk model untouched.

- **Path B (pre-optimizer signal)** — multiply the *predicted return
  vector* by `(1 + γ · cent)` and pass that into the optimizer. Lets
  the optimizer respect the diversification constraint while still
  seeing the hub-tilt as a return signal. Likely better-behaved on the
  S-M5 (CVaR) measure because the optimizer can still rebalance the
  risk model.

**Estimated cost of full S-v1 run** (per parent doc): 60 seeds × 2
regimes × 3 γ values × ~7 min ≈ 21 hrs of compute. The v1 module
shipped here is the prerequisite that lets that run be scheduled
without additional implementation work.

---

## Out of scope for v1

- **S-v2 (centrality-conditioned KF).** Augmenting the KF state with
  centrality is a structural change to the predictor stack — touches
  shared code; deferred to its own contract.
- **Mutual-information network (Kraskov k-NN).** Reuses Probe P toolkit;
  not implemented here.
- **DCCA (detrended cross-correlation, Peng et al 1994).** Same — Probe
  P territory.
- **Networkx integration.** Explicitly avoided to keep the probe
  dependency-free (NumPy + SciPy.csgraph only, both already in the
  project requirements).

---

## File manifest

| Path | Purpose |
|---|---|
| `python_refactor/src/probes/probe_s_centrality.py` | Pure functions: network build, eig + betweenness centrality, Gini, hub-tilt |
| `python_refactor/tests/test_probe_s_centrality.py` | 14 unit tests — diagonal, symmetry, star-graph centrality, Gini limits, hub-tilt monotonicity, simplex preservation |
| `docs/W22-PROBE-S-V1-CENTRALITY.md` | this document |

No modifications to shared code paths (`sms_emoa.py`,
`anticipatory_learning.py`, `thesis_aligned_experiment.py`, etc.) per
W22-MASTER-BACKLOG contained-probe discipline.

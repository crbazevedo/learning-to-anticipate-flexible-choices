# Validation results

🚧 **TEMPLATE — populated by W8.** Every numeric cell below is a
placeholder. Real numbers replace 🚧 markers when W8 executes the
[analytics plan](ANALYTICS-PLAN.md) against the output of the
[validation matrix](EXPERIMENT-VALIDATION-PLAN.md).

The structure of this file is settled (it mirrors
`ANALYTICS-PLAN.md`); only the cells need filling. That separation
makes W8 a mechanical execution against a settled design.

---

## 0. Reproducibility receipt

```
git_sha:    28bc9062
uv_lock:    sha256:aad2ff82cb4f0901
generated:  2026-05-17T03:44:38Z
n_runs:     2220
seeds:      [1, ..., 30] (30 seeds)
command:    python -m experiments.report --summary docs/VALIDATION-SUMMARY.csv --runs results/ --template docs/VALIDATION-RESULTS.md --output docs/VALIDATION-RESULTS.populated.md
```

---

## 1. Executive summary

**The W13-3 single-shot 80/20 evaluation reports S2 < S0 (−17.59%,
p < 0.0001), inconsistent with the paper headline. BUT the evaluation
methodology itself diverges from the thesis and may explain the result.**

Receipt (30 seeds × 1000 MC × single 80/20 split of the paper window):

| Scenario | Mean OOS EFHV | Std (n=30) | Direction vs S0 |
|---|---|---|---|
| **S0** (myopic baseline ≈ SMS/RDM) | **5.12e-05** | 4.06e-06 | — |
| **S2** (anticipatory + max-Hypv DM ≈ ASMS/mHDM) | **4.22e-05** | 3.68e-06 | **−17.59%** |

Welch t ≈ 9, df ≈ 60. Direction reverses the paper's Table 7.2 claim
that ASMS/mHDM beats SMS/RDM in 70% of scenario × dataset combinations.

### Methodology caveat (W13-CARRY-2 — the load-bearing one)

The thesis (§7.2.2) uses **online walk-forward evaluation**: at each
rolling investment period t, the algorithm trains on data up to t,
extracts the Estimated Maximal Flexible Choice (EMFC) `m̂_{u*_t}`,
implements that portfolio, and observes the sample-average future
hypervolume `Ŝ_{t+1}` at the NEXT period. Aggregation is over
(T − 1) × 30 seeds = ~720 observations across the rolling sweep.

**W13-3 instead used a single 80/20 train/test split** — 1 period
× 30 seeds = 30 observations. This:

- Defeats the anticipation property: the algorithm sees the whole
  training window at once rather than stepping forward through it
  with K∈{1,2,3} historical lookback. The paper's anticipation
  effect is driven by THE WALK FORWARD ITSELF + the K-historical
  window self-adjusting `λ_{t+h}` per Eq 7.16. Batch training
  collapses this temporal structure.
- Cuts statistical power by ~24×.
- Doesn't match the C++ rolling-step structure (which IS in the
  legacy code per the W13 sub-agent audit, even though that code
  computes only in-sample HV).

The S2 < S0 finding therefore says **the current pipeline + 80/20
single-shot methodology does not reproduce the paper claim**. It
does NOT say "the algorithm is broken." Disambiguating requires
the walk-forward implementation in W14.

### Candidate root causes (W13-CARRY-1 — secondary)

If walk-forward STILL shows S2 ≤ S0, ranked by plausibility:

(a) **K window-size mismatch** (highest). SCENARIOS maps
   `max_horizon` (paper Eq 14 H) but NOT K ∈ {0,1,2,3} from
   thesis §7.2.3 OAL historical-window. S2 may be running K=0
   effectively — equivalent to SMS/RDM baseline.
(b) Cardinality constraints (c_l=5, c_u=15) unenforced.
(c) OAL rate `λ_{t+h} = ½(λ^H + λ^K)` only half-firing.
(d) Data-window divergence (98 assets vs thesis 87; partial range).

**Headline number:** −17.59% per current methodology;
**agreement: methodologically incommensurable** until W14 lands
walk-forward. See [`OOS-EFHV-REPORT.md`](OOS-EFHV-REPORT.md).

---

## 2. Descriptive statistics (Table A)

Per (scenario × window × metric): mean, std, median, IQR, n_seeds.

| scenario | window | metric | mean | std | median | min | max | n |
|---|---|---|---|---|---|---|---|---|
| S0 | paper | algorithm.convergence_metric | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.diversity_metric | 0.2122 | 0.1029 | 0.2112 | 0.04716 | 0.4335 | 30 |
| S0 | paper | algorithm.future_robustness | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.generation | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.hypervolume | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.pareto_efficiency | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.pareto_front_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S0 | paper | algorithm.population_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S0 | paper | algorithm.solution_quality | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.spread_metric | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | algorithm.stochastic_quality | 0.4970 | 0.001124 | 0.4969 | 0.4950 | 0.4995 | 30 |
| S0 | paper | computational.evaluations_per_second | 53.70 | 7.148 | 56.43 | 35.90 | 57.85 | 30 |
| S0 | paper | computational.execution_time | 0.3815 | 0.07016 | 0.3544 | 0.3457 | 0.5570 | 30 |
| S0 | paper | computational.function_evaluations | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S0 | paper | computational.memory_usage_mb | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | computational.total_time | 0.6086 | 0.08545 | 0.5779 | 0.5561 | 0.8217 | 30 |
| S0 | paper | portfolio.calmar_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.concentration | 0.4110 | 0.05560 | 0.3948 | 0.3479 | 0.5518 | 30 |
| S0 | paper | portfolio.cumulative_return | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.cvar_95 | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.diversification | 0.5890 | 0.05560 | 0.6052 | 0.4482 | 0.6521 | 30 |
| S0 | paper | portfolio.information_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.max_drawdown | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.num_assets | 3.000 | 0 | 3.000 | 3.000 | 3.000 | 30 |
| S0 | paper | portfolio.portfolio_return | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.portfolio_value | 1.133 | 0 | 1.133 | 1.133 | 1.133 | 30 |
| S0 | paper | portfolio.sharpe_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.sortino_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.var_95 | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | portfolio.volatility | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | summary.final_hypervolume | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | summary.final_portfolio_value | 1.133 | 0 | 1.133 | 1.133 | 1.133 | 30 |
| S0 | paper | summary.max_drawdown | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | summary.pareto_front_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S0 | paper | summary.sharpe_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S0 | paper | summary.total_execution_time | 0.3815 | 0.07016 | 0.3544 | 0.3457 | 0.5570 | 30 |
| S0 | paper | summary.total_return | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.convergence_metric | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.diversity_metric | 0.2693 | 0.07857 | 0.2710 | 0.1340 | 0.3953 | 30 |
| S2 | paper | algorithm.future_robustness | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.generation | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.hypervolume | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.pareto_efficiency | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.pareto_front_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S2 | paper | algorithm.population_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S2 | paper | algorithm.solution_quality | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.spread_metric | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | algorithm.stochastic_quality | 0.4974 | 0.0009588 | 0.4974 | 0.4957 | 0.4994 | 30 |
| S2 | paper | computational.evaluations_per_second | 0.5377 | 0.03536 | 0.5253 | 0.5060 | 0.6347 | 30 |
| S2 | paper | computational.execution_time | 37.34 | 2.258 | 38.08 | 31.51 | 39.53 | 30 |
| S2 | paper | computational.function_evaluations | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S2 | paper | computational.memory_usage_mb | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | computational.total_time | 37.63 | 2.270 | 38.38 | 31.80 | 39.81 | 30 |
| S2 | paper | portfolio.calmar_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.concentration | 0.4446 | 0.1101 | 0.3988 | 0.3336 | 0.6604 | 30 |
| S2 | paper | portfolio.cumulative_return | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.cvar_95 | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.diversification | 0.5554 | 0.1101 | 0.6012 | 0.3396 | 0.6664 | 30 |
| S2 | paper | portfolio.information_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.max_drawdown | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.num_assets | 3.000 | 0 | 3.000 | 3.000 | 3.000 | 30 |
| S2 | paper | portfolio.portfolio_return | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.portfolio_value | 1.133 | 0 | 1.133 | 1.133 | 1.133 | 30 |
| S2 | paper | portfolio.sharpe_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.sortino_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.var_95 | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | portfolio.volatility | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | summary.final_hypervolume | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | summary.final_portfolio_value | 1.133 | 0 | 1.133 | 1.133 | 1.133 | 30 |
| S2 | paper | summary.max_drawdown | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | summary.pareto_front_size | 20.00 | 0 | 20.00 | 20.00 | 20.00 | 30 |
| S2 | paper | summary.sharpe_ratio | 0 | 0 | 0 | 0 | 0 | 30 |
| S2 | paper | summary.total_execution_time | 37.34 | 2.258 | 38.08 | 31.51 | 39.53 | 30 |
| S2 | paper | summary.total_return | 0 | 0 | 0 | 0 | 0 | 30 |

**Normality diagnostic** (per metric, per scenario): skewness 🚧;
kurtosis 🚧. Informs Welch-t vs Mann-Whitney choice in §3.

---

## 3. Inferential tests

### 3.1 Pairwise scenario comparison (Table B)

Mann-Whitney U one-sided + Wilcoxon signed-rank (paired-by-seed) +
Welch t (sanity).

| pair | window | metric | U | p (raw) | p (Holm) | Cohen's d | Cliff's δ | CLES |
|---|---|---|---|---|---|---|---|---|
| S0 vs S2 | paper | algorithm.convergence_metric | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.diversity_metric | 284.0 | 0.0144 | 0.476 | -0.623 | -0.369 | 0.316 |
| S0 vs S2 | paper | algorithm.future_robustness | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.generation | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.hypervolume | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.pareto_efficiency | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.pareto_front_size | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.population_size | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.solution_quality | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.spread_metric | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | algorithm.stochastic_quality | 380.0 | 0.304 | 1.00 | -0.325 | -0.156 | 0.422 |
| S0 vs S2 | paper | computational.evaluations_per_second | 900.0 | <0.001 | <0.001 | 10.518 | 1.000 | 1.000 |
| S0 vs S2 | paper | computational.execution_time | 0 | <0.001 | <0.001 | -23.138 | -1.000 | 0.000 |
| S0 vs S2 | paper | computational.function_evaluations | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | computational.memory_usage_mb | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | computational.total_time | 0 | <0.001 | <0.001 | -23.049 | -1.000 | 0.000 |
| S0 vs S2 | paper | portfolio.calmar_ratio | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.concentration | 434.0 | 0.819 | 1.00 | -0.385 | -0.036 | 0.482 |
| S0 vs S2 | paper | portfolio.cumulative_return | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.cvar_95 | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.diversification | 466.0 | 0.819 | 1.00 | 0.385 | 0.036 | 0.518 |
| S0 vs S2 | paper | portfolio.information_ratio | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.max_drawdown | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.num_assets | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.portfolio_return | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.portfolio_value | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.sharpe_ratio | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.sortino_ratio | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.var_95 | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | portfolio.volatility | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.final_hypervolume | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.final_portfolio_value | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.max_drawdown | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.pareto_front_size | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.sharpe_ratio | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |
| S0 vs S2 | paper | summary.total_execution_time | 0 | <0.001 | <0.001 | -23.138 | -1.000 | 0.000 |
| S0 vs S2 | paper | summary.total_return | 450.0 | 1.00 | 1.00 | 0.000 | 0.000 | 0.500 |

**Forest plot of effect sizes:** 🚧 → `docs/figures/F8_forest_effect_sizes.png`

---

## 4. Multi-factor analysis

### 4.1 One-way ANOVA over Multi-Horizon levels (Table C)

Subset: S1, S2, S3. Factor: Multi-Horizon ∈ {OFF, H=2, H=3}.

| window | metric | source | SS | df | MS | F | p |
|---|---|---|---|---|---|---|---|
| paper | final_wealth | Multi-Horizon | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| paper | final_wealth | Residual | 🚧 | 🚧 | 🚧 | — | — |
| paper | hypv | Multi-Horizon | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| … | … | … | … | … | … | … | … |

### 4.2 Synergy contrast (Table D)

Δ_interaction = (S2 − S0) − (S1 − S0). Bootstrap 95% CI.

| window | metric | Δ_interaction | 95% CI | interpretation |
|---|---|---|---|---|
| paper | final_wealth | 🚧 | [🚧, 🚧] | 🚧 (synergy / additive / sub-additive) |
| paper | hypv | 🚧 | [🚧, 🚧] | 🚧 |
| extended | final_wealth | 🚧 | [🚧, 🚧] | 🚧 |
| extended | hypv | 🚧 | [🚧, 🚧] | 🚧 |

### 4.3 Per-horizon ablation (Table E)

| comparison | window | metric | U | p | Cohen's d | interpretation |
|---|---|---|---|---|---|---|
| S2 (H=3) vs S3 (H=2) | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 |
| S4 (cov) vs S2 (mean-only) | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 |
| S2 vs S3 | paper | hypv | 🚧 | 🚧 | 🚧 | 🚧 |
| S4 vs S2 | paper | hypv | 🚧 | 🚧 | 🚧 | 🚧 |

---

## 5. Data segmentation

### 5.1 By calendar year (Heatmap C)

🚧 → `docs/figures/F9_year_heatmap.png`

> *Reader-facing summary: in which years does S2 − S0 cluster? Are
> there years where S0 outperforms? What do those years correspond to
> in market terms?*

### 5.2 By volatility regime (Table F + Figure F10)

| regime | scenario | final_wealth (mean ± std) | hypv (mean ± std) |
|---|---|---|---|
| low-vol | S0 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| low-vol | S1 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| low-vol | S2 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| medium-vol | S0 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| medium-vol | S1 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| medium-vol | S2 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| high-vol | S0 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| high-vol | S1 | 🚧 ± 🚧 | 🚧 ± 🚧 |
| high-vol | S2 | 🚧 ± 🚧 | 🚧 ± 🚧 |

🚧 → `docs/figures/F10_regime_bar_chart.png`

### 5.3 By sub-window (Table G)

| sub-window | scenario | final_wealth (mean ± std) |
|---|---|---|
| early | S0 | 🚧 ± 🚧 |
| early | S2 | 🚧 ± 🚧 |
| mid | S0 | 🚧 ± 🚧 |
| mid | S2 | 🚧 ± 🚧 |
| late | S0 | 🚧 ± 🚧 |
| late | S2 | 🚧 ± 🚧 |

🚧 → `docs/figures/G_sub_window_stability.png`

---

## 6. Visualization gallery

| Figure | Path | Status |
|---|---|---|
| F1 wealth_trajectory_per_scenario | `docs/figures/F1_wealth_trajectory.png` | 🚧 |
| F2 hypv_trajectory_per_scenario | `docs/figures/F2_hypv_trajectory.png` | 🚧 |
| F3 pareto_frontier_snapshot | `docs/figures/F3_pareto_snapshot.png` | 🚧 |
| F4 lambda_trajectory | `docs/figures/F4_lambda_trajectory.png` | 🚧 |
| F5 belief_trajectory | `docs/figures/F5_belief_trajectory.png` | 🚧 |
| F6 scenario_bar_chart | `docs/figures/F6_scenario_bar.png` | 🚧 |
| F7 pairwise_pvalue_heatmap | `docs/figures/F7_pvalue_heatmap.png` | 🚧 |
| F8 forest_plot_effect_sizes | `docs/figures/F8_forest_effect_sizes.png` | 🚧 |
| F9 year_heatmap | `docs/figures/F9_year_heatmap.png` | 🚧 |
| F10 regime_bar_chart | `docs/figures/F10_regime_bar_chart.png` | 🚧 |
| F11 qq_plot_metric | `docs/figures/F11_qq_plot.png` | 🚧 |
| F12 bootstrap_ci_ribbon | `docs/figures/F12_bootstrap_ribbon.png` | 🚧 |

---

## 7. Paper-comparison receipt (Table H)

Direct numerical comparison of S2 (paper-window, 30 seeds) vs paper's
reported FTSE-100 numbers (Table I + §VI text).

| metric | paper value | this implementation (S2, paper-window) | abs diff | rel diff | agree? |
|---|---|---|---|---|---|
| Final accumulated wealth (HDM, FTSE) | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Final accumulated wealth (MDM, FTSE) | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| POCID (HDM, FTSE) | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| Hypv at T (HDM, FTSE) | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

Agreement criterion (per ANALYTICS-PLAN.md §9): rel diff < 10% →
"agrees with paper"; 10-25% → "qualitative agreement"; > 25% →
"divergent — investigate".

---

## 8. Interpretation playbook walk-through

For each row of [ANALYTICS-PLAN.md §9](ANALYTICS-PLAN.md#9-honest-scars-interpretation-playbook):

- 🚧 *Did the predicted failure mode occur? What does the data say?*

---

## 9. Honest scars (encountered in execution)

🚧 *Populated by W8 — runtime issues, MC variance surprises,
unexpected metric distributions, etc.*

---

## 10. Conclusion

🚧 *One paragraph. The headline answer + one or two key qualifiers.
Lead with the empirical claim; back it with the evidence; flag the
honest scars.*

---

*Authored as W7-3. Numbers come in W8.*

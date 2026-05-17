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
git_sha:    🚧
uv_lock:    🚧
generated:  🚧
n_runs:     🚧 (target: 5 scenarios × 2 windows × 30 seeds = 300)
seeds:      🚧
command:    python -m experiments.validation_matrix [...]
            python -m experiments.aggregate_validation --input results/ --output VALIDATION-SUMMARY.csv
            python -m experiments.report
```

---

## 1. Executive summary

> 🚧 *One paragraph: did the wired pipeline reproduce the paper? Did
> multi-horizon add value? Where in the data does the effect live?
> What's the headline finding for a reader who reads only this
> paragraph?*

**Headline number:** S2 paper-window final wealth uplift vs S0 = **🚧%**
(paper Table I reports ~12%). Agreement: **🚧** (close / divergent / inconclusive).

---

## 2. Descriptive statistics (Table A)

Per (scenario × window × metric): mean, std, median, IQR, n_seeds.

| scenario | window | metric | mean | std | median | IQR | n |
|---|---|---|---|---|---|---|---|
| S0 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S1 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S2 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S3 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S4 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| … | … | hypv | 🚧 | … | … | … | … |
| … | … | pocid | 🚧 | … | … | … | … |
| … | … | sharpe | 🚧 | … | … | … | … |

**Normality diagnostic** (per metric, per scenario): skewness 🚧;
kurtosis 🚧. Informs Welch-t vs Mann-Whitney choice in §3.

---

## 3. Inferential tests

### 3.1 Pairwise scenario comparison (Table B)

Mann-Whitney U one-sided + Wilcoxon signed-rank (paired-by-seed) +
Welch t (sanity).

| pair | window | metric | U | p (raw) | p (Holm) | Cohen's d | Cliff's δ | CLES |
|---|---|---|---|---|---|---|---|---|
| S0 vs S1 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S0 vs S2 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S1 vs S2 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S1 vs S3 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S2 vs S3 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S2 vs S4 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| … | … | hypv | … | … | … | … | … | … |

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

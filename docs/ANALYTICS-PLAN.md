# Analytics plan

*Authored 2026-05-17 as W7-2. This is the **analysis framework** the
W8 execution wave will apply to the results produced by the W7-1
validation matrix. It complements
[`EXPERIMENT-VALIDATION-PLAN.md`](EXPERIMENT-VALIDATION-PLAN.md)
(the runs) with the analytics (the conclusions).*

---

## 1. Purpose + scope

Translate raw per-run metrics (the output of
`python -m experiments.validation_matrix`) into defensible answers to
five questions:

1. **Did the wiring deliver?** S1 (TIP-on) vs S0 (Markowitz baseline) on each metric — does anticipatory learning add value over a non-anticipatory baseline?
2. **Did multi-horizon deliver?** S2 (multi-horizon) vs S1 (single-horizon) — does multi-horizon convex combination (paper Eqs 14 + 15) add value over single-horizon TIP?
3. **Do the factors interact?** 2×2 factorial (TIP × Multi-Horizon) — is the combined effect additive, super-additive (synergy), or sub-additive?
4. **Where in the data does it work?** Segmented by year, market regime, sub-window — is the effect uniform or concentrated in specific conditions?
5. **Does it reproduce the paper?** S2 paper-window vs paper Table I numerical results — what's the absolute and relative agreement?

Each question maps to a specific section of this plan + a specific
deliverable in the eventual VALIDATION-RESULTS.md.

Status framing: this is a **plan**. Results land in W8. The plan
exists so W8 implementation is mechanical execution against a settled
analytical design rather than design-as-you-go.

---

## 2. Descriptive statistics

For each (scenario × window × metric) cell, the aggregator
([`aggregate_validation.py`](../python_refactor/experiments/aggregate_validation.py))
already computes:

| Statistic | Use |
|---|---|
| mean | Central tendency |
| std | Spread (Gaussian assumption) |
| median | Robust central tendency |
| min / max | Range |
| n_seeds | Sample size (≥ 30 per W6-3 §3) |
| seeds_used | Audit trail |

**Additional aggregations to compute in W8:**

| Statistic | Why |
|---|---|
| 95% bootstrap CI (10,000 resamples) | Non-parametric uncertainty interval — robust to non-Gaussian metric distributions (wealth is heavy-tailed) |
| Interquartile range (IQR) | Robust spread |
| Skewness + kurtosis | Diagnostic for normality assumption (informs choice of parametric vs non-parametric test in §3) |

All descriptive stats land in **Table A** of VALIDATION-RESULTS.md
(see §8 for the spec).

---

## 3. Inferential tests

### 3.1 Pairwise scenario comparison (primary question 1 + 2)

For each pair of scenarios (S0–S1, S0–S2, S1–S2, S1–S3, S2–S3, S2–S4)
× each window × each metric:

| Test | When | Why this test |
|---|---|---|
| **Mann-Whitney U (one-sided)** | Default. Compare medians without assuming normality. | Matches paper §V-D methodology. Robust to wealth-distribution heavy tails. |
| **Wilcoxon signed-rank** | Paired-by-seed comparison (same seed in both scenarios). | Removes seed variance from the comparison; tighter than unpaired Mann-Whitney when seeds are matched. |
| **t-test (Welch)** | Sanity check only — reported alongside U when normality is plausible (skewness < 1, kurtosis < 5). | Cross-validates U. Disagreement is itself a finding (non-normality). |

Significance threshold: **α = 0.05** with Bonferroni correction
across all pairwise tests (k = 6 pairs × 2 windows × 5 metrics → divide
α by 60 for the family-wise rate; per-test α' = 8.3 × 10⁻⁴).

Reported as **Table B** (pairwise p-values, corrected + uncorrected).

### 3.2 Effect size (orthogonal to p-values)

A p-value tells you whether the effect is detectable; an effect size
tells you whether it matters.

| Measure | Use |
|---|---|
| **Cohen's d** | Standardized mean difference. d > 0.8 = large, 0.5 = medium, 0.2 = small. Parametric. |
| **Cliff's δ** | Non-parametric effect size = P(X₁ > X₂) − P(X₁ < X₂). Robust. |
| **Common-language effect size (CLES)** | "If you pick one S2 run and one S1 run at random, P(S2 > S1) = …". Reader-friendly. |

Computed per scenario pair × metric. Reported alongside Table B.

### 3.3 Multi-comparison correction

Default: **Holm-Bonferroni** (uniformly more powerful than naive
Bonferroni; controls family-wise error rate). Reported both ways
so readers can see the difference correction makes.

---

## 4. Multi-factor analysis (primary question 3)

The 5 scenarios encode a 2×2 + baseline design:

|  | TIP off | TIP on (H=2) | TIP + Multi-Horizon |
|---|---|---|---|
| Markowitz | **S0** | — | — |
| Single-horizon | — | **S1** | — |
| Multi-horizon (H=2) | — | — | **S3** |
| Multi-horizon (H=3) | — | — | **S2** |
| Multi-horizon (H=3, explicit cov) | — | — | **S4** |

### 4.1 Two-way ANOVA on the {S1, S2, S3} subset

The factor design within the anticipatory-learning scenarios is:

- **Factor A**: Anticipation (TIP) — fixed at ON across S1/S2/S3
- **Factor B**: Multi-Horizon — OFF (S1) vs H=2 (S3) vs H=3 (S2)

Because TIP is constant within this subset, the ANOVA reduces to a
one-way over Multi-Horizon level. Reported as **Table C**.

### 4.2 Synergy detection — S0 vs S1 vs (S1+MH = S2)

Hypothesised: (S2 - S0) > (S1 - S0) + (Multi-Horizon-alone effect).
If true → synergy. If equal → additive. If less → sub-additive.

Quantified per metric as the **interaction contrast**:
`Δ_interaction = (S2 - S0) - (S1 - S0)`

Reported in **Table D** with bootstrap 95% CI on Δ_interaction.

### 4.3 Per-horizon ablation — S2 (H=3) vs S3 (H=2) vs S4 (H=3 + explicit cov)

Isolates the contributions of:
- Adding horizon h=2 vs h=1 (S2 vs S3)
- Adding explicit covariance threading on top of mean-only (S4 vs S2)

Reported in **Table E** with Mann-Whitney U + Cohen's d for each.

---

## 5. Data segmentation (primary question 4)

Same metrics, broken down along three orthogonal segmentations:

### 5.1 By calendar year

Aggregate per-period metrics into per-year buckets. Compute scenario
deltas (S2 − S0) per year. **Heatmap C** (years × scenarios) shows
where the effect lives. If the effect is uniform → robust to regime.
If concentrated in specific years → likely market-regime dependent.

### 5.2 By volatility regime

Compute realized volatility per investment period; bin into
{low, medium, high} (terciles). Aggregate metrics per (scenario,
regime) pair. **Table F** + **figure F**.

Hypothesis: anticipatory learning's value-add is larger in high-vol
regimes because that's where the predictability dial (TIP) has more
signal to extract.

### 5.3 By sub-window

Split the dataset into 3 sub-windows (early / mid / late thirds) and
re-run the validation matrix per sub-window. Tests temporal stability
of the effect. **Table G** + **figure G**.

---

## 6. Visualization catalog

Each figure has a single load-bearing purpose. No decorative plots.

| Figure | What | Use | Paper comparator |
|---|---|---|---|
| **F1: wealth_trajectory_per_scenario** | Mean wealth ± CI band over time, one line per scenario | Primary visual answer to Q1 + Q2 | Paper Fig 4 (wealth evolution) |
| **F2: hypv_trajectory_per_scenario** | E[Hypv] over time, one line per scenario | Q1 + Q2 in hypv space | Paper Fig 5 |
| **F3: pareto_frontier_snapshot** | Risk vs return, with predicted-frontier overlays, at three time slices | Qualitative shape check vs paper | Paper Fig 4(a/b/c/d) |
| **F4: lambda_trajectory** | λ over time, per scenario; histogram of λ | Diagnostic — proves the dial actually turns | Paper §VI-A "Estimated Confidence" |
| **F5: belief_trajectory** | v over time, per scenario; histogram of v | Diagnostic — should bottom at 0.5 in turbulent periods | (paper Eq 20 trajectory; not shown explicitly in paper) |
| **F6: scenario_bar_chart** | Final wealth (or any metric) by scenario; mean ± std | Comparison summary | Paper Table I (visual form) |
| **F7: pairwise_pvalue_heatmap** | Pairwise scenario × scenario significance matrix | Reader scans for which comparisons matter | — |
| **F8: forest_plot_effect_sizes** | Effect size (Cohen's d) for each scenario pair, with CI | Quick read of "which differences are meaningful" | — |
| **F9: year_heatmap** | Years × scenarios → mean wealth | Where in time the effect lives | — |
| **F10: regime_bar_chart** | Volatility regime × scenario → metric | Where in regime-space the effect lives | — |
| **F11: qq_plot_metric** | Q-Q plot for each metric per scenario | Normality diagnostic (informs test choice) | — |
| **F12: bootstrap_ci_ribbon** | Bootstrap-CI band around each scenario's wealth trajectory | Robust uncertainty visualization | — |

All figures: PNG, 300dpi, headless matplotlib (Agg backend). Each
saved as `docs/figures/F{N}_{name}.png` with a sibling `.md`
caption file for re-use.

---

## 7. Table specifications

| Table | Shape | Content |
|---|---|---|
| **A: descriptive_stats** | scenario × window × metric → (mean, std, median, IQR, n) | All-up summary |
| **B: pairwise_tests** | (scenario_i, scenario_j) × window × metric → (U, p, p_corrected, d, δ) | Pairwise inferential |
| **C: anova** | window × metric × source → (SS, df, MS, F, p) | Multi-Horizon-level ANOVA |
| **D: synergy_contrast** | window × metric → (Δ_interaction, 95% CI, interpretation) | Synergy detection |
| **E: horizon_ablation** | (S2 vs S3, S4 vs S2) × window × metric → (U, p, d) | Per-horizon ablation |
| **F: regime_segmentation** | regime × scenario × metric → mean ± std | Volatility-regime breakdown |
| **G: sub_window_stability** | sub_window × scenario × metric → mean ± std | Temporal stability |
| **H: paper_comparison** | metric → (paper Table I value, S2 paper-window value, abs diff, rel diff, agree?) | Direct paper-comparison receipt |

All tables: CSV (machine-readable) + Markdown rendering (in the
results doc). Numerical precision: 4 significant figures default;
p-values to 3 sig figs; 1 sig fig for very small p (< 0.001 → "<0.001").

---

## 8. Reproducibility receipts

Every table + figure in the results doc carries a receipt block:

```
Receipt:
  git_sha:   <8-char SHA>
  uv_lock:   sha256:<16-char hash>
  generated: 2026-05-17T03:45:00Z
  n_runs:    150 (5 scenarios × 30 seeds × 1 window)
  seeds:     [1, 2, ..., 30]
  command:   python -m experiments.aggregate_validation \
                 --input results/W8-paper/ --output docs/VALIDATION-SUMMARY.csv
```

The receipt is the contract: identical receipt → identical numbers
(modulo floating-point variance which W8 will quantify in a "machine
variance" appendix).

---

## 9. Honest scars (interpretation playbook)

What does each failure mode imply?

| Observed | Likely interpretation | Action |
|---|---|---|
| S1 ≤ S0 on wealth | TIP wiring fires but produces useless λ | Bisect: is λ stuck at 0? Is TIP stuck at 0.5 clamp? Revisit W4-1 clamp + W1-2 threading. |
| S2 < S1 | Multi-horizon override regresses on single-horizon | Investigate `_generate_predicted_solution`: are the predicted states drifting unrealistically? |
| S4 < S2 | Explicit covariance threading hurts | Σ_combined is becoming poorly-conditioned (or shrinking too aggressively) → revisit W5-2's w² weighting |
| Synergy detected (S2 − S0 > (S1 − S0) + MH-alone) | TIP and Multi-Horizon are co-amplifying | **This is the success case** — the wired pipeline is paper-faithful + value-additive |
| Per-year segmentation: effect concentrated in 1-2 years | The 12-year window includes specific regime shifts that drive the effect | Report honestly; cite the years; suggest a regime-stratified analysis as W9 |
| Paper-comparison Table H: large abs diff | Either the implementation diverged in W1-W5 or the paper used un-reported parameters | Investigation unit — read paper §V-D more carefully; check ε / N / G defaults |
| All scenarios overlap heavily (no significant differences) | Either the test is underpowered (need more seeds) or the implementation isn't producing differentiated behaviour | Increase N to 100 seeds before declaring "no effect" |

This playbook converts numbers into interpretable findings.

---

## 10. W8 execution checklist

To execute this plan, W8 needs:

| Surface | Status |
|---|---|
| `validation_matrix.py` per-run driver | ✅ W7-1 |
| `aggregate_validation.py` summary | ✅ W7-1 |
| `figures.py` plotting | ✅ W7-1 (per-run + comparison modes; F1-F6 + F11 covered; F7-F10 + F12 + table generators are W8 extension) |
| `tests.py` stats helpers (Mann-Whitney, Wilcoxon, ANOVA, Cohen's d, bootstrap CI) | 🚧 W8 NEW |
| `tables.py` table generators (A-H above) | 🚧 W8 NEW |
| `report.py` orchestrator that produces VALIDATION-RESULTS.md from CSVs | 🚧 W8 NEW |
| `pyproject.toml` adds `scipy.stats` to runtime deps (already available; verify) | 🚧 W8 verify |
| Run matrix: 5 scenarios × 2 windows × 30 seeds = 300 runs | 🚧 W8 execute |

Estimated W8 scope: 3-5 units.

---

## 11. Anti-goals (out of scope for W7 / W8 baseline)

- Bayesian posterior over scenario effects (deferred — pop-up via PyMC adds dep complexity)
- Time-series structural break detection (regime via volatility tertile is the W7 baseline)
- Causal inference (this is observational evaluation; causal claims would need RCT-style scenario randomisation across data subsets — out of scope)
- Cross-paper-baselines (NSGA-II, MOEA/D — adds N more scenarios, not in W6-3 matrix)

These would be substantial waves of their own. Mentioned here so they
don't quietly creep into W8.

---

*Authored by W7-2. Next action: when ready to execute, propose W8
with units that close §10's checklist. Plan above is the canonical
reference.*

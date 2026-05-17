# Experiment validation plan

*Authored 2026-05-17 as W6-3. This is a **plan**, not a record of
execution. The plan exists so a future wave can run the experiments
against a fully-specified target. Execution is queued as W7+ work.*

---

## 1. Purpose + scope

Validate the wired Python implementation against the empirical claims
of:

> Azevedo & Von Zuben (2015), *Learning to Anticipate Flexible Choices
> in Multiple Criteria Decision-Making Under Uncertainty.* IEEE
> Transactions on Cybernetics.
> DOI: [10.1109/TCYB.2015.2415732](https://doi.org/10.1109/TCYB.2015.2415732)

The 5-wave rehabilitation (W1–W5) closed every audit-flagged gap at
the **module + equation + wiring** level. This plan covers the
**experimental + empirical** validation that's the next discipline
tier: do the wired algorithms reproduce the paper's reported
performance on the paper's data?

In scope:
- FTSE-100 stock dataset (`python_refactor/data/ftse-updated/`)
- The 4 algorithmic variants the paper studies (HDM / MDM / RDM / SMS baseline)
- Hypv (hypervolume) + POCID (Prediction of Change in Direction) metrics
- Comparison against the paper's Section VI numerical results

Out of scope (for the validation plan; a future wave may extend):
- DJIA + HSI datasets (paper §V-A also reports these — extend after FTSE)
- Synthetic benchmark generator (paper §V-C)
- ANOVA + Mann-Whitney significance testing (paper §V-D)

---

## 2. Dataset

| Source | Path | Notes |
|---|---|---|
| FTSE-100 daily closes | `python_refactor/data/ftse-updated/FTSE_100_20121121_20241231.csv` | 12-year window, updated 2024-12-31 |
| FTSE-original (provenance) | `legacy-cpp/executable/data/ftse-original/` (98 CSVs) | The data the 2015 paper used; preserved for cross-check |

The paper used FTSE data from 2006-11-20 to 2012-12-31. The active
dataset extends to 2024-12-31 — validation runs split into two
windows:

- **Paper-window run** (2006-11-20 → 2012-12-31): direct comparison to
  paper's reported numbers.
- **Extended-window run** (2012-11-21 → 2024-12-31): test if the
  algorithm's relative performance holds out-of-sample.

---

## 3. Experiment matrix (2×2 + baseline)

5 scenarios per dataset window. Each runs at least **30 independent
seeds** per the paper's §V-D methodology.

| ID | Anticipatory learning | Multi-horizon | Notes |
|---|---|---|---|
| **S0** | OFF (Markowitz baseline) | — | Pareto frontier from sample μ, σ²; no anticipation |
| **S1** | ON (TIP integrated, H=2) | OFF | `learning.use_tip = true` — closes W1-2 wiring |
| **S2** | ON (TIP integrated, H=2) | ON (H=3) | `learning.use_multi_horizon = true` — closes W1-3 + W5-2 wiring; paper Eqs 14 + 15 drive |
| **S3** | ON (TIP integrated, H=2) | ON (H=2) | Baseline-multi-horizon control |
| **S4** | ON (TIP integrated, H=2) | ON (H=3, covariance threading explicit) | Tests W5-2 covariance contribution separately |

Each scenario runs ASMS-EMOA per the paper's §V-D protocol:
- Population N = 20
- Generations G = 30 per investment period
- Reference point z^ref = (0.2, 0.0)
- Crossover prob 0.2; mutation rate 0.3
- Binary tournament selection; ε = 0.99 feasibility

---

## 4. Expected outputs

For each scenario × dataset window:

| Output | What it measures | Compare to |
|---|---|---|
| **Final accumulated wealth `W_T`** | End-of-window wealth from following the algorithm's portfolio choices | Paper Table I (per algorithm × dataset) |
| **Pareto frontier evolution** | Risk vs return distribution at each rebalancing step | Paper Figures 4 + 7 (qualitative shape) |
| **Hypv trajectory** | E[Hypv] over time | Paper Figures 5 + 8 |
| **POCID** | Proportion of correct prediction of change in direction (Eq 35) | Paper §VI table |
| **Sharpe ratio** | Risk-adjusted return | Paper §VI-A baseline comparison |
| **Anticipation rate trajectory** | Live λ values produced by Eq (13) for S1-S4 | Diagnostic — should fall in [0, 0.5] per W4-1 clamp |
| **Belief coefficient trajectory** | Live v values produced by Eq (20) | Diagnostic — should fall in [0.5, 1.0] |

Outputs land at `python_refactor/results/<scenario>/<seed>/` as a tuple
of (CSV metrics, PNG figures, JSON manifest with seed + config).

---

## 5. Acceptance criteria

### Quantitative

| Metric | Threshold |
|---|---|
| Final wealth, S2 vs S0 (paper-window) | S2 > S0 by ≥ 10% on FTSE-100 (paper reports ~12%) |
| POCID, S2 (paper-window) | ≥ 0.55 (paper reports ~0.58) |
| Hypv at T, S1 vs S0 | S1 > S0 (proves TIP-arm wiring adds value) |
| Hypv at T, S2 vs S1 | S2 ≥ S1 (proves multi-horizon wiring at least doesn't hurt; ideally adds value) |
| Reproducibility | 5 independent runs with the same seed produce identical metrics |
| Test substrate (CI gate) | 162/162 curated tests still PASS post-validation run |

### Qualitative

- Pareto frontier shapes at t=1 and t=T qualitatively match paper Figures 4(a) + 4(d).
- Anticipation rate trajectory: λ stays > 0 in S1-S4 (proves TIP arm fires); periods of high uncertainty (TIP near 0.5) drive λ near 0 (the dial actually turns).
- Belief coefficient trajectory: v close to 1.0 during predictable regimes, drops toward 0.5 during regime change.

### Failure interpretation

- S1 ≤ S0 on hypv → TIP wiring is correct but the TIP-derived λ isn't producing useful anticipation; revisit W4-1 clamp or W1-2 threading.
- S2 < S1 → multi-horizon override (W4-2) is regressing on single-horizon; investigate predicted-state construction in `_generate_predicted_solution`.
- Reproducibility fails across seeds → an un-seeded `np.random` call survived W5-1; bisect.

---

## 6. Reproducibility recipe

```bash
# Environment
cd /path/to/learning-to-anticipate-flexible-choices
uv sync --frozen
cd python_refactor

# Single scenario (S2, paper-window, seed 42)
PYTHONPATH=. uv run python -m experiments.real_data_experiment \
  --scenario S2 \
  --dataset FTSE-100 \
  --window paper \
  --seed 42 \
  --output results/S2_paper_seed42/

# Full matrix (5 scenarios × 2 windows × 30 seeds = 300 runs)
PYTHONPATH=. uv run python -m experiments.validation_matrix \
  --output results/W7-validation/

# Aggregate + figures
PYTHONPATH=. uv run python -m experiments.aggregate_validation \
  --input results/W7-validation/ \
  --output docs/VALIDATION-RESULTS.md
```

The `--scenario`, `--window`, `validation_matrix`, and
`aggregate_validation` surfaces don't exist yet. Adding them is the
scaffold work for the W7 execution unit (see §8).

---

## 7. Comparison-to-paper anchor

| Paper section | What to reproduce | Implementation status |
|---|---|---|
| §IV-A Eq (11) | KF state vector | ✅ W1-1 |
| §IV-A Eq (12) | TIP definition | ✅ W1-4 (equation-level tests) |
| §IV-A.1 Eq (13) | λ from binary entropy | ✅ W1-4 (Eq-13 binary-entropy table tests) |
| §IV-A.2 Eq (14) | OAL multi-horizon convex combo | ✅ W4-2 (live) + W1-3 (eq test) |
| §IV-A.2 Eq (15) | Linear combo of Gaussians (covariance) | ✅ W5-2 (live + PSD test) |
| §V-D Step 2 (population estimation) | `learn_population` driving the run loop | ✅ W4-2 override |
| §VI-A "Estimated Confidence" subsection | Belief-coefficient trajectory comparison | 🚧 (requires execution) |
| Figures 4-8 + Table I | Empirical numerical results | 🚧 (requires execution) |

Implementation parity is COMPLETE. **Empirical parity is the
remaining validation tier.** That's W7's scope.

---

## 8. Execution scaffold (what W7 needs to build)

To run this plan end-to-end, the W7 execution wave needs to add:

| Surface | What it does | Estimated size |
|---|---|---|
| `experiments/validation_matrix.py` | Drive the 5×2×30 run matrix | M |
| `experiments/real_data_experiment.py --scenario/--window/--seed` | Extend existing flags | S |
| `experiments/aggregate_validation.py` | Aggregate per-seed runs into mean ± std metrics | M |
| `docs/VALIDATION-RESULTS.md` template | Final report skeleton | S |
| Pinned numpy random seed plumbing | Confirm seed is the only entropy source | S |
| Figure-generation helpers | PNG output per scenario | M |
| Reference-paper data snapshot | Pin to paper-window subset for direct numerical comparison | S |

Estimated W7 size: 5-7 units across 2-3 waves.

---

## 9. Honest scars

- **No empirical validation has run yet.** The plan above pins what
  validation looks like; until W7 ships, the implementation parity is
  the only verified claim.
- **Paper-window data**: the active dataset starts 2012-11-21, but the
  paper's reported FTSE results use 2006-11-20 → 2012-12-31. Direct
  numerical comparison requires either re-acquiring the older FTSE
  data OR rerunning the paper's algorithm on the newer window. Either
  decision adds methodological asymmetry; the validation plan should
  account for it in the receipts.
- **Monte-Carlo variance**: 30 seeds per the paper's §V-D is a
  conservative replication count. With modern hardware we can afford
  300+ seeds for tighter CI; the plan should explicitly budget how many.
- **Reproducibility of paper's results from current code**: the paper
  reports specific numbers (e.g. ~12% wealth uplift). If the wired
  implementation produces materially different numbers, that's either
  (a) a real algorithmic divergence we missed in W1-W5, or (b) the
  paper's experiments used parameter values not captured in code.
  Either case is interpretable; the plan must record which it turns
  out to be.

---

*Authored by W6-3. Next action: when ready to execute, propose W7 via
`dfg replan propose --action add wave-W7`. The plan above is the
canonical reference for what W7's units must deliver.*

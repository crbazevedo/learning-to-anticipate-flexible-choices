# learning-to-anticipate-flexible-choices

Reference Python implementation of:

> **Azevedo, C. R. B.,** & **Von Zuben, F. J.** (2015).
> *Learning to Anticipate Flexible Choices in Multiple Criteria
> Decision-Making Under Uncertainty.*
> IEEE Transactions on Cybernetics.
> DOI: [10.1109/TCYB.2015.2415732](https://doi.org/10.1109/TCYB.2015.2415732)

The paper proposes the **Anticipatory Stochastic Multi-objective
Optimization (AS-MOO)** model — a methodology for sequential
decision-making under uncertainty where preferences cannot be reliably
elicited up-front, and so *flexibility* (postponable preference
specification) is itself the objective.

This repository contains the canonical Python implementation, faithful
to the paper's notation, equation-level tested, and reproducible from a
pinned environment.

The canonical algorithmic spec is the paper PDF at
[`docs/paper.pdf`](docs/paper.pdf).

---

## Status

Public, actively maintained. v0.1 reflects the post-rehabilitation
state described in [`docs/VISION.md`](docs/VISION.md): every advanced
module (TIP, sliding-window Dirichlet, multi-horizon, belief
coefficient) is wired into the live experiment driver, and the
multi-horizon convex-combination (paper Eqs 14 + 15, mean + covariance)
drives the live run loop.

---

## Quick start

```bash
# clone
git clone https://github.com/crbazevedo/learning-to-anticipate-flexible-choices.git
cd learning-to-anticipate-flexible-choices

# install (uv: https://docs.astral.sh/uv/)
uv sync --frozen

# run the curated test set (162 tests, paper-equation-anchored)
cd python_refactor
PYTHONPATH=. uv run python -m pytest tests/test_kalman_filter.py \
  tests/test_tip_integration.py tests/test_belief_coefficient.py \
  tests/test_eq14_integration.py \
  tests/test_temporal_incomparability_probability.py \
  tests/test_multi_horizon_anticipatory.py \
  tests/test_sliding_window_dirichlet.py \
  tests/test_correspondence_integration.py \
  tests/test_correspondence_mapping.py \
  tests/test_enhanced_n_step_prediction.py -q
```

Expected: **162 passed**.

For full reproducibility of the paper's experiments, see
[`docs/EXPERIMENT-VALIDATION-PLAN.md`](docs/EXPERIMENT-VALIDATION-PLAN.md).

---

## Algorithm overview

AS-MOO models the decision-maker (DM) as choosing among a Pareto set
under stochastic + time-varying objectives. The implementation tracks
trade-off distributions over time via a Kalman Filter (objective space)
and a Sliding-Window Dirichlet model (search space), self-adjusts
anticipation rates from a Time Incomparability Probability (TIP), and
integrates these through ASMS-EMOA.

### Paper equations implemented

| Paper Eq | Concept | Implementation |
|---|---|---|
| **(11)** | KF state vector layout: `z_t^+ = (z_{1,t}...z_{m,t}, ż_{1,t}...ż_{m,t})` | `src/algorithms/kalman_filter.py` |
| **(12)** | Time Incomparability Probability (TIP) | `src/algorithms/temporal_incomparability_probability.py` |
| **(13)** | `λ^(H)_{t+h} = (1/(H-1))[1 - H(p_{t,t+h})]` from binary entropy of TIP | `src/algorithms/temporal_incomparability_probability.py` |
| **(14)** | OAL multi-horizon convex combination of anticipatory means | `src/algorithms/multi_horizon_anticipatory.py` |
| **(15)** | Linear combo of Gaussians stays Gaussian (covariance threading) | `src/algorithms/multi_horizon_anticipatory.py` |
| **(17)–(18)** | Sliding-Window Dirichlet recursive concentration update | `src/algorithms/sliding_window_dirichlet.py` |
| **(20)** | Belief coefficient `v_{t+1} = 1 - (1/2) H(p_{t-1,t})` | `src/algorithms/belief_coefficient.py` |
| **(22)** | MAP correction for predicted Dirichlet means | `src/algorithms/anticipatory_learning.py` |

The reconciliation between the PhD thesis's chapter-6 equation
numbering (6.x / 7.x) and the IEEE paper's (11)–(25) is at
[`thesis_codebase_analysis.md` §0](thesis_codebase_analysis.md).

---

## Project layout

```
learning-to-anticipate-flexible-choices/
├── README.md                            ← this file
├── pyproject.toml                       ← PEP 621 packaging
├── uv.lock                              ← reproducible install (uv)
├── Makefile                             ← `make ci` substrate gate
│
├── docs/
│   ├── paper.pdf                        ← canonical algorithmic spec (IEEE TCYB 2015)
│   ├── VISION.md                        ← vision + trajectory + provenance
│   └── EXPERIMENT-VALIDATION-PLAN.md    ← reproducibility recipe (W6-3)
│
├── python_refactor/                     ← active Python implementation
│   ├── src/
│   │   ├── algorithms/                  ← 16 modules (KF, TIP, Dirichlet, ASMS-EMOA, …)
│   │   ├── portfolio/                   ← portfolio + asset + statistics
│   │   ├── experiments/                 ← ExperimentManager + 3-tier learning configurator
│   │   └── config/                      ← thesis-aligned parameter layer
│   ├── experiments/                     ← experiment driver scripts (FTSE-100 / IBOVESPA / synthetic)
│   ├── tests/                           ← 162-test curated set; equation-anchored
│   └── data/ftse-updated/               ← active FTSE dataset (8.9 MB)
│
├── legacy-cpp/                          ← original C++ implementation (frozen, kept for provenance)
│
├── thesis_codebase_analysis.md          ← thesis ↔ paper equation reconciliation
├── HVDM_OPTIMIZATION_PLAN.md            ← extension proposals (post-paper)
└── 100_percent_adherence_backlog.md     ← PT-language thesis-adherence backlog
```

---

## Reproducibility

| Surface | Reproducible? |
|---|---|
| Python environment | ✅ `uv sync --frozen` from `uv.lock` (35 packages pinned) |
| Curated test set (162 tests) | ✅ deterministic; one seeded RNG in correspondence integration |
| Paper-equation regression | ✅ 5 equation-level test classes (Eq 11, 12, 13, 14, 15) |
| Live experiment outputs (paper figures + tables) | 🚧 plan documented in [`docs/EXPERIMENT-VALIDATION-PLAN.md`](docs/EXPERIMENT-VALIDATION-PLAN.md); execution in a future wave |

See the validation plan for dataset paths, seed conventions, and the
mapping between paper figures and runnable scripts.

---

## Tests

The curated set is 162 tests across 10 files. Every paper equation
implemented in [§Algorithm overview](#algorithm-overview) has at least
one **equation-level** assertion (not bounds-only) in:

- `test_kalman_filter.py::TestPaperEq11Canonical` (4 tests)
- `test_temporal_incomparability_probability.py::TestPaperEq12TIPKnownAnalytical` (3 tests)
- `test_temporal_incomparability_probability.py::TestPaperEq13LambdaBinaryEntropy` (4 tests)
- `test_eq14_integration.py::TestPaperEq14MultiHorizonConvexCombo` (4 tests)
- `test_multi_horizon_anticipatory.py::TestW5_2_CovarianceThreading` (3 tests, paper Eq 15)
- `test_belief_coefficient.py::TestW1_4_BeliefCoefficientKnownGaussians` (2 tests, paper Eq 20)

A future wave will add a GitHub Actions CI workflow to run this set
on every PR.

---

## Governance

This repository is governed by
[`dfg-harness`](https://github.com/korzainc/dfg-harness) under
methodology-as-code discipline: contract-first unit authoring,
paired plan-mutation + replan-accept ceremonies, ADR-004-framed
retrospectives, equation-level test gates. The substrate lives at
[`.dfg/`](.dfg/) for inspection.

This is not load-bearing for users of the algorithm — it's the
provenance trail for how the code reached its current state.

---

## Citation

If you use this implementation, please cite the paper:

```bibtex
@article{azevedo2015learning,
  title   = {Learning to Anticipate Flexible Choices in Multiple Criteria
             Decision-Making Under Uncertainty},
  author  = {Azevedo, Carlos R. B. and Von Zuben, Fernando J.},
  journal = {IEEE Transactions on Cybernetics},
  year    = {2015},
  doi     = {10.1109/TCYB.2015.2415732},
}
```

When citing the implementation specifically (rather than the
methodology), additionally cite this repository:

```bibtex
@software{azevedo2026learningimpl,
  title   = {learning-to-anticipate-flexible-choices: reference Python
             implementation},
  author  = {Azevedo, Carlos R. B.},
  year    = {2026},
  url     = {https://github.com/crbazevedo/learning-to-anticipate-flexible-choices},
}
```

---

## License

MIT — see [`pyproject.toml`](pyproject.toml) for the formal declaration.

---

## Acknowledgments

The 2015 paper was supported by FAPESP (grant 2012/16504-5), CNPq, and
CAPES. The Python implementation was rehabilitated and made publication-
ready in 2026 under `dfg-harness` governance — a 5-wave, 31-PR
campaign that closed every audit-flagged finding and surfaced (and
fixed) five dead-code-coming-alive bugs along the way. The
rehabilitation receipts are in [`docs/VISION.md`](docs/VISION.md) and
the per-wave retrospectives at [`.dfg/retrospectives/`](.dfg/retrospectives/).

— Carlos R. B. Azevedo · [renatoaz@gmail.com](mailto:renatoaz@gmail.com)

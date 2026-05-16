# learning-to-anticipate-flexible-choices

Reference Python implementation of:

> **Azevedo, C. R. B., & Von Zuben, F. J.** (2015). *Learning to Anticipate
> Flexible Choices in Multiple Criteria Decision-Making Under Uncertainty.*
> IEEE Transactions on Cybernetics. DOI:
> [10.1109/TCYB.2015.2415732](https://doi.org/10.1109/TCYB.2015.2415732)

The canonical algorithmic spec is the paper PDF at `docs/paper.pdf`.

## Status

🚧 **Private during publication-quality refactor.** This repo is being
hardened toward a stable public release. Until then it is governed by
[`dfg-harness`](https://github.com/korzainc/dfg-harness) under
contract-first, dual-critic, wave-by-wave PR discipline.

The state of the work is in `docs/VISION.md`. The active wave plan is
`.dfg/plan.yaml`.

## Layout

| Path | Contents |
|---|---|
| `docs/paper.pdf` | The canonical 2015 IEEE TCYB paper. |
| `docs/VISION.md` | Vision + trajectory + empirical receipts. |
| `python_refactor/src/` | The Python implementation (active). |
| `python_refactor/tests/` | Unit + integration tests. |
| `python_refactor/experiments/` | Experiment driver scripts. |
| `python_refactor/data/ftse-updated/` | Active FTSE dataset. |
| `legacy-cpp/` | Original C++ implementation (frozen, kept for provenance). |
| `.dfg/` | dfg-harness substrate — wave plan + agent contracts + state. |

## Quick start (subject to W1-3 import-fix landing first)

```bash
cd python_refactor
pip install -r requirements.txt
python main.py            # synthetic-data quick-run
python -m pytest tests/   # known: 17 collection errors pre-W1-3
```

## Provenance

Originally maintained as `crbazevedo/crbazevedo-phd-research`; renamed
to `learning-to-anticipate-flexible-choices` (the paper's title) on
2026-05-16 during the dfg-harness bootstrap. The prior snapshot
`crbazevedo/anticipatory-learning-asmoo` is archived.

— Carlos R. B. Azevedo

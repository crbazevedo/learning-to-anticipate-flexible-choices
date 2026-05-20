# W22 Probe V — Opt-in fix combination ablation framework

**Status:** framework SHIPPED (pure plumbing). Operator wires the real
ASMS runner via the `wealth_fn` plug-in; the framework itself runs
deterministic synthetic wealth for CI.

**Module:** `python_refactor/src/probes/probe_v_combination_ablation.py`
**Tests:** `python_refactor/tests/test_probe_v_combination_ablation.py`

---

## Hypothesis

> **H-Probe-V.** On FTSE (n ≥ 10 sample seeds), the combination of
> **2 or 3** of the four W22 opt-in structural fixes maximises final
> wealth. The baseline alone (zero opt-in fixes; ON-by-default fixes
> only — NC30-v2, NC29 w_0, NC29a γ^h) is **strictly sub-optimal**;
> the all-four-ON combination is **not** strictly optimal either
> (we expect at least one pairwise interaction to be negative, e.g.
> NC13b smooth-clamp ↔ NC31 conditional TIP).
>
> Falsifiable claim: there exists a non-baseline, non-full combination
> with wealth ≥ baseline + 3.0% **and** wealth ≥ full + 1.0%.

The four opt-in fixes (per `docs/W22-MASTER-BACKLOG.md` Section B):

| Fix | Surface | Default | Activation |
|---|---|---|---|
| NC13b smooth TIP clamp | `tip_calculator.py` | OFF | env `W22_NC13B_SMOOTH_CLAMP=1` |
| NC27-deep DirichletPosterior | `dirichlet_posterior.py` | OFF | env `W22_NC27_PREDICTOR=dirichlet_posterior` |
| NC30 c continuous variance penalty | `amfc_selector.py` | OFF | `dm_config['variance_penalty'] > 0` |
| NC31 TIP Defn 6.1 conditional | `tip_calculator.py` | OFF | env `W22_NC31_TIP_CONDITIONAL=1` |

There are 2⁴ = **16** possible combinations. Manually walking the
operator through 16 ASMS runs (≈ 30 min each on FTSE n=10) is a
~8 hr serial bill. The framework systematises the comparison:
enumeration + per-combo env/dm_config translation + result aggregation.

---

## Methodology

### Surface

```python
from src.probes.probe_v_combination_ablation import (
    OPT_IN_FIXES,            # canonical list
    ENV_VAR_MAP,             # 3-entry env-var map (NC30c absent — dm_config)
    enumerate_combinations,  # "full" / "pairwise" / "single"
    combination_to_env_dict,
    combination_to_dm_config,
    run_ablation_synthetic,
    subprocess_runner_skeleton,  # reference shape for the wealth_fn plug-in
    format_ablation_table,
)
```

### Strategies

| Strategy | Combinations | Use when |
|---|---|---|
| `"full"` | 16 (baseline + 15 non-empty subsets) | gold-standard once per release |
| `"pairwise"` | 7 (baseline + C(4,2)=6 pairs) | first pass — detects pairwise interactions |
| `"single"` | 5 (baseline + 4 singletons) | smoke — confirms each fix in isolation is non-degenerate |

### Per-combination translation

Given a fix-set (subset of `OPT_IN_FIXES`), the framework emits:

1. **Env-var dict** via `combination_to_env_dict(fix_set)`:
   only the 3 env-gated fixes contribute. The dict is empty for the
   baseline. NC30c never appears (it is not env-gated).

2. **dm_config dict** via `combination_to_dm_config(fix_set, base_dm_config)`:
   `variance_penalty = 1.0` iff NC30c is active, else the OFF-value
   (taken from `base_dm_config['variance_penalty']` if provided,
   defaulting to `0.0`). All other keys in `base_dm_config` pass
   through unchanged.

### Aggregation

`run_ablation_synthetic(combos, wealth_fn=...)` returns
`dict[frozenset, float]`. `format_ablation_table(results)` renders a
markdown table sorted by wealth descending, with the baseline row
annotated `◀ baseline` and an uplift column showing signed deltas vs
baseline.

---

## Success criteria

1. **Framework correctness** (CI):
   - `enumerate_combinations("full")` returns exactly 16 distinct subsets
   - The baseline (empty set) appears under every strategy
   - `combination_to_env_dict({...four fixes})` returns exactly the
     3-entry env dict (NC30c absent)
   - `combination_to_dm_config` correctly toggles `variance_penalty`
     and respects the operator-supplied OFF-value
   - `run_ablation_synthetic` returns a frozenset-keyed dict that
     always includes the baseline
   - `format_ablation_table` is sorted descending and highlights baseline

2. **Hypothesis verification** (operator pass):
   - Operator plugs the real ASMS runner into `wealth_fn`
   - Runs `strategy="full"` on FTSE with seeds `0..9`
   - Records the top-3 combinations by mean wealth and the bottom-3
   - The hypothesis is **upheld** iff there exists a non-baseline,
     non-full combination with mean wealth ≥ baseline + 3.0% AND
     ≥ full + 1.0%
   - The hypothesis is **falsified** iff the full-set strictly dominates
     OR the baseline ties the best non-baseline combination

---

## Operator integration sketch

The default `wealth_fn` is a deterministic hash-based mock that anchors
the baseline at `100.0` and spreads other combinations across
`[100.0, 200.0)`. Replace it with the real ASMS runner like this:

```python
import json
import re
from src.probes.probe_v_combination_ablation import (
    combination_to_dm_config,
    combination_to_env_dict,
    enumerate_combinations,
    run_ablation_synthetic,
    format_ablation_table,
    subprocess_runner_skeleton,
)


_WEALTH_RE = re.compile(r"FINAL_WEALTH\s*=\s*([0-9eE+\-.]+)")


def _parse_wealth(stdout: str) -> float:
    m = _WEALTH_RE.search(stdout)
    if not m:
        raise RuntimeError(f"no FINAL_WEALTH in stdout:\n{stdout[-500:]}")
    return float(m.group(1))


def asms_wealth_fn(fix_set: set[str]) -> float:
    """Operator's plug-in: spawn the real ASMS run for this combination."""
    dm_config = combination_to_dm_config(fix_set, base_dm_config={
        "z_ref": "data_derived",
        "w0_floor": 0.2,
    })
    return subprocess_runner_skeleton(
        fix_set,
        command=[
            "python", "run_asms.py",
            "--dataset", "ftse",
            "--n-seeds", "10",
            "--dm-config", json.dumps(dm_config),
        ],
        timeout_seconds=1800.0,  # 30 min per combo
        parse_wealth_from_stdout=_parse_wealth,
    )


if __name__ == "__main__":
    # First pass: pairwise (7 combos, ~3.5 hrs wall)
    combos = enumerate_combinations("pairwise")
    results = run_ablation_synthetic(combos, wealth_fn=asms_wealth_fn)
    print(format_ablation_table(results))
```

### Sub-papercut #25 protection

`subprocess_runner_skeleton` passes `stdin=subprocess.DEVNULL` to avoid
the 3h40m hang receipt from W63 release-prep (`doc.py` `sort -V` dead
code inheriting stdin). The operator's `wealth_fn` MUST preserve this
guard if it bypasses the skeleton.

### NC30c handling note

NC30c is the only fix whose activation is **not** env-var-gated — it
lives in `dm_config['variance_penalty']`. The framework keeps the two
surfaces (env vs dm_config) separated so the operator's runner can pipe
the dm_config through whatever CLI flag / config-file path the real
ASMS surface expects. The skeleton stashes `repr(dm_config)` in
`W22_PROBE_V_DM_CONFIG` as an env-var hint, but does not assume the
runner reads it from there.

---

## Out of scope (deferred)

- Multi-dataset sweep (FTSE × NASDAQ × HangSeng × SaoPaulo × EuroStoxx).
  The framework supports it trivially (loop `wealth_fn` over datasets);
  the timing budget belongs to the operator.
- Statistical significance bands (paired-t / bootstrap) on the wealth
  uplifts. The framework returns scalars; the operator's analysis layer
  can attach CIs.
- Interaction-effect ANOVA over the full 16. Out of scope for the
  framework; a separate probe (W) can layer that on top.

---

## Compliance

- NO modifications to shared code paths (`sms_emoa.py`,
  `anticipatory_learning.py`, `amfc_selector.py`,
  `dirichlet_posterior.py`, `tip_calculator.py`,
  `kalman_filter.py`, `multi_horizon_anticipatory.py`).
- Standard-library + `numpy` only.
- Framework ships with deterministic CI tests (mock wealth fn); the
  real-ASMS plumbing is operator-pluggable.

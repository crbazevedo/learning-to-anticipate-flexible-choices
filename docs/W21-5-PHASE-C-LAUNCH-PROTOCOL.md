# W21-5 Phase C launch protocol — 30-seed final ablation

*Drafted 2026-05-17 mid-Phase-B. Specifies the exact command + matrix
+ wall-clock + verdict criteria for the FINAL replication run.*

## Pre-launch checklist

Before launching Phase C, verify ALL of:

- [ ] Phase B (10-seed) complete with per_seed.json on disk
- [ ] Phase B verdict: which scenarios show direction-different-from-default
  at 95% CI? (none? some? all?)
- [ ] V6 implementation merged (PR #125)
- [ ] V5 + V7 implementations merged (PR #123)
- [ ] H=3 scenarios in validation_matrix.py (PR #123)
- [ ] CLI flags wired for closed-form A/B/C (PR #128)
- [ ] Operator GO signal

If any of these are FALSE: do NOT launch. Resolve first.

## Full ablation matrix (Phase C)

11 scenarios × 30 seeds = 330 jobs. Wall-clock estimate (per W21-1 data):
- ~1100s per ASMS job × 330 jobs / 5 workers = ~24,200s = **~6.7 hours**

| ID | Scenario name | Flags active | Hypothesis tested |
|---|---|---|---|
| V0 | SMS_RDM_K0 | (baseline) | (myopic baseline) |
| V1 | ASMS_mHDM_K3 | (all Python defaults) | (default ASMS reference) |
| V2 | ASMS_mHDM_K3_v2rate | use_v2_anticipative_rate | Reading E isolated |
| V3 | ASMS_mHDM_K3_v2stab | use_v2_stability_weighting | Reading F-INVERSION isolated |
| V4 | ASMS_mHDM_K3_v2both | rate + stab | Combined E+F-inv |
| V5 | ASMS_mHDM_K3_v2rate_noSqrt | rate + use_thesis_eq74_risk | W18-CARRY-1 sqrt removal |
| V6 | ASMS_mHDM_K3_v2rate_kfV6 | rate + use_v2_kf_lifecycle | Reading D per-call lifecycle |
| V7 | ASMS_mHDM_K3_v2rate_v2entropy | rate + use_v2_entropy_operators | W21-3 entropy operators |
| H3 | ASMS_mHDM_K3_H3 | max_horizon=3 | Two-step-ahead anticipation |
| H3v2 | ASMS_mHDM_K3_H3_v2rate | H=3 + rate | Combined H + E |
| ALL | ASMS_mHDM_K3_v2_kitchen_sink | rate + stab + sqrt + KF + entropy + H=3 | All flags combined |

The "ALL" scenario doesn't exist yet in validation_matrix.py — would
need to be added before Phase C launches:

```python
"ASMS_mHDM_K3_v2_kitchen_sink": {
    "name": "ASMS/mHDM K=3 + ALL W22 ablation flags (kitchen sink)",
    "learning": _asms_learning_config(K=3, use_v2_anticipative_rate=True, max_horizon=3),
    "algorithm_overrides": {
        "use_v2_stability_weighting": True,
        "use_thesis_eq74_risk": True,
        "use_v2_kf_lifecycle": True,
        "use_v2_entropy_operators": True,
    },
    "dm": "mHDM",
},
```

## Phase C launch command

```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate,ASMS_mHDM_K3_v2stab,ASMS_mHDM_K3_v2both,ASMS_mHDM_K3_v2rate_noSqrt,ASMS_mHDM_K3_v2rate_kfV6,ASMS_mHDM_K3_v2rate_v2entropy,ASMS_mHDM_K3_H3,ASMS_mHDM_K3_H3_v2rate,ASMS_mHDM_K3_v2_kitchen_sink \
    --seeds 1-30 \
    --train-window-days 378 --step-days 50 --n-mc 500 \
    --jobs 5 \
    --enforce-thesis-continuous-trades \
    --output ../docs/W21-5-FULL-ABLATION-VERDICT.md \
    --per-seed-json experiments/results/w21-5-ablation-phase-c/per_seed.json
```

Notes:
- `--n-mc 500` (vs Phase B's 200) for tighter MC variance on the final verdict
- `--jobs 5` is fixed by host resource budget; could increase to 8 if other smokes have completed
- Total wall-clock: ~6.7 hours at --jobs 5; ~4.2 hours at --jobs 8

## Verdict criteria (Phase C)

Per docs/W21-5-ABLATION-PROTOCOL.md + the Phase B revision:

**PAPER-REPLICATES (direction reverses):**
- Some V_k has Δ vs SMS > 0 with 95% CI excluding 0
- W21-6 framing 1 ("paper replicates with documented divergence chain")

**PARTIAL (≥ 80% gap closure but still V_k ≤ V0):**
- Some V_k achieves ≤ 20% of the default ASMS gap (vs n=10 default ≈ -9.92%, that means V_k > -2%)
- W21-6 framing 2 ("paper replicates with caveats")

**WEAK-PARTIAL (< 80% gap closure but > 20%):**
- Some V_k achieves between -20% and -7.9% (closes some but not most)
- W21-6 framing 2.5 ("ASMS approaches but doesn't reach paper claim")

**REFUTED (< 20% gap closure):**
- No V_k achieves > -7.9% (less than 20% of -9.92% default gap closed)
- W21-6 framing 3 ("paper headline doesn't replicate; full receipt provided")

Per W21-1 revision (docs/W21-1-HONEST-REVISION-N10-RESULTS.md):
the residual gap at n=10 after Reading E alone is approximately -7.94%,
i.e., the v2_rate scenario only closes 1.98pp of the 9.92pp gap. Phase C
must establish whether combined flags close substantially more.

## Decision points

1. **After Phase B completes**: if NO scenario shows direction-positive
   or substantial gap closure, the trajectory is strongly REFUTED. Operator
   may decide NOT to spend the 6.7h on Phase C.
2. **After Phase C completes**: the verdict matrix populates the W21-6
   synthesis. Operator decides publication-track per the matrix.
3. **If verdict is REFUTED**: the 14-check cross-validation + 4-estimator
   methodology comparison is itself a meaningful methodological contribution.
   "Even when documented divergences are closed, the paper's headline doesn't
   replicate" is a publishable finding.

## Risk mitigations

- **Process crashes mid-run**: per_seed.json writes only at end. To
  mitigate, consider chunking into 2 batches of 15 seeds each + concat
  the JSONs.
- **OOM on long sessions**: n_mc=500 with 11 scenarios × 30 seeds may
  pressure memory if any worker leaks. Monitor via `dfg trace export` or
  `ps`. Kill + restart if heap grows unboundedly.
- **Code drift mid-run**: Phase C uses the code state at LAUNCH time
  (per multiprocessing import semantics). Don't modify any source between
  launch and completion.

## Output artifacts

- `python_refactor/experiments/results/w21-5-ablation-phase-c/per_seed.json`
- `docs/W21-5-FULL-ABLATION-VERDICT.md`
- W21-6 synthesis fill-in (after Phase C analysis)

## What this enables for W21-6

W21-6 = synthesis + publication-track decision. With Phase C complete:
- Full verdict matrix (11 scenarios × 30 seeds × n_periods)
- Per-scenario 95% CI
- Direction verdict per scenario
- One of {PAPER-REPLICATES, PARTIAL, WEAK-PARTIAL, REFUTED}
- Operator decision: which publication framing to author

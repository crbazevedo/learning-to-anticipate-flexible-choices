# W21-5 full ablation protocol

*Drafted 2026-05-17 by W21-5 during W21-1 smoke wait. Defines the
keystone ablation matrix that produces the FINAL REPLICATION VERDICT.*

## Purpose

W21-5 is the KEYSTONE for the W7→W21 cross-validation chain. It runs
the variants identified by the 14-check audit (closed across W18-21)
on a sufficient seed count to produce a statistically meaningful
Δ(S2 − S0) per variant. The final verdict (PAPER-REPLICATES / PARTIAL /
REFUTED) is the answer to "does closing the documented divergence chain
reverse the W17-5 saturation result?"

## Variant matrix (8 variants)

| ID | Scenario name | Reading-E (v2 rate) | Reading-F (v2 stability = 1.0) | W18-CARRY-1 (sqrt removed) | Reading-D (KF lifecycle aligned) | Notes |
|---|---|---|---|---|---|---|
| V0 | SMS_RDM_K0 (baseline) | n/a | n/a | n/a | n/a | Myopic baseline |
| V1 | ASMS_mHDM_K3 (default) | ❌ Python Eq 7.16 | ❌ Python stability | ❌ sqrt() applied | ❌ Python lifecycle | Current default ASMS |
| V2 | ASMS_mHDM_K3_v2rate | ✅ | ❌ | ❌ | ❌ | W20-1 receipt; +7.82pp recovery |
| V3 | ASMS_mHDM_K3_v2stab | ❌ | ✅ | ❌ | ❌ | W21-1 isolated test |
| V4 | ASMS_mHDM_K3_v2both | ✅ | ✅ | ❌ | ❌ | W21-1 combined test |
| V5 | ASMS_mHDM_K3_v2both_noSqrt | ✅ | ✅ | ✅ | ❌ | Closes W18-CARRY-1 |
| V6 | ASMS_mHDM_K3_v2both_noSqrt_alignedKF | ✅ | ✅ | ✅ | ✅ | Closes Reading D |
| V7 | ASMS_mHDM_K3_v2entropy | ❌ | ❌ | ❌ | ❌ | + v2 entropy operators (W21-3) |

V5, V6, V7 require additional implementation work:
- V5: extend Python's `compute_risk` to optionally skip `sqrt` (W18-CARRY-1 flag)
- V6: extend Python's KF lifecycle to mirror v2's combined update→predict per period (Reading D / W19-D-CARRY-1)
- V7: port v2's `raise_entropy` + `lower_entropy` mutation operators to Python (W21-3 ablation candidate)

V0..V4 are already runnable on the current branch.

## Seed strategy

| Phase | Seeds | Variants | Purpose | Wall-clock estimate |
|---|---|---|---|---|
| Phase A (smoke) | 1-2 | V0..V4 | Directional verdict (W21-1 receipt) | ~15 min |
| Phase B (mid) | 1-10 | V0..V4 | Stability of direction with finite samples | ~80 min |
| Phase C (final) | 1-30 | V0..V6 (V7 pending impl) | Final replication verdict | ~5-7 hours |

Phase A is W21-1's smoke (running now). Phase C is W21-5's keystone run.

## Statistical protocol

Per variant V_k:
- 30 seeds × 23 rolling periods = ~690 observations
- Compute grand mean Ŝ_k = mean(efhv_mean_per_period_per_seed)
- Compute grand std σ_k = std(efhv_mean_per_period_per_seed)
- Compute Δ_k = (Ŝ_k - Ŝ_V0) / Ŝ_V0 (percentage vs SMS_RDM_K0 baseline)
- 95% CI: Δ_k ± 1.96 × σ_k / sqrt(n_obs)

Headline table format:

```
| Variant | Δ(S_k − S_0) | std | 95% CI | Verdict |
|---|---|---|---|---|
| V0 SMS_RDM_K0 | 0.00% | σ_0 | — | reference |
| V1 ASMS default | -X% | σ_1 | [-Xa, -Xb]% | LOSES |
| V2 +v2 rate | -Y% | σ_2 | [-Ya, -Yb]% | RECOVERS |
| V3 +v2 stab only | -Z% | σ_3 | [-Za, -Zb]% | UNKNOWN |
| V4 v2 rate + v2 stab | -W% | σ_4 | [-Wa, -Wb]% | COMBINED |
| V5 + no sqrt | … | … | … | … |
| V6 + KF aligned | … | … | … | … |
```

## Final verdict criteria

1. **PAPER-REPLICATES** — variant V_k exists with Δ_k > 0 (S_k > S_0)
   with 95% CI excluding 0 → paper headline replicates after closing
   the documented divergences in V_k.

2. **PARTIAL** — Δ_k ≤ 0 for all V_k but |Δ_k| << |Δ_V1|
   (gap closure ≥ 80%) → the divergences explain MOST of the gap but
   not all; remaining unexplained gap is the residual.

3. **REFUTED** — Δ_k ≤ Δ_V1 + ε for all V_k → none of the documented
   divergences materially close the gap; the paper's headline result
   may be unrecoverable from the available substrate.

## Reproducing (commands)

### Phase A (W21-1 smoke, currently running)
```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate,ASMS_mHDM_K3_v2both,ASMS_mHDM_K3_v2stab \
    --seeds 1-2 \
    --train-window-days 378 --step-days 50 --n-mc 200 \
    --jobs 5 \
    --enforce-thesis-continuous-trades \
    --output ../docs/READING-F-EXPERIMENTAL-TEST.md \
    --per-seed-json experiments/results/w21-1-reading-f-smoke/per_seed.json
```

### Phase B (mid 10 seeds)
```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate,ASMS_mHDM_K3_v2both,ASMS_mHDM_K3_v2stab \
    --seeds 1-10 \
    --train-window-days 378 --step-days 50 --n-mc 200 \
    --jobs 5 \
    --enforce-thesis-continuous-trades \
    --output ../docs/W21-5-ABLATION-PHASE-B.md \
    --per-seed-json experiments/results/w21-5-ablation-phase-b/per_seed.json
```

### Phase C (final 30 seeds, V0..V4 only — V5/V6/V7 need impl)
```bash
cd python_refactor && uv run python -m experiments.walk_forward_report \
    --scenarios SMS_RDM_K0,ASMS_mHDM_K3,ASMS_mHDM_K3_v2rate,ASMS_mHDM_K3_v2both,ASMS_mHDM_K3_v2stab \
    --seeds 1-30 \
    --train-window-days 378 --step-days 50 --n-mc 500 \
    --jobs 5 \
    --enforce-thesis-continuous-trades \
    --output ../docs/W21-5-FULL-ABLATION-VERDICT.md \
    --per-seed-json experiments/results/w21-5-ablation/per_seed.json
```

## Decision points (for operator)

| Phase | After-completion decision |
|---|---|
| A | Continue to Phase B (if results look bounded) or pivot (if smoke shows surprise) |
| B | Continue to Phase C (if 10-seed verdict is directionally stable) or extend variant matrix (V5/V6/V7) |
| C | Author W21-6 synthesis + publication-track decision based on final verdict |

## Dependencies

- W21-1 implementation merged (this PR #122)
- W21-1 smoke results (Phase A) for context
- Operator approval to launch Phase C (5-7 hour wall-clock)

## Output artifacts (target)

- `python_refactor/experiments/results/w21-5-ablation/per_seed.json`
- `docs/W21-5-FULL-ABLATION-VERDICT.md` (verdict table + analysis)
- `.dfg/retrospectives/W21/W21-5.md` (ADR-004 retro)

## Strategic context

This is the terminal experiment of the W7→W21 chain. After W21-5+W21-6:
- All 14 operator cross-checks (A-N) closed ✓
- All 6 Reading hypotheses (A-F) addressed ✓
- All 5 documented divergences in the bug catalog ablated ✓
- Final replication verdict shipped → operator decides publication track

# W22 NC Smoke Run Results

Generated: 2026-05-20T09:57:49.787624

## Per-config grand mean per scenario (n=seeds)

| Config | Scenario | n_seeds | Mean Ŝ | Std | Δ vs BASELINE | Wilcoxon p |
|---|---|---|---|---|---|---|
| BASELINE | ASMS_mHDM_K3_v2both | 5 | 3.1568e-04 | 1.1523e-05 | — | — |
| BASELINE | SMS_RDM_K0 | 5 | 3.3507e-04 | 1.9705e-05 | — | — |

## Notes
- Δ vs BASELINE: positive means config beats baseline
- Wilcoxon p-value < 0.05 → statistically significant difference (n_seeds≥5)
- Configs with `_status` marker (e.g., ANTICIPATORY_OPS) require
  integration into SMSEMOA before they can be smoke-tested

## NC paths reference
- TIP_CLEANUP = NC36 + NC13b + NC31 (all TIP improvements)
- FULL_STACK = TIP_CLEANUP + NC27_DEEP (high-confidence theory)
- ANTICIPATORY_OPS = Path C (NC34 + NC39); needs SMSEMOA wiring
- MULTI_PERIOD_AMFC = NC35; needs dm_type='Hv-DM-AMFC' + dm_config
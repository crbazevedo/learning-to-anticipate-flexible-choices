# W22 NC Smoke Run Results

Generated: 2026-05-20T20:11:31.599596

## Per-config grand mean per scenario (n=seeds)

| Config | Scenario | n_seeds | Mean Ŝ | Std | Δ vs BASELINE | Wilcoxon p |
|---|---|---|---|---|---|---|
| BASELINE | ASMS_mHDM_K3_v2both | 5 | 3.4346e-04 | 9.0268e-05 | — | — |
| BASELINE | SMS_RDM_K0 | 5 | 3.6989e-04 | 4.8276e-05 | — | — |
| NC27_DEEP | ASMS_mHDM_K3_v2both | 5 | 3.5031e-04 | 3.2654e-05 | +6.8488e-06 | 1.000 |
| NC27_DEEP | SMS_RDM_K0 | 5 | 3.8924e-04 | 3.9560e-05 | +1.9352e-05 | 0.312 |

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
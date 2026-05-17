---
wave: W16
gate_type: wave-gate
verdict: PASS-FURTHER-GAP-CLOSURE
date: 2026-05-17
units_completed: [W16-1, W16-2, W16-3, W16-4, W16-5]
units_carry_forward: [W16-1-CARRY-1, W16-2-CARRY-1, W16-3-CARRY-1, W16-4-CARRY-1, W16-5-CARRY-1, W16-5-CARRY-2, BACKLOG-H4, BACKLOG-H7, BACKLOG-M1, BACKLOG-M2, BACKLOG-M5, BACKLOG-M6, BACKLOG-M7, BACKLOG-M8, BACKLOG-M9, BACKLOG-M10, BACKLOG-M11, BACKLOG-M12, BACKLOG-M13, BACKLOG-M14, BACKLOG-L1, BACKLOG-L2, BACKLOG-L3, BACKLOG-L4, BACKLOG-L5, BACKLOG-L6, BACKLOG-L7, BACKLOG-L8]
verify:
  - "git log --oneline master | grep -q 'W16-1.*consumption'"
  - "git log --oneline master | grep -q 'W16-2.*txn costs'"
  - "git log --oneline master | grep -q 'W16-3.*extrema'"
  - "git log --oneline master | grep -q 'W16-4.*trace'"
  - "git log --oneline master | grep -q 'W16-5.*integration smoke'"
  - "test -f docs/OOS-EFHV-W16-INTEGRATION-SMOKE.md"
  - "grep -q '_compute_lambda_k' python_refactor/src/algorithms/anticipatory_learning.py"
  - "grep -q 'compute_thesis_transaction_cost' python_refactor/src/portfolio/portfolio.py"
  - "grep -q '_identify_protected_anchors' python_refactor/src/algorithms/sms_emoa.py"
  - "grep -q 'flush_lambda_trace_csv' python_refactor/src/algorithms/anticipatory_learning.py"
  - "grep -q 'THESIS_TABLE_71_BRACKETS' python_refactor/src/config/thesis_parameters.py"
  - "cd python_refactor && uv run python -m pytest tests/test_lambda_k_consumption.py tests/test_extrema_preservation.py tests/test_transaction_costs_in_hv.py tests/test_lambda_tip_trace.py -q"
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W16 closes 3 HIGH (H1 txn costs + H2 λ formula + H6 extrema) + 2 MEDIUM partial (M3+M4 trace-export scope). Plus W15-3-CARRY-1 consumption half (W16-1)."
  - "Empirical gap closure receipt at docs/OOS-EFHV-W16-INTEGRATION-SMOKE.md: W14-2 -24.86% → W15-5 -8.75% (65% closure) → W16-5 -5.53% (cumulative 77.7% closure since W14-2)."
  - "5 PRs merged: #89 W16-1 (H2 + W15-3-CARRY-1 consumption) + #90 W16-3 (H6) + #91 W16-2 (H1) + #92 W16-4 (M3+M4 partial) + #93 W16-5 (integration smoke). Plus W16 replan PR #88."
  - "Verdict: PASS-FURTHER-GAP-CLOSURE. Direction NOT reversed (S2 still < S0) but the gap dropped from -8.75% to -5.53% (further 37% closure). W17 keystone: BACKLOG H4 (asset universe 87 vs 98) — most likely remaining lever."

carry_forward:
  - id: W16-1-CARRY-1
    why: "λ^K formula is structurally correct per Eq 6.9 but the residual buffer (self._kf_residual_window) is never populated in production because nothing calls record_kf_residual. Result: λ^K falls back to the warm-up traditional-rate branch, not the K-period residual-sum branch. Production behavior approximately unchanged by W16-1 alone."
    next_action: "W17 minor: wire record_kf_residual into the Kalman update path (kalman_filter.py kalman_update returns innovation; expose it OR add a per-step hook). Trace assertions in W16-5-CARRY-1 will then directly verify λ^K fires."
  - id: W16-2-CARRY-1
    why: "u*_{t-1} threading uses 'first Pareto-front portfolio' as the implemented decision. Thesis §6.4 Eq 6.42 specifies argmax-EFHV (AMFC) selection."
    next_action: "W17: wire AMFC selection (BACKLOG M5) as u*_{t-1} selector; verify via per-period u*_{t-1} trace."
  - id: W16-3-CARRY-1
    why: "extrema_trace is populated on the SMSEMOA instance but not surfaced in metrics_collector's algorithm_metrics output. Downstream viz (W18) needs it."
    next_action: "W18: integrate extrema_trace into metrics_collector + render per-gen front-extent plots."
  - id: W16-4-CARRY-1
    why: "Same as W16-5-CARRY-1 (driver-side wiring)."
    next_action: "(see W16-5-CARRY-1)"
  - id: W16-5-CARRY-1
    why: "walk_forward_report.py doesn't pass --lambda-trace-csv-path through to run_walk_forward. As a result the smoke can only measure aggregate Δ(S2−S0); cannot directly assert 'both λ arms fire' for K>0 scenarios."
    next_action: "W17 minor (~10 min): add --lambda-trace-csv-path CLI arg to walk_forward_report.py; pass through to run_walk_forward (kwarg already exists per W16-4)."
  - id: W16-5-CARRY-2
    why: "Pre-existing AnticipatoryLearning class hierarchy bug: compute_transaction_cost lives on TIPIntegratedAnticipatoryLearning but tests call it on base class → 14-test cascade failure in test_anticipatory_learning.py. Untouched by W16."
    next_action: "W17 hygiene: move compute_transaction_cost up to base class OR rationalize the existing simple-turnover model vs W16-2's bracket model (likely the simple model is dead code now)."
  - id: BACKLOG-H4
    why: "Asset universe 87 vs 98 mismatch. legacy-cpp/ftse-original has 98 CSVs but thesis specifies d=87. 11 extras may include delisted/illiquid pollutants. Most likely lever for closing the remaining 5.53% gap."
    next_action: "W17 keystone: identify the 87 thesis-faithful subset (cross-ref thesis-cited dataset URL if available; otherwise drop the 11 with most missing/quirky data); document the asset list explicitly."

---

# W16-gate — WAVE 16 CLOSE: 77.7% cumulative gap closure since W14-2

## What W16 delivered

| Unit | PR | BACKLOG closed |
|---|---|---|
| W16-1 | #89 | H2 (λ formula completeness) + W15-3-CARRY-1 consumption half |
| W16-2 | #91 | H1 (transaction costs in HV objective) |
| W16-3 | #90 | H6 (rank-1 extrema preservation) |
| W16-4 | #92 | M3 + M4 (partial — trace export scope) |
| W16-5 | #93 | Integration smoke + gap-closure receipt |

Plus W16 replan PR #88.

## Gap closure receipt (full chain)

| Stage | Δ(S2 − S0) | Cumulative closure vs W14-2 baseline |
|---|---|---|
| W13-3 single-shot 80/20 | -17.59% | (baseline) |
| W14-2 walk-forward (pre-W15) | -24.86% | (methodology ruled out) |
| W15-5 walk-forward (W15 BLOCKERS) | -8.75% | 65% closure |
| **W16-5 walk-forward (W16 λ^K + txn + extrema)** | **-5.53%** | **77.7% cumulative closure** |

Wall-clock improved 55% (390s vs W15-5's 858s) — unexpected positive
side-effect of W16 stabilization.

## Verdict

**PASS-FURTHER-GAP-CLOSURE.** Direction still wrong (S2 < S0) but
the gap dropped a further 3.22pp on top of W15's 16.11pp closure.
Cumulative 77.7% closure since W14-2 leaves us within 5.53pp of
reversal.

## Critical path to paper-replication

W17 keystone = **BACKLOG H4** (asset universe 87 vs 98). The 11
extra assets may include delisted/illiquid pollutants — clearest
candidate for the remaining 5.53% gap.

Secondary W17 candidates:
- H7 (correspondence mapping wiring — §6.2.2)
- M5 (AMFC selection per Eq 6.42 → wires argmax-EFHV as u*_{t-1}
  per W16-2-CARRY-1)
- W16-1-CARRY-1 (wire record_kf_residual so λ^K actually fires in
  production)
- W16-5-CARRY-1 (trace CSV path through walk_forward_report so
  smoke directly asserts both λ arms fire)
- Full 30-seed run for statistical rigor

If S2 > S0 post-W17, the W7→W17 chain is structurally complete and
the paper claim replicates.

## 5 PRs shipped in W16

#89, #90, #91, #92, #93 + #88 replan. All contracts followed the
updated BACKLOG §6 template (plain-string must_read + body markdown
with verbatim thesis excerpts) — the W15-gate schema fix in PR #87
paid off.

All 56 W15+W16 thesis-spec tests green throughout the wave.

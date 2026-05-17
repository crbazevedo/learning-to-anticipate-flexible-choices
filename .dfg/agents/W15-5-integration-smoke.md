---
id: W15-5
role: verifier
name: Integration smoke with W15 BLOCKERS fixed
purpose: "Verify W15-1+W15-2+W15-3 combined fix: re-run 2-seed walk-forward smoke on {SMS_RDM_K0, ASMS_mHDM_K3}; report Δ(S2 vs S0); commit empirical receipt."
wave: W15
unit: W15-5
depends_on: [W15-1, W15-2, W15-3]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - path: docs/BACKLOG.md
      sections: ["§1.1 B1", "§1.1 B3", "§1.1 B4", "§1.1 B2", "§1.2 H3", "§1.2 H5"]
      reason: "All W15 BLOCKERs whose combined effect this smoke measures"
    - path: docs/OOS-EFHV-WALK-FORWARD-REPORT.md
      reason: "Baseline (W14-2) for comparison — pre-BLOCKERs-fix walk-forward"
    - path: python_refactor/experiments/walk_forward_report.py
      reason: "Smoke driver (reused as-is)"
output_contract:
  files:
    - docs/OOS-EFHV-W15-INTEGRATION-SMOKE.md
    - .dfg/retrospectives/W15/W15-5.md
  branch_name: feat/w15-5-integration-smoke
  acceptance: >
    Re-run W14-2 walk-forward smoke with SMS_RDM_K0 + ASMS_mHDM_K3
    (now using new operators, cardinality projection, z_ref=(0.0,0.2),
    paper date range 2006-2012). Empirical Δ direction reported
    honestly. Retro qualitative inspection.
dispatch_instructions: |
  Run smoke (2 seeds × 2 scenarios × paper-window date range).
  Use same shape as W14-2 smoke: 200 MC, 2 workers.

  Expected wall-clock: shorter than W14-2's 41 min because paper
  window is now ~1500 days instead of ~2600 (fewer rolling periods
  per seed).

  Report direction (S2 vs S0) honestly. If S2 > S0, BLOCKERS fixed
  the algorithm and paper claim direction replicates. If S2 ≤ S0,
  remaining issues are in subsequent waves (W15-3-CARRY-1 λ^K
  consumption + W16 instrumentation + W17 advanced + W18 report).
---

# W15-5 — Integration smoke

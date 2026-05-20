---
wave: W21
gate_type: wave-gate
verdict: DEFERRED
date: 2026-05-20
units_completed: []
units_carry_forward: [W21-1, W21-2, W21-3, W21-4, W21-5, W21-6]
note: "W21 superseded by W22 inspection-driven push. Reading-F findings partially absorbed into W22 Probe Z empirical (NEUTRAL n=5 p=0.485). Publication-track decision deferred pending W22-7 NC27-deep cross-instance verdict."
---

# W21 gate checkpoint — DEFERRED

**Verdict:** DEFERRED

**Reason:** W21 (Reading-F experimental test + cross-checks L/M/N + 30-seed
final replication) was planned but superseded by the 2026-05-18..20 W22
push. W21-1's specific deliverable (use_v2_stability_weighting flag) IS
already shipped in `sms_emoa.py`. The 30-seed final replication was
reduced in scope by W22 to a paired n=10 PO(8,1.0) empirical for
NC27-deep (documented in CHANGES_LOG.md).

**Action:** marked DEFERRED. Reading-F findings partially absorbed into
W22 Probe Z empirical (NEUTRAL n=5 p=0.485). Publication-track decision
(W21-6) is deferred pending W22-7 NC27-deep cross-instance verdict.

**Audit trail:** see docs/W22-NC-SMOKE-EMPIRICAL-FINAL.md, Probe Z
results (commit fb525c2), drift registry.

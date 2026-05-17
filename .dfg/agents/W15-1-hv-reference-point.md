---
id: W15-1
role: code-fixer
name: Thread z_ref=(0.2, 0.0) into SMSEMOA (BACKLOG B1)
purpose: "Closes BACKLOG B1. Replace sms_emoa.py defaults R1=0.0, R2=1.0 with thesis z_ref=(0.2, 0.0)^T per §7.2.3 p.147."
wave: W15
unit: W15-1
depends_on: []
blocks: [W15-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - docs/THESIS-INDEX.md
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/experiments/oos_evaluator.py
output_contract:
  files:
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/tests/test_sms_emoa_reference_point.py
  branch_name: feat/w15-1-hv-reference-point
  acceptance: >
    SMSEMOA constructor accepts z_ref tuple; defaults to thesis values
    (0.2, 0.0) per §7.2.3 p.147. Both HV computation methods use it
    consistently. ≥ 3 regression tests covering thesis defaults +
    override + computed HV at new reference.
dispatch_instructions: |
  Closes BACKLOG: B1.

  Surgical changes:
    1. SMSEMOA.__init__ signature: replace
         `reference_point_1: float = 0.0, reference_point_2: float = 1.0`
       with
         `z_ref: tuple[float, float] = (0.2, 0.0)`
       (thesis: risk_max, return_min). Update self.R1/R2 set from z_ref[0]/z_ref[1]
       BUT KEEP the semantic — risk axis MIN (R2 = "max risk acceptable"),
       ROI axis MAX from R1 (the lower bound on ROI).
    2. Verify call sites at lines 384, 409 still use R1/R2 correctly
       (they should — only the DEFAULTS change).
    3. Confirm oos_evaluator.py W13-2's z_ref=(0.2, 0.0) is consistent.

  What NOT to do:
    - Don't refactor R1/R2 → z_ref everywhere if the semantic differs;
      this is a default-value fix, not a structural refactor.
    - Don't touch the algorithm logic — only the reference point.

  PR body must echo the thesis excerpt (§7.2.3 p.147) per BACKLOG §6
  grounding discipline.
---

# W15-1 — Thread z_ref=(0.2, 0.0) into SMSEMOA

Closes BACKLOG.md items: **B1** — Internal HV reference point wrong scale.

## Thesis grounding

**§7.2.3 ASMS Parameters (p. 147)**:
> "Finally, the reference point for computing Hypv was set to z^ref = (0.2, 0.0)^T in terms of risk and return, coinciding with the objective space feasibility boundaries (maximum risk of 20% and minimum mean return of 0%)."

Pre-W15-1 defaults: R1=0.0 (ROI), R2=1.0 (risk) — selection pressure
points at the wrong corner of the objective space.

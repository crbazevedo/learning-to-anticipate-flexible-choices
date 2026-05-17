---
id: W20-1
role: experimenter
name: Reading-E experimental test (use_v2_anticipative_rate flag + W17-5 smoke re-run)
purpose: "Closes W19-4-CARRY-1. Experimentally test whether v2's monotonic anticipative-rate formula (1 - TIP) instead of Python's thesis-Eq-7.16 (1/2)(λ^H + λ^K) reverses the persistent S2 ≤ S0 result on the W17-5 fixture. If S2 > S0 with v2 formula, paper replicates."
wave: W20
unit: W20-1
depends_on: []
blocks: [W20-6]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md
    - docs/W19-CROSS-VALIDATION-SYNTHESIS.md
    - python_refactor/src/algorithms/anticipatory_learning.py
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_use_v2_anticipative_rate.py
    - docs/READING-E-EXPERIMENTAL-TEST.md
  branch_name: feat/w20-1-reading-e-experimental-test
  acceptance: >
    Python flag `use_v2_anticipative_rate=True` switches
    compute_anticipatory_learning_rate to v2's `alpha = 1 - TIP`
    formula. W17-5 smoke re-run with flag enabled; Δ(S2 − S0) reported
    honestly with comparison vs prior smokes (W17-5: -8.72%; W17-5-CARRY-1:
    -11.79%). If S2 > S0, Reading E confirmed + paper replicates.
dispatch_instructions: |
  Closes W19-4-CARRY-1 (KEYSTONE for W20).

  Surgical workflow:

  1. In `python_refactor/src/algorithms/anticipatory_learning.py`,
     `AnticipatoryLearning.__init__` add kwarg:
       `use_v2_anticipative_rate: bool = False`

  2. In `compute_anticipatory_learning_rate`, when
     `self.use_v2_anticipative_rate is True`, use:
       lambda_combined = 1.0 - tip_value
     INSTEAD of the Eq 7.16 form:
       lambda_combined = 0.5 * (lambda_h + lambda_k)

     Keep the trace plumbing intact; record lambda_h/lambda_k as
     usual but override the combined output.

  3. Propagate the flag through MultiHorizonAnticipatoryLearning +
     TIPIntegratedAnticipatoryLearning constructors so it can be
     enabled from the experiment_config.

  4. Add SCENARIOS variant ASMS_mHDM_K3_v2rate that's identical to
     ASMS_mHDM_K3 except `use_v2_anticipative_rate=True`.

  5. Run W17-5 smoke shape on {SMS_RDM_K0, ASMS_mHDM_K3, ASMS_mHDM_K3_v2rate}
     with 87-asset filter + 2 seeds × 200 MC. Compare Δ(S2 − S0) for
     ASMS_mHDM_K3 (Eq 7.16 formula) vs ASMS_mHDM_K3_v2rate (v2 formula).

  6. Tests at `python_refactor/tests/test_use_v2_anticipative_rate.py`:
     - default (flag=False) → Eq 7.16 formula
     - flag=True → 1 - TIP formula
     - both branches produce correct trace rows

  7. Receipt at docs/READING-E-EXPERIMENTAL-TEST.md:
     - Both formulas' λ outputs on production trace inputs
     - Smoke Δ(S2 − S0) for both scenarios
     - Verdict: PAPER-REPLICATES / READING-E-PARTIAL / READING-E-REFUTED

  Expected wall-clock: ~10-15 min smoke + 30 min code/tests/receipt.

  What NOT to do:
   - Don't modify the existing Eq 7.16 production path (flag-gated only)
   - Don't ship with non-default flag enabled (keep default=False)
   - Don't decide A/B/C/D readings — focus on E
---

# W20-1 — Reading E experimental test

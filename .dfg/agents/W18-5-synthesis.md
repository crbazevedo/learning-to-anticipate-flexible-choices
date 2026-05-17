---
id: W18-5
role: verifier
name: W18 synthesis — A+B+F findings + W19-W22 roadmap for C/D/E/G/H/I/J/K/L/M/N
purpose: "Synthesize W18-2/W18-3/W18-4 cross-check verdicts; produce wave-close receipt + roadmap deferring C/D/E/G/H/I/J/K/L/M/N to W19+ with unit-contract pre-stubs."
wave: W18
unit: W18-5
depends_on: [W18-2, W18-3, W18-4]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/CROSS-VALIDATION-A-RISK.md
    - docs/CROSS-VALIDATION-B-ROI.md
    - docs/CROSS-VALIDATION-F-TIP.md
output_contract:
  files:
    - docs/W18-CROSS-VALIDATION-SYNTHESIS.md
    - .dfg/retrospectives/W18/W18-5.md
  branch_name: feat/w18-5-synthesis
  acceptance: >
    W18 synthesis doc summarizes A/B/F verdicts + identifies which
    Reading (A=Python bug, B=C++ bug, C=both correct + structural)
    each check supports. Roadmap for W19-W22 with pre-stubbed unit
    contracts for the 10 remaining checks (C/D/E/G/H/I/J/K/L/M/N).
dispatch_instructions: |
  Closes W18 wave. Per operator directive: "investigate everything"
  — but acknowledge realistic scope (14 checks × 1-2hr each = 3 weeks).
  W18 covers harness + 3 most-strategic checks (A risk, B ROI, F TIP);
  W19-W22 inherit harness and chip away at remaining 11.

  Synthesis tasks:
   1. Aggregate A/B/F verdicts into a single table
   2. Determine which Reading each check supports:
      - Reading A: Python port has bugs that the cross-check exposes
      - Reading B: C++ reference has issues (less likely but documented)
      - Reading C: both agree → structural property of data/model
   3. Update W17-5 strategic-reading framework with W18 evidence
   4. Pre-stub W19-W22 unit contracts for C/D/E/G/H/I/J/K/L/M/N (one
      contract file per check, in .dfg/agents/W19-1-cross-check-C-bivariate.md,
      W19-2-cross-check-D-kf-gaussians.md, ...)
   5. Synthesis retro per ADR-004

  W18 synthesis MUST inform the Reading A vs Reading B vs Reading C
  decision — this is the operator-facing receipt of "is the paper
  replicable on this codebase, and if not, why".
---

# W18-5 — W18 synthesis

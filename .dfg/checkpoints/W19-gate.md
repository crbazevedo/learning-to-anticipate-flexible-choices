---
wave: W19
gate_type: wave-gate
verdict: PASS-READING-E-IDENTIFIED
date: 2026-05-17
units_completed: [W19-1, W19-2, W19-3, W19-4, W19-5]
units_carry_forward: [W19-4-CARRY-1, W19-4-CARRY-2, W19-D-CARRY-1, W20-cross-check-H, W20-cross-check-I, W20-cross-check-J, W20-cross-check-K, W21-cross-check-L, W21-cross-check-M, W21-cross-check-N]
verify:
  - "git log --oneline master | grep -iq 'W19-1.*bivariate'"
  - "git log --oneline master | grep -iq 'W19-2.*KF'"
  - "git log --oneline master | grep -iq 'substrate.*2015'"
  - "git log --oneline master | grep -iq 'W19-4.*anticipative'"
  - "test -f docs/CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md"
  - "test -f docs/CROSS-VALIDATION-D-KF-GAUSSIANS.md"
  - "test -f docs/CROSS-VALIDATION-V2-REASSESSMENT.md"
  - "test -f docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md"
  - "test -f docs/W19-CROSS-VALIDATION-SYNTHESIS.md"
  - "test -f legacy-cpp-v2/build/Makefile"
  - "test -f legacy-cpp-v2/source/dirichlet.cpp"
  - "test -f legacy-cpp-v2/build/drivers/dirichlet_driver.cpp"
  - "cd python_refactor && uv run python -m pytest tests/test_cross_check_bivariate_gaussian.py tests/test_cross_check_kf.py tests/test_cross_check_dirichlet.py tests/test_cross_check_anticipative_rate.py -q"
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W19 closes 4 of remaining 11 operator cross-checks (C bivariate, D KF, E Dirichlet, G anticipative rate). 7 remain (H, I, J, K, L, M, N) for W20-W21."
  - "MID-WAVE substrate correction: original W19-1/W19-2 ran vs legacy-cpp/ (2013); operator flagged the correct oracle is legacy-cpp-v2/ (2015 paper-companion). Mid-wave PR #113 imported v2; PR #114 re-ran all prior checks vs v2."
  - "MAJOR finding: W19-4 G surfaced THREE-WAY formula divergence in anticipative rate. v1 (tent entropy) + Python (Shannon Eq 7.16) both collapse to 0 at TIP=0.5; v2 (monotonic 1-p) maintains alpha=0.5. v2 is paper-headline-generating; Python is thesis-faithful. **Reading E** is the strongest hypothesis for the W17-5 saturation chain."
  - "6 PRs merged: #110 replan + #111 W19-1 + #112 W19-2 + #113 substrate import + #114 v2 re-assessment + #115 W19-4 G."
  - "Verdict: PASS-READING-E-IDENTIFIED. W20 keystone (W19-4-CARRY-1): implement use_v2_anticipative_rate flag + re-run W17-5 smoke; if S2 > S0, paper replicates with formula adjustment."

carry_forward:
  - id: W19-4-CARRY-1
    why: "Reading E experimental test. v2's monotonic anticipative-rate formula keeps anticipation alive at TIP=0.5 (W17-5 saturation regime) while Python's thesis Eq 7.16 collapses to 0. Add flag use_v2_anticipative_rate=True in Python + re-run W17-5 smoke. If S2 > S0, paper replicates with formula adjustment."
    next_action: "W20-1 (TOP priority): implement flag + smoke + receipt."
  - id: W19-4-CARRY-2
    why: "Thesis Eq 7.16 (1/2) factor semantics unclear. Python reads it as averaging that halves λ; v2 may have read it differently. Resolve via thesis text re-read."
    next_action: "W20-2: re-read thesis §7.2.3 p.146 + IEEE paper Eq 13 line-by-line."
  - id: W19-D-CARRY-1
    why: "Production state-evolution divergence between Python (update-only per generation) and v2 (combined update→predict per period). Adds a Reading D dimension orthogonal to E."
    next_action: "W20+: trace KF state per period in both implementations on same fixture; quantify divergence."
  - id: W20-cross-check-H
    why: "Operator check H: Dirichlet variant. May dedupe with E given W19-3 closure."
    next_action: "W20-3: verify dedupe or distinct scope."
  - id: W20-cross-check-I
    why: "Operator check I: anticipative distributions from OAL. Downstream of D + E."
    next_action: "W20-4: cross-validate OAL output distributions."
  - id: W20-cross-check-J
    why: "Operator check J: crowding distance for NDS. Independent of saturation chain."
    next_action: "W20-5: harness-pattern check."
  - id: W20-cross-check-K
    why: "Operator check K: expected HV contribution. Likely AGREE; independent verification."
    next_action: "W21-1: harness-pattern check."
  - id: W21-cross-check-L
    why: "Operator check L: NDS algorithm. Independent of saturation chain."
    next_action: "W21-2."
  - id: W21-cross-check-M
    why: "Operator check M: mutation + simplex/parent-correlation."
    next_action: "W21-3."
  - id: W21-cross-check-N
    why: "Operator check N: SBX vs uniform crossover. Note W15-2 already switched Python to uniform per thesis §7.2.3 p.147."
    next_action: "W21-4."

---

# W19-gate — WAVE 19 CLOSE: Reading E identified (anticipative-rate formula divergence)

## What W19 delivered

| Unit | PR | Headline finding |
|---|---|---|
| W19-1 | #111 | Cross-check C bivariate — AGREE machine precision (both v1 + v2) |
| W19-2 | #112 | Cross-check D KF — AGREE vs v1; **revised: DISAGREE vs v2 (lifecycle reversed)** |
| (substrate) | #113 | Imported legacy-cpp-v2/ (2015 paper-companion) as the correct oracle |
| (re-assessment) | #114 | 4 verdicts confirmed/revised; **W19-3 Dirichlet AGREE machine precision** |
| W19-4 | #115 | **Cross-check G — THREE-WAY formula divergence; Reading E identified** |
| W19-5 | THIS | Synthesis + W20-W21 roadmap |

Plus W19 replan PR #110.

## The Reading E finding

At TIP=0.50 (the W17-5 production saturation regime):

| Source | alpha / λ | Behavior |
|---|---|---|
| v1 (tent entropy) | 0.0000 | COLLAPSES |
| **v2 (monotonic 1−p)** | **0.5000** | **MAINTAINS ACTIVE ANTICIPATION** |
| Python λ^H per Eq 6.6 | 0.0000 | COLLAPSES |
| Python λ Eq 7.16 (λ^K=0) | 0.0000 | COLLAPSES |

**v2 is the OUTLIER**. It's the paper-headline-generating code, and
it keeps anticipation alive in EXACTLY the regime where v1 and Python
both collapse to zero. **Python implements thesis Eq 7.16 verbatim but
the paper appears to have used the simpler v2 monotonic formula**.

This unifies the W17-5 → W18-4 → W19 saturation chain: the persistent
S2 ≤ S0 in our smokes is most plausibly explained by Python
collapsing anticipation in saturation per Eq 7.16, while the paper's
ASMS > SMS result was generated with v2's monotonic formula that
maintains anticipation strength of 0.5 in the same regime.

## Verdict

**PASS-READING-E-IDENTIFIED.** W19 advanced the diagnostic chain from
"structural data property (Reading C)" to "anticipative-rate formula
divergence (Reading E)". The substrate update was load-bearing.

## W20 keystone

**W20-1 = W19-4-CARRY-1**: Implement Python flag `use_v2_anticipative_rate=True` + re-run W17-5 smoke.

| Outcome | Interpretation |
|---|---|
| S2 > S0 | Paper replicates with formula adjustment; (1/2) in thesis Eq 7.16 was mis-interpreted by Python |
| S2 ≤ S0 still | Other factors needed; deeper diagnosis required |

Either way the publication framing is clear and the answer is bounded.

## Test plan
- [x] dfg validate clean
- [x] dfg wave close W19 → WaveGatePass
- [x] All PRs merged (#110-#115 + this)
- [x] All 5 retros at .dfg/retrospectives/W19/W19-{1..5}.md
- [x] W19 synthesis at docs/W19-CROSS-VALIDATION-SYNTHESIS.md
- [x] Substrate provenance docs updated (THESIS-INDEX, BACKLOG, both legacy-cpp READMEs)

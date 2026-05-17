# W21 cross-validation synthesis — TERMINAL WAVE OF W7→W21 CHAIN

*Drafted 2026-05-17 by W21-6 (scaffold). Final synthesis after
W21-5 ablation completes.*

> **STATUS**: TEMPLATE. Receipt sections (W21-1 result, W21-5 ablation
> verdict) will be filled when the experiments complete. The framework,
> Reading-table, and bug catalog reflect the WAVE 21 state as of the
> 14-of-14 cross-checks closure.

## Wave summary

W21 closes the W7→W21 cross-validation chain by:
1. **W21-1** experimentally testing the Reading-F (Δ_S stability
   weighting) hypothesis identified in W20-5 + INVERTED per
   docs/W20-5-CORRECTION-READING-F-INVERSION.md.
2. **W21-2, W21-3, W21-4** closing the final 3 of 14 operator
   cross-checks (L NDS, M mutation, N crossover).
3. **W21-5** running the full ablation matrix on 30 seeds to produce
   the FINAL REPLICATION VERDICT.
4. **W21-6** synthesizing the verdict + surfacing the publication-track
   decision.

## Verdict table

| Unit | Subject | Verdict | Headline |
|---|---|---|---|
| W21-1 | Reading-F experimental test | **TBD** | (smoke result will populate) |
| W21-2 | L (NDS algorithm) | ✅ AGREE on core | Constraint layer intentionally diverges (v2 epsilon-feasibility vs Python max-cardinality) |
| W21-3 | M (mutation operator) | ⚠️ STRUCTURAL DIVERGENCE | v2 has 4 operators (thesis 2 + entropy 2 not in thesis) |
| W21-4 | N (crossover) | ✅ HONEST CORRECTION | Both use uniform; the "C++ uses SBX" assumption was wrong |
| W21-5 | Full ablation matrix | **TBD** | (30-seed run will produce final verdict) |
| W21-6 | Synthesis + publication decision | **THIS** | — |

## Reading framework — FINAL after W21

| Reading | Status |
|---|---|
| A wrong metric | RULED OUT (Ŝ matches paper Table 7.2 form) |
| B replication failure (random) | UNLIKELY (multiple AGREE checks; deterministic operators agree) |
| C structural data property | PARTIALLY HOLDS (TIP saturation is real) |
| D KF production state-evolution divergence | W21-5 ablation tests this |
| **E** anticipative-rate formula divergence | **PROVEN W20-1: +7.82pp recovery** |
| **F (INVERTED W20-5)** Δ_S stability-weighting | **TBD** — W21-1 smoke + W21-5 ablation |

## Bug catalog — FINAL

| # | Side | Where | Status after W21 |
|---|---|---|---|
| 1 | v1 (legacy-cpp) | autocorr comma-op | Off-headline. Bug confirmed. |
| 2 | Python | compute_risk adds sqrt() not in thesis Eq 7.4 | On-headline. W18-CARRY-1. W21-5 ablation tests removal. |
| 3 | Python | KF production state-evolution divergence (update-only-per-generation vs combined update→predict per period) | On-headline. Reading D. W21-5 ablation tests alignment. |
| 4 | Python | Eq 7.16 anticipative rate collapses vs v2 monotonic | **PROVEN W20-1: +7.82pp recovery via use_v2_anticipative_rate flag** |
| 5 | Python | Stability multiplier depresses Δ_S vs v2's no-op | INVERTED per W20-5 correction. W21-1 tests via use_v2_stability_weighting flag (forces stability=1.0). |
| 6 | v2 (legacy-cpp-v2) | Stability field declared but never reassigned (dead code) | Off-headline finding. Mutual-skepticism caveat. |
| 7 | v2 | 2 entropy mutation operators (raise/lower) not in thesis | Off-headline finding. W21-3. May contribute to v2's paper-headline behavior. |
| 8 | docs (audit chain) | "C++ uses SBX" propagated through W18-W20 without verification | DOCUMENTATION SCAR. W21-4 corrected: both use uniform. |
| 9 | docs (audit chain) | W20-5 fabricated portfolio.cpp:595 stability formula | DOCUMENTATION SCAR. W20-5-CORRECTION-READING-F-INVERSION.md transparent disclosure. |

## Cumulative gap-closure receipt (W13 → W21)

| Phase | Δ(S2 − S0) | Δ vs prior | What changed |
|---|---|---|---|
| W14-2 baseline | -24.86% | — | initial walk-forward |
| W15-5 BLOCKERS | -8.75% | +16.11pp | data + KF + dominance fixes |
| W16-5 algo fixes | -5.53% | +3.22pp | λ^K + txn + extrema |
| W17-5 clean data | -8.72% | -3.19pp | 87-asset thesis-faithful filter |
| W17-5-CARRY-1 RESMOKE | -11.79% | -3.07pp | Eq 7.16 activated (over-corrected) |
| **W20-1 v2-rate** | **-4.04%** | **+7.75pp** | Reading E (use_v2_anticipative_rate flag) |
| **W21-1 v2-rate + v2-stab** | **TBD** | **TBD** | Reading F (use_v2_stability_weighting flag) |
| **W21-5 final ablation** | **TBD** | **TBD** | Final replication verdict |

Cumulative since W14-2 baseline (as of W20-1): 83.7% gap closure.
Target after W21: 100% (verdict = paper replicates) OR documented
residual hypothesis (verdict = partial or refuted).

## 14 of 14 operator cross-checks — FINAL

| # | Wave | Check | Verdict | Source receipt |
|---|---|---|---|---|
| A | W18-2 | Risk | Python adds sqrt() not in thesis Eq 7.4 (W18-CARRY-1) | CROSS-VALIDATION-A-RISK.md |
| B | W18-3 | ROI | AGREE 1e-12 | CROSS-VALIDATION-B-ROI.md |
| F | W18-4 | TIP | STRUCTURAL saturation (Reading C) | CROSS-VALIDATION-F-TIP.md |
| C | W19-1 | Bivariate Gaussian inputs | AGREE machine precision | CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md |
| D | W19-2 | KF Gaussians (predictive) | DISAGREE vs v2 (lifecycle reversed) | CROSS-VALIDATION-D-KF-GAUSSIANS.md |
| E | W19-3 | Dirichlet filter | AGREE machine precision | (re-verified in CROSS-VALIDATION-V2-REASSESSMENT.md) |
| G | W19-4 | Anticipative rate end-to-end | THREE-WAY formula divergence | CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md |
| H | W20-2 | Dirichlet variant | DUPLICATE of E | CROSS-VALIDATION-H-DIRICHLET-VARIANT.md |
| I | W20-3 | OAL anticipative distributions | STRUCTURAL DIVERGENCE | CROSS-VALIDATION-I-ANTICIPATIVE-DISTRIBUTIONS.md |
| J | W20-4 | Crowding distance | NUMERICAL DIVERGENCE; same ranking | CROSS-VALIDATION-J-CROWDING-DISTANCE.md |
| K | W20-5 | Expected HV / Δ_S | STRUCTURAL (INVERTED W20-5-CORRECTION) | CROSS-VALIDATION-K-EXPECTED-HV.md + W20-5-CORRECTION-READING-F-INVERSION.md |
| L | W21-2 | NDS algorithm | AGREE on core | CROSS-VALIDATION-L-NDS.md |
| M | W21-3 | Mutation operator | STRUCTURAL DIVERGENCE (v2 has 2 extra) | CROSS-VALIDATION-M-MUTATION.md |
| N | W21-4 | SBX vs uniform crossover | DOCUMENTATION SCAR CORRECTED; both uniform | CROSS-VALIDATION-N-CROSSOVER.md |

**ALL 14 CHECKS CLOSED.** The 14-check audit chain is structurally
complete.

## Publication-track decision menu (for operator)

Conditional on W21-5 verdict:

### If PAPER-REPLICATES (some V_k > V0 with 95% CI)
> "This Python re-implementation of Azevedo & Von Zuben's (2015) ASMS-EMOA
> for portfolio anticipatory learning replicates the paper's headline
> result of ASMS > SMS after closing a documented chain of N divergences
> between the thesis (faithful Python port) and the paper-companion C++
> release. The dominant divergence was Eq 7.16's `(1/2)(λ^H + λ^K)`
> anticipative-rate factor, which collapses to 0 in the saturation
> regime (TIP ≈ 0.5) that dominates the FTSE 2006-2012 data; the paper
> appears to have used the simpler monotonic `α = 1 - TIP` formula
> that maintains anticipation at 0.5 in saturation. [Additional
> divergences closed: ...]. Methodological contribution: 14-check
> cross-validation against the paper-companion v2 substrate documents
> the full receipt."

### If PARTIAL (≥ 80% gap closure but still V_k ≤ V0)
> "We document a chain of N divergences between the thesis Python port
> and the v2 paper-companion C++ release. Closing these divergences
> recovers ≥ 80% of the W17-5 saturation gap (from −X% to −Y%) but
> the paper's headline ASMS > SMS direction is NOT fully reversed in
> our 30-seed walk-forward replication. Possible residual hypotheses
> include [...]. Methodological contribution: 14-check cross-validation."

### If REFUTED (none of the variants close meaningful gap)
> "We attempted to replicate the IEEE TCYB 2015 paper's headline
> ASMS > SMS result by porting the thesis algorithm to Python and
> cross-validating against the v2 paper-companion C++ release at
> 14 operator-defined checkpoints. We document N divergences in the
> codebases. Closing any combination of these divergences in a 30-seed
> walk-forward evaluation does NOT reproduce the paper's headline
> direction. The paper's specific seed/run/parameter setup may not be
> reconstructible from the public artifacts. Methodological contribution:
> the 14-check audit + receipt + final ablation matrix is itself a
> meaningful contribution to reproducibility-as-a-discipline literature."

## Strategic position

After W21:
- 14 of 14 operator cross-checks complete
- 9 documented divergences (5 on-headline Python, 1 off-headline v1 C++, 2 off-headline v2 C++, 2 documentation scars from audit chain itself)
- 5 readings (A through F) addressed, with E PROVEN and F TBD pending W21-1 smoke
- Ablation matrix protocol defined (W21-5)
- Publication-track framings drafted (3 options ready for operator)

## Next steps (post-W21)

- W21-5 implementation phase: V5 (sqrt removal flag), V6 (KF lifecycle alignment), V7 (entropy operators) require additional code work
- W21-5 Phase C run: 30-seed sweep with full V0..V6 (or V7) matrix; ~5-7 hours wall-clock
- W21-6 final synthesis + publication-track decision
- Post-W21: publication writing (operator decides framing); thesis re-read (W19-4-CARRY-2) for (1/2) factor semantics

## Output artifacts (when complete)

- `docs/READING-F-EXPERIMENTAL-TEST.md` (W21-1 receipt; smoke result)
- `docs/W21-5-FULL-ABLATION-VERDICT.md` (final ablation table)
- `docs/W21-CROSS-VALIDATION-SYNTHESIS.md` (this, final fill)
- `docs/PUBLICATION-TRACK-DECISION.md` (W21-6 surface to operator)
- `.dfg/retrospectives/W21/W21-{1..6}.md` (W21-2/3/4 already written)
- `.dfg/checkpoints/W21-gate.md` (wave-close ceremony artifact)

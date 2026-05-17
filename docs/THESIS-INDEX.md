# Thesis index — Azevedo (PhD, UNICAMP 2014)

*Authored 2026-05-17 during W13 (mean OOS future hypervolume) to
formally anchor the codebase on the originating PhD thesis. The IEEE
TCYB 2015 paper (`docs/paper.pdf`) is the condensed version; the
thesis is the canonical long-form source for derivations, future-work
suggestions, broader theory, and the **out-of-sample future
hypervolume** evaluation protocol that the W13 wave operationalizes.*

---

## §0 Bibliographic citation

**Carlos Renato Belo Azevedo. _Anticipation in Multiple Criteria
Decision-Making Under Uncertainty_ (Portuguese title: _Antecipação na
Tomada de Decisão com Múltiplos Critérios sob Incerteza_). Doctoral
thesis, School of Electrical and Computer Engineering, University of
Campinas (UNICAMP), 2014. Advisor: Prof. Dr. Fernando José Von Zuben.
~219 pages.**

- DOI: [10.47749/T/UNICAMP.2012.938003](https://doi.org/10.47749/T/UNICAMP.2012.938003)
- Local copy: [`docs/Azevedo_CarlosRenatoBelo_D.pdf`](Azevedo_CarlosRenatoBelo_D.pdf) (17.8 MB)

**Relation to the IEEE paper.** The IEEE TCYB 2015 paper at
`docs/paper.pdf` (DOI [10.1109/TCYB.2015.2415732](https://doi.org/10.1109/TCYB.2015.2415732))
condenses chapters 5–7 of this thesis. **When the two disagree, the
thesis wins** — it has the unabridged derivations and the explicit
experimental protocol.

**Companion C++ codebases (added 2026-05-17 during W18 substrate update).**
TWO C++ implementations are vendored:

| Path | Era | Status |
|---|---|---|
| `../legacy-cpp/` | 2013, thesis-companion (pre-paper) | Historical provenance only; **NOT cross-validation oracle** |
| `../legacy-cpp-v2/` | 2015, paper-companion release (GitHub `crbazevedo/anticipatory-learning-asmoo` @ `6643c92`) | **Cross-validation oracle for the Python port** |

The Python port (`../python_refactor/`) was built against the 2015 v2
code, which adds the explicit `dirichlet.cpp` module that the 2013
version lacks. W18 cross-validation work initially used `legacy-cpp/`
and surfaced false-positive structural divergences; PR (W18-substrate-import)
imported `legacy-cpp-v2/` as the correct reference. See
`../legacy-cpp-v2/README.md` for the full diff.

---

## §1 Authoritative chapter map (for codebase work)

| Codebase concern | Thesis chapter | Pages |
|---|---|---|
| Foundational theory: uncertainty, MAP, KF, Dirichlet priors | §2 Representing, Measuring, and Handling Uncertainty | 27–53 |
| MOO foundations: hypervolume, S-metric, flexibility, μ-distributions | §3 Multi-Objective Optimization Under Uncertainty | 55–74 |
| Markowitz baseline + preemptive trade-off + cardinality + regularization | §4 Preemptive Multi-Objective Strategies for Active Asset Allocation | 75–94 |
| **The AS-MOO model itself** (formal definition + time-linkage) | §5 Hypervolume-Based Anticipatory MCDM | 97–113 |
| **Online Anticipatory Learning (OAL) rule + ASMS-EMOA algorithm** | §6 Learning to Anticipate Flexible Trade-off Choices | 115–137 |
| **Out-of-Sample Future Average Hypervolume — the validation protocol** | §7.2.2 + §7.3.3 | 143–146, 158–168 |
| Case study experimental design + 4 algorithmic variants | §7.1 + §7.2.3 | 139–141, 145–148 |
| Suggested future research directions | §8.3 | 192–197 |

---

## §2 The Out-of-Sample Future Average Hypervolume protocol (the W13 anchor)

**Source:** thesis §7.2.2 (Experimental Methodology) and §7.3.3
(Effects on the Out-Of-Sample Future Average Hypervolume).

### §2.1 Definition

For each rolling investment period `t` (T=25 total, 24 for FTSE):

**Step 4 — Eq (7.10):** Apply the **out-of-sample future test state
parameters** χ_{t+1} (post-hoc only; never enters the optimizer; see
footnote 2 on p. 144) to evaluate the trained Pareto-flexible set
Û_t^{N*}:

```
ẑ_{t+1} = f(Û_t^{N*}, χ_{t+1}, m̂_{u*_t})
```

**Eq (7.11):** Compute the sample-average future hypervolume:

```
Ŝ_{t+1} = (1/E) Σ_{e=1}^E S(ẑ_{e,t+1}, z_ref)
```

with:

| Symbol | Value / meaning |
|---|---|
| `Û_t^{N*}` | trained Pareto-flexible set at period t (N=20 portfolios) |
| `χ_{t+1}` | (μ_{t+1}, Σ_{t+1}) MLE-fit Gaussian on the next 50-day held-out block |
| `m̂_{u*_t}` | mean portfolio implemented (mHDM) — the maximum-Hypv decision from Û_t^{N*} |
| `E` | **1000** Monte-Carlo simulated daily return scenarios per portfolio |
| `S(·, z_ref)` | S-metric (hypervolume indicator) against the reference point |
| `z_ref` | **(0.2, 0.0)^T** — risk_max=0.2, return_min=0.0 (§7.2.3 ASMS Parameters) |

Average across T periods and 30 random seeds → grand-mean OOS future
average hypervolume per scenario × dataset.

### §2.2 Critical distinction from in-sample EFHV

The thesis explicitly separates two metrics:

1. **In-sample expected future hypervolume** (the algorithm's internal
   prediction). Monte-Carlo MC over the Pareto front's current Kalman-state
   projection. *Computed inside* `sms_emoa._compute_expected_future_hypervolume`
   and used as a tiebreaker within the optimizer (§6 OAL).

2. **Out-of-sample future average hypervolume `Ŝ_{t+1}`** (Eqs 7.10–7.11).
   *Post-hoc* evaluation against the future-period MLE-fit Gaussian.
   Never enters the optimizer. This is the paper's **headline metric**
   for the four-way ANOVA in Table 7.2.

**The C++ legacy ships ONLY in-sample HV** (per the W13 sub-agent
audit). The OOS protocol above is exclusively thesis-grounded;
the W13-2 Python evaluator implements it from first principles.

### §2.3 Headline finding (§7.3.3, paraphrased)

The hypothesis is that ASMS variants achieve higher `Ŝ_{t+1}` than
myopic SMS, AND that mHDM (implemented-maximum-Hypv decision) beats
RDM (random decision from the Pareto-flexible set). Reported (Table 7.2,
p. 159):

- **DM factor** statistically significant on `Ŝ_{t+1}` in 6/7 artificial
  scenarios (85%) and 4/12 real-world test cases.
- **Anticipation factor** (window size K > 0 vs K = 0) significant in
  5/7 artificial scenarios (71%).
- **15 out of 20** (instance × DM) combinations show ≥ 1 anticipatory
  variant outperforming its myopic counterpart for some K > 0 (70%).
- Real-world ANOVA: DM significant on DJI; both factors near-zero
  effect on HSI; FTSE intermediate (Fig 7.15h–j).

### §2.4 The four algorithmic variants (§7.1.1)

| Variant | Anticipation (Factor 1) | DM (Factor 2) | Maps to validation_matrix SCENARIO |
|---|---|---|---|
| **ASMS/mHDM** | ON (K > 0) | mHDM | **S2 (paper headline)** |
| ASMS/RDM | ON (K > 0) | random from Pareto-flexible | S1 (TIP only — partial match; S1 currently uses H=2 not K controller) |
| SMS/mHDM | OFF (K = 0) | mHDM | (no current scenario) |
| **SMS/RDM** | OFF (K = 0) | random | **S0 (myopic baseline)** |

Note: the current `validation_matrix.SCENARIOS` dict is loosely aligned
to these four variants. A future unit may re-key SCENARIOS to match
the canonical {ASMS,SMS} × {mHDM,RDM} factorial precisely.

---

## §3 Datasets (§7.2.3)

| Dataset | d (assets) | Periods T | Window |
|---|---|---|---|
| Static mean PO + 6 nonstationary PO variants {2,4,8} × {0.5,1.0} | 30 (artificial) | 25 | 50-day lagged returns rolling |
| **FTSE-100** | **87** | **24** | 20/11/2006–31/12/2012 |
| Dow Jones (DJI) | 30 | 25 | same |
| Hang Seng (HSI) | 49 | 25 | same |

**Disagreement note**: thesis §7.2.3 says FTSE has d=87 assets; the
legacy C++ dataset at `legacy-cpp/executable/data/ftse-original/`
ships 98 CSVs. Likely either (a) data-vendor delta over time, or (b)
the 98 includes 11 that didn't survive the FTSE-100 composition during
the experimental window. Reconcile in a future unit if it becomes
load-bearing.

---

## §4 ASMS parameters (§7.2.3)

| Parameter | Value |
|---|---|
| Population size N | 20 |
| Generations per period | 30 |
| Anticipation horizon H | 2 (one-step-ahead) |
| Window size K (anticipation factor) | {0, 1, 2, 3} (0 = myopic SMS) |
| Cardinality constraint | c_l = 5, c_u = 15 |
| Mutation rate | 0.3 |
| Crossover probability | 0.2 |
| Selection | binary tournament over Pareto dominance on mean vectors; expected Hypv contribution as tiebreaker |
| Reference point `z_ref` | (0.2, 0.0)^T (risk, return) |
| MC sample size E | 1000 |
| OAL rate λ_{t+h} | ½(λ_{t+h}^H + λ_{t+h}^K) — Eq (7.16) |
| Initial wealth | 10,000.00 |
| Transaction cost h | Brazilian Securities Commission table (§7.2.3 Tab 7.1) |

---

## §5 Auxiliary metrics defined alongside `Ŝ_{t+1}`

- **POCID (§7.2.2, Eq 7.13):** Percentage Of Change In Direction.
  Counts how often the KF prediction agrees in DIRECTION with the
  next-period realized objective change. Used to assess Bayesian-
  tracking quality (§7.3.2).
- **Coherence (Eq 7.14):** mean cosine similarity of each DD portfolio
  to the population centroid in the (d−1)-simplex; assesses search-
  space convergence over rolling periods (§7.3.4).
- **Cardinality:** number of non-zero asset weights per portfolio,
  bounded by [c_l, c_u]. Reported alongside `Ŝ_{t+1}` to show how
  portfolio sparsity correlates with future flexibility (§7.3.4).
- **Accumulated Wealth `W_t` (Eq 7.12):** mean observed return for
  implemented portfolio minus transaction costs of rebalancing.
  Reported as a sanity-check that hypervolume gains translate to
  realized wealth (§7.3.5).

The Pearson correlation table (Tab 7.3, p. 170) shows `Ŝ_{t+1}` is
**positively correlated with coherence** (8/10 instances significant)
and **negatively correlated with cardinality** (5/10 significant) —
fewer, more-coherent portfolios tend to preserve more future
flexibility.

---

## §6 Open research directions (from §8.3 — "Suggestions of Further Research on AS-MOO Solvers")

| § | Title | One-line summary |
|---|---|---|
| 8.3.1 | Anticipating Future Preferences | Allow the DM's preferences themselves to drift; treat them as a tracked stochastic variable |
| 8.3.2 | Regret Bounds for Online Hypervolume Maximization | Theoretical regret-minimization analysis of the OAL update rule |
| 8.3.3 | Anticipatory Anomaly Prevention | Apply AS-MOO to early-warning systems (anomaly trajectories as future Pareto frontiers) |
| 8.3.4 | Anticipatory Resource Allocation | Multi-resource scheduling under stochastic demand with future-flexibility maximization |
| 8.3.5 | Anticipatory Hyper-Heuristics | Online selection over a pool of MOO operators given anticipatory state |
| 8.3.6 | Automation of Flexible and Sustainable Planning Systems | Generalize the methodology beyond finance: planning under sustainability constraints |
| 8.3.7 | Sustainable Planning: Models, Measurements, and Concepts | A research-program statement positioning AS-MOO inside sustainability science |

These titles are useful prompts for future-wave scoping when the
core validation track (W13 → W14 → W15) lands and the codebase is
ready for extension experiments. Carlos's `Personal/Professional/IP/`
knowledge graph may already overlap with several of these — worth
cross-checking on the next research-strategy turn.

---

## §7 Key supplementary references (excerpted from thesis references list, pp. 198+)

This is a curated subset relevant to the codebase, not the full ~250
references. For the complete list, see thesis pp. 198–217.

### §7.1 Algorithmic backbone

- **[Beume et al. 2007]** N. Beume, B. Naujoks, M. Emmerich. *SMS-EMOA:
  Multiobjective selection based on dominated hypervolume.* European
  Journal of Operational Research, 181(3):1653–1669. — the SMS-EMOA
  base algorithm.
- **[Deb et al. 2002]** *A fast and elitist multiobjective genetic
  algorithm: NSGA-II.* IEEE TEVC 6(2):182–197. — the alternative
  base; thesis §3.4.1.
- **[Kalman 1960]** *A new approach to linear filtering and prediction
  problems.* J. Basic Eng. 82(1):35–45. — KF foundation.

### §7.2 MOO under uncertainty + flexibility

- **[Hughes 2001]** *Evolutionary multi-objective ranking with
  uncertainty and noise.* EMO 2001. — predecessor for stochastic
  ranking.
- **[Branke & Mattfeld 2005]** *Anticipation and flexibility in dynamic
  scheduling.* IJPR 43(15):3103–3129. — the original "flexibility as a
  proxy for unknown future preferences" framing the thesis builds on.
- **[Auger et al. 2009]** *Theory of the hypervolume indicator:
  Optimal μ-distributions and the choice of the reference point.* FOGA.
  — theoretical grounding for the choice of z_ref.

### §7.3 Anticipation in biology + cognition (motivation pool, §1.5)

- **[Rosen 1985]** *Anticipatory Systems.* Pergamon. — the foundational
  text on anticipatory systems theory.
- **[Butz et al. 2003]** *Anticipatory Behavior in Adaptive Learning
  Systems.* Springer. — connects anticipation to RL.
- **[Friston 2010]** *The free-energy principle.* Nature Reviews
  Neuroscience 11(2):127–138. — anticipation as built-in feature of
  the brain (thesis §1.5.3).

### §7.4 Financial application backbone

- **[Markowitz 1952]** *Portfolio selection.* J. Finance 7(1):77–91. —
  the MVP problem the thesis lifts to multi-period stochastic.
- **[Sharpe 1994]** *The Sharpe ratio.* J. Portfolio Mgmt 21(1):49–58.
- **[Anagnostopoulos & Mamanis 2010]** *A portfolio optimization
  approach using NSGA-II.* — multi-objective portfolio precedent the
  thesis explicitly extends.

---

## §8 How to use this index

When opening a new unit (W13+) that touches the algorithmic core, the
optimizer behavior, the validation protocol, or the experimental
methodology:

1. **Open the thesis first** — at the chapter pointed to by §1.
2. **Cite the exact equation number** in the contract / retro
   (e.g., "thesis Eq 7.11" for OOS HV computation).
3. **Quote the parameter values** (e.g., E=1000, z_ref=(0.2, 0.0))
   to avoid drift across units.
4. **Cross-check against the IEEE paper** at `docs/paper.pdf` —
   if they disagree, thesis wins (§0 note).
5. **If the unit prompts a follow-up research direction**, note it
   against §6 above so we don't lose the lead.

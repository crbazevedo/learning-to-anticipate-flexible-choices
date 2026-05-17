---
id: W15-3
role: code-fixer
name: SCENARIOS K-aware re-key + date range + H=2 (BACKLOG B2+H3+H5)
purpose: "Closes BACKLOG B2 (thread K∈{0,1,2,3}) + H3 (paper-window date range 2006-2012) + H5 (H=2 fixed across anticipatory variants) per thesis §7.1.1 + §7.2.3 + Eq 7.16."
wave: W15
unit: W15-3
depends_on: []
blocks: [W15-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - path: docs/BACKLOG.md
      sections: ["§1.1 B2 — K (OAL window-size) not threaded", "§1.2 H3 — Date range mismatch", "§1.2 H5 — H per scenario varies"]
      reason: "Catalog entries — read FIRST; grounded specs"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [140]
      sections: ["§7.1.1 Investigated Algorithmic Variants"]
      excerpt: "The anticipation factor is controlled by four levels of window size (K): K ∈ {0, 1, 2, 3}. For K = 0, we have the myopic baseline SMS-EMOA (SMS, in short) for which constant predictions are made... For K > 0, the past solutions and objective distributions obtained in the latest K periods serve as input to the KF and DD tracking methods, in which case anticipation is enabled."
      reason: "B2 grounding — K is the dial we're not turning"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [146]
      sections: ["§7.2.3 Bayesian Tracking Parameters + Eq (7.16)"]
      excerpt: "λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K)), for which the anticipation horizon is H = 2 (one-step-ahead prediction)."
      reason: "H5 grounding (H=2 fixed) + B2 (λ^K is the K-arm)"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [145, 146]
      sections: ["§7.2.3 Artificial and Real-World Datasets"]
      excerpt: "The real-world scenarios are composed of daily adjusted close prices collected between 20/11/2006 – 31/12/2012... Each investment period comprises one and a half year worth of daily returns data. Between each period, the 50 oldest lagged returns are discarded and the 50 latest ones are included... Thus, the period t = 1 in FTSE comprises data from 20/11/2006 – 20/05/2008, t = 2 to 01/02/2007 – 30/07/2008, and so forth."
      reason: "H3 grounding — date range 2006-2012, NOT 2003-2012"
    - path: python_refactor/experiments/validation_matrix.py
      reason: "Module being edited — SCENARIOS + WINDOWS dicts"
    - path: python_refactor/src/algorithms/anticipatory_learning.py
      reason: "Receives K from SCENARIOS; need to thread into compute_anticipatory_learning_rate"
output_contract:
  files:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_scenarios_k_aware.py
  branch_name: feat/w15-3-scenarios-k-aware-date-range
  acceptance: >
    SCENARIOS factorial: {ASMS,SMS} × {mHDM,RDM} × K∈{0,1,2,3} per
    thesis Fig 7.15. K threaded through to AnticipatoryLearning
    constructor + consumed (or honestly scarred if structural blocker).
    WINDOWS paper-window date_range sliced to 2006-11-20 → 2012-12-31
    per thesis §7.2.3 p.145. H=2 fixed across all anticipatory
    variants per thesis Eq 7.16. ≥ 4 regression tests.
dispatch_instructions: |
  Closes BACKLOG: B2 + H3 + H5.

  SCENARIOS re-key (Fig 7.15 framing):
    SMS_RDM_K0       (= baseline myopic; K=0 implies no anticipation)
    ASMS_mHDM_K1     (anticipation enabled, max-Hypv DM, K=1)
    ASMS_mHDM_K2
    ASMS_mHDM_K3     (the paper headline configuration)
    ASMS_RDM_K1      (anticipation, random DM)
    ASMS_RDM_K2
    ASMS_RDM_K3
    SMS_mHDM_K0      (myopic, max-Hypv DM — DM-only factor)

  Keep legacy S0..S4 aliases for backward compat (alias dict).

  Pass K via 'window_size' to AnticipatoryLearning constructor.
  Verify the constructor accepts window_size (it does, per
  experiment_manager.py line ~253 in W1-2). Verify it's CONSUMED
  in compute_anticipatory_learning_rate (W13-CARRY-1 suspected
  not — if so, file as W15-3-CARRY-1 and ship the threading
  half; the consumption fix is a follow-on).

  WINDOWS update:
    paper: date_start='2006-11-20', date_end='2012-12-31'
    (asset_files_glob unchanged — still legacy-cpp/.../table*.csv)

  H=2 fixed: remove max_horizon variation from SCENARIOS;
  AnticipatoryLearning(window_size=K, max_horizon=2,
  monte_carlo_samples=500).

  What NOT to do:
    - Don't touch sms_emoa.py (W15-1, W15-2).
    - Don't touch operators.py (W15-2).
    - Don't ship analytics (W18).

  PR body MUST echo the thesis excerpts per BACKLOG §6.
---

# W15-3 — SCENARIOS K-aware + date range + H=2 (BACKLOG B2+H3+H5)

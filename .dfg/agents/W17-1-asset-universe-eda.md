---
id: W17-1
role: experimenter
name: H4 asset universe EDA + continuous-trades filter (87-asset subset)
purpose: "Closes BACKLOG H4. EDA of 98-asset FTSE legacy-cpp universe over 2006-2012; identify and drop assets without continuous daily trades; document the resulting 87-asset thesis-faithful subset."
wave: W17
unit: W17-1
depends_on: []
blocks: [W17-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/src/experiments/data_loader.py
output_contract:
  files:
    - docs/H4-ASSET-UNIVERSE-EDA.md
    - python_refactor/src/experiments/data_loader.py
    - python_refactor/experiments/results/h4-eda/asset_universe_87.json
    - python_refactor/tests/test_asset_universe_filter.py
  branch_name: feat/w17-1-asset-universe-eda
  acceptance: >
    EDA doc receipt at docs/H4-ASSET-UNIVERSE-EDA.md with: (1) distribution
    of trading-coverage per asset over the paper window 2006-11-20→
    2012-12-31, (2) list of assets with discontinuous trades (zero closing
    prices / missing days / dissolution events), (3) the resulting
    thesis-faithful 87-asset subset (or honest scar if filter yields a
    different N). Asset filter in data_loader gated by opt-in flag
    `enforce_thesis_continuous_trades`. ≥ 3 regression tests covering
    the filter + the asset count + the persistence of the asset list.
dispatch_instructions: |
  Closes BACKLOG: H4.

  EDA scope (the operator's verbatim spec):
    "keep only the assets which had continuous daily closing prices
     trades throughout the full dataset period"

  Surgical workflow:

  1. EDA notebook (could be plain Python script) at
     python_refactor/scripts/h4_asset_universe_eda.py:
       a. Load all 98 CSVs from legacy-cpp/executable/data/ftse-original/
       b. For each asset, restrict to the paper window 2006-11-20 →
          2012-12-31
       c. Compute trading-coverage % per asset (non-NaN close + > 0 close)
       d. Flag assets with: (i) gaps > N business days, (ii) zero/NaN
          mid-window, (iii) dissolution (zero from a date onwards)
       e. Output: trading-coverage histogram + per-asset summary CSV
       f. Apply the continuous-trades filter; record the resulting
          subset (target N≈87 per thesis §7.2.3 p.145)

  2. Result-of-truth artifact:
        python_refactor/experiments/results/h4-eda/asset_universe_87.json
     Schema:
        {
          "criterion": "continuous_daily_close_prices_2006_11_20_to_2012_12_31",
          "n_original": 98,
          "n_filtered": 87,  // or honest number
          "dropped_assets": [{"file": "table (X).csv", "reason": "...",
                              "first_zero_date": "..."}],
          "kept_assets": ["table (0).csv", ...],
          "window_start": "2006-11-20",
          "window_end": "2012-12-31"
        }

  3. data_loader filter: add `enforce_thesis_continuous_trades: bool=False`
     kwarg to the loader (or equivalent surface). When True, load only
     the kept_assets per the JSON. Default False to preserve current
     behavior (98 assets); SCENARIOS / WINDOWS / experiment_config flip
     to True per-scenario as appropriate.

  4. EDA receipt at docs/H4-ASSET-UNIVERSE-EDA.md:
       - Methodology + criterion (verbatim)
       - Trading-coverage distribution
       - Per-asset table for the dropped assets (with reasons)
       - Honest scar if N≠87 (could be 86, 89, etc.; explain)
       - Thesis cross-reference: §7.2.3 p.145 "d = 87 for FTSE"
       - Forward link: W17-5 smoke uses this subset

  5. Tests at python_refactor/tests/test_asset_universe_filter.py:
       - asset_universe_87.json exists + has correct schema
       - data_loader honors enforce_thesis_continuous_trades=True
         (loads only kept_assets)
       - Default behavior unchanged when flag is False (still 98 assets)

  What NOT to do:
    - Don't change SCENARIOS to default to 87-asset (defer to W17-5
      smoke; opt-in for backward compat).
    - Don't add data imputation for the gaps (drop is faithful to
      thesis criterion).
    - Don't ship analytics figures (W18).
    - Don't touch sms_emoa.py / operators / etc. — this is data-layer.

  PR body MUST echo thesis §7.2.3 p.145 "d = 87 for FTSE" verbatim
  per BACKLOG §6.
---

# W17-1 — H4 asset universe EDA + continuous-trades filter

Closes BACKLOG.md items: **H4**.

## Thesis grounding

**§7.2.3 "Artificial and Real-World Datasets", p. 145** — verbatim:
> "All benchmarks provide d = 30 simulated assets for composing the
>  portfolios, whereas for the real-world instances we have **d = 87
>  for FTSE**; d = 30 for DJI; and d = 49 for HSI, which are
>  represented in a (d − 1)–Simplex space."

> "The real-world scenarios are composed of daily adjusted close prices
>  collected between 20/11/2006 – 31/12/2012, from which 50 days lagged
>  returns were computed..."

**Footnote 3 (p. 145)**: "The data used in our experiments is publicly
available at..." [URL truncated in thesis PDF; needs follow-up]

## Operator criterion (verbatim, from session)

> "keep only the assets which had continuous daily closing prices
>  trades throughout the full dataset period"

## Why this is the W17 keystone

W16-5 smoke result: Δ(S2 − S0) = -5.53% after closing all W15
BLOCKERS + W16 algorithm fixes (77.7% cumulative closure since
W14-2 baseline). The remaining 5.53pp gap most likely traces to
data-pollution: legacy-cpp/ftse-original has 98 CSVs but thesis
specifies d=87. The 11 extras may include delisted assets that
suddenly go to zero mid-window — exactly the pollution that
breaks Kalman/Dirichlet estimators and corrupts the anticipatory
chain.

## Files to touch

- `python_refactor/scripts/h4_asset_universe_eda.py` — NEW EDA script
- `python_refactor/src/experiments/data_loader.py` — add filter flag
- `python_refactor/experiments/results/h4-eda/asset_universe_87.json` —
  result-of-truth artifact
- `docs/H4-ASSET-UNIVERSE-EDA.md` — EDA receipt
- `python_refactor/tests/test_asset_universe_filter.py` — NEW; ≥ 3 tests

## Acceptance

- EDA doc with trading-coverage distribution + dropped-asset table
- asset_universe_87.json artifact with the kept subset
- data_loader honors `enforce_thesis_continuous_trades` flag
- ≥ 3 regression tests
- Honest scar if N≠87 (e.g., 86 or 89) with explanation

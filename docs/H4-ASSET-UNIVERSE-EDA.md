# BACKLOG H4 closure — FTSE asset universe EDA (W17-1)

*Authored 2026-05-17. Closes BACKLOG H4.*

## Verdict

✅ **EXACT MATCH** — the 98-asset legacy-cpp/ftse-original universe filters to **exactly 87 thesis-faithful assets** under the operator's continuous-trades criterion in the paper window 2006-11-20 → 2012-12-31.

11 assets dropped: 10 listed mid-window (first trade > 5 business days after 2006-11-20), 1 with a zero closing price (dissolution event).

## Thesis grounding

**§7.2.3 "Artificial and Real-World Datasets", p. 145** — verbatim:
> "All benchmarks provide d = 30 simulated assets for composing the
>  portfolios, whereas for the real-world instances we have **d = 87
>  for FTSE**; d = 30 for DJI; and d = 49 for HSI, which are
>  represented in a (d − 1)–Simplex space."

> "The real-world scenarios are composed of daily adjusted close prices
>  collected between 20/11/2006 – 31/12/2012, from which 50 days lagged
>  returns were computed..."

## Operator criterion (verbatim, from session)

> "keep only the assets which had continuous daily closing prices
>  trades throughout the full dataset period"

## Operationalization

A FTSE asset CSV passes the continuous-trades filter iff (in the
paper window 2006-11-20 → 2012-12-31):

1. **No zero closes** (`Close > 0` throughout in-window rows)
2. **No NaN closes**
3. **No negative closes**
4. **No dissolution event** (no tail run of zero/NaN closes — would
   indicate the asset was delisted mid-window)
5. **First in-window trade within 5 business days of window start**
   (i.e., the asset was already publicly traded at 2006-11-20;
   excludes assets that listed mid-period)

The 5-business-day tolerance accounts for assets first trading on
or shortly after the window start, including bank-holiday slack.

## EDA results

### In-window row count distribution

```
count      98.000000
mean     1445.948980
std       263.046245
min       256.000000
25%      1504.000000
50%      1521.000000
75%      1527.500000
max      1558.000000
```

The distribution is bimodal — 87 assets in the "full window" cluster
(≥ 1500 rows) and 11 assets with much shorter in-window presence
(min 256 rows = ~1 trading year).

### Histogram (n_rows bins)

| Range | Count |
|---|---|
| (0, 500] | 4 |
| (500, 1000] | 1 |
| (1000, 1400] | 6 |
| (1400, 1500] | 0 |
| (1500, 1525] | 61 |
| (1525, 1560] | 26 |

The gap at (1400, 1500] is the natural cut — there are no
"borderline" assets in the 1400-1500 row range.

### Filter result

| Metric | Value |
|---|---|
| n_original | 98 |
| n_kept | **87** |
| n_dropped | 11 |
| Thesis target (d) | **87** ✅ **EXACT MATCH** |

### Dropped assets (11)

| File | Drop reason |
|---|---|
| table (25).csv | n_zero_close=1 (dissolution event) |
| table (28).csv | listed_mid_window first_date=2007-12-07 |
| table (29).csv | listed_mid_window first_date=2011-11-07 |
| table (32).csv | listed_mid_window first_date=2008-05-09 |
| table (35).csv | listed_mid_window first_date=2011-05-19 |
| table (37).csv | listed_mid_window first_date=2007-05-15 |
| table (54).csv | listed_mid_window first_date=2007-08-16 |
| table (61).csv | listed_mid_window first_date=2011-10-28 |
| table (64).csv | listed_mid_window first_date=2007-10-25 |
| table (74).csv | listed_mid_window first_date=2008-12-05 |
| table (93).csv | listed_mid_window first_date=2011-07-01 |

These 11 assets pollute the optimization in two distinct ways:

1. **Mid-window listings** (10 of 11): zero/NaN returns for the
   PRE-listing portion of the window → Kalman estimator initialized
   on a sample where this asset is silent → covariance estimate
   for this asset is degenerate → portfolio weights on it are
   spurious. The Dirichlet predictor then projects this
   spuriousness forward, corrupting the anticipatory chain.

2. **Dissolution event** (1 of 11, table (25)): asset goes to zero
   mid-window → return becomes -100% on the day of the zero → the
   Kalman residual blows up → λ^K (per Eq 6.9) saturates →
   anticipation collapses.

Both failure modes are exactly the kind of data-quality issue that
the operator's criterion was designed to exclude, and exactly the
kind that explains why our pre-W17 smokes had S2 < S0: the
anticipatory machinery was working correctly on data it shouldn't
have been ingesting.

## Artifacts

- `python_refactor/experiments/results/h4-eda/asset_universe_87.json` —
  result-of-truth artifact (kept_assets + dropped_assets)
- `python_refactor/experiments/results/h4-eda/asset_universe_summary.csv` —
  per-asset diagnostic table
- `python_refactor/scripts/h4_asset_universe_eda.py` — reproducible
  EDA script

## Reproducing

```bash
cd python_refactor && uv run python scripts/h4_asset_universe_eda.py
```

## Data-loader integration

`python_refactor/src/experiments/data_loader.py:load_asset_data`
now accepts `enforce_thesis_continuous_trades: bool = False`. When
True, restricts the expanded asset_files glob to the 87 kept_assets
from the artifact.

Default remains False for backward compatibility — SCENARIOS /
WINDOWS / experiment_config opt in per scenario as appropriate.

W17-5 integration smoke turns the flag on for the headline
ASMS_mHDM_K3 + SMS_RDM_K0 scenarios and re-measures Δ(S2 − S0).

## Forward link

This is the W17 keystone closure. The 11 polluting assets are the
clearest remaining hypothesis for the residual -5.53% Δ(S2 − S0)
gap measured at W16-5. If the 87-asset universe flips direction in
W17-5, the paper-replication chain is structurally complete.

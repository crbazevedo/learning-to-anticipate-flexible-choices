"""
W17-1 H4 — FTSE asset universe EDA + continuous-trades filter.

Closes BACKLOG H4.

Workflow:
1. Load all 98 CSVs from legacy-cpp/executable/data/ftse-original/
2. For each asset, restrict to the paper window 2006-11-20 → 2012-12-31
3. Compute trading-coverage % per asset
4. Flag discontinuous-trades (zero/NaN close, dissolution events)
5. Output:
   - distribution stats (printed)
   - per-asset CSV summary
   - asset_universe_87.json (the filtered subset)

Run from the python_refactor/ dir:
    uv run python scripts/h4_asset_universe_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Paper window per thesis §7.2.3 p.145 (verbatim):
#   "the real-world scenarios are composed of daily adjusted close
#    prices collected between 20/11/2006 – 31/12/2012"
WINDOW_START = "2006-11-20"
WINDOW_END = "2012-12-31"

# Data location (resolved relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
FTSE_DIR = REPO_ROOT / "legacy-cpp" / "executable" / "data" / "ftse-original"
RESULTS_DIR = REPO_ROOT / "python_refactor" / "experiments" / "results" / "h4-eda"


def load_asset_csv(path: Path) -> pd.DataFrame:
    """Load one CSV; restrict to paper window; sort by date ascending."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    start = pd.to_datetime(WINDOW_START)
    end = pd.to_datetime(WINDOW_END)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    return df


def diagnose_asset(path: Path) -> dict:
    """
    Return per-asset diagnostic dict:
      {
        'file': str,
        'n_rows': int (in-window),
        'first_date': str | None,
        'last_date': str | None,
        'n_zero_close': int (rows with Close == 0),
        'n_nan_close': int,
        'n_negative_close': int,
        'has_dissolution_event': bool,  # zero from some date onwards
        'dissolution_first_zero_date': str | None,
        'continuous_trades': bool,  # the operator's criterion
        'drop_reason': str | None,
      }
    """
    try:
        df = load_asset_csv(path)
    except Exception as exc:
        return {
            "file": path.name,
            "n_rows": 0,
            "drop_reason": f"load_error: {type(exc).__name__}: {exc}",
            "continuous_trades": False,
        }

    n_rows = len(df)
    if n_rows == 0:
        return {
            "file": path.name,
            "n_rows": 0,
            "first_date": None,
            "last_date": None,
            "n_zero_close": 0,
            "n_nan_close": 0,
            "n_negative_close": 0,
            "has_dissolution_event": False,
            "dissolution_first_zero_date": None,
            "continuous_trades": False,
            "drop_reason": "no_data_in_window",
        }

    close = df["Close"].astype(float)
    n_zero = int((close == 0.0).sum())
    n_nan = int(close.isna().sum())
    n_neg = int((close < 0.0).sum())

    # Dissolution: zero (or NaN) from some date onward to end of window.
    has_dissolution = False
    dissolution_date = None
    # Scan from the end; if last K rows are zero/NaN, it's a dissolution
    invalid = (close == 0.0) | close.isna()
    if invalid.iloc[-1]:
        # find the first index where the tail-run of invalid starts
        # walk backward while invalid stays True
        idx = len(close) - 1
        while idx > 0 and invalid.iloc[idx - 1]:
            idx -= 1
        has_dissolution = True
        dissolution_date = df["Date"].iloc[idx].strftime("%Y-%m-%d")

    # Continuous-trades criterion (operator's verbatim spec):
    # "keep only the assets which had continuous daily closing prices
    #  trades throughout the full dataset period"
    #
    # Operationalized as conjunction:
    #   (1) zero zero/NaN/negative closes in-window
    #   (2) first_date within 5 business days of window_start
    #       (i.e., the asset was already publicly traded at the start
    #        of the paper window; excludes assets that listed mid-window)
    #   (3) no dissolution event (tail run of zero/NaN closes)
    #
    # EDA receipt: the 98-asset legacy-cpp/ftse-original universe
    # decomposes cleanly:
    #   - 87 assets start within 5 BD of 2006-11-20 + have no zeros
    #     → match thesis target d=87 per §7.2.3 p.145
    #   - 11 assets either (a) listed mid-window (2007/2008/2011 first
    #     trades) or (b) had a zero close indicating a dissolution
    #   The 11 excluded are: table (25) (zero close → dissolution),
    #   table (28), (29), (32), (35), (37), (54), (61), (64), (74),
    #   (93) (all listed mid-window).
    drop_reason = None
    continuous_trades = True
    if n_zero > 0:
        continuous_trades = False
        drop_reason = f"n_zero_close={n_zero}"
    elif n_nan > 0:
        continuous_trades = False
        drop_reason = f"n_nan_close={n_nan}"
    elif n_neg > 0:
        continuous_trades = False
        drop_reason = f"n_negative_close={n_neg}"
    elif has_dissolution:
        continuous_trades = False
        drop_reason = f"dissolution_at={dissolution_date}"
    else:
        # Check the first_date proximity to window_start (operationalizing
        # "continuous trades throughout the full dataset period").
        first_date = df["Date"].iloc[0]
        window_start = pd.to_datetime(WINDOW_START)
        # Allow up to 5 business days slack (holidays / mid-month listings
        # close to 2006-11-20 — empirical tolerance from EDA).
        bd_gap = pd.bdate_range(window_start, first_date).size - 1
        if bd_gap > 5:
            continuous_trades = False
            drop_reason = (
                f"listed_mid_window first_date={first_date.strftime('%Y-%m-%d')} "
                f"(bd_gap={bd_gap} > 5 from window_start)"
            )

    return {
        "file": path.name,
        "n_rows": n_rows,
        "first_date": df["Date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": df["Date"].iloc[-1].strftime("%Y-%m-%d"),
        "n_zero_close": n_zero,
        "n_nan_close": n_nan,
        "n_negative_close": n_neg,
        "has_dissolution_event": has_dissolution,
        "dissolution_first_zero_date": dissolution_date,
        "continuous_trades": continuous_trades,
        "drop_reason": drop_reason,
    }


def main():
    print(f"# W17-1 H4 EDA — FTSE asset universe")
    print(f"# Window: {WINDOW_START} → {WINDOW_END} (per thesis §7.2.3 p.145)")
    print(f"# Data dir: {FTSE_DIR}")

    csvs = sorted(FTSE_DIR.glob("table*.csv"))
    print(f"# Found {len(csvs)} CSVs")

    diagnostics = [diagnose_asset(p) for p in csvs]

    # Distribution of in-window row count
    n_rows = pd.Series([d["n_rows"] for d in diagnostics])
    print(f"\n## In-window row count distribution")
    print(n_rows.describe().to_string())

    # Continuous-trades flag distribution
    continuous = sum(1 for d in diagnostics if d["continuous_trades"])
    dropped = len(diagnostics) - continuous
    print(f"\n## Continuous-trades filter result")
    print(f"  kept (continuous): {continuous}")
    print(f"  dropped: {dropped}")

    # Drop-reason breakdown
    drop_reasons: dict[str, int] = {}
    for d in diagnostics:
        if d.get("drop_reason"):
            # bucket by reason prefix
            key = d["drop_reason"].split("=")[0]
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
    if drop_reasons:
        print(f"\n## Drop reasons breakdown")
        for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")

    # Persist per-asset CSV summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(diagnostics)
    summary_csv = RESULTS_DIR / "asset_universe_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote per-asset summary: {summary_csv}")

    # Persist the result-of-truth artifact
    kept_assets = [d["file"] for d in diagnostics if d["continuous_trades"]]
    dropped_assets = [
        {"file": d["file"], "reason": d.get("drop_reason", "unknown"),
         "first_zero_date": d.get("dissolution_first_zero_date"),
         "n_rows_in_window": d["n_rows"]}
        for d in diagnostics if not d["continuous_trades"]
    ]
    artifact = {
        "criterion": "continuous_daily_close_prices_2006_11_20_to_2012_12_31",
        "criterion_verbatim": (
            "keep only the assets which had continuous daily closing "
            "prices trades throughout the full dataset period"
        ),
        "thesis_target_d": 87,
        "thesis_target_section": "§7.2.3 p.145 'd = 87 for FTSE'",
        "n_original": len(csvs),
        "n_kept": len(kept_assets),
        "n_dropped": len(dropped_assets),
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "kept_assets": kept_assets,
        "dropped_assets": dropped_assets,
    }
    artifact_path = RESULTS_DIR / "asset_universe_87.json"
    with open(artifact_path, "w") as fh:
        json.dump(artifact, fh, indent=2, default=str)
    print(f"Wrote artifact: {artifact_path}")

    print(f"\n## Summary vs thesis target (d=87)")
    print(f"  kept = {len(kept_assets)}; thesis target = 87")
    if len(kept_assets) == 87:
        print("  EXACT MATCH ✅")
    else:
        print(f"  delta = {len(kept_assets) - 87} (honest scar — explain in EDA doc)")


if __name__ == "__main__":
    main()

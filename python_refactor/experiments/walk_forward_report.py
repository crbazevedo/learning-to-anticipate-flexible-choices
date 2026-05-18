"""
W14-2 walk-forward 30-seed orchestrator + report.

Runs S0 + S2 (or any scenario list) × N seeds × T rolling periods
via W14-1 walk_forward.run_walk_forward; aggregates per seed +
per scenario; emits Markdown report; updates VALIDATION-RESULTS.

Parallelizes (scenario, seed) pairs via ProcessPoolExecutor
(same shape as W12-1 sweep).

CLI:
    python -m experiments.walk_forward_report \
        --scenarios S0,S2 \
        --seeds 1-30 \
        --train-window-days 378 \
        --step-days 50 \
        --n-mc 1000 \
        --output docs/OOS-EFHV-WALK-FORWARD-REPORT.md \
        --per-seed-json /tmp/w14-2-per-seed.json \
        --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_paper_window_returns(
    enforce_thesis_continuous_trades: bool = False,
) -> pd.DataFrame:
    """W17-5: optional 87-asset thesis-faithful filter (BACKLOG H4 closure).

    When enforce_thesis_continuous_trades=True, restricts the load to
    the 87 assets per docs/H4-ASSET-UNIVERSE-EDA.md. Default unchanged
    (98 assets) for backward compatibility with W12-W16 reports.
    """
    from experiments.validation_matrix import WINDOWS
    from src.experiments.data_loader import DataLoader
    window = WINDOWS["paper"]
    loader = DataLoader()
    return loader.load_asset_data(
        [window["asset_files_glob"]],
        date_range={"start": window["date_start"], "end": window["date_end"]},
        assets=[],
        enforce_thesis_continuous_trades=enforce_thesis_continuous_trades,
    )


def _run_one(scenario: str, seed: int,
              full_returns_pickle: bytes,
              train_window_days: int, step_days: int,
              n_mc: int,
              lambda_trace_csv_path: str | None = None,
              use_closed_form_efhv: bool = False,
              use_closed_form_expectation_efhv: bool = False,
              use_v2_per_front_efhv: bool = False) -> dict[str, Any]:
    """ProcessPoolExecutor entry: unpickle returns + run one
    (scenario, seed) walk-forward, return aggregate.

    W17-3 (closes W16-5-CARRY-1 + W16-4-CARRY-1): optional
    lambda_trace_csv_path forwarded to run_walk_forward (kwarg from
    W16-4). When set, the K-period λ trace is appended to the CSV
    per period via ExperimentManager.
    """
    import pickle
    from experiments.walk_forward import (aggregate_per_seed,
                                            run_walk_forward)
    full_returns = pickle.loads(full_returns_pickle)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    per_period = run_walk_forward(
        scenario=scenario, seed=seed,
        full_returns=full_returns,
        train_window_days=train_window_days,
        step_days=step_days,
        n_mc=n_mc,
        rng=rng,
        lambda_trace_csv_path=lambda_trace_csv_path,
        use_closed_form_efhv=use_closed_form_efhv,
        use_closed_form_expectation_efhv=use_closed_form_expectation_efhv,
        use_v2_per_front_efhv=use_v2_per_front_efhv,
    )
    agg = aggregate_per_seed(per_period)
    return {
        "scenario": scenario, "seed": seed,
        "per_period": per_period,
        "n_periods_ok": agg["n_periods_ok"],
        "n_periods_total": agg["n_periods_total"],
        "grand_mean": agg["grand_mean"],
        "grand_std": agg["grand_std"],
        "wall_clock_s": time.time() - t0,
    }


def parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    if "," in spec:
        return [int(s.strip()) for s in spec.split(",") if s.strip()]
    return [int(spec)]


def _welch_t(a: list[float], b: list[float]) -> tuple[float, int, str]:
    """Welch's t-statistic + approximate df + direction string."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[np.isfinite(a_arr)]
    b_arr = b_arr[np.isfinite(b_arr)]
    if len(a_arr) < 2 or len(b_arr) < 2:
        return float("nan"), 0, "insufficient-data"
    m_a, m_b = a_arr.mean(), b_arr.mean()
    v_a, v_b = a_arr.var(ddof=1), b_arr.var(ddof=1)
    n_a, n_b = len(a_arr), len(b_arr)
    se = np.sqrt(v_a / n_a + v_b / n_b)
    if se == 0:
        return float("nan"), 0, "zero-variance"
    t = (m_a - m_b) / se
    # Welch-Satterthwaite df
    df = ((v_a / n_a + v_b / n_b) ** 2
          / ((v_a / n_a) ** 2 / max(1, n_a - 1)
              + (v_b / n_b) ** 2 / max(1, n_b - 1)))
    direction = ("a > b" if t > 0 else
                  "a < b" if t < 0 else "equal")
    return float(t), int(df), direction


def _format_report(per_scenario_means: dict[str, list[float]],
                    scenarios: list[str],
                    n_seeds: int,
                    train_window_days: int,
                    step_days: int,
                    n_mc: int,
                    wall_clock_s: float) -> str:
    lines: list[str] = []
    lines.append("# Walk-Forward Out-of-Sample Future Hypervolume — paper headline (proper methodology)")
    lines.append("")
    lines.append("*Generated by `python_refactor/experiments/walk_forward_report.py` (W14-2).*")
    lines.append("*Implements thesis §7.2.2 rolling-period protocol — see "
                  "[`docs/THESIS-INDEX.md`](THESIS-INDEX.md).*")
    lines.append("")
    lines.append("## Protocol (thesis §7.2.3)")
    lines.append("")
    lines.append(f"- **Train window**: {train_window_days} business days (~1.5 years)")
    lines.append(f"- **Step**: {step_days} business days (~2.5 months)")
    lines.append(f"- **MC scenarios per period (E)**: {n_mc}")
    lines.append(f"- **Reference point z_ref**: (0.2, 0.0) — risk_max=0.2, return_min=0.0")
    lines.append(f"- **Scenarios**: {', '.join(scenarios)}")
    lines.append(f"- **Seeds**: {n_seeds}")
    lines.append(f"- **Wall-clock**: {wall_clock_s:.1f}s")
    lines.append("")
    lines.append("## Aggregated results (per scenario, across seeds × periods)")
    lines.append("")
    lines.append("| scenario | n_seeds | grand mean Ŝ | std | median | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in scenarios:
        means = per_scenario_means.get(s, [])
        finite = [m for m in means if np.isfinite(m)]
        if not finite:
            lines.append(f"| {s} | 0 | nan | nan | nan | nan | nan |")
            continue
        arr = np.asarray(finite, dtype=float)
        lines.append(
            f"| {s} | {len(finite)} | {arr.mean():.6g} | "
            f"{arr.std(ddof=1) if len(arr) >= 2 else 0:.6g} | "
            f"{np.median(arr):.6g} | {arr.min():.6g} | {arr.max():.6g} |"
        )
    lines.append("")
    if "S0" in per_scenario_means and "S2" in per_scenario_means:
        s0 = per_scenario_means["S0"]
        s2 = per_scenario_means["S2"]
        s0_arr = np.asarray([m for m in s0 if np.isfinite(m)], dtype=float)
        s2_arr = np.asarray([m for m in s2 if np.isfinite(m)], dtype=float)
        if len(s0_arr) >= 2 and len(s2_arr) >= 2:
            delta = s2_arr.mean() - s0_arr.mean()
            rel = (delta / s0_arr.mean() * 100) if abs(s0_arr.mean()) > 1e-12 else float("nan")
            t, df, _ = _welch_t(list(s2_arr), list(s0_arr))
            sign = "+" if delta > 0 else ""
            lines.append("## Headline observation (paper §V-D claim direction)")
            lines.append("")
            lines.append(
                f"**S2 (anticipatory) grand mean Ŝ = {s2_arr.mean():.6g}**"
            )
            lines.append(
                f"**S0 (myopic) grand mean Ŝ = {s0_arr.mean():.6g}**"
            )
            lines.append("")
            lines.append(
                f"Delta (S2 − S0) = **{sign}{delta:.6g}** ({sign}{rel:.2f}%); "
                f"Welch t = **{t:.3g}**, df ≈ {df}. "
                f"Paper claim direction (Table 7.2): S2 > S0. "
                f"Empirical direction here: **"
                f"{'S2 > S0 (consistent)' if delta > 0 else 'S2 ≤ S0 (inconsistent — W15 candidate)'}**."
            )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W14-2 walk-forward 30-seed orchestrator.")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--train-window-days", type=int, default=378)
    parser.add_argument("--step-days", type=int, default=50)
    parser.add_argument("--n-mc", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-seed-json", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    # W17-3 (closes W16-5-CARRY-1 + W16-4-CARRY-1): optional λ trace
    # CSV path. When set, ExperimentManager flushes per-period λ + TIP
    # rows to this CSV via the W16-4 plumbing. Smokes that pass this
    # arg can then directly assert "both λ arms fire" rather than
    # infer from aggregate HV.
    parser.add_argument("--lambda-trace-csv-path", type=Path, default=None,
                          help="If set, ExperimentManager appends per-period "
                               "(period, generation, solution_rank, λ^H, λ^K, "
                               "λ, TIP, lambda_k_branch) rows to this CSV.")
    # W17-5 (BACKLOG H4): opt-in 87-asset thesis-faithful filter
    # (see docs/H4-ASSET-UNIVERSE-EDA.md).
    parser.add_argument("--enforce-thesis-continuous-trades",
                          action="store_true",
                          help="Restrict asset universe to the 87 thesis-faithful "
                               "assets per docs/H4-ASSET-UNIVERSE-EDA.md. "
                               "Default is the full 98-asset legacy archive.")
    # W22 OOS eHypV estimator variants (operator directive 2026-05-17):
    # 4-way calibration: MC (default) + closed-form-point (Option A)
    # + closed-form-expectation (Option B) + v2-per-front (Option C).
    # These flags are MUTUALLY EXCLUSIVE — only one closed-form variant
    # may be active per smoke. When any is set, n_mc is IGNORED.
    parser.add_argument("--use-closed-form-efhv",
                          action="store_true",
                          help="W22 Option A: skip bootstrap MC in "
                               "compute_oos_efhv; use single full-window MLE "
                               "(μ̂, Σ̂) per period for a deterministic Ŝ "
                               "point-estimate. Faster + zero MC noise; not "
                               "directly comparable to the W14-2 MC chain.")
    parser.add_argument("--use-closed-form-expectation-efhv",
                          action="store_true",
                          help="W22 Option B: true closed-form expected Ŝ via "
                               "per-portfolio Black-Scholes-style truncated-mean "
                               "call/put pricing on (ROI, risk) Gaussians. "
                               "Honest scar: no Pareto overlap correction → "
                               "over-estimates vs true cloud HV.")
    parser.add_argument("--use-v2-per-front-efhv",
                          action="store_true",
                          help="W22 Option C: lift v2's per-front Δ_S formula "
                               "(asms_emoa.cpp:380+) directly to OOS "
                               "aggregation. Honest scar: v2 actually uses Δ_S "
                               "to RANK solutions, not as HV estimator; sum is "
                               "a heuristic.")
    args = parser.parse_args(argv)

    # Validate mutual exclusivity of W22 estimator flags
    _w22_flags_active = sum([
        args.use_closed_form_efhv,
        args.use_closed_form_expectation_efhv,
        args.use_v2_per_front_efhv,
    ])
    if _w22_flags_active > 1:
        print(
            "[wf-report] ERROR: --use-closed-form-efhv, "
            "--use-closed-form-expectation-efhv, and --use-v2-per-front-efhv "
            "are mutually exclusive. Pick at most one per smoke.",
            file=sys.stderr,
        )
        return 2

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = parse_seeds(args.seeds)
    print(f"[wf-report] loading paper-window returns…", file=sys.stderr)
    full_returns = _load_paper_window_returns(
        enforce_thesis_continuous_trades=args.enforce_thesis_continuous_trades,
    )
    print(f"[wf-report] returns shape: {full_returns.shape}", file=sys.stderr)
    if args.enforce_thesis_continuous_trades:
        print(f"[wf-report] H4 87-asset filter ENABLED → {full_returns.shape[1]} assets",
              file=sys.stderr)

    import pickle
    full_returns_pickle = pickle.dumps(full_returns)

    pairs = [(s, sd) for s in scenarios for sd in seeds]
    total = len(pairs)
    print(f"[wf-report] dispatching {total} (scenario, seed) pairs "
          f"over {args.jobs} workers…", file=sys.stderr)

    per_seed_results: list[dict[str, Any]] = []
    t0 = time.time()
    # W17-3: convert Path → str for ProcessPoolExecutor pickling (Paths
    # are picklable but stringifying upfront avoids any worker-side
    # cwd surprises).
    lambda_trace_csv_path_str = (
        str(args.lambda_trace_csv_path) if args.lambda_trace_csv_path else None
    )
    if lambda_trace_csv_path_str:
        print(f"[wf-report] λ trace CSV → {lambda_trace_csv_path_str}", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_run_one, s, sd, full_returns_pickle,
                          args.train_window_days, args.step_days, args.n_mc,
                          lambda_trace_csv_path_str,
                          args.use_closed_form_efhv,
                          args.use_closed_form_expectation_efhv,
                          args.use_v2_per_front_efhv): (s, sd)
            for (s, sd) in pairs
        }
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            print(f"[wf-report {done}/{total}] {res['scenario']}/seed{res['seed']} "
                  f"  {res['n_periods_ok']}/{res['n_periods_total']} periods ok  "
                  f"grand_mean={res['grand_mean']:.6g}  "
                  f"({res['wall_clock_s']:.1f}s)", file=sys.stderr)
            per_seed_results.append(res)
    wall_clock_s = time.time() - t0
    print(f"[wf-report] done in {wall_clock_s:.1f}s", file=sys.stderr)

    per_scenario_means: dict[str, list[float]] = {s: [] for s in scenarios}
    for r in per_seed_results:
        per_scenario_means[r["scenario"]].append(r["grand_mean"])

    report = _format_report(
        per_scenario_means, scenarios, len(seeds),
        args.train_window_days, args.step_days, args.n_mc, wall_clock_s)
    args.output.write_text(report)
    print(f"[wf-report] wrote {args.output}", file=sys.stderr)

    if args.per_seed_json:
        # Convert non-serializable per_period entries to floats.
        slim = []
        for r in per_seed_results:
            slim.append({
                "scenario": r["scenario"], "seed": r["seed"],
                "n_periods_ok": r["n_periods_ok"],
                "n_periods_total": r["n_periods_total"],
                "grand_mean": r["grand_mean"],
                "grand_std": r["grand_std"],
                "wall_clock_s": r["wall_clock_s"],
            })
        args.per_seed_json.write_text(json.dumps(slim, indent=2, default=float))
        print(f"[wf-report] wrote {args.per_seed_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

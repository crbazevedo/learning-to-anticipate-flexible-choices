"""
W13-3 OOS EFHV report orchestrator.

For each (scenario, seed):
  1. Run validation_matrix in-process to obtain a trained Pareto front
  2. Extract portfolio weights from each Pareto solution
  3. Load OOS (out-of-sample) asset returns from the extended-window
     CSV via data_loader
  4. Compute OOS Future Hypervolume per thesis Eqs (7.10)+(7.11) via
     experiments.oos_evaluator.compute_oos_efhv

Aggregates per-scenario means + stds and writes a small summary
markdown that can be folded into VALIDATION-RESULTS.populated.md §1.

CLI:
    python -m experiments.oos_report \\
        --scenarios S0,S2 \\
        --seeds 1-30 \\
        --in-sample-window paper \\
        --oos-window extended \\
        --n-mc 1000 \\
        --output docs/OOS-EFHV-REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.oos_evaluator import compute_oos_efhv
from experiments.sweep import parse_seeds


REPO_ROOT = Path(__file__).parents[1]


def _load_window_returns(window_id: str) -> pd.DataFrame:
    """Load full asset returns for the chosen window via data_loader."""
    from experiments.validation_matrix import WINDOWS
    from src.experiments.data_loader import DataLoader

    if window_id not in WINDOWS:
        raise ValueError(f"unknown window {window_id}; choices: {list(WINDOWS)}")
    window = WINDOWS[window_id]
    loader = DataLoader()
    returns_df = loader.load_asset_data(
        [window["asset_files_glob"]],
        date_range={"start": window["date_start"], "end": window["date_end"]},
        assets=[],
    )
    if returns_df.empty:
        raise RuntimeError(f"data_loader returned empty for window {window_id}")
    return returns_df


def split_temporal(returns_df: pd.DataFrame,
                    train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: first `train_frac` rows for training,
    remaining (held-out) for OOS evaluation.

    Preserves the same asset universe in both portions — critical
    because the trained Pareto weights are (n_assets,) and must
    match the OOS returns' column count."""
    n = len(returns_df)
    split = int(n * train_frac)
    train = returns_df.iloc[:split].copy()
    oos = returns_df.iloc[split:].copy()
    return train, oos


def _run_one_scenario_seed(scenario: str, in_sample_window: str,
                              seed: int, train_returns: pd.DataFrame
                              ) -> tuple[list[np.ndarray], int]:
    """Run SMS-EMOA in-process against pre-split train returns; return
    (pareto_weights, n_assets).

    W13-3 deviation from validation_matrix.run_one: data['assets']
    is the TRAIN portion of the in-sample window (not the full window).
    This enforces a true train/OOS temporal split using the SAME
    asset universe — without it, the OOS evaluator would see a
    different number of columns than the trained portfolio weights.
    """
    from experiments.validation_matrix import build_experiment_config
    from src.experiments.experiment_manager import ExperimentManager

    config = build_experiment_config(scenario, in_sample_window, seed)
    np.random.seed(seed)
    suite_config = {
        "experiment_name": f"oos_report_{scenario}_seed{seed}",
        "description": "W13-3 in-process train (temporal split)",
        "version": "W13-3",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    mgr = ExperimentManager(suite_config)
    # Bypass mgr._load_experiment_data: inject the train portion directly.
    n_assets = train_returns.values.shape[1]
    data = {"assets": train_returns, "market": pd.DataFrame(),
            "config": config.get("data", {})}
    results = mgr._run_algorithm(config, data)
    pareto = results.get("pareto_front", [])
    weights = []
    for sol in pareto:
        w = getattr(sol.P, "investment", None)
        if w is None:
            continue
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape == (n_assets,):
            weights.append(w_arr)
    return weights, n_assets


def _aggregate(results: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Per-scenario summary stats."""
    summary: dict[str, dict[str, float]] = {}
    for scenario, efhvs in results.items():
        arr = np.asarray(efhvs, dtype=float)
        n = len(arr)
        summary[scenario] = {
            "n_seeds": int(n),
            "mean": float(arr.mean()) if n else float("nan"),
            "std": float(arr.std(ddof=1)) if n >= 2 else 0.0,
            "median": float(np.median(arr)) if n else float("nan"),
            "min": float(arr.min()) if n else float("nan"),
            "max": float(arr.max()) if n else float("nan"),
        }
    return summary


def _format_report(summary: dict[str, dict[str, float]],
                    scenarios: list[str],
                    seeds: list[int],
                    in_sample_window: str,
                    oos_window: str,
                    n_mc: int) -> str:
    """Render a small Markdown report summarising mean OOS EFHV per scenario."""
    lines: list[str] = []
    lines.append("# Out-of-Sample Future Hypervolume — paper headline metric")
    lines.append("")
    lines.append("*Generated by `python_refactor/experiments/oos_report.py` (W13-3).")
    lines.append("Implements thesis Eqs (7.10)+(7.11) — see "
                  "[`docs/THESIS-INDEX.md`](THESIS-INDEX.md).*")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- **In-sample window** (training): `{in_sample_window}`")
    lines.append(f"- **Out-of-sample window** (evaluation): `{oos_window}`")
    lines.append(f"- **Scenarios**: {', '.join(scenarios)}")
    lines.append(f"- **Seeds**: {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    lines.append(f"- **MC scenarios per seed (E)**: {n_mc}")
    lines.append(f"- **Reference point z_ref**: (0.2, 0.0) — risk_max=0.2, return_min=0.0")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| scenario | n_seeds | mean OOS EFHV | std | median | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in scenarios:
        if s not in summary:
            continue
        row = summary[s]
        lines.append(
            f"| {s} | {row['n_seeds']} | "
            f"{row['mean']:.6g} | {row['std']:.6g} | "
            f"{row['median']:.6g} | {row['min']:.6g} | {row['max']:.6g} |"
        )
    lines.append("")
    # Headline observation (empirical, no assertion).
    if "S0" in summary and "S2" in summary:
        s0_mean = summary["S0"]["mean"]
        s2_mean = summary["S2"]["mean"]
        if np.isfinite(s0_mean) and np.isfinite(s2_mean):
            delta = s2_mean - s0_mean
            rel = (delta / s0_mean * 100) if abs(s0_mean) > 1e-12 else float("nan")
            sign = "+" if delta > 0 else ""
            lines.append("## Headline observation (paper §V-D claim direction)")
            lines.append("")
            lines.append(
                f"**S2 (anticipatory + max-Hypv DM = ASMS/mHDM) mean OOS EFHV = "
                f"{s2_mean:.6g}**"
            )
            lines.append(
                f"**S0 (myopic baseline = SMS/RDM) mean OOS EFHV = {s0_mean:.6g}**"
            )
            lines.append("")
            lines.append(
                f"Delta (S2 − S0) = **{sign}{delta:.6g}** ({sign}{rel:.2f}%). "
                f"Paper claim direction (Table 7.2): S2 > S0. "
                f"Empirical direction here: **"
                f"{'S2 > S0 (consistent)' if delta > 0 else 'S2 ≤ S0 (inconsistent)'}**."
            )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W13-3 OOS EFHV report orchestrator.")
    parser.add_argument("--scenarios", required=True,
                          help="Comma-separated scenario ids (e.g. 'S0,S2').")
    parser.add_argument("--seeds", required=True,
                          help="Seed spec (e.g. '1-30' or '1,2,5').")
    parser.add_argument("--in-sample-window", default="paper",
                          help="Window id used for training (e.g. 'paper'). "
                               "Temporally split by --train-frac.")
    parser.add_argument("--train-frac", type=float, default=0.8,
                          help="Fraction of the in-sample-window's rows used "
                               "for training; rest is held out for OOS evaluation.")
    parser.add_argument("--oos-window", default=None,
                          help="(optional) Alternative window id for OOS — must "
                               "have same asset count. Default: held-out tail of "
                               "--in-sample-window.")
    parser.add_argument("--n-mc", type=int, default=1000,
                          help="Number of MC scenarios per seed (E in thesis Eq 7.11).")
    parser.add_argument("--output", type=Path, required=True,
                          help="Path to write the OOS-EFHV report Markdown.")
    parser.add_argument("--per-seed-json", type=Path, default=None,
                          help="Optional per-seed result JSON dump for downstream stats.")
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = parse_seeds(args.seeds)
    n_mc = args.n_mc

    print(f"[oos-report] loading in-sample window '{args.in_sample_window}'…",
          file=sys.stderr)
    full_returns = _load_window_returns(args.in_sample_window)
    print(f"[oos-report] full window shape: {full_returns.shape}", file=sys.stderr)
    train_returns, default_oos_returns = split_temporal(
        full_returns, train_frac=args.train_frac)
    print(f"[oos-report] temporal split @ {args.train_frac*100:.0f}%: "
          f"train={train_returns.shape}, oos={default_oos_returns.shape}",
          file=sys.stderr)
    if args.oos_window is None:
        oos_returns = default_oos_returns
        oos_window_label = f"{args.in_sample_window}[tail{(1-args.train_frac)*100:.0f}%]"
    else:
        oos_returns = _load_window_returns(args.oos_window)
        oos_window_label = args.oos_window
        print(f"[oos-report] overriding OOS with window '{args.oos_window}', "
              f"shape: {oos_returns.shape}", file=sys.stderr)

    rng = np.random.default_rng(0)
    per_seed: dict[str, list[dict[str, Any]]] = {s: [] for s in scenarios}
    per_scenario: dict[str, list[float]] = {s: [] for s in scenarios}

    total = len(scenarios) * len(seeds)
    done = 0
    t0 = time.time()
    for scenario in scenarios:
        for seed in seeds:
            done += 1
            t_seed = time.time()
            try:
                weights, n_assets = _run_one_scenario_seed(
                    scenario, args.in_sample_window, seed, train_returns)
            except Exception as exc:
                print(f"[oos-report {done}/{total}] {scenario}/seed{seed} "
                      f"TRAIN ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                per_seed[scenario].append({"seed": seed, "error": str(exc)})
                continue
            # Asset-count alignment: temporal split preserves the same
            # 98-asset universe, so this should always match.
            if oos_returns.shape[1] != n_assets:
                # Defensive — only fires when --oos-window override has a
                # different asset count. Truncate; honest scar (asset identity
                # alignment is approximate when column counts differ).
                use_oos = oos_returns.iloc[:, :n_assets]
            else:
                use_oos = oos_returns
            efhv = compute_oos_efhv(
                pareto_weights=weights,
                oos_returns=use_oos,
                n_samples=n_mc,
                rng=rng,
            )
            per_seed[scenario].append({
                "seed": seed,
                "n_pareto": len(weights),
                "efhv_mean": efhv["efhv_mean"],
                "efhv_std": efhv["efhv_std"],
            })
            per_scenario[scenario].append(efhv["efhv_mean"])
            elapsed = time.time() - t_seed
            print(f"[oos-report {done}/{total}] {scenario}/seed{seed}  "
                  f"|pareto|={len(weights)}  EFHV={efhv['efhv_mean']:.6g}  "
                  f"({elapsed:.1f}s)", file=sys.stderr)

    print(f"[oos-report] done in {(time.time()-t0):.1f}s", file=sys.stderr)

    summary = _aggregate(per_scenario)
    report = _format_report(summary, scenarios, seeds,
                              args.in_sample_window, oos_window_label, n_mc)
    args.output.write_text(report)
    print(f"[oos-report] wrote {args.output}", file=sys.stderr)

    if args.per_seed_json is not None:
        args.per_seed_json.write_text(json.dumps(per_seed, indent=2, default=float))
        print(f"[oos-report] wrote {args.per_seed_json}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""W22 NC Smoke Runner — FTSE / PO empirical sweep across opt-in NC configs.

Per operator directive 2026-05-20: validate the synthetic-vs-empirical claims
for opt-in NCs against real data. Converts "operator manual FTSE sweep" into
"one CLI command across all NC configurations."

Each NC configuration is a set of env-var assignments that ACTIVATE specific
opt-in fixes. The runner spawns a subprocess per (config × seed) so each
inherits its own environment cleanly. Results aggregated into a markdown
table for direct ratification analysis.

Usage:
    cd python_refactor
    uv run python -m scripts.nc_smoke_runner \\
        --configs BASELINE,NC36,TIP_CLEANUP,NC27_DEEP,FULL_STACK \\
        --seeds 1,2,3,4,5 \\
        --output ../results/nc_smoke_$(date +%Y%m%d).json

Configurations (presets):
- BASELINE: all NC opt-ins OFF (current production behavior)
- NC36: W22_NC36_TIP_ANALYTICAL=1 (analytical TIP)
- TIP_CLEANUP: NC36 + NC13b + NC31 (all three TIP fixes)
- NC27_DEEP: W22_NC27_PREDICTOR=dirichlet_posterior (stateful posterior)
- FULL_STACK: TIP_CLEANUP + NC27_DEEP (high-confidence theory upgrades)
- ANTICIPATORY_OPS: Path C — NC34+NC39 (currently un-wired into SMSEMOA;
  this config is documented but not yet executable until integration lands)

Output: per-config per-seed final wealth/Ŝ aggregates + Wilcoxon paired tests
vs BASELINE.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Predefined NC configurations
NC_CONFIGS = {
    "BASELINE": {
        # No opt-in env vars set → production defaults
    },
    "NC36": {
        "W22_NC36_TIP_ANALYTICAL": "1",
    },
    "NC13b": {
        "W22_NC13B_SMOOTH_CLAMP": "1",
    },
    "NC31": {
        "W22_NC31_TIP_CONDITIONAL": "1",
    },
    "TIP_CLEANUP": {
        "W22_NC36_TIP_ANALYTICAL": "1",
        "W22_NC13B_SMOOTH_CLAMP": "1",
        "W22_NC31_TIP_CONDITIONAL": "1",
    },
    "NC27_DEEP": {
        "W22_NC27_PREDICTOR": "dirichlet_posterior",
    },
    "NC27_LN": {
        "W22_NC27_PREDICTOR": "logistic_normal",
    },
    "FULL_STACK": {
        "W22_NC36_TIP_ANALYTICAL": "1",
        "W22_NC13B_SMOOTH_CLAMP": "1",
        "W22_NC31_TIP_CONDITIONAL": "1",
        "W22_NC27_PREDICTOR": "dirichlet_posterior",
    },
    "ANTICIPATORY_OPS": {
        # Path C — requires NC34/NC39 integration into SMSEMOA (not yet wired)
        # When integrated: would set W22_NC34_ENABLE=1, W22_NC39_ENABLE=1
        # Currently a placeholder for documentation
        "_status": "REQUIRES_INTEGRATION",
    },
    "MULTI_PERIOD_AMFC": {
        # Requires Hv-DM-AMFC dm_type + dm_config['horizon_accumulated']=3
        # Doesn't fit env-var contract; documented for operator
        "_status": "REQUIRES_DM_CONFIG_CHANGE",
    },
}


def _run_subprocess_for_config(
    config_name: str,
    env_overrides: dict,
    scenarios: list[str],
    seeds: list[int],
    n_mc: int,
    train_window_days: int,
    step_days: int,
    po_dir: Path | None = None,
    timeout_seconds: int = 7200,
) -> list[dict]:
    """Run walk_forward_po_smoke as a subprocess with custom env overrides.

    Each subprocess inherits the full os.environ, then sets the NC-specific
    env vars. This ensures clean isolation between NC configurations even
    when they share a single python process pool.
    """
    if env_overrides.get("_status") == "REQUIRES_INTEGRATION":
        print(f"[nc-smoke] SKIPPING {config_name}: requires SMSEMOA integration",
              file=sys.stderr)
        return []
    if env_overrides.get("_status") == "REQUIRES_DM_CONFIG_CHANGE":
        print(f"[nc-smoke] SKIPPING {config_name}: requires dm_config change",
              file=sys.stderr)
        return []

    env = copy.deepcopy(os.environ)
    for k, v in env_overrides.items():
        if not k.startswith("_"):
            env[k] = str(v)

    # Build command: invoke walk_forward_po_smoke as a module
    cmd = [
        "uv", "run", "python", "-m", "experiments.walk_forward_po_smoke",
        "--scenarios", ",".join(scenarios),
        "--seeds", ",".join(str(s) for s in seeds),
        "--n-mc", str(n_mc),
        "--train-window-days", str(train_window_days),
        "--step-days", str(step_days),
        "--output", f"/tmp/nc_smoke_inner_{config_name}_{int(time.time())}.json",
    ]
    if po_dir is not None:
        cmd.extend(["--po-dir", str(po_dir)])

    t0 = time.time()
    print(f"[nc-smoke] Running {config_name}: env={list(env_overrides.keys())}",
          file=sys.stderr)
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout_seconds, check=False, cwd=Path(__file__).parent.parent,
        )
    except subprocess.TimeoutExpired:
        print(f"[nc-smoke] TIMEOUT for {config_name} after {timeout_seconds}s",
              file=sys.stderr)
        return []
    wall = time.time() - t0
    if result.returncode != 0:
        print(f"[nc-smoke] ERROR in {config_name} (rc={result.returncode}):",
              file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return []

    # Parse output JSON
    output_path = None
    for tok in cmd:
        if tok.startswith("/tmp/nc_smoke_inner_"):
            output_path = Path(tok)
            break
    if output_path is None or not output_path.exists():
        print(f"[nc-smoke] {config_name}: output file not found", file=sys.stderr)
        return []

    with open(output_path) as f:
        inner_results = json.load(f)
    print(f"[nc-smoke] {config_name} done in {wall:.1f}s ({len(inner_results)} seed × scenario rows)",
          file=sys.stderr)
    return [
        {**row, "config": config_name, "config_env": env_overrides, "wall_total": wall}
        for row in inner_results
    ]


def _aggregate_by_config(all_results: list[dict]) -> dict[str, dict]:
    """Per-config: per-scenario mean/std across seeds + delta-vs-baseline."""
    by_config = {}
    for row in all_results:
        cfg = row.get("config", "UNKNOWN")
        scenario = row.get("scenario", "UNKNOWN")
        gm = row.get("grand_mean", float("nan"))
        if cfg not in by_config:
            by_config[cfg] = {}
        if scenario not in by_config[cfg]:
            by_config[cfg][scenario] = []
        if not np.isnan(gm):
            by_config[cfg][scenario].append(gm)
    # Mean / std per config × scenario
    summary = {}
    for cfg, scn_dict in by_config.items():
        summary[cfg] = {}
        for scn, vals in scn_dict.items():
            arr = np.array(vals)
            summary[cfg][scn] = {
                "n_seeds": len(arr),
                "mean": float(np.mean(arr)) if len(arr) > 0 else float("nan"),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                "min": float(np.min(arr)) if len(arr) > 0 else float("nan"),
                "max": float(np.max(arr)) if len(arr) > 0 else float("nan"),
                "raw": arr.tolist(),
            }
    return summary


def _wilcoxon_vs_baseline(summary: dict, baseline_cfg: str = "BASELINE") -> dict:
    """Pairwise Wilcoxon test of each non-baseline config vs BASELINE per scenario.

    Returns dict[(config, scenario)] → {p_value, mean_diff, n}.
    Requires scipy; degrades gracefully if missing.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {}
    if baseline_cfg not in summary:
        return {}
    tests = {}
    for cfg, scn_dict in summary.items():
        if cfg == baseline_cfg:
            continue
        for scn, stats in scn_dict.items():
            baseline_vals = summary[baseline_cfg].get(scn, {}).get("raw", [])
            cfg_vals = stats.get("raw", [])
            if len(baseline_vals) != len(cfg_vals) or len(baseline_vals) < 2:
                tests[(cfg, scn)] = {
                    "p_value": float("nan"),
                    "mean_diff": float("nan"),
                    "n": len(cfg_vals),
                    "note": "insufficient or unaligned paired samples",
                }
                continue
            try:
                stat, p = wilcoxon(cfg_vals, baseline_vals)
            except ValueError:
                p = float("nan")
            tests[(cfg, scn)] = {
                "p_value": float(p),
                "mean_diff": float(np.mean(cfg_vals) - np.mean(baseline_vals)),
                "n": len(cfg_vals),
            }
    return tests


def _format_markdown(summary: dict, tests: dict) -> str:
    """Render a markdown comparison report."""
    lines = [
        "# W22 NC Smoke Run Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Per-config grand mean per scenario (n=seeds)",
        "",
        "| Config | Scenario | n_seeds | Mean Ŝ | Std | Δ vs BASELINE | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in sorted(summary.keys()):
        for scn in sorted(summary[cfg].keys()):
            s = summary[cfg][scn]
            test = tests.get((cfg, scn), {})
            mean_diff = test.get("mean_diff", "—")
            p_value = test.get("p_value", "—")
            if isinstance(mean_diff, float):
                mean_diff_str = f"{mean_diff:+.4e}"
            else:
                mean_diff_str = str(mean_diff)
            if isinstance(p_value, float):
                p_str = f"{p_value:.3f}" if not np.isnan(p_value) else "NaN"
            else:
                p_str = str(p_value)
            lines.append(
                f"| {cfg} | {scn} | {s['n_seeds']} | {s['mean']:.4e} | "
                f"{s['std']:.4e} | {mean_diff_str} | {p_str} |"
            )
    lines.extend([
        "",
        "## Notes",
        "- Δ vs BASELINE: positive means config beats baseline",
        "- Wilcoxon p-value < 0.05 → statistically significant difference (n_seeds≥5)",
        "- Configs with `_status` marker (e.g., ANTICIPATORY_OPS) require",
        "  integration into SMSEMOA before they can be smoke-tested",
        "",
        "## NC paths reference",
        "- TIP_CLEANUP = NC36 + NC13b + NC31 (all TIP improvements)",
        "- FULL_STACK = TIP_CLEANUP + NC27_DEEP (high-confidence theory)",
        "- ANTICIPATORY_OPS = Path C (NC34 + NC39); needs SMSEMOA wiring",
        "- MULTI_PERIOD_AMFC = NC35; needs dm_type='Hv-DM-AMFC' + dm_config",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="BASELINE,NC36,TIP_CLEANUP,FULL_STACK",
                         help="Comma-separated config names (presets in NC_CONFIGS)")
    parser.add_argument("--seeds", default="1,2,3,4,5",
                         help="Comma-separated seed integers")
    parser.add_argument("--scenarios", default="ASMS_mHDM_K3_v2both,SMS_RDM_K0")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--train-window-days", type=int, default=378)
    parser.add_argument("--step-days", type=int, default=50)
    parser.add_argument("--po-dir", type=str, default=None,
                         help="PO data directory (default: data/synthetic-po-8-1.0)")
    parser.add_argument("--output", default=None,
                         help="Output JSON path (required unless --list-configs)")
    parser.add_argument("--markdown-output", default=None,
                         help="Optional markdown summary path (default: <output>.md)")
    parser.add_argument("--timeout-per-config-seconds", type=int, default=7200)
    parser.add_argument("--list-configs", action="store_true",
                         help="Print available config presets and exit")
    args = parser.parse_args(argv)

    if args.list_configs:
        print("Available NC configurations:")
        for name, env in NC_CONFIGS.items():
            status = env.get("_status", "ACTIVE")
            print(f"  {name}: [{status}]")
            for k, v in env.items():
                if not k.startswith("_"):
                    print(f"    {k}={v}")
        return 0

    if args.output is None:
        print("ERROR: --output is required unless --list-configs", file=sys.stderr)
        return 2
    configs = args.configs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = args.scenarios.split(",")

    # Validate configs
    unknown = [c for c in configs if c not in NC_CONFIGS]
    if unknown:
        print(f"ERROR: unknown configs: {unknown}", file=sys.stderr)
        print(f"Available: {list(NC_CONFIGS.keys())}", file=sys.stderr)
        return 2

    po_dir = Path(args.po_dir) if args.po_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    all_results = []
    for cfg_name in configs:
        env_overrides = NC_CONFIGS[cfg_name]
        results = _run_subprocess_for_config(
            cfg_name, env_overrides, scenarios, seeds,
            args.n_mc, args.train_window_days, args.step_days,
            po_dir=po_dir,
            timeout_seconds=args.timeout_per_config_seconds,
        )
        all_results.extend(results)

    total_wall = time.time() - t_start

    summary = _aggregate_by_config(all_results)
    tests = _wilcoxon_vs_baseline(summary, baseline_cfg="BASELINE")

    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "configs": configs,
            "seeds": seeds,
            "scenarios": scenarios,
            "n_mc": args.n_mc,
            "train_window_days": args.train_window_days,
            "step_days": args.step_days,
            "total_wall_seconds": total_wall,
        },
        "raw_results": all_results,
        "summary": summary,
        "wilcoxon_tests": {f"{c}::{s}": v for (c, s), v in tests.items()},
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[nc-smoke] Wrote {output_path} ({len(all_results)} rows, {total_wall:.1f}s)",
          file=sys.stderr)

    md_path = args.markdown_output or str(output_path).replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(_format_markdown(summary, tests))
    print(f"[nc-smoke] Wrote markdown summary {md_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
W12-1 multi-seed sweep driver.

Dispatches validation_matrix runs across the scenarios × windows ×
seeds Cartesian product, in parallel via concurrent.futures.

CLI:
    python -m experiments.sweep \\
        --scenarios S0,S2 \\
        --windows paper \\
        --seeds 1-30 \\
        --output results/W12 \\
        --jobs 4

Output layout (one directory per tuple):
    results/W12/
    ├── S0_paper_seed1/
    │   ├── metrics.csv
    │   └── manifest.json
    ├── S0_paper_seed2/
    │   ├── ...
    ...

This is compatible with `experiments.aggregate_validation` which walks
the tree, reads `manifest.status`, and aggregates per (scenario, window,
metric) into VALIDATION-SUMMARY.csv.

The dispatch is shell-out + ProcessPoolExecutor rather than calling
run_one in-process because numpy / scipy / pandas occasionally have
non-fork-safe state and the per-run isolation is cheap (~0.5-2s
overhead per run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_seeds(spec: str) -> list[int]:
    """Parse seed spec: '1-30', '1,2,5,10', or '30'.

    >>> parse_seeds("1-3")
    [1, 2, 3]
    >>> parse_seeds("1,5,10")
    [1, 5, 10]
    >>> parse_seeds("7")
    [7]
    """
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    if "," in spec:
        return [int(s.strip()) for s in spec.split(",") if s.strip()]
    return [int(spec)]


def enumerate_tuples(scenarios: list[str], windows: list[str],
                      seeds: list[int]) -> list[tuple[str, str, int]]:
    """Cartesian product over the sweep spec."""
    return [(s, w, seed) for s in scenarios for w in windows for seed in seeds]


def _run_one_subprocess(scenario: str, window: str, seed: int,
                          output_root: Path) -> tuple[str, str, int, int, str]:
    """Spawn a validation_matrix subprocess for one (scenario, window, seed).

    Returns:
        (scenario, window, seed, exit_code, stderr_tail) — stderr_tail
        is the last 200 chars on failure (empty on success).
    """
    output_dir = output_root / f"{scenario}_{window}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "experiments.validation_matrix",
        "--scenario", scenario, "--window", window,
        "--seed", str(seed), "--output", str(output_dir),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parents[1]),
            capture_output=True, text=True, timeout=600,
        )
        tail = "" if result.returncode == 0 else result.stderr[-200:]
        return scenario, window, seed, result.returncode, tail
    except subprocess.TimeoutExpired:
        return scenario, window, seed, 124, "TimeoutExpired"
    except Exception as e:
        return scenario, window, seed, 1, f"{type(e).__name__}: {e}"


def run_sweep(scenarios: list[str], windows: list[str], seeds: list[int],
               output_root: Path, jobs: int) -> int:
    """Run the full sweep. Returns 0 if all OK, 1 if any failed."""
    tuples = enumerate_tuples(scenarios, windows, seeds)
    total = len(tuples)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {total} runs across "
          f"{len(scenarios)} scenarios × {len(windows)} windows × "
          f"{len(seeds)} seeds; {jobs} parallel workers", file=sys.stderr)

    ok_count = 0
    err_count = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_one_subprocess, s, w, seed, output_root): (s, w, seed)
            for (s, w, seed) in tuples
        }
        for fut in as_completed(futures):
            s, w, seed, code, tail = fut.result()
            done = ok_count + err_count + 1
            if code == 0:
                ok_count += 1
                print(f"[sweep {done}/{total}] OK  {s}/{w}/seed{seed}",
                      file=sys.stderr)
            else:
                err_count += 1
                print(f"[sweep {done}/{total}] ERR {s}/{w}/seed{seed} "
                      f"(exit {code}): {tail}", file=sys.stderr)

    print(f"[sweep] done: {ok_count}/{total} ok, {err_count} errors",
          file=sys.stderr)
    return 0 if err_count == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W12-1 multi-seed sweep driver over validation_matrix runs.")
    parser.add_argument("--scenarios", required=True,
                          help="Comma-separated scenario ids (e.g. 'S0,S2').")
    parser.add_argument("--windows", required=True,
                          help="Comma-separated window ids (e.g. 'paper' or 'paper,extended').")
    parser.add_argument("--seeds", required=True,
                          help="Seed spec: '1-30' (range), '1,2,5,10' (list), or '30' (single).")
    parser.add_argument("--output", type=Path, required=True,
                          help="Results tree root (will be created if missing).")
    parser.add_argument("--jobs", type=int, default=4,
                          help="Number of parallel worker processes.")
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    seeds = parse_seeds(args.seeds)
    return run_sweep(scenarios, windows, seeds, args.output, args.jobs)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

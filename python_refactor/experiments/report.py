"""
W8-4 report orchestrator.

Composes the W8 analytics layer (stats helpers + table generators)
into a populated VALIDATION-RESULTS.md.

Reads:
    - VALIDATION-SUMMARY.csv  (from aggregate_validation.py)
    - results/                (per-run metrics.csv + manifest.json tree)
    - docs/VALIDATION-RESULTS.md  (template with 🚧 placeholders)

Writes:
    - docs/VALIDATION-RESULTS.populated.md

Substitution strategy:
    For each Table A-H section in the template, locate the section
    header, build the corresponding data rows from the summary CSV /
    pairwise tests / per-run files, call generate_table_X from
    experiments.tables, and replace the placeholder table body with
    the generated string. Sections with no data keep their 🚧
    placeholders (graceful incompleteness).

Receipt block per ANALYTICS-PLAN.md §8 populates section §0 with
git SHA + uv.lock fingerprint + generated timestamp + n_runs +
seed list + reproducibility command.

CLI:
    python -m experiments.report \\
        --summary docs/VALIDATION-SUMMARY.csv \\
        --runs results/ \\
        --template docs/VALIDATION-RESULTS.md \\
        --output docs/VALIDATION-RESULTS.populated.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments import stats, tables


# ---------------------------------------------------------------------------
# Summary CSV reader
# ---------------------------------------------------------------------------

def read_summary(summary_path: Path) -> list[dict[str, Any]]:
    """Read VALIDATION-SUMMARY.csv emitted by aggregate_validation.py.

    Returns a list of row-dicts with numeric fields coerced to float
    where possible.
    """
    rows: list[dict[str, Any]] = []
    if not summary_path.exists():
        return rows
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ("mean", "std", "median", "min", "max"):
                if k in row and row[k] != "":
                    try:
                        row[k] = float(row[k])
                    except (TypeError, ValueError):
                        pass
            if "n_seeds" in row and row["n_seeds"] != "":
                try:
                    row["n_seeds"] = int(row["n_seeds"])
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def read_run_values(runs_root: Path) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    """Walk runs_root and collect per-(scenario, window, metric) → [(seed, value)].

    Mirrors aggregate_validation.collect_metrics but keeps per-seed values
    needed for pairwise tests / synergy contrast / bootstrap CI.
    """
    bucket: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    if not runs_root.exists():
        return bucket
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.csv"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "ok":
            continue
        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("scenario", ""), row.get("window", ""), row.get("metric", ""))
                try:
                    value = float(row.get("value", "nan"))
                    seed = int(row.get("seed", "0"))
                except (TypeError, ValueError):
                    continue
                bucket.setdefault(key, []).append((seed, value))
    return bucket


# ---------------------------------------------------------------------------
# Builders — turn summary + per-run buckets into the row-dicts that the
# Table A-H generators expect.
# ---------------------------------------------------------------------------

def build_table_a_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Table A — pass through summary CSV rows verbatim."""
    return list(summary)


def build_table_b_rows(per_run: dict[tuple[str, str, str], list[tuple[int, float]]],
                        scenario_pairs: list[tuple[str, str]] | None = None,
                        windows: list[str] | None = None,
                        metrics: list[str] | None = None) -> list[dict[str, Any]]:
    """Table B — pairwise Mann-Whitney U + effect sizes.

    Iterates all (pair, window, metric) combinations present in `per_run`.
    Applies Holm-Bonferroni correction across the family at the end.
    """
    if scenario_pairs is None:
        scenarios = sorted({s for (s, _, _) in per_run.keys()})
        scenario_pairs = list(itertools.combinations(scenarios, 2))
    if windows is None:
        windows = sorted({w for (_, w, _) in per_run.keys()})
    if metrics is None:
        metrics = sorted({m for (_, _, m) in per_run.keys()})

    raw: list[dict[str, Any]] = []
    p_values: list[float] = []
    for (s_i, s_j), window, metric in itertools.product(scenario_pairs, windows, metrics):
        x = [v for (_, v) in per_run.get((s_i, window, metric), [])]
        y = [v for (_, v) in per_run.get((s_j, window, metric), [])]
        if len(x) < 2 or len(y) < 2:
            continue
        U, p_raw = stats.mann_whitney_u(x, y)
        d = stats.cohens_d(x, y)
        delta = stats.cliffs_delta(x, y)
        # CLES = P(X > Y) — same definition as U / (n_x * n_y).
        cles = float(U) / (len(x) * len(y))
        raw.append({
            "pair": f"{s_i} vs {s_j}", "window": window, "metric": metric,
            "U": U, "p_raw": p_raw, "p_holm": p_raw,  # filled below
            "cohens_d": d, "cliffs_delta": delta, "cles": cles,
        })
        p_values.append(p_raw)

    if p_values:
        adjusted = stats.holm_bonferroni(p_values)
        for row, p_adj in zip(raw, adjusted):
            row["p_holm"] = p_adj
    return raw


def build_table_d_rows(per_run: dict[tuple[str, str, str], list[tuple[int, float]]],
                        windows: list[str] | None = None,
                        metrics: list[str] | None = None,
                        n_bootstrap: int = 10_000) -> list[dict[str, Any]]:
    """Table D — synergy contrast Δ_interaction = (S2 − S0) − (S1 − S0).

    Hardcodes the S0/S1/S2 triplet from ANALYTICS-PLAN.md §4.2; if any of
    the three is missing for a given (window, metric), the row is skipped.
    """
    if windows is None:
        windows = sorted({w for (_, w, _) in per_run.keys()})
    if metrics is None:
        metrics = sorted({m for (_, _, m) in per_run.keys()})

    rows: list[dict[str, Any]] = []
    for window, metric in itertools.product(windows, metrics):
        s0_vals = [v for (_, v) in per_run.get(("S0", window, metric), [])]
        s1_vals = [v for (_, v) in per_run.get(("S1", window, metric), [])]
        s2_vals = [v for (_, v) in per_run.get(("S2", window, metric), [])]
        if not (s0_vals and s1_vals and s2_vals):
            continue
        delta = (sum(s2_vals) / len(s2_vals)) - 2 * (sum(s1_vals) / len(s1_vals)) + (sum(s0_vals) / len(s0_vals))
        # Bootstrap CI on the contrast — resample each scenario's seed pool
        # independently. Cheap version: pool deltas at scenario means via
        # bootstrap_ci on a fabricated delta-sample (1 value); replaced
        # by per-pair seed pairing if seeds align in a later iteration.
        import numpy as np
        rng = np.random.default_rng(0)
        n = min(len(s0_vals), len(s1_vals), len(s2_vals))
        deltas = np.empty(n_bootstrap)
        s0_arr = np.asarray(s0_vals)
        s1_arr = np.asarray(s1_vals)
        s2_arr = np.asarray(s2_vals)
        for i in range(n_bootstrap):
            i0 = rng.integers(0, len(s0_arr), n)
            i1 = rng.integers(0, len(s1_arr), n)
            i2 = rng.integers(0, len(s2_arr), n)
            deltas[i] = s2_arr[i2].mean() - 2 * s1_arr[i1].mean() + s0_arr[i0].mean()
        ci_lo, ci_hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        if ci_lo > 0:
            interp = "synergy"
        elif ci_hi < 0:
            interp = "sub-additive"
        else:
            interp = "additive (CI spans 0)"
        rows.append({
            "window": window, "metric": metric,
            "delta_interaction": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "interpretation": interp,
        })
    return rows


# ---------------------------------------------------------------------------
# Receipt block
# ---------------------------------------------------------------------------

def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
        return out[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _uv_lock_sha(repo_root: Path) -> str:
    lock = repo_root / "uv.lock"
    if not lock.exists():
        # Try python_refactor/uv.lock as fallback (per W6-1).
        lock = repo_root / "python_refactor" / "uv.lock"
    if not lock.exists():
        return "unknown"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def build_receipt_block(summary: list[dict[str, Any]],
                         repo_root: Path,
                         command: str | None = None,
                         now: _dt.datetime | None = None) -> str:
    """Build the §0 reproducibility-receipt block per ANALYTICS-PLAN §8."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    seeds: set[int] = set()
    for r in summary:
        for s in str(r.get("seeds_used", "")).split(","):
            s = s.strip()
            if s.isdigit():
                seeds.add(int(s))
    n_runs = sum(int(r.get("n_seeds", 0) or 0) for r in summary)
    seeds_list = sorted(seeds)
    seeds_str = (f"[{seeds_list[0]}, ..., {seeds_list[-1]}] ({len(seeds_list)} seeds)"
                  if len(seeds_list) > 4 else str(seeds_list))
    cmd = command or "python -m experiments.report --summary docs/VALIDATION-SUMMARY.csv --runs results/ --template docs/VALIDATION-RESULTS.md --output docs/VALIDATION-RESULTS.populated.md"
    return (
        "```\n"
        f"git_sha:    {_git_sha(repo_root)}\n"
        f"uv_lock:    sha256:{_uv_lock_sha(repo_root)}\n"
        f"generated:  {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"n_runs:     {n_runs}\n"
        f"seeds:      {seeds_str}\n"
        f"command:    {cmd}\n"
        "```"
    )


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------

# Each section is identified by its header. Mapping: header pattern → (builder, generator).
_SECTION_MAP: list[tuple[str, str]] = [
    (r"## 2\. Descriptive statistics \(Table A\)", "A"),
    (r"### 3\.1 Pairwise scenario comparison \(Table B\)", "B"),
    (r"### 4\.1 One-way ANOVA over Multi-Horizon levels \(Table C\)", "C"),
    (r"### 4\.2 Synergy contrast \(Table D\)", "D"),
    (r"### 4\.3 Per-horizon ablation \(Table E\)", "E"),
    (r"### 5\.2 By volatility regime \(Table F \+ Figure F10\)", "F"),
    (r"### 5\.3 By sub-window \(Table G\)", "G"),
    (r"## 7\. Paper-comparison receipt \(Table H\)", "H"),
]


def _replace_table_block(template: str, section_pattern: str, new_table: str) -> str:
    """Replace the Markdown-table block following `section_pattern`.

    Locates the section header, then finds the first Markdown table
    (line starting with `|`) below it, and replaces the contiguous
    table block with `new_table`. If no header / no table found,
    returns the template unchanged.
    """
    m = re.search(section_pattern, template)
    if not m:
        return template
    # Find the first table line at or after the header.
    rest = template[m.end():]
    table_start_match = re.search(r"^\|", rest, re.MULTILINE)
    if not table_start_match:
        return template
    abs_start = m.end() + table_start_match.start()
    # Find the end of the table — first non-table line (blank or non-|).
    lines = template[abs_start:].split("\n")
    end_offset = 0
    for line in lines:
        if line.startswith("|") or line.strip() == "":
            end_offset += len(line) + 1
            if line.strip() == "" and end_offset > 1:
                # blank line ends the table
                end_offset -= 1
                break
        else:
            break
    # Trim the trailing blank line we may have consumed.
    abs_end = abs_start + end_offset
    return template[:abs_start] + new_table + "\n" + template[abs_end:]


def populate(template: str,
              summary: list[dict[str, Any]],
              per_run: dict[tuple[str, str, str], list[tuple[int, float]]],
              receipt: str) -> str:
    """Populate the template, replacing placeholders where data exists."""
    out = template

    # Receipt block — replace section §0 fenced block.
    receipt_pattern = re.compile(r"```\ngit_sha:.*?```", re.DOTALL)
    if receipt_pattern.search(out):
        out = receipt_pattern.sub(receipt, out)

    # Table generators keyed by letter.
    gen_map = {
        "A": (tables.generate_table_a, build_table_a_rows(summary)),
        "B": (tables.generate_table_b, build_table_b_rows(per_run)),
        # C, E, F, G, H require per-run extras (ANOVA / ablation / regimes /
        # sub-windows / paper comparison) that aren't yet emitted by the
        # W7-1 scaffold — leave 🚧 markers intact (graceful incompleteness).
        "D": (tables.generate_table_d, build_table_d_rows(per_run)),
    }

    for pattern, letter in _SECTION_MAP:
        gen, rows = gen_map.get(letter, (None, []))
        if gen is None or not rows:
            continue
        rendered = gen(rows)
        out = _replace_table_block(out, pattern, rendered)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W8-4 validation-results report orchestrator.")
    parser.add_argument("--summary", type=Path, required=True,
                          help="VALIDATION-SUMMARY.csv path")
    parser.add_argument("--runs", type=Path, required=True,
                          help="results/ tree root (per-seed manifests + metrics)")
    parser.add_argument("--template", type=Path, required=True,
                          help="VALIDATION-RESULTS.md template path")
    parser.add_argument("--output", type=Path, required=True,
                          help="populated output file path")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(),
                          help="repo root for git SHA + uv.lock fingerprint")
    args = parser.parse_args(argv)

    summary = read_summary(args.summary)
    per_run = read_run_values(args.runs)
    template = args.template.read_text()
    receipt = build_receipt_block(summary, args.repo_root)
    populated = populate(template, summary, per_run, receipt)
    args.output.write_text(populated)
    print(f"[report] wrote {args.output} ({len(populated)} bytes, "
          f"{len(summary)} summary rows, {sum(len(v) for v in per_run.values())} per-run values)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

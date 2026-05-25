# W22 Probe AA — AMFC-vs-Hv-DM Telemetry Analysis Harness

**Status:** SHIPPED (analysis harness + synthetic smoke driver).
**Scope:** standalone module under `python_refactor/src/probes/` consuming
the telemetry produced by `select_amfc(..., collect_telemetry=True)` in
`python_refactor/src/algorithms/amfc_selector.py`. **No shared code is
modified by this probe** — it only reads telemetry the selector already
emits and renders comparisons.

## Hypothesis

**H-Probe-AA**: On real walk-forward data, the AMFC selector
(`Hv-DM-AMFC`) makes a measurably different decision from the legacy
Hv-DM selector on a non-trivial fraction of periods, and when it
diverges it consistently prefers candidates whose forecast distribution
is more certain.

Two falsifiable sub-claims:

1. **H-Probe-AA-disagreement**: AMFC disagrees with Hv-DM on
   **≥ 30 % of periods** on real-data walk-forward runs (FTSE / NASDAQ
   benchmark configurations). On synthetic Pareto fronts W22 Inspection 6
   already measured 82 % argmax disagreement; we expect production data
   to land lower (real KFs converge, ties are less frequent) but still
   well above 0 %.
2. **H-Probe-AA-low-variance**: On periods where AMFC picks a candidate
   different from Hv-DM, the AMFC pick has **forecast variance trace
   less than the population mean** more often than chance. Operationally
   captured via the
   `amfc_picks_more_certain_than_average` flag on the analyzer summary.

## Success criteria

Probe AA is considered to have produced a usable analysis substrate when:

- `summarize_telemetry` accepts the canonical telemetry list emitted by
  `amfc_selector.get_amfc_telemetry()` and returns the
  `SUMMARY_KEYS` closed-enum dict — including `agreement_rate`,
  `roi_delta_mean`, `tie_break_fire_rate`,
  `mean_amfc_pick_forecast_variance`, and the
  `amfc_picks_more_certain_than_average` flag.
- `format_summary_markdown` renders a markdown table embeddable in a
  W22 report; empty input renders an explicit "no telemetry records"
  placeholder.
- `compare_telemetry_summaries` produces a side-by-side markdown table
  with a Δ column (B − A) for ablation reporting.
- Telemetry IO (`save_telemetry_to_json`, `load_telemetry_from_json`)
  roundtrips losslessly.
- All Probe AA regression tests pass (see
  `python_refactor/tests/test_probe_aa_telemetry_analyzer.py`).

The empirical thresholds for H-Probe-AA-disagreement (≥ 30 %) and
H-Probe-AA-low-variance (flag = YES on majority of real-data runs)
are evaluated in a follow-up wired-walk-forward step that is the
operator's decision per the W22 master backlog (Section D — Probe AA
row, this row).

## Running the synthetic smoke script

The smoke script generates **synthetic** populations (no real ASMS run)
to verify the analyzer pipeline end-to-end:

```bash
# From the repo root
uv run python python_refactor/scripts/probe_aa_run_telemetry_smoke.py \
    --num-periods 30 \
    --num-solutions-per-period 6 \
    --seed 42
```

Outputs:

- Markdown summary printed to stdout (via `format_summary_markdown`).
- Raw telemetry persisted to
  `results/probe_aa_telemetry_42.json` (filename is parameterised by
  `--seed`).

To produce a second run for comparison:

```bash
uv run python python_refactor/scripts/probe_aa_run_telemetry_smoke.py \
    --num-periods 30 --num-solutions-per-period 6 --seed 7
```

Both runs land under `results/`; load them via
`load_telemetry_from_json` and pass through `summarize_telemetry` +
`compare_telemetry_summaries` to produce an ablation table.

## Wiring into production (operator decision — out of scope here)

To exercise the harness on a real walk-forward run:

1. Before the walk-forward loop, call
   `reset_amfc_telemetry()`.
2. Ensure `select_amfc(...)` is called with `collect_telemetry=True` for
   every period of interest (the dispatcher resolves this for
   `Hv-DM-AMFC`; the telemetry kwarg may already be wired in
   `experiments/walk_forward.py` for production runs).
3. After the loop, call `get_amfc_telemetry()` and pass the list through
   `summarize_telemetry` + `format_summary_markdown`.

Per the operator's directive on the W22 Probe AA backlog row, this
wiring is intentionally NOT performed by this probe ship — the analysis
substrate is delivered first; the live-data exercise is gated on the
operator's call to run it on the FTSE / NASDAQ benchmarks.

## File index

| File | Role |
|---|---|
| `python_refactor/src/probes/probe_aa_amfc_telemetry_analyzer.py` | analyzer module |
| `python_refactor/scripts/probe_aa_run_telemetry_smoke.py` | synthetic CLI smoke driver |
| `python_refactor/tests/test_probe_aa_telemetry_analyzer.py` | regression tests |
| `docs/W22-PROBE-AA-AMFC-TELEMETRY.md` | this contract / report |

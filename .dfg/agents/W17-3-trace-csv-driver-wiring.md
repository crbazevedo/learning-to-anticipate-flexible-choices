---
id: W17-3
role: code-fixer
name: Thread --lambda-trace-csv-path through walk_forward_report (W16-5-CARRY-1)
purpose: "Closes W16-5-CARRY-1 + W16-4-CARRY-1. Add CLI arg to walk_forward_report.py + thread to run_walk_forward so smokes emit lambda_tip_trace.csv and can directly assert 'both λ arms fire'."
wave: W17
unit: W17-3
depends_on: []
blocks: [W17-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - python_refactor/experiments/walk_forward_report.py
    - python_refactor/experiments/walk_forward.py
output_contract:
  files:
    - python_refactor/experiments/walk_forward_report.py
    - python_refactor/tests/test_walk_forward_report_lambda_trace.py
  branch_name: feat/w17-3-trace-csv-driver-wiring
  acceptance: >
    walk_forward_report.py exposes --lambda-trace-csv-path CLI arg
    (default None); when set, threads to run_walk_forward via the
    existing kwarg from W16-4. Trace CSV is emitted with header +
    rows. ≥ 2 regression tests covering CLI arg parsing + path
    propagation (mock the run_walk_forward call).
dispatch_instructions: |
  Closes W16-5-CARRY-1 + W16-4-CARRY-1.

  Background. W16-4 added `lambda_trace_csv_path` kwarg to
  run_walk_forward AND ExperimentManager honors data["lambda_trace_csv_path"].
  What's missing is the driver-side CLI flag in walk_forward_report.

  Surgical workflow:

  1. argparse: add `--lambda-trace-csv-path` (Path, default=None) to
     walk_forward_report.py's argparser

  2. Thread args.lambda_trace_csv_path through to run_walk_forward(...)
     in the ProcessPoolExecutor submission

  3. Important: each (scenario, seed) pair must write to its own CSV
     OR all share one and append (W16-4 supports append). For
     simplicity start with all-share-one + append; if races appear
     in production, switch to per-pair CSVs (filed as scar).

  4. Tests at python_refactor/tests/test_walk_forward_report_lambda_trace.py:
       - CLI parser accepts --lambda-trace-csv-path
       - default is None
       - when set, the value propagates to run_walk_forward kwarg
         (use unittest.mock to capture)

  What NOT to do:
    - Don't change run_walk_forward signature (W16-4 already added
      the kwarg).
    - Don't change ExperimentManager (W16-4 already wired data dict).
    - Don't add per-pair CSV split (file as scar if races appear).
    - Don't touch sms_emoa / operators / anticipatory.

  PR body MUST cite W16-5 retro's W16-5-CARRY-1 entry verbatim.
---

# W17-3 — Thread --lambda-trace-csv-path through walk_forward_report

Closes W16-5-CARRY-1 + W16-4-CARRY-1.

## Why

W16-5 retro: "trace assertions DEFERRED. walk_forward_report.py
doesn't yet pass --lambda-trace-csv-path through to run_walk_forward
(W16-4 kwarg exists; just driver-side wiring needed). Future smokes
should directly assert 'both λ arms fire' rather than infer."

This unit ships the ~10 minutes of driver wiring. With W17-2 also
landing (record_kf_residual wired), W17-5 can directly assert
mean(λ^K) > 0 for K=3 scenarios.

## Files to touch

- `python_refactor/experiments/walk_forward_report.py` — CLI +
  thread kwarg
- `python_refactor/tests/test_walk_forward_report_lambda_trace.py` —
  NEW; ≥ 2 tests

## Acceptance

- `--lambda-trace-csv-path` CLI arg exposed
- Path propagates to run_walk_forward
- ≥ 2 tests

---
id: W2-2
role: bug-fixer
name: Test-file + thesis_aligned_experiment import-style sweep
purpose: "Fix 5 test files + 1 source file using broken `from algorithms.X` style. Closes W1-3-CARRY-1/2."
wave: W2
unit: W2-2
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/tests/test_multi_horizon_anticipatory.py
    - python_refactor/tests/test_enhanced_n_step_prediction.py
    - python_refactor/tests/test_correspondence_integration.py
    - python_refactor/tests/test_sliding_window_dirichlet.py
    - python_refactor/tests/test_correspondence_mapping.py
    - python_refactor/src/experiments/thesis_aligned_experiment.py
output_contract:
  files:
    - python_refactor/tests/test_multi_horizon_anticipatory.py
    - python_refactor/tests/test_enhanced_n_step_prediction.py
    - python_refactor/tests/test_correspondence_integration.py
    - python_refactor/tests/test_sliding_window_dirichlet.py
    - python_refactor/tests/test_correspondence_mapping.py
    - python_refactor/src/experiments/thesis_aligned_experiment.py
  branch_name: feat/w2-2-import-style-sweep
  acceptance: >
    Every test file uses `from src.algorithms.X` (W1-2/W1-3/W1-4 pattern);
    thesis_aligned_experiment.py uses `from ..algorithms.X` (relative
    within src/experiments/). `grep "from algorithms\\." python_refactor/`
    returns 0 hits in any *.py file after the unit. All previously-
    collecting tests continue to pass; previously-broken collect-error
    tests either collect cleanly OR surface a different pre-existing
    issue (documented as inherited carry).
dispatch_instructions: |
  Mechanical sweep. For each file:
   - tests/*.py: `from algorithms.X` → `from src.algorithms.X`;
     drop the `sys.path.insert(...)` shim if present.
   - src/experiments/thesis_aligned_experiment.py: `from algorithms.X`
     → `from ..algorithms.X` (matches the relative-import convention
     used elsewhere in src/).
  Verify in `.venv-w1` with PYTHONPATH=. that all 6 files at least
  COLLECT cleanly; surface any newly-visible per-test failures as
  inherited carries (those are W2-2's surface, not regressions).
---

# W2-2 — Test-file + thesis_aligned_experiment import-style sweep

Closes W1-3-CARRY-1 (5 test files) + W1-3-CARRY-2 (thesis_aligned_experiment).

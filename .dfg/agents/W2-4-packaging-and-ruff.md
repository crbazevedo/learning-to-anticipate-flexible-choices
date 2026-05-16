---
id: W2-4
role: tooling-author
name: pyproject.toml + ruff fix on 3 legacy files
purpose: "Add packaging substrate (pyproject.toml) + close 11 F401 ruff errors in 3 legacy files. Closes packaging gap."
wave: W2
unit: W2-4
depends_on: []
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/requirements.txt  # 5 runtime + 2 dev deps to lift into pyproject
    - python_refactor/data_analysis.py  # legacy F401 errors
    - python_refactor/ftse_data_downloader.py
    - python_refactor/ftse_data_downloader_v2.py
output_contract:
  files:
    - pyproject.toml
    - python_refactor/data_analysis.py
    - python_refactor/ftse_data_downloader.py
    - python_refactor/ftse_data_downloader_v2.py
  branch_name: feat/w2-4-packaging-and-ruff
  acceptance: >
    pyproject.toml at repo root declares the project metadata
    (name=learning-to-anticipate-flexible-choices, dynamic-version
    OK, runtime deps from requirements.txt). Ruff F401 count drops
    by 11 across the 3 legacy files (verified via
    `ruff check --select F401 python_refactor/data_analysis.py
    python_refactor/ftse_data_downloader*.py` returning 0 errors).
    No file outside output_contract is touched.
dispatch_instructions: |
  1. Author pyproject.toml at repo root with PEP 621 metadata:
     - name = "learning-to-anticipate-flexible-choices"
     - version = "0.1.0"
     - description = "Python implementation of Azevedo & Von Zuben (2015) anticipatory MCDM"
     - readme = "README.md"
     - requires-python = ">=3.10"
     - dependencies from requirements.txt (numpy, pandas, scipy, matplotlib, scikit-learn)
     - optional-dependencies.dev = pytest + pytest-cov + ruff
     - tool.ruff.lint section with sensible defaults
     - Reference docs/paper.pdf in URLs section
  2. Apply `ruff check --select F401 --fix python_refactor/data_analysis.py python_refactor/ftse_data_downloader*.py` to close F401 errors in scope.
  3. Verify scope: only the 3 legacy files + pyproject.toml are modified.
---

# W2-4 — pyproject.toml + ruff fix on 3 legacy files

Adds the packaging substrate the audit flagged as missing + closes
the 11 F401 errors in legacy files (data_analysis, ftse_data_downloader
v1+v2). File-disjoint from W2-1 / W2-2 / W2-3.

Does NOT add uv.lock — locked installs are a separate concern.

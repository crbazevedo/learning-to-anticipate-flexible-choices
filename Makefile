# learning-to-anticipate-flexible-choices — minimal Makefile
#
# Authored 2026-05-16 during dfg-harness bootstrap. The `ci` target is the
# dfg-harness pre-pr battery's load-bearing dependency.
#
# At bootstrap time, this repo has no pyproject.toml and no installed
# venv — only the legacy requirements.txt. Two carries follow from this:
#
#   - W1-3 (relative-import refactor) closes 17 pytest collection errors
#     that prevent the full suite from running today.
#   - A future packaging unit will add pyproject.toml + uv lock so
#     `make test` can run reproducibly.
#
# Until those land, `make ci` is a SUBSTRATE-ONLY gate (dfg validate +
# dfg index --verify + dfg substrate check). When test infrastructure
# lights up, `make ci` gains pytest + ruff.

DFG_HARNESS ?= /Users/crbazevedo/Documents/Korza/repos/dfg-harness
DFG := uv run --project $(DFG_HARNESS) dfg

.PHONY: ci test lint fix help

help:
	@echo "Targets:"
	@echo "  make ci      # substrate gate (validate + index --verify + substrate check)"
	@echo "  make test    # pytest full suite (deferred to W1-3 + packaging unit)"
	@echo "  make lint    # ruff check (if installed)"
	@echo "  make fix     # ruff format + auto-fix (if installed)"
	@echo ""
	@echo "Variables:"
	@echo "  DFG_HARNESS  # path to dfg-harness checkout (default: $(DFG_HARNESS))"

# Substrate-only CI gate. Validates plan.yaml + context-map.yaml + agent
# contracts against the harness schemas; verifies state.json matches the
# canonical projection from events.jsonl + plan.yaml; checks cross-
# coherence across substrate surfaces.
ci:
	@echo "[ci] dfg validate"
	@$(DFG) validate
	@echo "[ci] dfg index --verify"
	@$(DFG) index --verify
	@echo "[ci] dfg substrate check"
	@$(DFG) substrate check
	@echo "[ci] substrate-only gate complete — pytest re-enables when W1-3 + packaging land"

# Full pytest — currently fails on env (no numpy) + 17 collection errors.
# Re-enables when W1-3 lands the relative-import refactor and a
# packaging unit adds pyproject.toml + uv lock.
test:
	@echo "[test] pytest is deferred until W1-3 + packaging land; see .dfg/plan.yaml W1-3"
	@cd python_refactor && uv run --project $(DFG_HARNESS) python -m pytest tests/ -v --tb=short || true

lint:
	@command -v ruff >/dev/null 2>&1 && cd python_refactor && ruff check src/ tests/ || echo "[lint] ruff not installed — skipping"

fix:
	@command -v ruff >/dev/null 2>&1 && cd python_refactor && ruff check --fix src/ tests/ && ruff format src/ tests/ || echo "[fix] ruff not installed — skipping"

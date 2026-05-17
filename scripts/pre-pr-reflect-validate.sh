#!/usr/bin/env bash
# W9-3 — retro-hygiene gate
#
# Retires sub-papercut #23: retro frontmatter wrong-keys silently
# skipping ReflectionEmit. For every retro present at
# .dfg/retrospectives/W*/W*-*.md, runs `dfg reflect --validate <unit-id>`
# and surfaces failures.
#
# Exit codes:
#   0 — every retro on disk has a matching ReflectionEmit event
#   1 — at least one retro is on disk but ReflectionEmit was skipped
#       (usually wrong frontmatter keys; sub-papercut #23 recurrence)
#
# Usage:
#   bash kit/scripts/pre-pr-reflect-validate.sh [--wave WAVE_ID]
#
# Wire into pre-pr:
#   make ci can add: bash kit/scripts/pre-pr-reflect-validate.sh

set -euo pipefail

# Default: validate only the latest wave directory (the one currently
# being worked on). Older waves may have a backlog of non-canonical-key
# retros that pre-date this gate; use --all to sweep them.
WAVE_FILTER=""
SWEEP_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wave) WAVE_FILTER="$2"; shift 2 ;;
    --all)  SWEEP_ALL=1; shift ;;
    *) shift ;;
  esac
done
if [[ -z "${WAVE_FILTER}" && "${SWEEP_ALL}" -eq 0 ]]; then
  # Auto-pick latest wave by numeric sort.
  WAVE_FILTER=$(ls .dfg/retrospectives 2>/dev/null | grep -E '^W[0-9]+$' | sort -V | tail -1 || true)
fi

# Resolve dfg invocation. PREFER the dfg-harness uv-project checkout
# because it bundles kit/SCHEMAS that `dfg reflect --validate` needs
# for structural-validity checks. Fall back to PATH dfg only if no
# checkout is found (works for repos that vendor the schemas).
DFG_HARNESS_ROOT="${DFG_HARNESS_ROOT:-/Users/crbazevedo/Documents/Korza/repos/dfg-harness}"
if [[ -d "${DFG_HARNESS_ROOT}" ]]; then
  DFG=(uv run --project "${DFG_HARNESS_ROOT}" dfg)
elif command -v dfg >/dev/null 2>&1; then
  DFG=(dfg)
else
  echo "[reflect-validate] dfg not on PATH and DFG_HARNESS_ROOT=${DFG_HARNESS_ROOT} not found"
  exit 2
fi

retros_root=".dfg/retrospectives"
if [[ ! -d "${retros_root}" ]]; then
  echo "[reflect-validate] no .dfg/retrospectives/ — nothing to validate"
  exit 0
fi

failures=0
checked=0
for retro in $(find "${retros_root}" -name 'W*-*.md' | sort); do
  base=$(basename "${retro}" .md)
  # Expect base = W<wave>-<unit>, e.g. W8-1.
  if [[ ! "${base}" =~ ^W[0-9]+-[0-9]+$ ]]; then
    continue
  fi
  wave="${base%-*}"
  if [[ -n "${WAVE_FILTER}" && "${wave}" != "${WAVE_FILTER}" ]]; then
    continue
  fi
  checked=$((checked + 1))
  if ! "${DFG[@]}" reflect --validate "${base}" >/dev/null 2>&1; then
    echo "[FAIL] ${base}: retro on disk but no ReflectionEmit event"
    echo "       likely cause: frontmatter uses non-canonical keys"
    echo "       fix: ADR-004 keys are what_worked, what_broke,"
    echo "            what_youd_change, assumption_to_challenge"
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -gt 0 ]]; then
  echo ""
  echo "[reflect-validate] ${failures}/${checked} retro(s) FAILED validation"
  echo "[reflect-validate] sub-papercut #23 has recurred — fix retro frontmatter keys"
  exit 1
fi

echo "[reflect-validate] ${checked} retro(s) all have matching ReflectionEmit events"
exit 0

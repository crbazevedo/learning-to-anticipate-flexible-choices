"""W22 Probe V — opt-in fix combination ablation framework.

Per ``docs/W22-MASTER-BACKLOG.md`` Section B and Section D (Probe V row):
the W22 inspection cycle shipped FOUR opt-in structural fixes that are
all gated OFF by default::

    NC13b  smooth TIP clamp           env  W22_NC13B_SMOOTH_CLAMP=1
    NC27-deep DirichletPosterior      env  W22_NC27_PREDICTOR=dirichlet_posterior
    NC30 c continuous variance penalty   dm_config['variance_penalty'] > 0
    NC31  TIP Defn 6.1 conditional    env  W22_NC31_TIP_CONDITIONAL=1

There are 2**4 = 16 possible combinations (15 non-trivial + 1 baseline).
Manually running each through ASMS is expensive. This module ships the
ABLATION FRAMEWORK that systematizes the comparison:

* enumerate combinations (full / pairwise / single-fix)
* translate each combination into a (env-var dict, dm_config dict) pair
* hand it to a wealth-measurement function (mock by default; operator
  plugs in the real ASMS runner later)
* aggregate the results into a markdown comparison table

The framework deliberately does NOT touch any shared code path
(``sms_emoa.py``, ``anticipatory_learning.py``, ``amfc_selector.py``,
``dirichlet_posterior.py``, ``tip_calculator.py``, ``kalman_filter.py``).
Standard-library + numpy only.

The mock wealth function is a deterministic hash-based synthetic — it
exists so the framework's plumbing is exercised end-to-end in CI without
requiring an ASMS run. The real signal lives in the operator's
plug-in.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from itertools import combinations
from typing import Callable, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Canonical list of the four W22 opt-in structural fixes.
# Order is fixed so enumerate_combinations() output is reproducible across
# Python versions (set iteration order is otherwise insertion-order, but the
# *test* uses set semantics so the constant itself is the source of truth).
# ---------------------------------------------------------------------------
OPT_IN_FIXES: list[str] = [
    "NC13b_smooth_clamp",
    "NC27_deep_posterior",
    "NC30c_variance_penalty",
    "NC31_tip_conditional",
]


# Env-var mapping for the THREE env-gated fixes.  NC30c is intentionally NOT
# in this dict — it lives in ``dm_config['variance_penalty']`` and is handled
# by :func:`combination_to_dm_config`.
ENV_VAR_MAP: dict[str, tuple[str, str]] = {
    "NC13b_smooth_clamp": ("W22_NC13B_SMOOTH_CLAMP", "1"),
    "NC27_deep_posterior": ("W22_NC27_PREDICTOR", "dirichlet_posterior"),
    "NC31_tip_conditional": ("W22_NC31_TIP_CONDITIONAL", "1"),
}


# ---------------------------------------------------------------------------
# Combination enumeration
# ---------------------------------------------------------------------------
def enumerate_combinations(strategy: str = "full") -> list[set[str]]:
    """Enumerate fix-set combinations under a chosen strategy.

    Parameters
    ----------
    strategy : str
        One of:

        * ``"full"``      — all 2**4 = 16 combinations (baseline + every
          non-empty subset of OPT_IN_FIXES)
        * ``"pairwise"``  — baseline + each unordered pair (C(4,2) = 6)
        * ``"single"``    — baseline + each fix in isolation (4)

    Returns
    -------
    list[set[str]]
        Each element is a (possibly empty) subset of ``OPT_IN_FIXES``.
        The empty set (baseline) is always the first element.

    Raises
    ------
    ValueError
        If ``strategy`` is not recognised.
    """
    baseline: set[str] = set()
    if strategy == "full":
        result: list[set[str]] = [baseline]
        for r in range(1, len(OPT_IN_FIXES) + 1):
            for combo in combinations(OPT_IN_FIXES, r):
                result.append(set(combo))
        return result
    if strategy == "pairwise":
        result = [baseline]
        for combo in combinations(OPT_IN_FIXES, 2):
            result.append(set(combo))
        return result
    if strategy == "single":
        result = [baseline]
        for fix in OPT_IN_FIXES:
            result.append({fix})
        return result
    raise ValueError(
        f"strategy must be 'full', 'pairwise', or 'single'; got {strategy!r}"
    )


# ---------------------------------------------------------------------------
# Combination → runtime configuration
# ---------------------------------------------------------------------------
def combination_to_env_dict(fix_set: set[str]) -> dict[str, str]:
    """Translate a fix-set into a dict of env-var assignments.

    NC30c is skipped because it is not env-gated — it lives in dm_config.

    Parameters
    ----------
    fix_set : set[str]
        Subset of ``OPT_IN_FIXES``. Unknown entries are ignored (so the
        caller can hand us a superset without first filtering).

    Returns
    -------
    dict[str, str]
        ``{env_var_name: env_var_value}`` for every fix in ``fix_set``
        that has an env-var mapping. Empty dict for the baseline.
    """
    env: dict[str, str] = {}
    for fix in fix_set:
        if fix in ENV_VAR_MAP:
            name, value = ENV_VAR_MAP[fix]
            env[name] = value
    return env


def combination_to_dm_config(
    fix_set: set[str], base_dm_config: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Build a dm_config dict that reflects whether NC30c is active.

    Parameters
    ----------
    fix_set : set[str]
        Subset of ``OPT_IN_FIXES``.
    base_dm_config : Mapping[str, object] | None
        Starting dm_config to merge into. If provided AND it contains a
        ``"variance_penalty"`` key, that key acts as the OFF-value (i.e.,
        the value used when NC30c is NOT in ``fix_set``). If absent, the
        OFF-value is 0.0 and the ON-value is 1.0.

        Any other keys in ``base_dm_config`` are passed through unchanged.

    Returns
    -------
    dict[str, object]
        A NEW dict (the input is not mutated) with the appropriate
        ``"variance_penalty"`` value plus all other base keys.
    """
    out: dict[str, object] = dict(base_dm_config) if base_dm_config else {}
    off_value: float = float(out.get("variance_penalty", 0.0))
    on_value: float = 1.0
    out["variance_penalty"] = on_value if "NC30c_variance_penalty" in fix_set else off_value
    return out


# ---------------------------------------------------------------------------
# Wealth measurement — mock + real-runner plumbing
# ---------------------------------------------------------------------------
def _mock_wealth_fn(fix_set: set[str]) -> float:
    """Deterministic synthetic wealth derived from the fix-set.

    Uses SHA1 of the sorted fix-set so the value is stable across Python
    versions / hash-randomisation. The output range is ~[100.0, 200.0]
    so it reads like a wealth-in-percent in the comparison table.

    The baseline (empty set) deliberately maps to a known anchor (100.0)
    so tests can pin it.
    """
    if not fix_set:
        return 100.0
    key = ",".join(sorted(fix_set)).encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    # Take the first 8 hex chars → up-to-32-bit int → map to [100, 200).
    bucket = int(digest[:8], 16) % 10_000
    return 100.0 + bucket / 100.0


def run_ablation_synthetic(
    combinations_to_run: list[set[str]],
    wealth_fn: Callable[[set[str]], float] | None = None,
) -> dict[frozenset, float]:
    """Run the ablation framework, returning per-combination wealth.

    Parameters
    ----------
    combinations_to_run : list[set[str]]
        Combinations to evaluate (typically the output of
        :func:`enumerate_combinations`).
    wealth_fn : Callable[[set[str]], float] | None
        Callable mapping a fix-set to a wealth scalar. Defaults to
        :func:`_mock_wealth_fn` (the deterministic synthetic). Operators
        plug in the real ASMS runner here — see
        :func:`subprocess_runner_skeleton` for the recommended shape.

    Returns
    -------
    dict[frozenset, float]
        Mapping from ``frozenset(fix_set)`` (hashable) to wealth.
    """
    fn = wealth_fn if wealth_fn is not None else _mock_wealth_fn
    return {frozenset(combo): float(fn(combo)) for combo in combinations_to_run}


def subprocess_runner_skeleton(
    fix_set: set[str],
    command: list[str],
    base_dm_config: Mapping[str, object] | None = None,
    base_env: Mapping[str, str] | None = None,
    timeout_seconds: float = 600.0,
    parse_wealth_from_stdout: Callable[[str], float] | None = None,
) -> float:
    """Subprocess-runner skeleton that an operator can wire to real ASMS.

    The framework does NOT invoke this by default — it exists as the
    reference shape for ``wealth_fn`` plug-ins. Given a fix-set:

    1. Translate to env-vars (:func:`combination_to_env_dict`).
    2. Translate to dm_config (:func:`combination_to_dm_config`); the
       caller is responsible for piping ``dm_config`` into ``command``
       (e.g. via a CLI flag or a temp JSON file) — this skeleton does
       not assume any particular surface.
    3. Spawn ``command`` with the combined env.
    4. Parse the wealth-scalar from stdout via ``parse_wealth_from_stdout``
       (the operator's parser — knows the ASMS log format).

    Parameters
    ----------
    fix_set : set[str]
        Subset of OPT_IN_FIXES.
    command : list[str]
        argv to spawn (e.g. ``[sys.executable, "run_asms.py", "--ftse"]``).
    base_dm_config : Mapping[str, object] | None
        dm_config to merge NC30c into (used by the caller; the skeleton
        only computes it for the operator's convenience and returns it
        in the env under the key ``W22_PROBE_V_DM_CONFIG`` as a hint).
    base_env : Mapping[str, str] | None
        Extra env vars (e.g. ``PYTHONPATH``). The framework adds these
        AFTER copying ``os.environ`` so they override the inherited env.
    timeout_seconds : float
        Subprocess wall-clock cap. Default 10 min.
    parse_wealth_from_stdout : Callable[[str], float] | None
        Operator's parser. If None, returns float('nan') so the operator
        notices they forgot to wire it.

    Returns
    -------
    float
        Wealth scalar parsed from stdout, or NaN if no parser supplied.

    Notes
    -----
    Sub-papercut #25 (subprocess.run inheriting stdin): we pass
    ``stdin=subprocess.DEVNULL`` to avoid the hang receipt from W63
    release-prep.
    """
    # Build the env: start from current os.environ, layer fix-set vars, then
    # any operator overrides.
    env: dict[str, str] = dict(os.environ)
    env.update(combination_to_env_dict(fix_set))
    # NC30c hint — operator's parser can read this if they want.
    dm_config = combination_to_dm_config(fix_set, base_dm_config=base_dm_config)
    env["W22_PROBE_V_DM_CONFIG"] = repr(dm_config)
    if base_env:
        env.update(base_env)

    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,  # sub-papercut #25 guard
        timeout=timeout_seconds,
        check=False,
    )
    if parse_wealth_from_stdout is None:
        return float("nan")
    return float(parse_wealth_from_stdout(completed.stdout))


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------
def _label_combo(combo: frozenset) -> str:
    """Human-readable label for a fix-set in the comparison table."""
    if not combo:
        return "(baseline)"
    return " + ".join(sorted(combo))


def format_ablation_table(results: dict[frozenset, float]) -> str:
    """Render a markdown comparison table sorted by wealth descending.

    The baseline row (empty fix-set) is HIGHLIGHTED with a ``◀ baseline``
    annotation so the operator can read the uplift at a glance.

    Parameters
    ----------
    results : dict[frozenset, float]
        Output of :func:`run_ablation_synthetic`.

    Returns
    -------
    str
        Markdown-formatted table (header + rows + trailing newline).
    """
    if not results:
        return "| rank | combination | wealth | uplift vs baseline |\n|---|---|---|---|\n"

    baseline_wealth: float = float(results.get(frozenset(), float("nan")))
    # Sort by wealth DESC. Ties broken by lexicographic label for determinism.
    rows = sorted(
        results.items(),
        key=lambda item: (-item[1], _label_combo(item[0])),
    )
    lines = ["| rank | combination | wealth | uplift vs baseline |", "|---|---|---|---|"]
    for rank, (combo, wealth) in enumerate(rows, start=1):
        label = _label_combo(combo)
        if combo == frozenset():
            label = f"{label}  ◀ baseline"
            uplift_str = "—"
        elif np.isnan(baseline_wealth):
            uplift_str = "n/a"
        else:
            uplift = wealth - baseline_wealth
            uplift_str = f"{uplift:+.2f}"
        lines.append(f"| {rank} | {label} | {wealth:.2f} | {uplift_str} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point — quick smoke for the operator.
# ---------------------------------------------------------------------------
def _cli_smoke() -> int:
    combos = enumerate_combinations("full")
    results = run_ablation_synthetic(combos)
    sys.stdout.write(format_ablation_table(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually
    raise SystemExit(_cli_smoke())

"""W22 alt-signal probes.

Per ``docs/W22-ALT-SIGNAL-PROBES.md`` — contained standalone modules that
explore alternative signals (per-asset prediction, centrality tilts,
causal graphs, ...) WITHOUT touching shared code paths
(``sms_emoa.py``, ``anticipatory_learning.py``, etc.).

Probes here are intentionally side-effect-free: each module exposes a
class or function with explicit ``observe(...)`` / ``predict_*(...)``
methods so an experiment harness can drive them and a portfolio
optimizer can consume their output via a future, separate wiring step
(see per-probe docs for the integration sketch).
"""

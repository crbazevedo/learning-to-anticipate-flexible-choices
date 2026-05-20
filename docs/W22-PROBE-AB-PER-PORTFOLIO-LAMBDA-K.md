# W22 Probe AB — λ^K per-portfolio differentiation analyzer

**Status:** module shipped, integration deferred
**Module:** `python_refactor/src/probes/probe_ab_per_portfolio_lambda_k.py`
**Tests:** `python_refactor/tests/test_probe_ab_per_portfolio_lambda_k.py` (19 passing)
**Backlog row:** `docs/W22-MASTER-BACKLOG.md` Section C row AB
**Inspection reference:** `docs/W22-INSPECTIONS-SYNTHESIS.md` — "Anticipation
rate is poorly differentiated per period" / Inspection 6

---

## Hypothesis

**AB-H1.** A per-portfolio λ^K (computed from per-portfolio KF residual
windows) provides **meaningful discrimination** across portfolios in
heterogeneous-tracking regimes, where the current shared-window λ^K
provides exactly zero per-portfolio discrimination.

**AB-H1a (necessary condition).** In homogeneous tracking regimes —
where all portfolios share the same KF tracking quality — per-portfolio
λ^K collapses to the shared-window baseline (within noise). The
per-portfolio mode adds no spurious signal when the population is
genuinely undifferentiated.

**AB-H1b (sufficient condition).** As per-portfolio mean residuals
diverge (synthetic ``heterogeneity → 1.0``), the per-portfolio λ^K
range exceeds the **MODEST** discrimination band (≥ 0.05), and at
sufficient signal-to-noise (low intra-window σ) reaches **STRONG**
(≥ 0.15).

## Inspection 6 reference (exact citation)

From `docs/W22-INSPECTIONS-SYNTHESIS.md`:

> ### "Anticipation rate" is poorly differentiated per period
> - **λ^K** is solution-invariant per period (constant across portfolios)
> - **λ^H** is TIP-modulated but TIP often saturates at [0.05, 0.95] (NC13a)
> - The (1/(H-1)) prefactor in Eq 6.6 is constant per horizon
>
> Net: in saturated regimes, every solution gets approximately the SAME
> anticipation rate, and Δ_S argmax reduces to current-state argmax with
> a uniform anticipation offset. The "Flexible" in AMFC degenerates.

The probe is the **falsifier** for the implicit claim that re-shaping
λ^K to be per-portfolio would meaningfully restore the "Flexible" in
AMFC.

## Canonical formula (mirrored bit-for-bit from production)

Source of truth: `anticipatory_learning._compute_lambda_k(...)` lines
759-810 (non-empty branch).

```text
residual_sum = sum(window)
scale        = max(1.0, mean(window))
normalized   = 1 - exp(-residual_sum / (len(window) * scale))
lambda_k     = 0.5 * normalized              # ∈ [0, 0.5]
```

The probe replicates this formula in `_lambda_k_from_window(...)` and
exposes it via three callers:

| Caller | Window source | Output |
|---|---|---|
| `compute_shared_lambda_k(window)` | one shared K-period buffer (production) | scalar in [0, 0.5] |
| `compute_per_portfolio_lambda_k(windows_per_portfolio)` | per-portfolio K-period buffers | `np.ndarray` of shape `(n_portfolios,)` |
| `compare_lambda_k_modes(...)` | both | closed-enum dict (see below) |

Empty windows return 0.0. The probe deliberately does **not**
replicate the per-solution traditional-rate fallback used in production
warm-up (`anticipatory_learning._compute_lambda_k` lines 788-795) — that
fallback exists for backward-compat in periods before K real residuals
have been recorded, and is orthogonal to the per-portfolio
discrimination question the probe interrogates.

## Methodology

1. **Synthetic residual windows.**
   `synthesize_residual_windows(n_portfolios, window_size, mean_residual,
   std_residual, heterogeneity, rng)` produces per-portfolio K-period
   windows. The ``heterogeneity`` axis is the load-bearing knob:
   * ``heterogeneity = 0`` → all portfolio means equal ``mean_residual``
     (homogeneous tracking regime).
   * ``heterogeneity = 1.0`` → portfolio means drawn from
     ``N(mean_residual, mean_residual)`` then clipped positive (wide
     spread; heterogeneous tracking regime).
2. **Two-mode comparison.** `compare_lambda_k_modes(windows)` computes:
   * `shared_lambda_k` — λ^K from the **concatenated** windows
     (closest analytic stand-in for the production shared buffer).
   * `per_portfolio_lambda_k` — one λ^K per portfolio.
   * `per_portfolio_std` — std across the per-portfolio λ^K array.
   * `per_portfolio_range` — max − min across the per-portfolio λ^K
     array.
   * `discrimination_significance` — closed-enum label from the
     **range** value:

     | Range | Label |
     |---|---|
     | `< 0.05` | `NEGLIGIBLE` |
     | `[0.05, 0.15)` | `MODEST` |
     | `>= 0.15` | `STRONG` |
3. **Falsification protocol.** Run `compare_lambda_k_modes` across a
   heterogeneity sweep ``{0.0, 0.1, 0.3, 0.5, 0.7, 1.0}`` with fixed
   ``mean_residual``, ``std_residual``, ``n_portfolios``, ``window_size``.
   AB-H1 holds iff `discrimination_significance` is `NEGLIGIBLE` at
   heterogeneity = 0 and transitions to `MODEST` / `STRONG` at high
   heterogeneity.

## Success criteria

| Criterion | Source of truth |
|---|---|
| Shared-window formula matches production line-for-line | `test_compute_shared_lambda_k_matches_canonical_formula` |
| Shared-window output bounded in `[0, 0.5]` over wide scales | `test_compute_shared_lambda_k_bounded_in_zero_half` |
| Per-portfolio output shape `(n_portfolios,)` | `test_compute_per_portfolio_lambda_k_returns_array` |
| Per-portfolio mode collapses to shared when homogeneous | `test_per_portfolio_matches_shared_when_homogeneous` |
| Per-portfolio std exceeds NEGLIGIBLE band at heterogeneity=1 | `test_per_portfolio_differs_from_shared_when_heterogeneous` |
| `compare_lambda_k_modes` returns closed-enum keys | `test_compare_modes_returns_all_keys` |
| Significance label is `NEGLIGIBLE` at heterogeneity=0 | `test_discrimination_negligible_at_zero_heterogeneity` |
| Significance label is `MODEST`/`STRONG` at heterogeneity=1 | `test_discrimination_strong_at_high_heterogeneity` |
| Synthesizer shape, non-negativity, reproducibility | `test_synthesize_*` (5 tests) |

## Integration sketch (future, separate operator decision)

The probe ships ONLY analyzer + tests. Wiring it into production
requires a paired-diff on shared code paths and is OUT OF SCOPE for
this contract. The minimal sketch:

1. **Per-portfolio residual recording in `AnticipatoryLearning`.**
   Today `record_kf_residual(residual_squared_sum)` appends one scalar
   to a single buffer `self._kf_residual_window`. A per-portfolio
   variant would add::

       self._kf_residual_window_per_portfolio: dict[int, list[float]] = {}

       def record_kf_residual_per_portfolio(self, portfolio_id: int, r2: float):
           buf = self._kf_residual_window_per_portfolio.setdefault(portfolio_id, [])
           buf.append(float(r2))
           if len(buf) > self.window_size:
               self._kf_residual_window_per_portfolio[portfolio_id] = buf[-self.window_size:]

2. **Per-portfolio λ^K consumption inside `_compute_lambda_k`.**
   The non-empty branch (lines 797-810) reads a single window. The
   per-portfolio variant would key by `solution.portfolio_id` (or
   equivalent) and call `_lambda_k_from_window(...)` on that
   portfolio's window — with the same warm-up fallback to the
   per-solution traditional rate when its window is empty.
3. **Solution / Portfolio surface.** Solutions / portfolios need a
   stable identifier (`portfolio_id`) so the residual buffer can be
   keyed across generations. The hash of decision-vector indices is
   one option; explicit assignment at population init is cleaner.
4. **Per-portfolio residual emission from the walk-forward driver.**
   Whoever calls `record_kf_residual(sum_of_squared_innovations)`
   today needs to be refactored to emit one residual per portfolio
   (today the sum is aggregated across all portfolios / both
   objectives).
5. **Regression test (gate before merge).** Use this probe's
   `compare_lambda_k_modes` on real walk-forward residuals to confirm
   the new per-portfolio λ^K actually shows `MODEST`/`STRONG`
   discrimination on FTSE data. If real data shows `NEGLIGIBLE`
   discrimination, the refactor adds complexity without signal — AB-H1
   falsified on real data and the refactor should be aborted.

## Out-of-scope (explicitly NOT shipped)

* No modification to `anticipatory_learning.py`,
  `sms_emoa.py`, `kalman_filter.py`, or any other shared code.
* No `Solution` / `Portfolio` schema change.
* No empirical FTSE run — production residuals are not exposed
  per-portfolio yet, so the empirical part of AB-H1 cannot be
  evaluated until the integration sketch is funded.
* No replacement of the production warm-up fallback (per-solution
  traditional rate) — that is orthogonal to per-portfolio
  discrimination and outside the contract scope.

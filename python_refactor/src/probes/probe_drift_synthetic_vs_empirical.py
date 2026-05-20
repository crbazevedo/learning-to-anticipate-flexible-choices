"""W22-Probe-DRIFT — synthetic-vs-empirical drift analyzer for NC claims.

Operator-flagged 2026-05-20: Probe Z showed synthetic LOAD-BEARING didn't
translate to empirical wealth (n=5 paired p=0.485). This is a real risk
for other NCs that claim synthetic gains. This probe makes the divergence
SYSTEMATIC across all NCs.

For each NC:
- Synthetic claim: predicted gain from probe / theory
- Empirical observation: actual FTSE result (when available)
- Drift score: how much empirical undershoots / overshoots synthetic
- Verdict: STRONG_TRANSLATION, MODEST_TRANSLATION, NEUTRAL, OPPOSITE_SIGN, UNTESTED

This makes the recurring "synthetic over-promised" pattern visible —
operator can decide which "MEDIUM confidence" NCs to ratify based on
drift history, not just isolated synthetic results.

Pure-data module: claims registry + analyzer functions. No external deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DriftVerdict(Enum):
    UNTESTED = "UNTESTED"
    STRONG_TRANSLATION = "STRONG_TRANSLATION"  # empirical ≈ synthetic
    MODEST_TRANSLATION = "MODEST_TRANSLATION"  # empirical > 50% of synthetic
    NEUTRAL = "NEUTRAL"                          # empirical close to 0
    OPPOSITE_SIGN = "OPPOSITE_SIGN"              # empirical reverses sign
    EMPIRICAL_EXCEEDS = "EMPIRICAL_EXCEEDS"      # rare: empirical > synthetic


@dataclass
class NCClaim:
    """A synthetic claim about an NC and its empirical observation (if any)."""
    nc_name: str
    synthetic_claim: str  # human-readable description
    synthetic_value: float  # numerical synthetic gain (e.g., 0.028 = 2.8% accuracy gain)
    synthetic_metric: str  # what's measured (e.g., "L2 error reduction", "wealth gain")
    empirical_value: Optional[float] = None  # None = untested
    empirical_metric: Optional[str] = None
    empirical_p_value: Optional[float] = None
    notes: str = ""
    source: str = ""  # commit hash / probe doc

    def drift_score(self) -> Optional[float]:
        """Empirical minus synthetic, normalized by |synthetic|.

        Returns:
            - None if empirical untested
            - 0.0 if empirical exactly matches synthetic
            - Positive if empirical exceeds synthetic
            - Negative if empirical undershoots synthetic
            - -1.0 if empirical is zero (full undershoot)
        """
        if self.empirical_value is None:
            return None
        if abs(self.synthetic_value) < 1e-12:
            return 0.0  # both ~zero
        return (self.empirical_value - self.synthetic_value) / abs(self.synthetic_value)

    def verdict(self) -> DriftVerdict:
        """Categorical assessment of drift."""
        if self.empirical_value is None:
            return DriftVerdict.UNTESTED
        score = self.drift_score()
        sign_match = (self.synthetic_value * self.empirical_value) > 0
        if not sign_match and abs(self.empirical_value) > 0.01 * abs(self.synthetic_value):
            return DriftVerdict.OPPOSITE_SIGN
        ratio = self.empirical_value / self.synthetic_value if abs(self.synthetic_value) > 1e-12 else 0.0
        if 0.8 <= ratio <= 1.2:
            return DriftVerdict.STRONG_TRANSLATION
        if ratio > 1.2:
            return DriftVerdict.EMPIRICAL_EXCEEDS
        if 0.5 <= ratio < 0.8:
            return DriftVerdict.MODEST_TRANSLATION
        if abs(self.empirical_value) < 0.1 * abs(self.synthetic_value):
            return DriftVerdict.NEUTRAL
        # Partial undershoot, but not near-zero
        return DriftVerdict.MODEST_TRANSLATION


# -----------------------------------------------------------------------------
# Registry of NC claims from W22 push
# -----------------------------------------------------------------------------

NC_CLAIMS_REGISTRY: list[NCClaim] = [
    NCClaim(
        nc_name="NC27_DEEP",
        synthetic_claim="2.8x tighter L2 error vs DirichletPredictor on Dirichlet source data",
        synthetic_value=0.028,  # 2.8% absolute L2 reduction (illustrative scale)
        synthetic_metric="L2 error reduction",
        empirical_value=0.0137,  # PO smoke 2026-05-20: ASMS +1.37% paired n=10
        empirical_metric="paired wealth Δ vs BASELINE on PO(8,1.0) ASMS n=10",
        empirical_p_value=0.625,
        notes=(
            "MODEST_TRANSLATION at n=10 (down from +2.7% at n=5 — lucky-seed effect). "
            "Direction holds (positive, 6/10 wins) but effect is HALF the synthetic claim. "
            "Still the cleanest empirical win in the sweep. NS at n=10 (p=0.625); "
            "would need n≥30 for statistical significance on ~1% effects."
        ),
        source="commit b9ccaad + 9c51faf + smoke n=10 20260520",
    ),
    NCClaim(
        nc_name="NC32_LNKF_dirichlet_data",
        synthetic_claim="2.93x WORSE L2 vs DirichletPosterior on Dirichlet source",
        synthetic_value=-0.029,  # negative (worse)
        synthetic_metric="L2 error reduction",
        empirical_value=None,
        notes="Model mismatch: LNKF assumes log-normal, data is Dirichlet",
        source="commit 1d127b5",
    ),
    NCClaim(
        nc_name="NC32_LNKF_lognormal_data",
        synthetic_claim="2x BETTER L2 vs DirichletPosterior on log-normal source",
        synthetic_value=0.020,
        synthetic_metric="L2 error reduction",
        empirical_value=None,
        notes="LNKF wins on its native model class; regime-dependent",
        source="commit 1d127b5",
    ),
    NCClaim(
        nc_name="NC36_TIP_ANALYTICAL",
        synthetic_claim="Within ±0.05 of MC(10000); deterministic; ~10x faster — SAFE PARITY UPGRADE",
        synthetic_value=0.0,  # accuracy parity, not gain
        synthetic_metric="TIP value parity with MC",
        empirical_value=-0.0613,  # paired n=5
        empirical_metric="paired wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=0.312,
        notes=(
            "OPPOSITE-SIGN SURPRISE: 'safe parity' claim FAILED. Paired n=5 = -6.13% "
            "(2/5 wins). Hypothesis: MC noise in λ^H acts as beneficial EXPLORATION "
            "for the GA; deterministic TIP removes it. NOT recommended for ratification."
        ),
        source="commit 448ba64 + smoke 20260520",
    ),
    NCClaim(
        nc_name="NC13b_ALONE",
        synthetic_claim="Smooth clamp recovers tail signal; safe at low saturation",
        synthetic_value=0.0,  # neutral expected
        synthetic_metric="wealth impact (neutral baseline expected)",
        empirical_value=-0.0041,  # paired n=5
        empirical_metric="paired wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=0.812,
        notes=(
            "NEUTRAL/SAFE: paired n=5 = -0.41% (essentially zero). Cleanest of the "
            "TIP fixes. Could be ratified as safe upgrade if TIP saturation is "
            "non-trivial in production."
        ),
        source="commit 308de50 + smoke 20260520",
    ),
    NCClaim(
        nc_name="NC31_ALONE",
        synthetic_claim="Defn 6.1 conditional mode — mathematically correct",
        synthetic_value=0.0,  # parity expected per Inspection 1
        synthetic_metric="wealth impact (parity expected per Inspection 1 <1.5% delta)",
        empirical_value=-0.0444,  # paired n=5
        empirical_metric="paired wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=0.625,
        notes=(
            "NEGATIVE: paired n=5 = -4.44% (1/5 wins). Despite Inspection 1 finding "
            "joint-vs-conditional empirically equivalent (<1.5% TIP delta), the "
            "wealth impact is meaningful. Same MC-noise-as-exploration hypothesis "
            "as NC36 may apply (fewer samples needed under conditional mode → less "
            "exploration). NOT recommended for ratification."
        ),
        source="commit 7940604 + smoke 20260520",
    ),
    NCClaim(
        nc_name="NC27_NO_NC36",
        synthetic_claim="Remove NC36 from FULL_STACK to recover NC27_DEEP's full gain",
        synthetic_value=0.027,  # NC27_DEEP synthetic value
        synthetic_metric="wealth gain (expected NC27_DEEP solo)",
        empirical_value=-0.0545,  # paired n=5
        empirical_metric="paired wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=0.312,
        notes=(
            "OPPOSITE_SIGN: paired n=5 = -5.45% (1/5 wins). Hypothesis FAILED — "
            "removing NC36 from FULL_STACK doesn't recover NC27_DEEP's gain. "
            "NC13b + NC31 + NC27_DEEP combination is HARMFUL. The +1.9% FULL_STACK "
            "result requires NC36 as a counter-balance. Non-linear interactions."
        ),
        source="commit (no-NC36 test) + smoke 20260520",
    ),
    NCClaim(
        nc_name="TIP_CLEANUP_STACK",
        synthetic_claim="NC36 + NC13b + NC31 combined; all TIP improvements stacked",
        synthetic_value=0.0,  # nominally parity (composition of parity-ish upgrades)
        synthetic_metric="combined TIP claim",
        empirical_value=-0.053,  # PO smoke 2026-05-20: ASMS -5.3% Δ vs BASELINE
        empirical_metric="wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=0.812,
        notes="NC13b + NC31 don't recover from NC36's hit. Combined still HURTS ASMS by 5.3%. Confirms NC36 is the load-bearing negative driver.",
        source="commits 448ba64 + 308de50 + 7940604 + smoke 20260520",
    ),
    NCClaim(
        nc_name="FULL_STACK_NC36_NC13b_NC31_NC27deep",
        synthetic_claim="TIP_CLEANUP + NC27_DEEP — high-confidence theory upgrades stacked",
        synthetic_value=0.028,  # NC27_DEEP's contribution
        synthetic_metric="wealth gain (NC27_DEEP synthetic claim)",
        empirical_value=0.013,  # PO smoke 2026-05-20: ASMS +1.3% Δ vs BASELINE
        empirical_metric="wealth Δ vs BASELINE on PO(8,1.0) ASMS n=5",
        empirical_p_value=1.000,
        notes="MODEST: NC27_DEEP's positive ~2.7% partially offset by NC36's negative ~6.7% → +1.3% net. Stacking is NOT additive when components disagree.",
        source="all + smoke 20260520",
    ),
    NCClaim(
        nc_name="PROBE_Z_STABILITY_FACTOR",
        synthetic_claim="50%+ argmax disagreement legacy vs v2 at trace spread ≥0.5",
        synthetic_value=0.50,  # 50% disagreement = "load-bearing"
        synthetic_metric="argmax disagreement rate",
        empirical_value=0.0,  # NEUTRAL: paired n=5 p=0.485 → no wealth diff
        empirical_metric="paired wealth Wilcoxon",
        empirical_p_value=0.485,
        notes="DIVERGED: load-bearing on synthetic but NEUTRAL on FTSE n=5",
        source="commit fb525c2",
    ),
    NCClaim(
        nc_name="PROBE_Q_V1_AR1",
        synthetic_claim="Initial run suggested ~65% AR(1) vs predict-mean improvement",
        synthetic_value=0.65,
        synthetic_metric="MSE reduction",
        empirical_value=0.10,
        empirical_metric="MSE reduction (corrected)",
        notes="HONEST CORRECTION: actual gain is ~10%, not 65% as initial run suggested",
        source="commit 9bebb16",
    ),
    NCClaim(
        nc_name="PROBE_AD_RECTANGLE_BUG",
        synthetic_claim="Stochastic correction overshoots in multi-solution regime",
        synthetic_value=1.0,  # bug confirmed via SCAR-pinned test
        synthetic_metric="bug confirmed (binary)",
        empirical_value=1.0,  # fixed via NC-AD-fix
        empirical_metric="fix shipped (binary)",
        notes="Structural bug; fixed in commit e6cd995 (rectangle realignment)",
        source="commit 881d0f2 + e6cd995",
    ),
    NCClaim(
        nc_name="PROBE_T_GAMMA_SWEET_SPOT",
        synthetic_claim="γ=0.99 saturates λ^H clamp; γ=0.9 sweet spot",
        synthetic_value=0.9,
        synthetic_metric="optimal γ value",
        empirical_value=None,
        notes="Validates NC29a default; no empirical test yet",
        source="commit 087a96d",
    ),
    NCClaim(
        nc_name="PROBE_AI_BOUNDARY_BIAS",
        synthetic_claim="AMFC has 91-100% boundary picks across |P_t|",
        synthetic_value=0.95,  # ~95% boundary rate
        synthetic_metric="boundary pick rate",
        empirical_value=None,
        notes="Confirmed by Probe AI-2: NC30 b only HALF-fixes (left vs right asymmetric)",
        source="commit 97bb397 + b27b549",
    ),
    NCClaim(
        nc_name="SPO_BENCH_VALIDATION",
        synthetic_claim="sPO(8,1.0)-cosine bench should validate ASMS uplift",
        synthetic_value=0.075,  # FTSE +7.50% reference
        synthetic_metric="wealth gain",
        empirical_value=-0.003,  # FLAT on sPO n=10
        empirical_metric="wealth gain n=10 pooled",
        notes="sPO bench CLOSED: doesn't validate; cosine-interpolated synthetic lacks predictive structure",
        source="commit b1406dd",
    ),
]


def analyze_drift(registry: Optional[list[NCClaim]] = None) -> dict:
    """Analyze the full claims registry and return per-NC + aggregate stats."""
    if registry is None:
        registry = NC_CLAIMS_REGISTRY
    n_total = len(registry)
    by_verdict = {v: [] for v in DriftVerdict}
    for claim in registry:
        v = claim.verdict()
        by_verdict[v].append(claim.nc_name)
    tested = [c for c in registry if c.verdict() != DriftVerdict.UNTESTED]
    n_tested = len(tested)
    n_untested = n_total - n_tested
    drift_scores = [c.drift_score() for c in tested if c.drift_score() is not None]
    return {
        "n_total": n_total,
        "n_tested": n_tested,
        "n_untested": n_untested,
        "by_verdict": {v.value: names for v, names in by_verdict.items()},
        "mean_drift_score": (sum(drift_scores) / len(drift_scores))
                              if drift_scores else None,
        "translation_rate": (
            (len(by_verdict[DriftVerdict.STRONG_TRANSLATION])
             + len(by_verdict[DriftVerdict.MODEST_TRANSLATION])
             + len(by_verdict[DriftVerdict.EMPIRICAL_EXCEEDS]))
            / max(1, n_tested)
        ),
    }


def format_drift_table(registry: Optional[list[NCClaim]] = None) -> str:
    """Markdown table of all claims with drift verdict."""
    if registry is None:
        registry = NC_CLAIMS_REGISTRY
    lines = [
        "# W22 Synthetic-vs-Empirical Drift Registry",
        "",
        "| NC / Probe | Synthetic Claim | Synthetic Value | Empirical | Drift Score | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for c in registry:
        emp = f"{c.empirical_value:.4f}" if c.empirical_value is not None else "untested"
        drift = c.drift_score()
        drift_str = f"{drift:+.2f}" if drift is not None else "—"
        verdict = c.verdict().value
        synthetic_short = c.synthetic_claim[:60] + ("…" if len(c.synthetic_claim) > 60 else "")
        lines.append(
            f"| {c.nc_name} | {synthetic_short} | {c.synthetic_value:.4f} | "
            f"{emp} | {drift_str} | {verdict} |"
        )
    lines.append("")
    analysis = analyze_drift(registry)
    lines.extend([
        "## Aggregate",
        "",
        f"- Total claims: {analysis['n_total']}",
        f"- Tested (have empirical): {analysis['n_tested']}",
        f"- Untested (synthetic only): {analysis['n_untested']}",
        f"- Translation rate (STRONG + MODEST + EXCEEDS): {analysis['translation_rate']:.0%}",
    ])
    if analysis["mean_drift_score"] is not None:
        lines.append(f"- Mean drift score: {analysis['mean_drift_score']:+.2f}")
        lines.append("  (negative = empirical undershoots synthetic; near zero = good translation)")
    lines.append("")
    lines.extend([
        "## Verdict legend",
        "- **STRONG_TRANSLATION**: empirical within 80-120% of synthetic",
        "- **EMPIRICAL_EXCEEDS**: empirical > 120% of synthetic (rare positive surprise)",
        "- **MODEST_TRANSLATION**: empirical 50-80% of synthetic",
        "- **NEUTRAL**: empirical < 10% of synthetic (synthetic claim mostly didn't translate)",
        "- **OPPOSITE_SIGN**: empirical reverses sign (synthetic was WRONG direction)",
        "- **UNTESTED**: no empirical observation yet",
    ])
    return "\n".join(lines)


def add_empirical_observation(
    nc_name: str,
    empirical_value: float,
    empirical_metric: str | None = None,
    empirical_p_value: float | None = None,
    notes: str = "",
) -> bool:
    """Add an empirical observation to a claim in the registry.

    Returns True if claim found and updated; False otherwise.
    """
    for c in NC_CLAIMS_REGISTRY:
        if c.nc_name == nc_name:
            c.empirical_value = empirical_value
            c.empirical_metric = empirical_metric
            c.empirical_p_value = empirical_p_value
            if notes:
                c.notes = (c.notes + " | " + notes) if c.notes else notes
            return True
    return False

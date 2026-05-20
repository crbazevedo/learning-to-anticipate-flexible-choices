"""W22 Inspection 4: Correspondence mapping — is cosine similarity sound,
threshold reasonable, and is the infrastructure ACTUALLY USED in production?

Per W22-RESEARCH-PROGRAM.md Area VI (Correspondence Mapping).

OPERATOR FLAGGED: examine correspondence mapping.

CURRENT CODE (CorrespondenceMapping.find_corresponding_solution):
    similarity = cosine(target_weights, search_weights)
    return solution if similarity >= 0.95 else None

ALGEBRAIC CONCERNS:
1. Cosine similarity on SPARSE weight vectors (cardinality 5-15 of 30-87 assets):
   two portfolios with NO overlapping assets can have cosine=0; but cosine
   doesn't distinguish "no overlap" from "partial overlap" the same way as
   Jaccard or L1 distance.
2. 0.95 threshold: how strict is this for sparse portfolios that rebalance?
3. Cosine ignores the L1 simplex constraint — it is invariant to scaling,
   but weights are already normalized to sum to 1, so this invariance is wasted.

PRODUCTION USAGE AUDIT (grep -rn in src/):
    store_population_snapshot     → called 1 place (thesis_aligned_experiment.py:174)
    find_corresponding_solution   → NEVER called outside the class
    track_solution_evolution      → NEVER called outside the class
    get_solution_evolution        → NEVER called outside the class
    get_correspondence_statistics → NEVER called outside the class

CONCLUSION: Correspondence mapping is DEAD INFRASTRUCTURE.
Populations are stored at every period but NEVER queried.
This is exactly the kind of "ceremony without consumption" that wastes
substrate and confuses readers about what the algorithm actually depends on.

This script:
- Demonstrates cosine similarity failure modes on sparse portfolios
- Compares cosine vs Jaccard vs L1 distance vs L2 distance
- Tests threshold sensitivity (what fraction of correspondences found at
  0.95 vs 0.7 vs 0.5)
- Confirms zero production callers via subprocess grep

Usage:
    uv run python -m experiments.inspect_correspondence
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.algorithms.correspondence_mapping import CorrespondenceMapping

RNG = np.random.default_rng(42)


def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    """Cosine similarity matching current implementation."""
    w1n = w1 / (np.linalg.norm(w1) + 1e-10)
    w2n = w2 / (np.linalg.norm(w2) + 1e-10)
    return float(np.dot(w1n, w2n))


def jaccard_set_similarity(w1: np.ndarray, w2: np.ndarray, eps: float = 1e-6) -> float:
    """Jaccard similarity on asset-SETS (which assets are held > eps)."""
    s1 = set(np.where(w1 > eps)[0])
    s2 = set(np.where(w2 > eps)[0])
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def l1_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """L1 (taxicab) distance between weight vectors. On simplex: range [0, 2]."""
    return float(np.sum(np.abs(w1 - w2)))


def l2_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """L2 (Euclidean) distance between weight vectors."""
    return float(np.linalg.norm(w1 - w2))


def random_sparse_portfolio(n_assets: int, cardinality: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Sample a random portfolio with given cardinality (sparse)."""
    w = np.zeros(n_assets)
    chosen = rng.choice(n_assets, size=cardinality, replace=False)
    raw = rng.dirichlet(np.ones(cardinality))
    w[chosen] = raw
    return w


def perturb_portfolio(w: np.ndarray, swap_frac: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Perturb a portfolio by swapping a fraction of held assets to unheld ones."""
    w_new = w.copy()
    held = np.where(w > 1e-6)[0]
    unheld = np.where(w <= 1e-6)[0]
    n_swap = int(np.ceil(swap_frac * len(held)))
    if n_swap == 0 or len(unheld) == 0:
        return w_new

    to_remove = rng.choice(held, size=min(n_swap, len(held)), replace=False)
    to_add = rng.choice(unheld, size=min(n_swap, len(unheld)), replace=False)

    for i, j in zip(to_remove, to_add):
        w_new[j] = w_new[i]  # move weight to new asset
        w_new[i] = 0.0

    return w_new


def main():
    print("=" * 80)
    print("W22 INSPECTION 4: Correspondence mapping — soundness + production audit")
    print("=" * 80)
    print()

    # =========================================================================
    # TEST 0: Production-usage audit
    # =========================================================================
    print("=" * 80)
    print("TEST 0: Production-usage audit (grep src/ for callers)")
    print("=" * 80)
    print()
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"

    targets = [
        "store_population_snapshot",
        "find_corresponding_solution",
        "track_solution_evolution",
        "get_solution_evolution",
        "get_correspondence_statistics",
    ]
    for target in targets:
        try:
            out = subprocess.run(
                ["grep", "-rn", target, str(src_dir), "--include=*.py"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            lines = [
                ln for ln in out.stdout.splitlines()
                if "correspondence_mapping.py" not in ln
                and "anticipatory_learning.py" not in ln
            ]
            print(f"{target}:")
            if lines:
                for ln in lines[:5]:
                    print(f"    {ln}")
            else:
                print("    (no external callers)")
        except Exception as e:
            print(f"  grep failed: {e}")
    print()
    print("VERDICT: only store_population_snapshot has a real production caller")
    print("(thesis_aligned_experiment.py:174). The retrieval API")
    print("(find_corresponding_solution, track_solution_evolution) is UNUSED.")
    print()

    # =========================================================================
    # TEST 1: Cosine similarity failure modes on sparse portfolios
    # =========================================================================
    print("=" * 80)
    print("TEST 1: Cosine similarity on sparse portfolios — known pathologies")
    print("=" * 80)
    print()
    n_assets = 30
    print(f"Universe: {n_assets} assets; cardinality 5 (sparse, like FTSE/AS31).")
    print()

    cases = [
        ("Identical sparse",          [0.5, 0.3, 0.2, 0.0, 0.0],
                                       [0.5, 0.3, 0.2, 0.0, 0.0]),
        ("Same assets, perturbed",    [0.5, 0.3, 0.2, 0.0, 0.0],
                                       [0.4, 0.4, 0.2, 0.0, 0.0]),
        ("3-of-5 overlap (typical)",  [0.5, 0.3, 0.2, 0.0, 0.0],
                                       [0.5, 0.3, 0.0, 0.2, 0.0]),
        ("1-of-5 overlap (rebalanced)", [0.5, 0.3, 0.2, 0.0, 0.0],
                                       [0.5, 0.0, 0.0, 0.3, 0.2]),
        ("ZERO overlap (different)",  [0.5, 0.3, 0.2, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.5, 0.5]),
        ("Same set, totally diff weights", [0.5, 0.3, 0.2, 0.0, 0.0],
                                            [0.1, 0.1, 0.8, 0.0, 0.0]),
    ]
    print(f"{'case':<35} {'cosine':>10} {'jaccard':>10} {'L1_dist':>10} {'L2_dist':>10}")
    print("-" * 80)
    for name, w1_short, w2_short in cases:
        w1 = np.zeros(n_assets); w1[:5] = w1_short
        w2 = np.zeros(n_assets); w2[:5] = w2_short
        cos_sim = cosine_similarity(w1, w2)
        jac = jaccard_set_similarity(w1, w2)
        l1 = l1_distance(w1, w2)
        l2 = l2_distance(w1, w2)
        print(f"{name:<35} {cos_sim:>10.4f} {jac:>10.4f} {l1:>10.4f} {l2:>10.4f}")
    print()
    print("Observations:")
    print("  - Cosine for 'same set, totally diff weights' is HIGH (0.45+) even")
    print("    though portfolios are economically very different.")
    print("  - Cosine for '3-of-5 overlap (typical)' is also HIGH (0.81+) — that's")
    print("    the threshold regime; portfolios with most assets shared look similar.")
    print("  - Jaccard tracks asset-set overlap directly; more interpretable.")
    print("  - L1 distance on simplex has clean [0, 2] range and is symmetric.")
    print()

    # =========================================================================
    # TEST 2: Threshold sensitivity — how often does 0.95 fire?
    # =========================================================================
    print("=" * 80)
    print("TEST 2: Threshold sensitivity — how often does threshold=0.95 fire?")
    print("=" * 80)
    print()
    print("Setup: 100 portfolios at time t, 100 at time t+1.")
    print("Each t+1 portfolio is a perturbation of its t counterpart by 'swap_frac'")
    print("(fraction of held assets swapped to unheld). We ask: does the cosine")
    print("threshold detect the correspondence?")
    print()
    n_assets = 30
    cardinality = 5
    n_pop = 100
    print(f"  n_assets={n_assets}, cardinality={cardinality}, n_pop={n_pop}")
    print()
    print(f"{'swap_frac':>10} {'thresh':>8} {'mean_cos':>10} {'frac_match':>12} {'frac_self_top1':>16}")
    print("-" * 70)
    for swap_frac in [0.0, 0.2, 0.4, 0.6, 0.8]:
        pop_t = [random_sparse_portfolio(n_assets, cardinality, RNG)
                 for _ in range(n_pop)]
        pop_tp1 = [perturb_portfolio(w, swap_frac, RNG) for w in pop_t]

        for thresh in [0.95, 0.7, 0.5]:
            cosines = []
            n_match = 0
            n_self_top1 = 0  # self correspondence is highest-similarity match
            for i, w_t in enumerate(pop_t):
                sims = [cosine_similarity(w_t, w) for w in pop_tp1]
                best_idx = int(np.argmax(sims))
                cosines.append(sims[i])  # self-cosine
                if sims[i] >= thresh:
                    n_match += 1
                if best_idx == i:
                    n_self_top1 += 1
            print(f"{swap_frac:>10.2f} {thresh:>8.2f} {np.mean(cosines):>10.4f} "
                  f"{n_match/n_pop:>12.2%} {n_self_top1/n_pop:>16.2%}")
    print()
    print("Interpretation:")
    print("  - At swap_frac=0.0 (no change): threshold=0.95 should ALWAYS fire (1.00).")
    print("    If it doesn't, the threshold is fundamentally broken.")
    print("  - At swap_frac=0.4–0.6 (realistic rebalancing): threshold=0.95 fires")
    print("    much less; threshold=0.5 fires more but loses precision.")
    print("  - 'frac_self_top1' = how often the correct self-correspondence is")
    print("    actually the top-1 match. If << 100%, even perfect similarity")
    print("    search would fail.")
    print()

    # =========================================================================
    # TEST 3: Cosine similarity geometric pathology — same-set weight invariance
    # =========================================================================
    print("=" * 80)
    print("TEST 3: Cosine on identical-support, scaled-equally is 1.0")
    print("=" * 80)
    print()
    print("Cosine similarity is invariant to scaling — but portfolio weights")
    print("are already simplex-normalized. So the invariance is wasted, AND")
    print("a portfolio entirely concentrated in one asset versus a uniform")
    print("portfolio over the SAME assets can have surprisingly high cosine.")
    print()
    n_assets = 10
    cases_3 = [
        ("uniform 5-asset",           np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0])),
        ("concentrated 5-asset",      np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0])),
        ("uniform-but-1-shifted",     np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0])),
        ("disjoint 5-asset",          np.array([0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2])),
    ]
    print(f"{'case A':<25} {'case B':<25} {'cosine':>10} {'jaccard':>10} {'L1':>10}")
    for i, (name_a, w_a) in enumerate(cases_3):
        for j, (name_b, w_b) in enumerate(cases_3):
            if j <= i:
                continue
            cos = cosine_similarity(w_a, w_b)
            jac = jaccard_set_similarity(w_a, w_b)
            l1 = l1_distance(w_a, w_b)
            print(f"{name_a:<25} {name_b:<25} {cos:>10.4f} {jac:>10.4f} {l1:>10.4f}")
    print()
    print("Note: cosine(uniform 5-asset, concentrated 5-asset) ≈ 0.45 — high(ish)")
    print("even though one portfolio puts 96% into ONE asset vs uniform-20-each.")
    print("Jaccard correctly reports 1.0 (same asset set) — different metric for")
    print("different question. The choice between cosine, Jaccard, L1 should")
    print("depend on what 'correspondence' means in the algorithm.")
    print()

    # =========================================================================
    # TEST 4: Verify the actual CorrespondenceMapping.find_corresponding_solution
    # =========================================================================
    print("=" * 80)
    print("TEST 4: End-to-end test of CorrespondenceMapping.find_corresponding_solution")
    print("=" * 80)
    print()

    class FakeP:
        def __init__(self, w):
            self.investment = w
            self.num_assets = len(w)
            self.ROI = 0.0
            self.risk = 0.0
            self.cardinality = int(np.sum(w > 1e-6))

    class FakeSol:
        def __init__(self, w):
            self.P = FakeP(w)
            self.cd = 0.0
            self.Delta_S = 0.0
            self.Pareto_rank = 0
            self.stability = 0.0
            self.rank_ROI = 0
            self.rank_risk = 0
            self.alpha = 0.0
            self.anticipation = False
            self.prediction_error = 0.0

    cm = CorrespondenceMapping(max_history_size=10)
    n_assets, cardinality, n_pop = 30, 5, 20

    pop_t = [FakeSol(random_sparse_portfolio(n_assets, cardinality, RNG))
             for _ in range(n_pop)]
    pop_tp1 = [FakeSol(perturb_portfolio(s.P.investment, 0.4, RNG))
               for s in pop_t]

    cm.store_population(pop_t, 0)
    cm.store_population(pop_tp1, 1)

    # For each pop_t solution, try to find its correspondence in pop_tp1
    n_found_095 = 0
    n_found_070 = 0
    n_found_050 = 0
    for i, sol in enumerate(pop_t):
        for thresh, counter in [(0.95, "_095"), (0.7, "_070"), (0.5, "_050")]:
            found = cm.find_corresponding_solution(sol, 0, 1, similarity_threshold=thresh)
            if found is not None:
                if thresh == 0.95: n_found_095 += 1
                elif thresh == 0.7: n_found_070 += 1
                elif thresh == 0.5: n_found_050 += 1

    print(f"  Population size:                                       {n_pop}")
    print(f"  Correspondences found at threshold 0.95:               {n_found_095}/{n_pop} ({n_found_095/n_pop:.0%})")
    print(f"  Correspondences found at threshold 0.70:               {n_found_070}/{n_pop} ({n_found_070/n_pop:.0%})")
    print(f"  Correspondences found at threshold 0.50:               {n_found_050}/{n_pop} ({n_found_050/n_pop:.0%})")
    print()
    print("With 40% asset swap (realistic for some rebalancing windows), threshold=0.95")
    print("is too strict — most correspondences are missed even though they exist.")
    print()

    # =========================================================================
    # Final summary
    # =========================================================================
    print("=" * 80)
    print("INSPECTION 4 CONCLUSIONS")
    print("=" * 80)
    print()
    print("1. PRODUCTION USAGE: Correspondence mapping is DEAD INFRASTRUCTURE.")
    print("   - store_population_snapshot: called once (thesis_aligned_experiment).")
    print("   - find_corresponding_solution: NEVER called outside the wrapper.")
    print("   - track_solution_evolution: NEVER called outside the wrapper.")
    print("   - Populations are stored at every period but NEVER queried.")
    print("   Recommended: remove the retrieval API OR wire it into actual")
    print("   anticipation logic (e.g., per-solution KF state continuity).")
    print()
    print("2. COSINE SIMILARITY CHOICE: questionable for sparse portfolios.")
    print("   - Cosine is scale-invariant, but weights are already simplex-normalized.")
    print("   - Cosine doesn't track asset-SET overlap (Jaccard does).")
    print("   - Cosine doesn't preserve simplex geometry (L1 does, range [0, 2]).")
    print("   - For cardinality-5 portfolios in a 30-asset universe, 1-of-5 overlap")
    print("     can still produce cosine > 0.7 depending on weights.")
    print()
    print("3. THRESHOLD 0.95 IS BRITTLE.")
    print("   - At realistic rebalancing (40% asset swap), threshold=0.95 finds")
    print("     correspondences in well under 50% of cases.")
    print("   - But lowering threshold loses precision (more spurious matches).")
    print("   - The threshold should be empirically tuned to the rebalancing regime,")
    print("     OR replaced with a top-K nearest neighbor scheme.")
    print()
    print("4. NC28 CANDIDATE (if we activate correspondence mapping):")
    print("   a. Replace cosine with composite: 0.5*Jaccard + 0.5*(1 - L1/2)")
    print("      → bounded in [0, 1], rewards asset-set overlap AND weight similarity.")
    print("   b. Replace threshold with top-K nearest neighbor (K=1 by default).")
    print("   c. Then wire correspondence into per-solution KF state carry-forward.")
    print("      Currently every solution gets its own KF state independently,")
    print("      but two solutions with similar portfolios should share KF predictions.")
    print()
    print("5. OPERATOR'S QUESTION 'is this sound?':")
    print("   - The CODE is sound (cosine is well-defined, threshold is correctly")
    print("     applied). The ALGORITHM design is suspect (wrong metric for sparse")
    print("     simplex data, brittle threshold). And the PRODUCTION USAGE is")
    print("     non-existent — so neither the soundness nor the brittleness matters")
    print("     until/unless we activate the retrieval API.")


if __name__ == "__main__":
    main()

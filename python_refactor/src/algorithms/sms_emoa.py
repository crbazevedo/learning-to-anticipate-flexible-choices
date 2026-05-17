"""
SMS-EMOA (S-metric Selection Evolutionary Multi-Objective Algorithm)

Revised implementation to include:
- Stochastic Pareto frontiers over time
- Dirichlet MAP filtering for portfolios
- Expected future hypervolume metrics
- Anticipatory learning integration
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .solution import Solution
from .anticipatory_learning import AnticipatoryLearning

class StochasticParams:
    """Stochastic parameters for portfolio state tracking."""
    
    def __init__(self, solution: Solution):
        """
        Initialize stochastic parameters from solution's Kalman state.
        
        Args:
            solution: Solution with Kalman filter state
        """
        # Extract covariance matrix from Kalman state
        kalman_state = solution.P.kalman_state
        P = kalman_state.P
        
        # Extract parameters
        self.cov = P[0, 1]  # Covariance between ROI and risk
        self.var_ROI = P[0, 0]  # Variance of ROI
        self.var_risk = P[1, 1]  # Variance of risk
        
        # Compute correlation
        self.corr = self.cov / (np.sqrt(self.var_ROI) * np.sqrt(self.var_risk)) if self.var_ROI > 0 and self.var_risk > 0 else 0.0
        
        # Variance ratio
        self.var_ratio = np.sqrt(self.var_ROI) / np.sqrt(self.var_risk) if self.var_risk > 0 else 0.0
        
        # Conditional parameters (assuming independence)
        self.conditional_mean_ROI = solution.P.ROI
        self.conditional_var_ROI = (1.0 - self.corr**2) * self.var_ROI
        self.conditional_mean_risk = solution.P.risk
        self.conditional_var_risk = (1.0 - self.corr**2) * self.var_risk

class SMSEMOA:
    """SMS-EMOA with stochastic Pareto frontiers and anticipatory learning."""
    
    def __init__(self, population_size: int = 100, generations: int = 200,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 z_ref: tuple[float, float] = (0.0, 0.2),
                 # Deprecated kwargs kept for callers that still pass them:
                 reference_point_1: float | None = None,
                 reference_point_2: float | None = None):
        """
        Initialize SMS-EMOA algorithm.

        Args:
            population_size: Size of the population
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            tournament_size: Tournament selection size
            z_ref: tuple (return_min, risk_max) — the HV reference point
                in the order (R1=return_min, R2=risk_max) used internally.

                **W15-1 (BACKLOG B1)**: thesis §7.2.3 p.147 specifies
                ``z^ref = (0.2, 0.0)^T`` "in terms of risk and return"
                meaning risk_max=0.2 and return_min=0.0. Mapping to the
                INTERNAL (R1, R2) ordering used by
                _compute_hypervolume_contributions_class
                (``(ROI - R1) * (R2 - risk)``): R1=return_min=0.0,
                R2=risk_max=0.2. Pre-W15-1 default was R2=1.0
                — far beyond the thesis feasibility boundary
                of 20% risk — which allowed high-risk solutions to
                accumulate positive HV contribution + skewed selection
                pressure outside the feasible region.

                Verbatim thesis quote (§7.2.3 p.147):
                "Finally, the reference point for computing Hypv was
                set to z^ref = (0.2, 0.0)^T in terms of risk and
                return, coinciding with the objective space feasibility
                boundaries (maximum risk of 20% and minimum mean
                return of 0%)."
            reference_point_1: DEPRECATED — use z_ref. Kept for
                backward compatibility; if supplied overrides z_ref[0].
            reference_point_2: DEPRECATED — use z_ref. Kept for
                backward compatibility; if supplied overrides z_ref[1].
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        # Resolve z_ref with deprecated-kwarg overrides.
        r1 = reference_point_1 if reference_point_1 is not None else z_ref[0]
        r2 = reference_point_2 if reference_point_2 is not None else z_ref[1]
        self.R1 = r1  # ROI reference point (return_min in thesis terms)
        self.R2 = r2  # Risk reference point (risk_max in thesis terms)
        self.z_ref = (r1, r2)  # canonical exposure for downstream consumers
        
        # Anticipatory learning components
        self.anticipatory_learning = None
        
        # Population and Pareto front tracking
        self.population = []
        self.pareto_front = []
        self.hypervolume_history = []
        self.stochastic_hypervolume_history = []

        # W16-3: per-generation [ROI_max, ROI_min, risk_max, risk_min]
        # trace, computed from rank-1 (Pareto front) only. Used to
        # diagnose whether the front extent shrinks over generations
        # (which it should NOT — extrema preservation is the W16-3
        # fix). Each entry is a dict {gen, roi_max, roi_min,
        # risk_max, risk_min}.
        self.extrema_trace: List[Dict[str, float]] = []

        # W16-2 (BACKLOG H1): previous-period implemented portfolio
        # u*_{t-1}. When set, _evaluate_solution subtracts the thesis
        # Table 7.1 transaction cost h(u_t, u*_{t-1}) from the ROI
        # objective so the optimizer sees and avoids high-churn
        # rebalancings (per thesis §7.2 Eqs (7.4)-(7.5)).
        #
        # None on the first period (no prior portfolio yet → no
        # rebalancing → no cost). Caller (walk_forward driver) calls
        # set_previous_weights(u_prev) before each period's optimization.
        self.previous_weights: Optional[np.ndarray] = None
        self.portfolio_value: float = 1.0  # default wealth scale

        # Performance tracking
        self.function_evaluations = 0
        self.current_generation = 0

    def set_previous_weights(self, weights: Optional[np.ndarray],
                              portfolio_value: float = 1.0) -> None:
        """
        Set the previous-period implemented portfolio u*_{t-1} (W16-2).

        The walk-forward driver calls this before each period's
        optimization, passing the prior period's realized maximal
        flexible choice. None on period 1 (no prior period).

        Args:
            weights: u*_{t-1} length-d array (or None on period 1)
            portfolio_value: total wealth (for sizing the bracket lookup)
        """
        if weights is None:
            self.previous_weights = None
        else:
            self.previous_weights = np.asarray(weights, dtype=float).copy()
        self.portfolio_value = float(portfolio_value)
        
    def set_learning(self, learning: AnticipatoryLearning):
        """Set anticipatory learning component."""
        self.anticipatory_learning = learning
    
    def run(self, data: Dict[str, Any]) -> List[Solution]:
        """
        Run SMS-EMOA algorithm.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Final population
        """
        # Initialize population
        self._initialize_population(data)
        
        # Main evolution loop
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Apply anticipatory learning if enabled
            if self.anticipatory_learning is not None:
                self._apply_anticipatory_learning(generation)
            
            # Run one generation
            self._run_generation()
            
            # Track hypervolume
            self._track_hypervolume()
            
            # Log progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Population size = {len(self.population)}, "
                      f"Pareto front size = {len(self.pareto_front)}")
        
        return self.population
    
    def _initialize_population(self, data: Dict[str, Any]):
        """Initialize population with random solutions.

        W13-1: set Portfolio class-level statistics (mean_ROI,
        median_ROI, covariance, robust_covariance) from the asset
        returns BEFORE constructing Solution objects. Pre-W13-1
        these were left as None, so Solution.__init__'s guarded
        `Portfolio.compute_efficiency(self.P)` call silently
        skipped, leaving every solution with ROI=0 and risk=0.
        Downstream `_compute_hypervolume` then iterated a Pareto
        front of zeros and produced hypervolume=0 by construction
        (the W12-CARRY-1 family — degenerate objectives, not
        decoupled evaluator).
        """
        from ..portfolio.portfolio import Portfolio

        # Resolve asset returns from data['assets'] (DataFrame from
        # the W9-2 data_loader pivot, already returns-shaped after
        # W11-2 sanitation).
        assets_df = data.get('assets')
        if assets_df is not None and hasattr(assets_df, 'values') and not assets_df.empty:
            returns = assets_df.values
            num_assets = returns.shape[1]
            Portfolio.mean_ROI = Portfolio.estimate_assets_mean_ROI(returns)
            Portfolio.median_ROI = Portfolio.estimate_assets_median_ROI(returns)
            Portfolio.covariance = Portfolio.estimate_covariance(
                Portfolio.mean_ROI, returns)
            # Robust covariance fallback: use sample cov (a proper
            # MCD/MVE estimator can land in a later wave).
            Portfolio.robust_covariance = Portfolio.covariance
        else:
            num_assets = data.get('num_assets', 3)

        self.population = []
        for _ in range(self.population_size):
            solution = Solution(num_assets=num_assets)

            # W15-2: project initial weights to thesis cardinality
            # [c_l=5, c_u=15] + non-negative simplex BEFORE evaluating.
            # Solution.__init__ produces random fully-dense weights;
            # without projection, initial population starts with
            # cardinality ≈ n_assets (98 for paper window), violating
            # thesis §7.2 Eq (7.3) cardinality constraint.
            from .operators import project_to_simplex
            solution.P.investment = project_to_simplex(
                solution.P.investment, rng=np.random.default_rng(self.function_evaluations))
            # Re-compute portfolio efficiency on the projected weights.
            from ..portfolio.portfolio import Portfolio
            if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
                Portfolio.compute_efficiency(solution.P)

            # Initialize Kalman filter state
            self._initialize_kalman_state(solution, data)

            # Evaluate solution
            self._evaluate_solution(solution, data)

            self.population.append(solution)
            self.function_evaluations += 1
    
    def _initialize_kalman_state(self, solution: Solution, data: Dict[str, Any]):
        """Initialize Kalman filter state for solution."""
        # Initialize with current portfolio state
        kalman_state = solution.P.kalman_state
        
        # State vector: [ROI, risk, ROI_velocity, risk_velocity]
        kalman_state.x = np.array([solution.P.ROI, solution.P.risk, 0.0, 0.0])
        
        # Initial covariance matrix
        kalman_state.P = np.array([
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0]
        ])
        
        # State transition matrix (constant velocity model)
        kalman_state.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement matrix (observe ROI and risk)
        kalman_state.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Measurement noise covariance
        kalman_state.R = np.array([
            [0.01, 0.0],
            [0.0, 0.01]
        ])
    
    def _evaluate_solution(self, solution: Solution, data: Dict[str, Any]):
        """Evaluate solution and update objectives.

        W16-2 (BACKLOG H1): if self.previous_weights is set, subtract
        the thesis Table 7.1 transaction cost from the OBJECTIVE ROI
        so the optimizer sees the rebalancing cost. The PORTFOLIO
        ROI (portfolio.ROI) is preserved as gross-of-cost for
        downstream reporting (so wealth-evaluation code still has
        the un-netted ROI).

        Per thesis §7.2 Eq (7.4)-(7.5), p. 142:
            z_t|u*_{t-1} = g(u_t, χ_t) + h(u_t, u*_{t-1})
        i.e. the optimizer's z-vector ROI component is
        net-of-transaction-cost.
        """
        portfolio = solution.P

        # Update Kalman filter with current observation if available
        if hasattr(solution.P, 'kalman_state') and solution.P.kalman_state is not None:
            from .kalman_filter import kalman_update
            measurement = np.array([portfolio.ROI, portfolio.risk])
            kalman_update(solution.P.kalman_state, measurement)

        # W16-2: compute net-of-cost ROI for the NDS/HV objective.
        # Gross ROI stays on portfolio.ROI for reporting; only the
        # objective vector (used by dominance + HV contribution)
        # uses ROI_net.
        roi_objective = portfolio.ROI
        if self.previous_weights is not None:
            from ..portfolio.portfolio import Portfolio
            txn_cost = Portfolio.compute_thesis_transaction_cost(
                weights_new=np.asarray(portfolio.investment, dtype=float),
                weights_prev=self.previous_weights,
                portfolio_value=self.portfolio_value,
            )
            roi_objective = portfolio.ROI - txn_cost
            # Stash on solution for diagnostics (W16-5 trace + W18 viz)
            solution.transaction_cost = txn_cost
            solution.roi_gross = portfolio.ROI
            solution.roi_net = roi_objective
        else:
            solution.transaction_cost = 0.0
            solution.roi_gross = portfolio.ROI
            solution.roi_net = portfolio.ROI

        # Store objectives (NET-of-cost ROI per Eq 7.4-7.5; risk unchanged)
        solution.objectives = [roi_objective, portfolio.risk]

        # Compute stability (simplified)
        portfolio.stability = 1.0 / (1.0 + np.std(solution.P.investment))
    
    def _apply_anticipatory_learning(self, generation: int):
        """Apply anticipatory learning to population.

        W11-1 (closes W10-CARRY-2): the pre-W11-1 loop always called
        learn_single_solution, which is defined only on the base
        AnticipatoryLearning class. TIPIntegratedAnticipatoryLearning
        and MultiHorizonAnticipatoryLearning override learn_population
        (paper Eq 13 / Eq 14 / Eq 15) but NOT learn_single_solution,
        so all "learning-enabled" scenarios collapsed to base behavior
        (S1=S2=S3=S4 produced identical metrics).

        Fix: detect whether the learner overrides learn_population
        (not inherited from base) and call it once for the unlearned
        subset; fall back to the per-solution loop only when no
        override exists.
        """
        if self.anticipatory_learning is None:
            return

        unlearned = [s for s in self.population
                     if not getattr(s, 'anticipation', False)]
        if not unlearned:
            return

        # The base AnticipatoryLearning class does NOT define
        # learn_population — only subclasses (TIPIntegrated, MultiHorizon)
        # do. So `hasattr(learner, 'learn_population')` is a sufficient
        # detector: True → subclass-defined → route through population
        # entry point (paper Eq 13 TIP arm + Eq 14/15 multi-horizon
        # convex combination). False → base → per-solution fallback.
        #
        # Note: we re-check whether learn_population was defined on a
        # non-base class via __mro__ walk, BUT skipping that check
        # when the base lacks it entirely (the actual project shape)
        # is equivalent: any class with learn_population is a subclass
        # override of nothing.
        learner = self.anticipatory_learning
        if hasattr(learner, 'learn_population'):
            # Subclass-defined entry point (paper Eq 13/14/15).
            learner.learn_population(unlearned, generation)
            return

        # Fallback: per-solution loop (the base / legacy path).
        for solution in unlearned:
            learner.learn_single_solution(solution, generation)
    
    def _run_generation(self):
        """Run one generation of SMS-EMOA."""
        # Fast non-dominated sorting
        pareto_ranks = self._fast_non_dominated_sort()
        
        # Compute hypervolume contributions
        if self.anticipatory_learning is not None:
            self._compute_stochastic_hypervolume_contributions()
        else:
            self._compute_hypervolume_contributions()
        
        # Tournament selection
        parent1_idx = self._tournament_selection()
        parent2_idx = self._tournament_selection()
        
        # W15-2: switch to thesis-faithful operators per §7.2.3 p. 147
        # ("We utilized uniform crossover over the mean DD vectors. For
        # mutation, we randomly choose between (1) modifying the
        # non-zero weights; or (2) adding/removing assets..."). Legacy
        # SBX `crossover` and `mutation` retained in operators.py for
        # backward compat but no longer used here.
        from .operators import (
            thesis_dual_mode_mutation, thesis_uniform_crossover,
        )
        offspring1, offspring2 = thesis_uniform_crossover(
            self.population[parent1_idx],
            self.population[parent2_idx],
            p=self.crossover_rate,
        )
        offspring1 = thesis_dual_mode_mutation(offspring1)
        offspring2 = thesis_dual_mode_mutation(offspring2)
        
        # Add offspring to population
        self.population.append(offspring1)
        self.population.append(offspring2)
        
        # Remove worst solution based on hypervolume contribution
        self._remove_worst_solution()
        
        # Update Pareto front
        self._update_pareto_front()
    
    def _fast_non_dominated_sort(self) -> int:
        """Perform fast non-dominated sorting."""
        pareto_ranks = [0] * len(self.population)
        
        for i, solution1 in enumerate(self.population):
            for j, solution2 in enumerate(self.population):
                if i != j:
                    if self._dominates(solution1, solution2):
                        pareto_ranks[j] += 1
        
        # Assign ranks
        for i, solution in enumerate(self.population):
            solution.Pareto_rank = pareto_ranks[i]
        
        # Return number of fronts (unique ranks)
        return len(set(pareto_ranks))
    
    def _dominates(self, solution1: Solution, solution2: Solution) -> bool:
        """Check if solution1 dominates solution2."""
        # Get objectives from portfolio
        roi1, risk1 = solution1.P.ROI, solution1.P.risk
        roi2, risk2 = solution2.P.ROI, solution2.P.risk
        
        # Check if solution1 is better in at least one objective
        # and not worse in any objective
        better_in_one = False
        
        # Higher ROI is better
        if roi1 > roi2:
            better_in_one = True
        elif roi1 < roi2:
            return False
        
        # Lower risk is better
        if risk1 < risk2:
            better_in_one = True
        elif risk1 > risk2:
            return False
        
        return better_in_one
    
    def _compute_hypervolume_contributions(self):
        """Compute hypervolume contributions for each solution."""
        # Group solutions by Pareto rank
        pareto_classes = {}
        for solution in self.population:
            rank = solution.Pareto_rank
            if rank not in pareto_classes:
                pareto_classes[rank] = []
            pareto_classes[rank].append(solution)
        
        # Compute contributions for each class
        for rank, solutions in pareto_classes.items():
            self._compute_hypervolume_contributions_class(solutions)
    
    def _compute_stochastic_hypervolume_contributions(self):
        """Compute stochastic hypervolume contributions considering future uncertainty."""
        # Group solutions by Pareto rank
        pareto_classes = {}
        for solution in self.population:
            rank = solution.Pareto_rank
            if rank not in pareto_classes:
                pareto_classes[rank] = []
            pareto_classes[rank].append(solution)
        
        # Compute stochastic contributions for each class
        for rank, solutions in pareto_classes.items():
            self._compute_stochastic_hypervolume_contributions_class(solutions)
    
    def _compute_hypervolume_contributions_class(self, solutions: List[Solution]):
        """Compute hypervolume contributions for a Pareto class."""
        if len(solutions) == 1:
            # Single solution in class
            solution = solutions[0]
            solution.hypervolume_contribution = (solution.P.ROI - self.R1) * (self.R2 - solution.P.risk)
            return
        
        # Sort by ROI (ascending)
        solutions.sort(key=lambda s: s.P.ROI)
        
        # Compute contributions
        for i, solution in enumerate(solutions):
            if i == 0:
                # First solution
                next_solution = solutions[i + 1]
                solution.hypervolume_contribution = (solution.P.ROI - self.R1) * (self.R2 - solution.P.risk)
            elif i == len(solutions) - 1:
                # Last solution
                prev_solution = solutions[i - 1]
                solution.hypervolume_contribution = (solution.P.ROI - prev_solution.P.ROI) * (self.R2 - solution.P.risk)
            else:
                # Middle solution
                prev_solution = solutions[i - 1]
                next_solution = solutions[i + 1]
                solution.hypervolume_contribution = (solution.P.ROI - next_solution.P.ROI) * (prev_solution.P.risk - solution.P.risk)
            
            # Apply stability factor
            solution.hypervolume_contribution *= solution.stability
    
    def _compute_stochastic_hypervolume_contributions_class(self, solutions: List[Solution]):
        """Compute stochastic hypervolume contributions considering uncertainty."""
        if len(solutions) == 1:
            # Single solution with uncertainty
            solution = solutions[0]
            stoch_params = StochasticParams(solution)
            
            mean_delta_ROI = stoch_params.conditional_mean_ROI - self.R1
            mean_delta_risk = self.R2 - stoch_params.conditional_mean_risk
            var_delta_ROI = stoch_params.conditional_var_ROI
            var_delta_risk = stoch_params.conditional_var_risk
            
            # Expected hypervolume contribution
            solution.hypervolume_contribution = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI) / (var_delta_ROI + var_delta_risk)
            solution.hypervolume_contribution *= solution.stability
            return
        
        # Sort by ROI
        solutions.sort(key=lambda s: s.P.ROI)
        
        # Compute stochastic contributions
        for i, solution in enumerate(solutions):
            stoch_params = StochasticParams(solution)
            
            if i == 0:
                # First solution
                next_solution = solutions[i + 1]
                next_stoch_params = StochasticParams(next_solution)
                
                mean_delta_ROI = stoch_params.conditional_mean_ROI - next_stoch_params.conditional_mean_ROI
                mean_delta_risk = self.R2 - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI + next_stoch_params.conditional_var_ROI
                var_delta_risk = stoch_params.conditional_var_risk
                
            elif i == len(solutions) - 1:
                # Last solution
                prev_solution = solutions[i - 1]
                prev_stoch_params = StochasticParams(prev_solution)
                
                mean_delta_ROI = stoch_params.conditional_mean_ROI - self.R1
                mean_delta_risk = prev_stoch_params.conditional_mean_risk - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI
                var_delta_risk = prev_stoch_params.conditional_var_risk + stoch_params.conditional_var_risk
                
            else:
                # Middle solution
                prev_solution = solutions[i - 1]
                next_solution = solutions[i + 1]
                prev_stoch_params = StochasticParams(prev_solution)
                next_stoch_params = StochasticParams(next_solution)
                
                # Compute mean delta product
                mean_delta_ROI = stoch_params.conditional_mean_ROI - next_stoch_params.conditional_mean_ROI
                mean_delta_risk = prev_stoch_params.conditional_mean_risk - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI + next_stoch_params.conditional_var_ROI
                var_delta_risk = prev_stoch_params.conditional_var_risk + stoch_params.conditional_var_risk
            
            # Expected hypervolume contribution
            solution.hypervolume_contribution = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI) / (var_delta_ROI + var_delta_risk)
            solution.hypervolume_contribution *= solution.stability
    
    def _tournament_selection(self) -> int:
        """Perform tournament selection based on hypervolume contribution."""
        # Select random individuals (handle case where tournament_size > population_size)
        tournament_size = min(self.tournament_size, len(self.population))
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        
        # Find the best based on hypervolume contribution
        best_idx = indices[0]
        best_contribution = self.population[best_idx].hypervolume_contribution
        
        for idx in indices[1:]:
            contribution = self.population[idx].hypervolume_contribution
            if contribution > best_contribution:
                best_contribution = contribution
                best_idx = idx
        
        return best_idx
    
    def _remove_worst_solution(self):
        """
        Remove the solution with the lowest hypervolume contribution.

        W16-3: rank-1 extrema (argmax(ROI), argmin(risk)) are protected
        from removal so the HV bounding box doesn't shrink over
        generations. Per standard SMS-EMOA practice (Beume et al. 2007
        EJOR 181(3):1653-1669) the HV indicator is most sensitive to
        anchor solutions which define the bounding box; pruning them
        introduces sawtooth instability in the per-gen HV trajectory.

        Edge cases:
          - Rank-1 with < 3 solutions: fall back to no protection
            (otherwise reduce can't proceed at all).
          - Rank-1 anchors that happen to coincide (same solution is
            both argmax(ROI) AND argmin(risk)): protect that single
            solution.
        """
        while len(self.population) > self.population_size:
            # ── W16-3: identify protected anchor indices ──────────
            protected = self._identify_protected_anchors()

            # Find worst solution among non-protected
            worst_idx = -1
            worst_contribution = float('inf')

            for i, solution in enumerate(self.population):
                if i in protected:
                    continue
                contribution = getattr(solution, 'hypervolume_contribution', float('inf'))
                if contribution < worst_contribution:
                    worst_contribution = contribution
                    worst_idx = i

            # Edge case: every solution is protected (rank-1 too small).
            # Fall back to the original behavior (pick globally worst,
            # may evict an anchor — better than infinite loop).
            if worst_idx == -1:
                worst_idx = 0
                worst_contribution = getattr(self.population[0], 'hypervolume_contribution', float('inf'))
                for i, solution in enumerate(self.population):
                    contribution = getattr(solution, 'hypervolume_contribution', float('inf'))
                    if contribution < worst_contribution:
                        worst_contribution = contribution
                        worst_idx = i

            # Remove worst solution
            self.population.pop(worst_idx)

    def _identify_protected_anchors(self) -> set:
        """
        Identify rank-1 extrema (argmax(ROI), argmin(risk)) — W16-3.

        Returns:
            set of population indices to protect from reduce-step eviction.
            Empty set if rank-1 has < 3 solutions (then protection is
            counter-productive — would block reduce entirely).
        """
        # Pareto front (rank 0 = first non-dominated front; some code
        # paths use rank 0, others rank 1 — be permissive).
        rank1 = [
            (i, s) for i, s in enumerate(self.population)
            if getattr(s, 'Pareto_rank', None) in (0, 1)
        ]
        if len(rank1) < 3:
            # Edge case: rank-1 too small to protect anchors safely.
            return set()

        # argmax(ROI) and argmin(risk) within rank-1.
        argmax_roi_idx = max(rank1, key=lambda pair: pair[1].P.ROI)[0]
        argmin_risk_idx = min(rank1, key=lambda pair: pair[1].P.risk)[0]
        return {argmax_roi_idx, argmin_risk_idx}
    
    def _update_pareto_front(self):
        """Update the Pareto front."""
        self.pareto_front = [s for s in self.population if s.Pareto_rank == 0]
    
    def _track_hypervolume(self):
        """Track hypervolume over generations."""
        if self.pareto_front:
            # Compute current hypervolume
            hypervolume = self._compute_hypervolume()
            self.hypervolume_history.append(hypervolume)

            # Compute expected future hypervolume if using anticipatory learning
            if self.anticipatory_learning is not None:
                future_hypervolume = self._compute_expected_future_hypervolume()
                self.stochastic_hypervolume_history.append(future_hypervolume)

            # W16-3: per-generation Pareto front extrema trace
            rois = [s.P.ROI for s in self.pareto_front]
            risks = [s.P.risk for s in self.pareto_front]
            self.extrema_trace.append({
                "gen": self.current_generation,
                "roi_max": float(max(rois)),
                "roi_min": float(min(rois)),
                "risk_max": float(max(risks)),
                "risk_min": float(min(risks)),
                "front_size": len(self.pareto_front),
            })
    
    def _compute_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front."""
        if not self.pareto_front:
            return 0.0
        
        # Sort by ROI
        sorted_front = sorted(self.pareto_front, key=lambda s: s.P.ROI)
        
        hypervolume = 0.0
        prev_roi = self.R1
        
        for solution in sorted_front:
            roi = solution.P.ROI
            risk = solution.P.risk
            hypervolume += (roi - prev_roi) * (self.R2 - risk)
            prev_roi = roi
        
        return hypervolume
    
    def _compute_expected_future_hypervolume(self) -> float:
        """Compute expected future hypervolume considering uncertainty."""
        if not self.pareto_front:
            return 0.0
        
        # Compute expected hypervolume using Monte Carlo sampling
        num_samples = 1000
        total_hypervolume = 0.0
        
        for _ in range(num_samples):
            # Sample future states for each solution
            future_front = []
            for solution in self.pareto_front:
                # Sample from Kalman filter prediction
                from .kalman_filter import kalman_prediction
                kalman_prediction(solution.P.kalman_state)
                # Paper Eq (11) ordering: x_next = [ROI, risk, ROI_vel, risk_vel]
                # Prior to W1-1 this read x_next[2] (= ROI_velocity, NOT risk)
                # under sms_emoa.py's own paper-canonical F matrix.
                future_roi = solution.P.kalman_state.x_next[0]
                future_risk = solution.P.kalman_state.x_next[1]
                
                # Create temporary solution for hypervolume computation
                temp_solution = Solution(num_assets=len(solution.P.investment))
                temp_solution.P.ROI = future_roi
                temp_solution.P.risk = future_risk
                future_front.append(temp_solution)
            
            # Compute hypervolume for this sample
            if future_front:
                sorted_front = sorted(future_front, key=lambda s: s.P.ROI)
                sample_hypervolume = 0.0
                prev_roi = self.R1
                
                for solution in sorted_front:
                    roi = solution.P.ROI
                    risk = solution.P.risk
                    sample_hypervolume += (roi - prev_roi) * (self.R2 - risk)
                    prev_roi = roi
                
                total_hypervolume += sample_hypervolume
        
        return total_hypervolume / num_samples
    
    def get_pareto_front(self) -> List[Solution]:
        """Get current Pareto front."""
        return self.pareto_front
    
    def get_hypervolume(self) -> float:
        """Get current hypervolume."""
        return self._compute_hypervolume()
    
    def get_expected_future_hypervolume(self) -> float:
        """Get expected future hypervolume."""
        return self._compute_expected_future_hypervolume()
    
    def get_function_evaluations(self) -> int:
        """Get number of function evaluations."""
        return self.function_evaluations 
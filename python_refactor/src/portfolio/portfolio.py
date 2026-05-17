"""
Portfolio class and calculations.

This module replicates the C++ portfolio struct and related calculations
including ROI, risk, covariance estimation, and efficiency computations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass
from .asset import Asset


@dataclass
class KalmanParams:
    """Kalman filter parameters equivalent to C++ Kalman_params."""
    x: np.ndarray  # State vector
    P: np.ndarray  # Error covariance matrix
    F: np.ndarray  # State transition matrix
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    H: np.ndarray  # Observation matrix


class Portfolio:
    """
    Python equivalent of the C++ portfolio struct.
    
    This class manages portfolio weights, calculates efficiency metrics,
    and maintains state for optimization algorithms.
    """
    
    # Static variables (equivalent to C++ static members)
    training_start_date: Optional[pd.Timestamp] = None
    training_end_date: Optional[pd.Timestamp] = None
    validation_start_date: Optional[pd.Timestamp] = None
    validation_end_date: Optional[pd.Timestamp] = None
    window_size: int = 20
    tr_period: int = 0
    vl_period: int = 0
    available_assets: List[Asset] = []
    available_assets_size: int = 0
    
    # Data matrices
    covariance: Optional[np.ndarray] = None
    robust_covariance: Optional[np.ndarray] = None
    mean_ROI: Optional[np.ndarray] = None
    median_ROI: Optional[np.ndarray] = None
    tr_returns_data: Optional[np.ndarray] = None
    vl_returns_data: Optional[np.ndarray] = None
    current_returns_data: Optional[np.ndarray] = None
    complete_returns_data: Optional[np.ndarray] = None
    autocorrelation: Optional[np.ndarray] = None
    
    # Configuration
    max_cardinality: int = 10
    robustness: bool = False
    
    def __init__(self, num_assets: int):
        """
        Initialize portfolio with given number of assets.
        
        Args:
            num_assets: Number of assets in the portfolio
        """
        self.num_assets = num_assets
        
        # Portfolio metrics
        self.ROI: float = 0.0
        self.ROI_prediction: float = 0.0
        self.ROI_observed: float = 0.0
        self.risk: float = 0.0
        self.risk_prediction: float = 0.0
        self.risk_observed: float = 0.0
        self.robust_ROI: float = 0.0
        self.robust_risk: float = 0.0
        self.non_robust_ROI: float = 0.0
        self.non_robust_risk: float = 0.0
        self.cardinality: float = 0.0
        
        # Kalman filter state
        self.kalman_state: Optional[KalmanParams] = None
        self.error_covar: Optional[np.ndarray] = None
        self.error_covar_prediction: Optional[np.ndarray] = None
        
        # Investment weights (equivalent to C++ investment vector)
        self.investment: np.ndarray = np.zeros(num_assets)
    
    def init(self):
        """Initialize portfolio weights randomly."""
        # Generate random weights that sum to 1
        weights = np.random.random(self.num_assets)
        self.investment = weights / np.sum(weights)
    
    @classmethod
    def estimate_assets_mean_ROI(cls, returns_data: np.ndarray) -> np.ndarray:
        """
        Estimate mean ROI for all assets.
        
        Args:
            returns_data: Matrix of returns data (time x assets)
        
        Returns:
            Vector of mean ROIs for each asset
        """
        return np.mean(returns_data, axis=0)
    
    @classmethod
    def estimate_assets_median_ROI(cls, returns_data: np.ndarray) -> np.ndarray:
        """
        Estimate median ROI for all assets.
        
        Args:
            returns_data: Matrix of returns data (time x assets)
        
        Returns:
            Vector of median ROIs for each asset
        """
        return np.median(returns_data, axis=0)
    
    @classmethod
    def estimate_mean_ROI(cls, asset_idx: int, returns_data: np.ndarray) -> float:
        """
        Estimate mean ROI for a specific asset.
        
        Args:
            asset_idx: Index of the asset
            returns_data: Matrix of returns data
        
        Returns:
            Mean ROI for the specified asset
        """
        return np.mean(returns_data[:, asset_idx])
    
    @classmethod
    def estimate_median_ROI(cls, asset_idx: int, returns_data: np.ndarray) -> float:
        """
        Estimate median ROI for a specific asset.
        
        Args:
            asset_idx: Index of the asset
            returns_data: Matrix of returns data
        
        Returns:
            Median ROI for the specified asset
        """
        return np.median(returns_data[:, asset_idx])
    
    @classmethod
    def estimate_covariance(cls, mean_ROI: np.ndarray, returns_data: np.ndarray) -> np.ndarray:
        """
        Estimate covariance matrix using sample covariance.
        
        Args:
            mean_ROI: Vector of mean ROIs
            returns_data: Matrix of returns data
        
        Returns:
            Covariance matrix
        """
        # Center the data
        centered_data = returns_data - mean_ROI
        # Compute sample covariance
        n = returns_data.shape[0]
        covariance = (centered_data.T @ centered_data) / (n - 1)
        return covariance
    
    @classmethod
    def estimate_robust_covariance(cls, mean_ROI: np.ndarray, returns_data: np.ndarray) -> np.ndarray:
        """
        Estimate robust covariance matrix using median-based approach.
        
        Args:
            mean_ROI: Vector of mean ROIs
            returns_data: Matrix of returns data
        
        Returns:
            Robust covariance matrix
        """
        n_assets = returns_data.shape[1]
        n_obs = returns_data.shape[0]
        
        # Compute robust statistics for each asset
        medians = np.median(returns_data, axis=0)
        
        # Compute IQD (Interquartile Distance) for each asset
        q75 = np.percentile(returns_data, 75, axis=0)
        q25 = np.percentile(returns_data, 25, axis=0)
        IQDs = (q75 - q25) / 1.35  # Normalize to match standard deviation
        
        # Center data using medians
        transformed_data = returns_data - medians
        
        # Compute correlation matrix using robust approach
        correlation_matrix = np.ones((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    # Compute correlation using transformed data
                    numerator = np.sum(transformed_data[:, i] * transformed_data[:, j])
                    n_nonzero = np.sum(transformed_data[:, i] * transformed_data[:, j] != 0)
                    
                    if n_nonzero > 0:
                        # Use sine transformation for robust correlation
                        correlation_matrix[i, j] = np.sin(np.pi/2 * numerator / n_nonzero)
                    else:
                        correlation_matrix[i, j] = 0.0
        
        # Compute robust covariance
        covariance = np.outer(IQDs, IQDs) * correlation_matrix
        
        # Ensure positive definiteness using eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        
        # Project data onto eigenvectors
        projected_data = returns_data @ eigenvecs
        
        # Compute robust statistics for projected data
        projected_IQDs = np.percentile(projected_data, 75, axis=0) - np.percentile(projected_data, 25, axis=0)
        projected_IQDs /= 1.35
        
        # Sort and square the IQDs
        sorted_IQDs = np.sort(projected_IQDs)[::-1]
        sorted_IQDs = sorted_IQDs ** 2
        
        # Reconstruct covariance matrix
        D = np.diag(sorted_IQDs)
        covariance = eigenvecs @ D @ eigenvecs.T
        
        return covariance
    
    @classmethod
    def compute_ROI(cls, portfolio: 'Portfolio', mean_ROI: np.ndarray) -> float:
        """
        Compute portfolio ROI.
        
        Args:
            portfolio: Portfolio object
            mean_ROI: Vector of mean ROIs
        
        Returns:
            Portfolio ROI
        """
        return portfolio.investment @ mean_ROI
    
    @classmethod
    def compute_risk(cls, portfolio: 'Portfolio', covariance: np.ndarray) -> float:
        """
        Compute portfolio risk (standard deviation).
        
        Args:
            portfolio: Portfolio object
            covariance: Covariance matrix
        
        Returns:
            Portfolio risk (standard deviation)
        """
        variance = portfolio.investment @ covariance @ portfolio.investment
        return np.sqrt(max(variance, 0.0))  # Ensure non-negative
    
    @classmethod
    def compute_efficiency(cls, portfolio: 'Portfolio'):
        """
        Compute portfolio efficiency metrics (non-robust).
        """
        portfolio.ROI = portfolio.non_robust_ROI = cls.compute_ROI(portfolio, cls.mean_ROI)
        portfolio.risk = portfolio.non_robust_risk = cls.compute_risk(portfolio, cls.covariance)
        
        portfolio.robust_ROI = cls.compute_ROI(portfolio, cls.median_ROI)
        portfolio.robust_risk = cls.compute_risk(portfolio, cls.robust_covariance)
        portfolio.cardinality = cls.card(portfolio)
    
    @classmethod
    def compute_robust_efficiency(cls, portfolio: 'Portfolio'):
        """
        Compute portfolio efficiency metrics (robust).
        """
        portfolio.ROI = portfolio.robust_ROI = cls.compute_ROI(portfolio, cls.median_ROI)
        portfolio.risk = portfolio.robust_risk = cls.compute_risk(portfolio, cls.robust_covariance)
        
        portfolio.non_robust_ROI = cls.compute_ROI(portfolio, cls.mean_ROI)
        portfolio.non_robust_risk = cls.compute_risk(portfolio, cls.covariance)
        portfolio.cardinality = cls.card(portfolio)
    
    @classmethod
    def card(cls, portfolio: 'Portfolio') -> float:
        """
        Compute portfolio cardinality (number of non-zero weights).

        Args:
            portfolio: Portfolio object

        Returns:
            Portfolio cardinality
        """
        return np.sum(portfolio.investment > 1e-6)

    # ─── W16-2: thesis Table 7.1 transaction-cost computation ─────
    # Closes BACKLOG H1. The cost is computed per asset based on the
    # absolute traded value (|Δweight| × portfolio_value); each asset's
    # cost picks the bracket from Table 7.1 (p. 144) whose lower_bound
    # is met. Total cost is the sum across assets, returned as a
    # fraction of portfolio_value so it can be subtracted from the
    # ROI objective directly.
    @staticmethod
    def compute_thesis_transaction_cost(weights_new: np.ndarray,
                                        weights_prev: np.ndarray,
                                        portfolio_value: float = 1.0,
                                        brackets: list | None = None) -> float:
        """
        Compute total transaction cost per thesis Table 7.1 (W16-2).

        Per thesis §7.2 Eqs (7.4)-(7.5): the optimizer's ROI objective
        must net out h(u_t, u*_{t-1}), the brokerage fee on the trades
        implied by rebalancing from u*_{t-1} → u_t.

        Args:
            weights_new: target weights u_t (length d, sums to 1)
            weights_prev: previous-period implemented weights u*_{t-1}
                (length d, sums to 1)
            portfolio_value: total wealth (default 1.0; cost returned
                in same units)
            brackets: optional override of Table 7.1 brackets; default
                is loaded from ThesisParameters().THESIS_TABLE_71_BRACKETS

        Returns:
            Total transaction cost (fraction of portfolio_value),
            i.e. (sum of per-asset fees) / portfolio_value.

        Thesis grounding:
            §7.2 Eqs (7.4)-(7.5), p. 142:
                z_t|u*_{t-1} = g(u_t, χ_t) + h(u_t, u*_{t-1})
                h(u_t, u*_{t-1}) = (0, h)^T  where h is the trade cost

            §7.2.3 Table 7.1, p. 144 — Brazilian SEC fee schedule
            (Brackets in thesis_parameters.THESIS_TABLE_71_BRACKETS).
        """
        if weights_prev is None:
            return 0.0  # no previous → first period → no rebalancing
        weights_new = np.asarray(weights_new, dtype=float)
        weights_prev = np.asarray(weights_prev, dtype=float)
        if weights_new.shape != weights_prev.shape:
            return 0.0  # shape mismatch → no meaningful cost

        if brackets is None:
            # Lazy import to avoid circular dep at module load.
            from ..config.thesis_parameters import ThesisParameters
            brackets = ThesisParameters().THESIS_TABLE_71_BRACKETS

        # Per-asset absolute traded value.
        traded_value_per_asset = np.abs(weights_new - weights_prev) * portfolio_value

        total_fee = 0.0
        # Sort brackets descending by lower_bound so we pick the
        # FIRST bracket the value qualifies for.
        brackets_desc = sorted(brackets, key=lambda b: -b[0])

        for tv in traded_value_per_asset:
            if tv == 0.0:
                continue  # no trade, no fee (still skip bracket 0 fixed fee)
            for lower_bound, prop_fee, fixed_fee in brackets_desc:
                if tv >= lower_bound:
                    total_fee += tv * prop_fee + fixed_fee
                    break

        # Return as fraction of portfolio_value so it's directly
        # subtractable from a normalized ROI.
        return float(total_fee / portfolio_value) if portfolio_value > 0.0 else 0.0
    
    @classmethod
    def moving_average(cls, returns_data: np.ndarray) -> np.ndarray:
        """
        Compute moving average of returns data.
        
        Args:
            returns_data: Matrix of returns data
        
        Returns:
            Moving average matrix
        """
        n_assets = returns_data.shape[1]
        ma = np.zeros((cls.window_size, n_assets))
        
        lags = 2 * cls.window_size
        t_ini = returns_data.shape[0] - 2 * cls.window_size
        t_fin = returns_data.shape[0]
        
        for a in range(n_assets):
            # Compute weights from autocorrelation
            weights = cls.autocorrelation[:lags, a]
            positive_weights = weights[weights > 0]
            
            if len(positive_weights) == 1:
                # If only one positive weight, use uniform weights
                weights = np.ones(lags) / lags
            
            sum_w = np.sum(weights[weights > 0])
            
            for i in range(cls.window_size):
                ma[i, a] = 0.0
                
                for t in range(t_ini + i, t_fin + i):
                    index = t - (t_ini + i)
                    w = weights[lags - (index + 1)]
                    
                    if w < 0:
                        continue
                    
                    if t < t_fin:
                        ma[i, a] += w * returns_data[t, a]
                    else:
                        ma[i, a] += w * ma[t - t_fin, a]
                
                ma[i, a] /= sum_w
        
        return ma
    
    @classmethod
    def moving_median(cls, returns_data: np.ndarray) -> np.ndarray:
        """
        Compute moving median of returns data.
        
        Args:
            returns_data: Matrix of returns data
        
        Returns:
            Moving median matrix
        """
        n_assets = returns_data.shape[1]
        mm = np.zeros((cls.window_size, n_assets))
        
        for a in range(n_assets):
            for i in range(cls.window_size):
                t_ini = returns_data.shape[0] - 2 * cls.window_size
                t_fin = returns_data.shape[0]
                
                # Extract window of data
                window_data = returns_data[t_ini:t_fin, a]
                mm[i, a] = np.median(window_data)
        
        return mm
    
    @classmethod
    def sample_autocorrelation(cls, returns_data: np.ndarray, lags: int):
        """
        Compute sample autocorrelation for returns data.
        
        Args:
            returns_data: Matrix of returns data
            lags: Number of lags to compute
        """
        n_assets = returns_data.shape[1]
        cls.autocorrelation = np.zeros((lags, n_assets))
        
        for a in range(n_assets):
            returns_series = returns_data[:, a]
            mean_ret = np.mean(returns_series)
            variance = np.var(returns_series)
            
            if variance == 0:
                continue
            
            for k in range(lags):
                autocorr = 0.0
                for i in range(len(returns_series) - k):
                    autocorr += (returns_series[i] - mean_ret) * (returns_series[i + k] - mean_ret)
                
                cls.autocorrelation[k, a] = autocorr / (variance * (len(returns_series) - k))
    
    @classmethod
    def compute_statistics(cls, returns_data: np.ndarray):
        """
        Compute all portfolio statistics.
        
        Args:
            returns_data: Matrix of returns data
        """
        cls.current_returns_data = returns_data
        
        # Compute ROI estimates
        cls.mean_ROI = cls.estimate_assets_mean_ROI(returns_data)
        cls.median_ROI = cls.estimate_assets_median_ROI(returns_data)
        
        # Compute covariance matrices
        cls.covariance = cls.estimate_covariance(cls.mean_ROI, returns_data)
        cls.robust_covariance = cls.estimate_robust_covariance(cls.mean_ROI, returns_data)
        
        # Compute autocorrelation
        cls.sample_autocorrelation(returns_data, 2 * cls.window_size)
    
    def __repr__(self):
        return f"Portfolio(ROI={self.ROI:.4f}, risk={self.risk:.4f}, cardinality={self.cardinality:.1f})"

    @classmethod
    def evaluate_stability(cls, portfolio: 'Portfolio') -> float:
        """
        Evaluate portfolio stability using Kalman filter.
        
        Args:
            portfolio: Portfolio object
            
        Returns:
            Stability measure
        """
        if portfolio.kalman_state is None:
            return 1.0  # Default stability if no Kalman filter
        
        # Use prediction error as stability measure
        # Lower prediction error indicates higher stability
        prediction_error = np.trace(portfolio.kalman_state.P)
        stability = 1.0 / (1.0 + prediction_error)
        
        return stability

    @classmethod
    def initialize_kalman_filter(cls, portfolio: 'Portfolio', 
                               initial_roi: float = 0.0, 
                               initial_risk: float = 0.0) -> None:
        """
        Initialize Kalman filter for portfolio.
        
        Args:
            portfolio: Portfolio object
            initial_roi: Initial ROI value
            initial_risk: Initial risk value
        """
        from ..algorithms.kalman_filter import create_kalman_params
        portfolio.kalman_state = create_kalman_params(initial_roi, initial_risk) 
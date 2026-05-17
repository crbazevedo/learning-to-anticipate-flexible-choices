"""
Advanced statistical functions for portfolio optimization.

This module provides multivariate normal distribution functions,
entropy calculations, and other statistical utilities needed for
anticipatory learning and stochastic dominance calculations.
"""

import numpy as np
from typing import Tuple
from scipy.stats import multivariate_normal
import warnings


def multi_norm(mu: np.ndarray, Sigma: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Generate samples from multivariate normal distribution.
    
    Args:
        mu: Mean vector
        Sigma: Covariance matrix
        num_samples: Number of samples to generate
        
    Returns:
        Matrix of samples (dimensions x num_samples)
    """
    try:
        # Use scipy's multivariate normal for sampling
        mvn = multivariate_normal(mean=mu, cov=Sigma)
        samples = mvn.rvs(size=num_samples)
        
        # Reshape if needed (scipy returns 1D for single sample)
        if num_samples == 1:
            samples = samples.reshape(-1, 1)
        else:
            samples = samples.T  # Transpose to match C++ format
            
        return samples
        
    except Exception as e:
        # Fallback to manual implementation if scipy fails
        warnings.warn(f"Scipy multivariate_normal failed: {e}. Using manual implementation.")
        return _manual_multi_norm(mu, Sigma, num_samples)


def _manual_multi_norm(mu: np.ndarray, Sigma: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Manual implementation of multivariate normal sampling.
    
    Args:
        mu: Mean vector
        Sigma: Covariance matrix
        num_samples: Number of samples to generate
        
    Returns:
        Matrix of samples (dimensions x num_samples)
    """
    size = mu.shape[0]
    
    # Cholesky decomposition for covariance matrix
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(Sigma)
        eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive
        L = eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    # Generate standard normal samples
    Z = np.random.normal(0, 1, (size, num_samples))
    
    # Transform to desired distribution
    samples = L @ Z + mu.reshape(-1, 1)
    
    return samples


def normal_cdf(z: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Compute multivariate normal cumulative distribution function.
    
    Args:
        z: Upper bounds vector
        Sigma: Covariance matrix
        
    Returns:
        Probability P(X <= z)
    """
    try:
        # Handle 1D case
        if z.ndim == 1 and len(z) == 1:
            # For 1D case, use standard normal CDF
            from scipy.stats import norm
            return norm.cdf(z[0], loc=0, scale=np.sqrt(Sigma[0, 0]))
        
        # For 2D case, we can use scipy's multivariate_normal
        if z.shape[0] == 2:
            mvn = multivariate_normal(mean=np.zeros(2), cov=Sigma)
            return mvn.cdf(z)
        else:
            # For higher dimensions, use Monte Carlo approximation
            return _monte_carlo_cdf(z, Sigma)
    except Exception as e:
        warnings.warn(f"Scipy CDF failed: {e}. Using Monte Carlo approximation.")
        return _monte_carlo_cdf(z, Sigma)


def _monte_carlo_cdf(z: np.ndarray, Sigma: np.ndarray, num_samples: int = 10000) -> float:
    """
    Monte Carlo approximation of multivariate normal CDF.
    
    Args:
        z: Upper bounds vector
        Sigma: Covariance matrix
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Approximate probability P(X <= z)
    """
    # Handle 1D case
    if z.ndim == 1 and len(z) == 1:
        z = z.reshape(1)
    
    samples = multi_norm(np.zeros(z.shape[0]), Sigma, num_samples)
    
    # Count samples that satisfy X <= z
    count = 0
    for i in range(num_samples):
        # Handle both 1D and 2D sample formats
        if samples.ndim == 2:
            sample_i = samples[:, i]
        else:
            sample_i = samples[i] if i < len(samples) else samples[0]
        
        if np.all(sample_i <= z):
            count += 1
    
    return count / num_samples


def entropy(p: float) -> float:
    """
    Compute binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Args:
        p: Probability value
        
    Returns:
        Entropy value
    """
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def linear_entropy(p: float) -> float:
    """
    Compute linear entropy: L(p) = 2*min(p, 1-p).
    
    Args:
        p: Probability value
        
    Returns:
        Linear entropy value
    """
    if p <= 0.5:
        return 2.0 * p
    return 2.0 * (1.0 - p)


def compute_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix from covariance matrix.
    
    Args:
        covariance: Covariance matrix
        
    Returns:
        Correlation matrix
    """
    std_devs = np.sqrt(np.diag(covariance))
    std_dev_matrix = np.outer(std_devs, std_devs)
    
    # Avoid division by zero - set diagonal to 1 for zero variance
    correlation = np.zeros_like(covariance)
    for i in range(covariance.shape[0]):
        for j in range(covariance.shape[1]):
            if std_dev_matrix[i, j] == 0:
                correlation[i, j] = 1.0 if i == j else 0.0
            else:
                correlation[i, j] = covariance[i, j] / std_dev_matrix[i, j]
    
    return correlation


def compute_conditional_statistics(covariance: np.ndarray, mean: np.ndarray, 
                                 condition_idx: int) -> Tuple[float, float]:
    """
    Compute conditional mean and variance for a given variable.
    
    Args:
        covariance: Covariance matrix
        mean: Mean vector
        condition_idx: Index of variable to condition on
        
    Returns:
        Tuple of (conditional_mean, conditional_variance)
    """
    n = covariance.shape[0]
    
    if n == 1:
        return mean[0], covariance[0, 0]
    
    # Partition covariance matrix
    sigma_11 = covariance[condition_idx, condition_idx]
    sigma_12 = np.delete(covariance[condition_idx, :], condition_idx)
    sigma_21 = sigma_12.reshape(-1, 1)
    sigma_22 = np.delete(np.delete(covariance, condition_idx, axis=0), condition_idx, axis=1)
    
    # Compute conditional statistics
    if sigma_11 > 0:
        conditional_mean = mean[condition_idx]
        conditional_variance = sigma_11
    else:
        conditional_mean = 0.0
        conditional_variance = 1e-10  # Small positive value
    
    return conditional_mean, conditional_variance


def compute_stochastic_parameters(roi: float, risk: float, 
                                error_covariance: np.ndarray) -> dict:
    """
    Compute stochastic parameters for anticipatory learning.
    
    Args:
        roi: Current ROI value
        risk: Current risk value
        error_covariance: Error covariance matrix from Kalman filter
        
    Returns:
        Dictionary of stochastic parameters
    """
    # Extract variances and covariance
    var_roi = error_covariance[0, 0]
    var_risk = error_covariance[1, 1]
    cov = error_covariance[0, 1]
    
    # Compute correlation
    if var_roi > 0 and var_risk > 0:
        corr = cov / np.sqrt(var_roi * var_risk)
    else:
        corr = 0.0
    
    # Compute variance ratio
    if var_risk > 0:
        var_ratio = np.sqrt(var_roi) / np.sqrt(var_risk)
    else:
        var_ratio = 1.0
    
    # Compute conditional statistics
    conditional_mean_roi, conditional_var_roi = compute_conditional_statistics(
        error_covariance, np.array([roi, risk]), 0
    )
    conditional_mean_risk, conditional_var_risk = compute_conditional_statistics(
        error_covariance, np.array([roi, risk]), 1
    )
    
    return {
        'cov': cov,
        'var_roi': var_roi,
        'var_risk': var_risk,
        'corr': corr,
        'var_ratio': var_ratio,
        'conditional_mean_roi': conditional_mean_roi,
        'conditional_var_roi': conditional_var_roi,
        'conditional_mean_risk': conditional_mean_risk,
        'conditional_var_risk': conditional_var_risk
    }


def compute_non_dominance_probability(delta1: np.ndarray, delta2: np.ndarray, 
                                    covariance: np.ndarray) -> float:
    """
    Compute probability of non-dominance for anticipatory learning.
    
    Args:
        delta1: Delta vector for first scenario
        delta2: Delta vector for second scenario
        covariance: Combined covariance matrix
        
    Returns:
        Non-dominance probability
    """
    # Point of interest for Pr {Delta < u = [0 0]^T}
    u = np.zeros(2)
    
    # Compute probabilities using multivariate normal CDF
    prob1 = normal_cdf(u - delta1, covariance)
    prob2 = normal_cdf(u - delta2, covariance)
    
    # Non-dominance probability
    nd_probability = prob1 + prob2
    
    return nd_probability


def compute_confidence_alpha(nd_probability: float) -> float:
    """
    Compute confidence parameter alpha for anticipatory learning.
    
    Args:
        nd_probability: Non-dominance probability
        
    Returns:
        Confidence parameter alpha
    """
    # Alpha quantifies confidence in prediction
    # Higher probability should give higher confidence
    # Use the probability directly as confidence measure
    alpha = nd_probability
    return alpha 
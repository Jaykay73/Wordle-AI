"""
Statistical Analysis & Confidence Interval Utilities.
Computes bootstrap 95% confidence intervals and paired statistical comparisons.
"""

from typing import List, Tuple, Dict
import numpy as np


def compute_bootstrap_ci(
    data: List[float],
    n_bootstraps: int = 1000,
    ci: float = 95.0,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculates bootstrap mean and confidence interval [mean, ci_lower, ci_upper].
    """
    if not data:
        return 0.0, 0.0, 0.0

    arr = np.array(data)
    rng = np.random.RandomState(seed)
    means = []
    n = len(arr)

    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))

    mean_val = float(np.mean(arr))
    lower_p = (100.0 - ci) / 2.0
    upper_p = 100.0 - lower_p

    ci_lower = float(np.percentile(means, lower_p))
    ci_upper = float(np.percentile(means, upper_p))

    return mean_val, ci_lower, ci_upper

"""Package teneva, module core.stat: helper functions for processing statistics.

This module contains the helper functions for processing statistics, including
computation of the CDF function and its confidence bounds.

"""
import numpy as np


def cdf_confidence(x, alpha=0.05):
    """Construct a Dvoretzky-Kiefer-Wolfowitz confidence band for the CDF.

    Args:
        x (np.ndarray): the empirical distribution in the form of 1D np.ndarray
            of length "m".
        alpha (float): "alpha" for the "(1 - alpha)" confidence band.

    Returns:
        [np.ndarray, np.ndarray]: CDF lower and upper bounds in the form of 1D
        np.ndarray of the length "m".

    """
    eps = np.sqrt(np.log(2. / alpha) / (2 * len(x)))
    return np.clip(x - eps, 0, 1), np.clip(x + eps, 0, 1)


def cdf_getter(x):
    """Build the getter for CDF.

    Args:
        x (list or np.ndarray): one-dimensional points.

    Returns:
        function: the function that computes CDF values. Its input may be one
        point (float) or a set of points (1D np.ndarray). The output
        (corresponding CDF value/values) will have the same type.

    """
    x = np.array(x, copy=True)
    x.sort()
    y = np.linspace(1./len(x), 1, len(x))

    x = np.r_[-np.inf, x]
    y = np.r_[0, y]

    def cdf(z):
        return y[np.searchsorted(x, z, 'right') - 1]

    return cdf

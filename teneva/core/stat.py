"""Package teneva, module core.stat: helper functions for processing statistics.

This module contains the helper functions for processing statistics, including
computation of the CDF function and its confidence bounds.

"""
import numpy as np


def cdf_confidence(F, alpha=0.05):
    """Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the CDF.

    Args:
        F (np.ndarray): the empirical distributions in the form of 1D np.ndarray
            of length m.
        alpha (float): alpha for a (1 - alpha) confidence band.

    Returns:
        np.ndarray: the CDF lower bound in the form of 1D np.ndarray
            of length m.
        np.ndarray: the CDF upper bound in the form of 1D np.ndarray
            of length m.

    Note:
        The description of this algorithm is presented in the work: Wasserman
        L., "All of Nonparametric Statistics".

    """
    eps = np.sqrt(np.log(2. / alpha) / (2 * len(F)))
    return np.clip(F - eps, 0, 1), np.clip(F + eps, 0, 1)


def cdf_getter(x):
    """Build the getter for CDF.

    Args:
        x (np.ndarray): 1D points in the form of np.ndarray or list.

    Returns:
        function: the function that computes CDF values (input is point in the
            form of np.ndarray and output is the corresponding float CDF value).

    """
    x = np.array(x, copy=True)
    x.sort()
    y = np.linspace(1./len(x), 1, len(x))

    x = np.r_[-np.inf, x]
    y = np.r_[0, y]

    def cdf(z):
        return y[np.searchsorted(x, z, 'right') - 1]

    return cdf

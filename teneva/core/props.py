"""Package teneva, module core.props: various properties of TT-tensors'.

This module contains the basic properties of TT-tensors, including "mean",
"norm", "ranks", etc.

"""
import numpy as np


import teneva


def accuracy_on_data(Y, I_data, Y_data, e_trunc=None):
    """Compute the relative error of TT-tensor on the dataset.

    Args:
        I_data (np.ndarray): multi-indices for items of dataset in the form of
            array of the shape [samples, d].
        Y_data (np.ndarray): values for items related to I_data of dataset in
            the form of array of the shape [samples].
        e_trunc (float): optional truncation accuracy (> 0). If this parameter
            is set, then sampling will be performed from the rounded TT-tensor.

    Returns:
        float: the relative error.

    Note:
        If "I_data" or "Y_data" is not provided, the function will return "-1".

    """
    if I_data is None or Y_data is None:
        return -1.

    I_data = np.asanyarray(I_data, dtype=int)
    Y_data = np.asanyarray(Y_data, dtype=float)

    if e_trunc is not None:
        get = teneva.getter(teneva.truncate(Y, e_trunc))
    else:
        get = teneva.getter(Y)

    Z = np.array([get(i) for i in I_data])
    e = np.linalg.norm(Z - Y_data)
    e /= np.linalg.norm(Y_data)
    return e


def erank(Y):
    """Compute effective TT-rank of the given TT-tensor.

    Effective TT-rank r of a TT-tensor Y with shape [n_1, n_2, ..., n_d] and
    TT-ranks r_0, r_1, ..., r_d (r_0 = r_d = 1) is a solution of equation
    n_1 r + \sum_{\alpha=2}^{d-1} n_\alpha r^2 + n_d r =
    \sum_{\alpha=1}^{d} n_\alpha r_{\alpha-1} r_{\alpha}.

    The representation with a constant TT-rank r (r_0 = 1, r_1 = r_2 = ... =
    r_{d-1} = r, r_d = 1) yields the same total number of parameters as in the
    original decomposition of the tensor Y.

    Args:
        Y (list): TT-tensor.

    Returns:
        float: effective TT-rank.

    """
    d, n, r = len(Y), shape(Y), ranks(Y)
    sz = np.dot(n * r[0:d], r[1:])
    b = r[0] * n[0] + n[d-1] * r[d]
    a = np.sum(n[1:d-1])
    return (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)


def ranks(Y):
    """Return the TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: TT-ranks in form of the 1D array of ints of the length d+1,
        where "d" is a number of tensor dimensions (the first and last elements
        are equal 1).

    """
    return np.array([1] + [G.shape[2] for G in Y], dtype=int)


def shape(Y):
    """Return the shape of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: shape of the tensor in form of the 1D array of ints of the
        length "d", where "d" is a number of tensor dimensions.

    """
    return np.array([G.shape[1] for G in Y], dtype=int)


def size(Y):
    """Return the size (number of parameters) of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        int: total number of parameters in the TT-representation (it is a sum
        of sizes of all TT-cores).

    """
    return np.sum([G.size for G in Y])

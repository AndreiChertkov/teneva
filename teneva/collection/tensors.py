"""Package teneva, module collection.tensors: various useful TT-tensors.

This module contains the collection of functions for explicit construction of
various useful TT-tensors (delta function, polynomial and others).

"""
import numpy as np


import teneva


def tensor_const(n, v=1.):
    """Build tensor in the TT-format with all values equal to given number.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        v (float): all elements of the tensor will be equal to this value.

    Returns:
        list: TT-tensor.

    """
    Y = [np.ones([1, k, 1], dtype=float) for k in n]
    Y[-1] *= v
    return Y


def tensor_delta(n, ind, v=1.):
    """Build TT-tensor that is zero everywhere except for a given multi-index.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        ind (list, np.ndarray): the multi-index for nonzero element. It should
            be list or np.ndarray of the length "d".
        v (float): the value of the tensor at multi-index "ind".

    Returns:
        list: TT-tensor.

    """
    Y = []
    for i, k in enumerate(n):
        G = np.zeros((1, k, 1))
        G[0, ind[i], 0] = 1.
        Y.append(G)
    Y[-1][0, ind[-1], 0] = v
    return Y


def tensor_poly(n, shift=0., power=2, scale=1.):
    """Build TT-tensor that is polynomial like scale * (index + shift)^power.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        shift (float, list, np.ndarray): the shift value. It should be list or
            np.ndarray of the length "d". It may be also float value.
        power (int): the power of polynomial.
        scale (float): the scale.

    Returns:
        list: TT-tensor.

    """
    d = len(n)
    Y = []

    shift = teneva.grid_prep_opt(shift, d)

    def _get(m, j):
        return (m + shift[j])**power

    for j, k in enumerate(n):
        if j == 0:
            G = np.zeros((1, k, 2))
            for m in range(k):
                G[0, m, :] = np.array([1., _get(m, j)])

        if j > 0 and j < d-1:
            G = np.zeros((2, k, 2))
            for m in range(k):
                G[:, m, :] = np.array([
                    [1., _get(m, j)],
                    [0., 1.],
                ])

        if j == d - 1:
            G = np.zeros((2, k, 1))
            for m in range(k):
                G[:, m, 0] = np.array([_get(m, j) * scale, scale])

        Y.append(G)

    return Y

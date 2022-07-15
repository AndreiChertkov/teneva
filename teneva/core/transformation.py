"""Package teneva, module core.transformation: transformation of TT-tensors.

This module contains the functions for orthogonalization and truncation of the
TT-tensors.

"""
import numpy as np
import scipy as sp


from .svd import matrix_svd
from .tensor import copy
from .utils import _reshape


def orthogonalize(Y, k=None):
    """Orthogonalize TT-tensor.

    Args:
        Y (list): TT-tensor.
        k (int): the leading mode for orthogonalization. The TT-cores 0, 1, ...,
            k-1 will be left-orthogonalized and the TT-cores k+1, k+2, ..., d-1
            will be right-orthogonalized. It will be the last mode by default.

    Returns:
        list: orthogonalized TT-tensor.

    """
    d = len(Y)
    k = d - 1 if k is None else k

    if k is None or k < 0 or k > d-1:
        raise ValueError('Invalid mode number')

    Z = copy(Y)

    for i in range(k):
        orthogonalize_left(Z, i, inplace=True)

    for i in range(d-1, k, -1):
        orthogonalize_right(Z, i, inplace=True)

    return Z


def orthogonalize_left(Y, k, inplace=False):
    """Left-orthogonalization for TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): mode for orthogonalization (>= 0 and < d-1).
        inplace (bool): if flag is set, then the original TT-tensor (i.e.,
            the function argument will be transformed). Otherwise, a copy of
            the TT-tensor will be made.

    Returns:
        list: orthogonalized TT-tensor.

    """
    d = len(Y)

    if k is None or k < 0 or k >= d-1:
        raise ValueError('Invalid mode number')

    Z = Y if inplace else copy(Y)

    r1, n1, r2 = Z[k].shape
    G1 = _reshape(Z[k], (r1 * n1, r2))
    Q, R = np.linalg.qr(G1, mode='reduced')
    Z[k] = _reshape(Q, (r1, n1, Q.shape[1]))

    r2, n2, r3 = Z[k+1].shape
    G2 = _reshape(Z[k+1], (r2, n2 * r3))
    G2 = R @ G2
    Z[k+1] = _reshape(G2, (G2.shape[0], n2, r3))

    return Z


def orthogonalize_right(Y, k, inplace=False):
    """Right-orthogonalization for TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): mode for orthogonalization (> 0 and <= d-1).
        inplace (bool): if flag is set, then the original TT-tensor (i.e.,
            the function argument will be transformed). Otherwise, a copy of
            the TT-tensor will be made.

    Returns:
        list: orthogonalized TT-tensor.

    """
    d = len(Y)

    if k is None or k <= 0 or k > d-1:
        raise ValueError('Invalid mode number')

    Z = Y if inplace else copy(Y)

    r2, n2, r3 = Z[k].shape
    G2 = _reshape(Z[k], (r2, n2 * r3))
    L, Q = sp.linalg.rq(G2, mode='economic', check_finite=False)
    Z[k] = _reshape(Q, (Q.shape[0], n2, r3))

    r1, n1, r2 = Z[k-1].shape
    G1 = _reshape(Z[k-1], (r1 * n1, r2))
    G1 = G1 @ L
    Z[k-1] = _reshape(G1, (r1, n1, G1.shape[1]))

    return Z


def truncate(Y, e=1.E-10, r=1.E+12, orth=True):
    """Truncate (round) TT-tensor.

    Args:
        Y (list): TT-tensor wth overestimated ranks.
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum TT-rank of the result (> 0).
        orth (bool): if the flag is set, then tensor orthogonalization will be
            performed (it is True by default).

    Returns:
        list: TT-tensor, which is rounded up to a given accuracy "e" and
        satisfying the rank constraint "r".

    """
    d = len(Y)

    if orth:
        Z = orthogonalize(Y, d-1)
        e = e / np.sqrt(d-1) * np.linalg.norm(Z[-1])
    else:
        Z = copy(Y)

    for k in range(d-1, 0, -1):
        r1, n, r2 = Z[k].shape
        G = _reshape(Z[k], (r1, n * r2))
        U, V = matrix_svd(G, e, r)
        Z[k] = _reshape(V, (-1, n, r2))
        Z[k-1] = np.einsum('ijq,ql', Z[k-1], U, optimize=True)

    return Z

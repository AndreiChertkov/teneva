"""Package teneva, module core.transformation: transformation of TT-tensors.

This module contains the functions for orthogonalization and truncation of the
TT-tensors.

"""
import numpy as np
import scipy as sp


from .svd import matrix_svd


def orthogonalize(Y, k=None):
    """Orthogonalize TT-tensor (note that operation is performed inplace).

    Args:
        Y (list): TT-tensor.
        k (int): the leading mode for orthogonalization. The TT-cores 0, 1, ...,
            k-1 will be left-orthogonalized and the TT-cores k+1, k+2, ..., d-1
            will be right-orthogonalized. It will be the last mode by default.

    Returns:
        [np.ndarray, np.ndarray]: R and L factor matrices.

    """
    d = len(Y)
    k = d - 1 if k is None else k
    if k is None or k < 0 or k > d-1:
        raise ValueError('Invalid mode number')

    R = np.array([[1.]])
    for i in range(k):
        R = orthogonalize_left(Y, i)

    L = np.array([[1.]])
    for i in range(d-1, k, -1):
        L = orthogonalize_right(Y, i)

    return R, L


def orthogonalize_left(Y, k):
    """Left-orthogonalization for TT-tensor (operation is performed inplace).

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): mode for orthogonalization (>= 0 and < d-1).

    Returns:
        np.ndarray: R factor matrix.

    """
    d = len(Y)
    if k is None or k < 0 or k >= d-1:
        raise ValueError('Invalid mode number')

    r1, n1, r2 = Y[k].shape
    G1 = _reshape(Y[k], (r1 * n1, r2))
    Q, R = np.linalg.qr(G1, mode='reduced')
    Y[k] = _reshape(Q, (r1, n1, Q.shape[1]))

    r2, n2, r3 = Y[k+1].shape
    G2 = _reshape(Y[k+1], (r2, n2 * r3))
    G2 = R @ G2
    Y[k+1] = _reshape(G2, (G2.shape[0], n2, r3))

    return R

def orthogonalize_right(Y, k):
    """Right-orthogonalization for TT-tensor (operation is performed inplace).

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): mode for orthogonalization (> 0 and <= d-1).

    Returns:
        np.ndarray: L factor matrix.

    """
    d = len(Y)
    if k is None or k <= 0 or k > d-1:
        raise ValueError('Invalid mode number')

    r2, n2, r3 = Y[k].shape
    G2 = _reshape(Y[k], (r2, n2 * r3))
    L, Q = sp.linalg.rq(G2, mode='economic', check_finite=False)
    Y[k] = _reshape(Q, (Q.shape[0], n2, r3))

    r1, n1, r2 = Y[k-1].shape
    G1 = _reshape(Y[k-1], (r1 * n1, r2))
    G1 = G1 @ L
    Y[k-1] = _reshape(G1, (r1, n1, G1.shape[1]))

    return L


def truncate(Y, e, r=1.E+12, orth=True):
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
    Z = [G.copy() for G in Y]

    if orth:
        orthogonalize(Z)
        e = e / np.sqrt(d-1) * np.linalg.norm(Z[-1])

    for k in range(d-1, 0, -1):
        r1, n, r2 = Z[k].shape
        G = _reshape(Z[k], (r1, n * r2))
        U, V = matrix_svd(G, e, r)
        Z[k] = _reshape(V, (-1, n, r2))
        Z[k-1] = np.einsum('ijq,ql', Z[k-1], U, optimize=True)

    return Z


def _reshape(A, n):
    return np.reshape(A, n, order='F')

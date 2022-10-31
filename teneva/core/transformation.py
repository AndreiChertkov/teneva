"""Package teneva, module core.transformation: transformation of TT-tensors.

This module contains the functions for orthogonalization, truncation and
transformation into full (numpy) format of the TT-tensors.

"""
import numpy as np
import scipy as sp


from .act_one import copy
from .core import core_stab
from .svd import matrix_svd
from .utils import _reshape
import teneva


def full(Y):
    """Export TT-tensor to the full (numpy) format.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: multidimensional array related to the given TT-tensor.

    Note:
         This function can only be used for relatively small tensors, because
         the resulting tensor will have n^d elements and may not fit in memory
         for large dimensions.

    """
    Z = Y[0].copy()
    for i in range(1, len(Y)):
        Z = np.tensordot(Z, Y[i], 1)
        
    if Z.shape[0] == 1:
        Z = Z[0, ...]
        
    if Z.shape[-1] == 1:
        Z = Z[..., 0]
        
    return Z


def full_matrix(Y):
    """Export QTT-matrix to the full (numpy) format.

    Args:
        Y (list): TT-tensor of dimension "q" and mode size "4", which
            represents the QTT-matrix of the shape "2^q x 2^q".

    Returns:
        np.ndarray: the matrix of the shape "2^q x 2^q".

    Note:
         This function can only be used for relatively small mode size, because
         the resulting matrix may not fit in memory otherwise.

    """
    q = len(Y)

    Z = full(Y)
    Z = Z.reshape([2, 2]*q, order='F')

    prm = np.hstack((np.arange(0, 2*q, 2), np.arange(1, 2*q, 2)))
    Z = Z.transpose(prm).reshape(2**q, 2**q, order='F')

    return Z


def orthogonalize(Y, k=None, use_stab=False):
    """Orthogonalize TT-tensor.

    Args:
        Y (list): TT-tensor.
        k (int): the leading mode for orthogonalization. The TT-cores 0, 1, ...,
            k-1 will be left-orthogonalized and the TT-cores k+1, k+2, ..., d-1
            will be right-orthogonalized. It will be the last mode by default.
        use_stab (bool): if flag is set, then function will also return the
            second argument "p", which is the factor of 2-power.

    Returns:
        list: orthogonalized TT-tensor.

    """
    d = len(Y)
    k = d - 1 if k is None else k

    if k is None or k < 0 or k > d-1:
        raise ValueError('Invalid mode number')

    Z = copy(Y)
    p = 0

    for i in range(k):
        orthogonalize_left(Z, i, inplace=True)
        if use_stab:
            Z[i+1], p = core_stab(Z[i+1], p)

    for i in range(d-1, k, -1):
        orthogonalize_right(Z, i, inplace=True)
        if use_stab:
            Z[i-1], p = core_stab(Z[i-1], p)

    return (Z, p) if use_stab else Z


def orthogonalize_left(Y, i, inplace=False):
    """Left-orthogonalization for TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        i (int): mode for orthogonalization (>= 0 and < d-1).
        inplace (bool): if flag is set, then the original TT-tensor (i.e.,
            the function argument) will be transformed. Otherwise, a copy of
            the TT-tensor will be returned.

    Returns:
        list: TT-tensor with left orthogonalized i-th mode.

    """
    d = len(Y)

    if i is None or i < 0 or i >= d-1:
        raise ValueError('Invalid mode number')

    Z = Y if inplace else copy(Y)

    r1, n1, r2 = Z[i].shape
    G1 = _reshape(Z[i], (r1 * n1, r2))
    Q, R = np.linalg.qr(G1, mode='reduced')
    Z[i] = _reshape(Q, (r1, n1, Q.shape[1]))

    r2, n2, r3 = Z[i+1].shape
    G2 = _reshape(Z[i+1], (r2, n2 * r3))
    G2 = R @ G2
    Z[i+1] = _reshape(G2, (G2.shape[0], n2, r3))

    return Z


def orthogonalize_right(Y, i, inplace=False):
    """Right-orthogonalization for TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        i (int): mode for orthogonalization (> 0 and <= d-1).
        inplace (bool): if flag is set, then the original TT-tensor (i.e.,
            the function argument) will be transformed. Otherwise, a copy of
            the TT-tensor will be returned.

    Returns:
        list: TT-tensor with right orthogonalized i-th mode.

    """
    d = len(Y)

    if i is None or i <= 0 or i > d-1:
        raise ValueError('Invalid mode number')

    Z = Y if inplace else copy(Y)

    r2, n2, r3 = Z[i].shape
    G2 = _reshape(Z[i], (r2, n2 * r3))
    R, Q = sp.linalg.rq(G2, mode='economic', check_finite=False)
    Z[i] = _reshape(Q, (Q.shape[0], n2, r3))

    r1, n1, r2 = Z[i-1].shape
    G1 = _reshape(Z[i-1], (r1 * n1, r2))
    G1 = G1 @ R
    Z[i-1] = _reshape(G1, (r1, n1, G1.shape[1]))

    return Z


def truncate(Y, e=1.E-10, r=1.E+12, orth=True, use_stab=False, is_eigh=True):
    """Truncate (round) TT-tensor.

    Args:
        Y (list): TT-tensor wth overestimated ranks.
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum TT-rank of the result (> 0).
        orth (bool): if the flag is set, then tensor orthogonalization will be
            performed (it is True by default).
        use_stab (bool): if flag is set, then the additional stabilization will
            be used.

    Returns:
        list: TT-tensor, which is rounded up to a given accuracy "e" and
        satisfying the rank constraint "r".

    """
    d = len(Y)

    if orth:
        if use_stab:
            Z, p = orthogonalize(Y, d-1, True)
            e = e / np.sqrt(d-1) * np.linalg.norm(Z[-1]) # TODO!
        else:
            Z, p = orthogonalize(Y, d-1), 0
            e = e / np.sqrt(d-1) * np.linalg.norm(Z[-1])
    else:
        Z, p = copy(Y), 0

    for k in range(d-1, 0, -1):
        r1, n, r2 = Z[k].shape
        G = _reshape(Z[k], (r1, n * r2))
        if is_eigh:
            U, V = teneva.matrix_svd(G, e, r)
        else:
            U, V = teneva.matrix_skeleton(G, e, r, rel=False, give_to='r')
        Z[k] = _reshape(V, (-1, n, r2))
        Z[k-1] = np.einsum('ijq,ql', Z[k-1], U, optimize=True)

    if use_stab:
        for k in range(d):
            Z[k] *= 2**(p/d)

    return Z

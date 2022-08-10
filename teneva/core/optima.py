"""Package teneva, module core.optima: estimation of min and max of tensor.

This module contains the novel algorithm for computation of minimum and
maximum element of the given TT-tensor (function optima_tt).

"""
import numpy as np


from .act_one import copy
from .act_one import get
from .act_one import tt_to_qtt
from .act_two import mul
from .act_two import sub
from .grid import ind_qtt_to_tt
from .props import shape
from .transformation import orthogonalize
from .utils import _ones
from .utils import _range
from teneva import tensor_const


def optima_qtt(Y, k=100, e=1.E-12, r=100):
    """Find items which relate to min and max elements of the TT-tensor.

    The provided TT-tensor is transformed into the QTT-format and then
    "optima_tt" method is applied to the QTT-tensor. Note that this method
    support only the tensors with constant mode size, which is a power of two,
    i.e., the shape should be "[2^q, 2^q, ..., 2^q]".

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.
        e (float): desired approximation accuracy for the QTT-tensor (> 0).
        r (int, float): maximum rank for the SVD decomposition for the
            QTT-tensor(> 0).

    Returns:
        [np.ndarray, float, np.ndarray, float]: multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like "i_min, y_min, i_max, y_max".

    """
    n = shape(Y)
    for n_ in n[1:]:
        if n[0] != n_:
            msg = 'Invalid mode size (it should be equal for all modes)'
            raise ValueError(msg)

    n = n[0]
    q = int(np.log2(n))

    if 2**q != n:
        msg = 'Invalid mode size (it should be a power of two)'
        raise ValueError(msg)

    Z = tt_to_qtt(Y, e, r)

    i_min, y_min, i_max, y_max = optima_tt(Z, k)

    i_min = ind_qtt_to_tt(i_min, q)
    i_max = ind_qtt_to_tt(i_max, q)

    return i_min, y_min, i_max, y_max


def optima_tt(Y, k=100):
    """Find items which relate to min and max elements of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.

    Returns:
        [np.ndarray, float, np.ndarray, float]: multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like "i_min, y_min, i_max, y_max".

    """
    i1, y1 = optima_tt_max(Y, k)

    D = tensor_const(shape(Y), y1)
    Z = sub(Y, D)
    Z = mul(Z, Z)

    i2, _ = optima_tt_max(Z, k)
    y2 = get(Y, i2)

    if y2 > y1:
        return i1, y1, i2, y2
    else:
        return i2, y2, i1, y1


def optima_tt_beam(Y, k=100, ret_all=False, l2r=True):
    """Find maximum modulo value of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.
        ret_all (bool): if flag is set, then all "k" multi-indices will be
            returned. Otherwise, only best found multi-index will be returned.
        l2r (bool): if flag is set, hen the TT-cores are passed in the from
            left to right (that is, from the first to the last TT-core).
            Otherwise, the TT-cores are passed from right to left.

    Returns:
        np.ndarray: multi-index (array of length d) which relates to maximum
        modulo TT-tensor element if "ret_all" flag is not set. If "ret_all" flag
        is set, then it will be the set of "k" best multi-indices (array of the
        shape [k, d]).

    """
    Z, p = orthogonalize(Y, 0 if l2r else len(Y)-1, use_stab=True)
    p0 = p / len(Z) # Scale factor (2^p0) for each TT-core

    G = Z[0 if l2r else -1]
    r1, n, r2 = G.shape

    I = _range(n)
    Q = G.reshape(n, r2) if l2r else G.reshape(r1, n)

    Q *= 2**p0

    for G in (Z[1:] if l2r else Z[:-1][::-1]):
        r1, n, r2 = G.shape

        if l2r:
            Q = np.einsum('kr,riq->kiq', Q, G, optimize='optimal')
            Q = Q.reshape(-1, r2)
        else:
            Q = np.einsum('qir,rk->qik', G, Q, optimize='optimal')
            Q = Q.reshape(r1, -1)

        if l2r:
            I1 = np.kron(I, _ones(n))
            I2 = np.kron(_ones(I.shape[0]), _range(n))
        else:
            I1 = np.kron(_range(n), _ones(I.shape[0]))
            I2 = np.kron(_ones(n), I)
        I = np.hstack((I1, I2))

        q_max = np.max(np.abs(Q))
        norms = np.sum((Q/q_max)**2, axis=1 if l2r else 0)
        ind = np.argsort(norms)[:-(k+1):-1]

        I = I[ind]
        Q = Q[ind] if l2r else Q[:, ind]

        Q *= 2**p0

    return I if ret_all else I[0]


def optima_tt_max(Y, k=100):
    """Build a good approximation for the maximum modulo value of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.

    Returns:
        [np.ndarray, float]: multi-index (array of length d) which relates to
        maximum modulo TT-tensor element and the related value of the tensor
        item (float).

    """
    i_max_list = [
        optima_tt_beam(Y, k, l2r=True),
        optima_tt_beam(Y, k, l2r=False),
    ]

    y_max_list = [get(Y, i) for i in i_max_list]
    index_best = np.argmax([np.abs(y) for y in y_max_list])
    return i_max_list[index_best], y_max_list[index_best]

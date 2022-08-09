"""Package teneva, module core.optima: estimation of min and max of tensor.

This module contains the novel algorithm for computation of minimum and
maximum element of the given TT-tensor (function optima_tt).

"""
import numpy as np


from .act_one import copy
from .act_one import get
from .act_two import mul
from .act_two import sub
from .props import shape
from .transformation import orthogonalize
from .transformation import truncate
from .utils import _ones
from .utils import _range
from teneva import tensor_const
from teneva import tensor_delta


def optima_tt(Y, k=100, nswp=1, e=1.E-10):
    """Find items which relate to min and max elements of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.
        nswp (int): number of power-iterations (> 0; this is an experimental
            option, the default value "1" means no iterations).
        e (float): accuracy for intermediate truncations (> 0; this is only
            relevant if "nswp > 1").

    Returns:
        [np.ndarray, float, np.ndarray, float]: multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like "i_min, y_min, i_max, y_max".

    Note:
        As it turns out empirically, this function often gives the same result
        as if we converted the TT-tensor to the full format (i.e., teneva.full)
        and explicitly found its minimum and maximum elements. However, this
        values will not always correspond to the minimum and maximum of the
        original tensor (for which the TT-tensor is an approximation). In the
        latter case, the accuracy of the min/max will depend on the accuracy of
        the TT-approximation.

    """
    i_min, y_min, i_max, y_max = optima_tt_min(Y, k, nswp, e, with_max=True)

    if y_min < y_max:
        return i_min, y_min, i_max, y_max
    else:
        return i_max, y_max, i_min, y_min


def optima_tt_beam_left(Y, k=100, ort_num=-1, ret_all=False):
    """Find maximum modulo value of TT-tensor via right to left sweep."""
    Z = mul(Y, Y) # In old version was Z = copy(Y)

    if ort_num is not None:
        ort_num = ort_num if ort_num >= 0 else len(Z)-1
        Z, p = orthogonalize(Z, ort_num, use_stab=True)
    else:
        p = 0

    # Scale factor (2^p0) for each TT-core:
    p0 = p / len(Z)

    G = Z[-1]
    r1, n, r2 = G.shape
    Q = G.reshape(r1, n)
    Q *= 2**p0
    I = _range(n)

    for G in Z[:-1][::-1]:
        r1, n, r2 = G.shape
        Q = np.einsum('qir,rk->qik', G, Q, optimize='optimal').reshape(r1, -1)

        I1 = np.kron(_range(n), _ones(I.shape[0]))
        I2 = np.kron(_ones(n), I)
        I = np.hstack((I1, I2))

        q_max = np.max(np.abs(Q))
        norms = np.sum((Q/q_max)**2, axis=0)
        ind = np.argsort(norms)[:-(k+1):-1]
        I = I[ind]
        Q = Q[:, ind]

        Q *= 2**p0

    return I if ret_all else I[0]


def optima_tt_beam_right(Y, k=100, ort_num=0, ret_all=False):
    """Find maximum modulo value of TT-tensor via left to right sweep."""
    Z = mul(Y, Y) # In old version was Z = copy(Y)

    if ort_num is not None:
        ort_num = ort_num if ort_num >= 0 else len(Z)-1
        Z, p = orthogonalize(Z, ort_num, use_stab=True)
    else:
        p = 0

    # Scale factor (2^p0) for each TT-core:
    p0 = p / len(Z)

    G = Z[0]
    r1, n, r2 = G.shape
    Q = G.reshape(n, r2)
    Q *= 2**p0
    I = _range(n)

    for G in Z[1:]:
        r1, n, r2 = G.shape
        Q = np.einsum('kr,riq->kiq', Q, G, optimize='optimal').reshape(-1, r2)

        I1 = np.kron(I, _ones(n))
        I2 = np.kron(_ones(I.shape[0]), _range(n))
        I = np.hstack((I1, I2))

        q_max = np.max(np.abs(Q))
        norms = np.sum((Q/q_max)**2, axis=1)
        ind = np.argsort(norms)[:-(k+1):-1]
        I = I[ind]
        Q = Q[ind]

        Q *= 2**p0

    return I if ret_all else I[0]


def optima_tt_max(Y, k=100, nswp=1, e=1.E-10):
    """Build a good approximation for the maximum modulo value of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.
        nswp (int): number of power-iterations (> 0; this is an experimental
            option, the default value "1" means no iterations).
        e (float): accuracy for intermediate truncations (> 0; this is only
            relevant if "nswp > 1").

    Returns:
        [np.ndarray, float]: multi-index (array of length d) which relates to
        maximum modulo TT-tensor element and the related value of the tensor
        item (float).

    Note:
        As it turns out empirically, this function often gives the same result
        as if we converted the TT-tensor to the full format (i.e., teneva.full)
        and explicitly found its maximum modulo element. However, this values
        will not always correspond to the minimum of the original tensor (for
        which the TT-tensor is an approximation). In the latter case, the
        accuracy of the max will depend on the accuracy of the TT-approximation.

    """
    def find_index(Z):
        i_opt_list = [
            optima_tt_beam_right(Z, k),
            optima_tt_beam_left(Z, k),
        ]
        y_opt_list = [get(Z, i_opt) for i_opt in i_opt_list]
        ind = np.argmax([np.abs(y_opt) for y_opt in y_opt_list])
        return i_opt_list[ind]

    Z = copy(Y)

    i_max = None
    y_max = None

    for swp in range(nswp):
        i_max_new = find_index(Z)
        y_max_new = get(Y, i_max_new)

        if i_max is not None and np.max(np.abs(i_max_new - i_max)) == 0:
            print(f'Converged optima_tt_max (nswp = {swp+1})')
            break

        if y_max is None or abs(y_max_new) > abs(y_max):
            i_max = i_max_new.copy()
            y_max = y_max_new

        if nswp > 1:
            z = get(Z, i_max_new)
            D = tensor_delta(shape(Z), i_max_new, z)
            Z = sub(Z, D)
            Z = truncate(Z, e)

    return i_max, y_max


def optima_tt_min(Y, k=100, nswp=1, e=1.E-10, with_max=False):
    """Build a good approximation for the minimum modulo value of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items for each tensor mode.
        nswp (int): number of power-iterations (> 0; this is an experimental
            option, the default value "1" means no iterations).
        e (float): accuracy for intermediate truncations (> 0; this is only
            relevant if "nswp > 1").
        with_max (bool): if this (service) flag is True, then max-values will
            be also returned. Note that this function finds both min and value
            due to design.

    Returns:
        [np.ndarray, float]: multi-index (array of length d) which relates to
        minimum modulo TT-tensor element and the related value of the tensor
        item (float).

    Note:
        As it turns out empirically, this function often gives the same result
        as if we converted the TT-tensor to the full format (i.e., teneva.full)
        and explicitly found its minimum modulo element. However, this values
        will not always correspond to the minimum of the original tensor (for
        which the TT-tensor is an approximation). In the latter case, the
        accuracy of the min will depend on the accuracy of the TT-approximation.

    """
    i_max, y_max = optima_tt_max(Y, k, nswp, e)

    D = tensor_const(shape(Y), y_max)
    Z = sub(Y, D)
    # Z = truncate(Z, e)

    i_min, y_min_shifted = optima_tt_max(Z, k, nswp, e)
    y_min = y_min_shifted + y_max

    if with_max:
        return i_min, y_min, i_max, y_max
    else:
        return i_min, y_min

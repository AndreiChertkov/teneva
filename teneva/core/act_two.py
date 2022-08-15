"""Package teneva, module core.act_two: operations with a pair of TT-tensors.

This module contains the basic operations with a pair of TT-tensors (Y1, Y2),
including "add", "mul", "sub", etc.

"""
import numpy as np


from .act_one import copy
from .act_one import norm
from .core import core_stab
from .props import ranks
from .props import shape
from .utils import _is_num
from teneva import tensor_const


def accuracy(Y1, Y2):
    """Compute || Y1 - Y2 || / || Y2 || for tensors in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        float: the relative difference between two tensors.

    Note:
        Also, for convenience, tensors in numpy format can be passed, in this
        case the norm will be calculated using the function "linalg.norm".

    """
    if isinstance(Y1, np.ndarray):
        return np.linalg.norm(Y1 - Y2) / np.linalg.norm(Y2)

    z1, p1 = norm(sub(Y1, Y2), use_stab=True)
    z2, p2 = norm(Y2, use_stab=True)

    if p1 - p2 > 500:
        return 1.E+299
    if p1 - p2 < -500:
        return 0.

    c = 2.**(p1 - p2)

    if np.isinf(c) or np.isinf(z1) or np.isinf(z2) or abs(z2) < 1.E-100:
        return -1 # TODO: check

    return c * z1 / z2


def add(Y1, Y2):
    """Compute Y1 + Y2 in the TT-format.

    Args:
        Y1 (int, float, list): TT-tensor (or it may be int/float).
        Y2 (int, float, list): TT-tensor (or it may be int/float).

    Returns:
        list: TT-tensor, which represents the element wise sum of Y1 and Y2.
        If both Y1 and Y2 are numbers, then result will be int/float number.

    """
    if _is_num(Y1) and _is_num(Y2):
        return Y1 + Y2
    elif _is_num(Y1):
        Y1 = tensor_const(shape(Y2), Y1)
    elif _is_num(Y2):
        Y2 = tensor_const(shape(Y1), Y2)

    n, r1, r2, Y = shape(Y1), ranks(Y1), ranks(Y2), []
    for i, (G1, G2, k) in enumerate(zip(Y1, Y2, n)):
        if i == 0:
            G = np.concatenate([G1, G2], axis=2)
        elif i == len(n) - 1:
            G = np.concatenate([G1, G2], axis=0)
        else:
            r1_l, r1_r = r1[i:i+2]
            r2_l, r2_r = r2[i:i+2]
            Z1 = np.zeros([r1_l, k, r2_r])
            Z2 = np.zeros([r2_l, k, r1_r])
            L1 = np.concatenate([G1, Z1], axis=2)
            L2 = np.concatenate([Z2, G2], axis=2)
            G = np.concatenate([L1, L2], axis=0)
        Y.append(G)

    return Y


def mul(Y1, Y2):
    """Compute element wise product Y1 * Y2 in the TT-format.

    Args:
        Y1 (int, float, list): TT-tensor (or it may be int/float).
        Y2 (int, float, list): TT-tensor (or it may be int/float).

    Returns:
        list: TT-tensor, which represents the element wise product of Y1 and Y2.
        If both Y1 and Y2 are numbers, then result will be float number.

    """
    if _is_num(Y1) and _is_num(Y2):
        return Y1 * Y2

    if _is_num(Y1):
        Y = copy(Y2)
        Y[0] *= Y1
        return Y

    if _is_num(Y2):
        Y = copy(Y1)
        Y[0] *= Y2
        return Y

    Y = []
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        Y.append(G)

    return Y


def mul_scalar(Y1, Y2, use_stab=False):
    """Compute scalar product for Y1 and Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.
        use_stab (bool): if flag is set, then function will also return the
            second argument "p", which is the factor of 2-power.

    Returns:
        float: the scalar product.

    """
    v = None
    p = 0

    for i, (G1, G2) in enumerate(zip(Y1, Y2)):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        v = G.copy() if i == 0 else v @ G

        if use_stab:
            v, p = core_stab(v, p)

    v = v.item()

    return (v, p) if use_stab else v


def outer(Y1, Y2):
    """Compute outer product of two TT-tensors.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        list: TT-tensor, which is the outer product of Y1 and Y2.

    Note:
        See also "kron_many" function, which computes outer product of many
        given TT-tensors.

    """
    Y = copy(Y1)
    Y.extend(copy(Y2))
    return Y


def sub(Y1, Y2):
    """Compute Y1 - Y2 in the TT-format.

    Args:
        Y1 (int, float, list): TT-tensor (or it may be int/float).
        Y2 (int, float, list): TT-tensor (or it may be int/float).

    Returns:
        list: TT-tensor, which represents the result of the operation Y1 - Y2.
        If both Y1 and Y2 are numbers, then result will be float number.

    """
    if _is_num(Y1) and _is_num(Y2):
        return Y1 - Y2

    if _is_num(Y2):
        Y2 = tensor_const(shape(Y1), -1.*Y2)
    else:
        Y2 = copy(Y2)
        Y2[0] *= -1.

    return add(Y1, Y2)

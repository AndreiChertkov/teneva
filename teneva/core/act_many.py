"""Package teneva, module core.act_many: operations with a set of TT-tensors.

This module contains the basic operations with a set of multiple TT-tensors
(Y1, Y2, ...), including "add_many" and "kron_many".

"""
from .act_one import copy
from .act_two import add
from .transformation import truncate
from .utils import _is_num


def add_many(Y_many, e=1.E-10, r=1.E+12, trunc_freq=15):
    """Compute Y1 + Y2 + ... + Ym in the TT-format.

    Args:
        Y_many (list): the list of TT-tensors (some of them may be int/float).
        e (float): desired approximation accuracy (> 0). The result will be
            truncated to this accuracy.
        r (int, float): maximum rank of the result (> 0).
        trunc_freq (int): frequency of intermediate summation result truncation.

    Returns:
        list: TT-tensor, which represents the element wise sum of all given
        tensors. If all the tensors are numbers, then result will be int/float.

    """
    Y = copy(Y_many[0])
    for i, Y_curr in enumerate(Y_many[1:]):
        Y = add(Y, Y_curr)
        if not _is_num(Y) and (i+1) % trunc_freq == 0:
            Y = truncate(Y, e)
    return truncate(Y, e, r) if not _is_num(Y) else Y


def outer_many(Y_many):
    """Compute outer product of given TT-tensors.

    Args:
        Y_many (list): list of TT-tensors.

    Returns:
        list: TT-tensor, which is the outer product of given TT-tensors.

    Note:
        See also "outer" function, which outer kronecker product of two given
        TT-tensors.

    """
    if len(Y_many) == 0:
        return None

    Y = copy(Y_many[0])
    for Y_curr in Y_many[1:]:
        Y.extend(copy(Y_curr))

    return Y

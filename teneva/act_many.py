"""Package teneva, module act_many: operations with a set of TT-tensors.

This module contains the basic operations with a set of multiple TT-tensors
(Y1, Y2, ...), including "add_many" and "outer_many".

"""
import teneva


def add_many(Y_many, e=1.E-10, r=1.E+12, trunc_freq=15):
    """Compute Y1 + Y2 + ... + Ym in the TT-format.

    Args:
        Y_many (list): the list of TT-tensors (some of them may be int/float).
        e (float): desired approximation accuracy. The result will be truncated
            to this accuracy.
        r (int): maximum rank of the result.
        trunc_freq (int): frequency of intermediate summation result truncation.

    Returns:
        list: TT-tensor, which represents the element wise sum of all given
        tensors. If all the tensors are numbers, then result will be int/float.

    """
    Y = teneva.copy(Y_many[0])
    for i, Y_curr in enumerate(Y_many[1:]):
        Y = teneva.add(Y, Y_curr)
        if not teneva._is_num(Y) and (i+1) % trunc_freq == 0:
            Y = teneva.truncate(Y, e)
    return teneva.truncate(Y, e, r) if not teneva._is_num(Y) else Y


def outer_many(Y_many):
    """Compute outer product of given TT-tensors.

    Args:
        Y_many (list): list of TT-tensors.

    Returns:
        list: TT-tensor, which is the outer product of given TT-tensors.

    Note:
        See also "outer" function, which computes outer kronecker product of
        two given TT-tensors.

    """
    if len(Y_many) == 0:
        return None

    Y = teneva.copy(Y_many[0])
    for Y_curr in Y_many[1:]:
        Y.extend(teneva.copy(Y_curr))

    return Y

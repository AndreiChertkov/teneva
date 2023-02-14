"""Package teneva, module core.act_two: operations with a pair of TT-tensors.

This module contains the basic operations with a pair of TT-tensors (Y1, Y2),
including "add", "mul", "sub", etc.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def accuracy(Y1, Y2):
    """Compute || Y1 - Y2 || / || Y2 || for tensors in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        np.ndarray of size 1: the relative difference between two tensors.

    """
    z1, p1 = teneva.norm(sub(Y1, Y2), use_stab=True)
    z2, p2 = teneva.norm(Y2, use_stab=True)

    if p1 - p2 > 500:
        return 1.E+299
    if p1 - p2 < -500:
        return 0.

    c = 2.**(p1 - p2)

    if np.isinf(c) or np.isinf(z1) or np.isinf(z2) or abs(z2) < 1.E-100:
        return -1 # TODO: check

    return c * z1 / z2

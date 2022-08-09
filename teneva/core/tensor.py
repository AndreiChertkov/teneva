"""Package teneva, module core.tensor: basic operations with TT-tensors.

This module contains the basic operations and utilities for TT-tensors,
including "add", "get", "mul", etc.

"""
import numpy as np


def rand(n, r, f=np.random.randn):
    """Construct random TT-tensor.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        r (int, float, list, np.ndarray): TT-ranks of the tensor. It should be
            list or np.ndarray of the length d+1 with outer elements (first and
            last) equals to 1. If all inner TT-ranks are equal, it may be the
            int/float number.
        f (function): sampling function.

    Returns:
        list: TT-tensor.

    """
    n = np.asanyarray(n, dtype=int)
    d = n.size

    if isinstance(r, (int, float)):
        r = [1] + [int(r)] * (d - 1) + [1]
    r = np.asanyarray(r, dtype=int)

    ps = np.cumsum(np.concatenate(([1], n * r[0:d] * r[1:d+1])))
    ps = ps.astype(int)
    core = f(ps[d] - 1)

    Y = []
    for i in range(d):
        G = core[ps[i]-1:ps[i+1]-1]
        Y.append(G.reshape((r[i], n[i], r[i+1]), order='F'))

    return Y

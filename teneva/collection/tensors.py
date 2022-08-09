"""Package teneva, module collection.tensors: various useful TT-tensors.

This module contains the collection of functions for explicit construction of
various useful TT-tensors (delta function, polynomial and others).

"""
import numpy as np


import teneva


def tensor_const(n, v=1., i_non_zero=None, I_zero=None):
    """Build tensor in the TT-format with all values equal to given number.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        v (float): all elements of the tensor will be equal to this value.
        i_non_zero (list, np.ndarray): optional multi-index in the form of list
            or np.ndarray of the shape d, which is not affected by zero
            multi-indices (I_zero).
        I_zero (list, np.ndarray): optional list of lists or np.ndarray of the
            shape samples x d, which relates to multi-indices, where tensor
            should be zero (this ensures that the value at index "i_non_zero"
            is not affected).

    Returns:
        list: TT-tensor.

    """
    d = len(n)
    s = abs(v) / v if abs(v) > 1.E-16 else v
    v = abs(v)**(1./d) if abs(v) > 1.E-16 else 1.
    Y = [np.ones([1, k, 1]) * v for k in n]
    Y[-1] *= s

    if I_zero is not None:
        k = 0
        for i_zero in I_zero:
            skiped = 0
            while True:
                if i_non_zero is None or i_zero[k] != i_non_zero[k]:
                    Y[k][0, i_zero[k], 0] = 0.
                    k += 1
                    break
                else:
                    k += 1
                    skiped += 1
                    if skiped > d:
                        raise ValueError('Can not set zero items')
                    if k >= d:
                        k = 0
            if k >= d:
                k = 0

    return Y


def tensor_delta(n, i, v=1.):
    """Build TT-tensor that is zero everywhere except for a given multi-index.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        i (list, np.ndarray): the multi-index for nonzero element. It should
            be list or np.ndarray of the length "d".
        v (float): the value of the tensor at multi-index "i".

    Returns:
        list: TT-tensor.

    """
    d = len(n)
    s = abs(v) / v if abs(v) > 1.E-16 else v
    v = abs(v)**(1./d) if abs(v) > 1.E-16 else 1.
    Y = [np.zeros([1, k, 1]) for k in n]
    for j in range(d):
        Y[j][0, i[j], 0] = v
    Y[-1] *= s
    return Y


def tensor_poly(n, shift=0., power=2, scale=1.):
    """Build TT-tensor that is polynomial like scale * (index + shift)^power.

    Args:
        n (list, np.ndarray): shape of the tensor. It should be list or
            np.ndarray of the length "d", where "d" is a number of dimensions.
        shift (float, list, np.ndarray): the shift value. It should be list or
            np.ndarray of the length "d". It may be also float value.
        power (int): the power of polynomial.
        scale (float): the scale.

    Returns:
        list: TT-tensor.

    """
    d = len(n)
    Y = []

    shift = teneva.grid_prep_opt(shift, d)

    def _get(m, j):
        return (m + shift[j])**power

    for j, k in enumerate(n):
        if j == 0:
            G = np.zeros((1, k, 2))
            for m in range(k):
                G[0, m, :] = np.array([1., _get(m, j)])

        if j > 0 and j < d-1:
            G = np.zeros((2, k, 2))
            for m in range(k):
                G[:, m, :] = np.array([
                    [1., _get(m, j)],
                    [0., 1.],
                ])

        if j == d - 1:
            G = np.zeros((2, k, 1))
            for m in range(k):
                G[:, m, 0] = np.array([_get(m, j) * scale, scale])

        Y.append(G)

    return Y


def tensor_rand(n, r, f=np.random.randn):
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

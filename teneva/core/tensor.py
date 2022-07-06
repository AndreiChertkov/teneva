"""Package teneva, module core.tensor: basic operations with TT-tensors.

This module contains the basic operations and utilities for TT-tensors,
including "add", "mul", "norm", "mean", etc.

"""
import numba as nb
import numpy as np


import teneva


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

    return norm(sub(Y1, Y2)) / norm(Y2)


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
        Y1 = teneva.tensor_const(shape(Y2), Y1)
    elif _is_num(Y2):
        Y2 = teneva.tensor_const(shape(Y1), Y2)

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
            Y = teneva.truncate(Y, e)
    return teneva.truncate(Y, e, r) if not _is_num(Y) else Y


def copy(Y):
    """Return a copy of the given TT-tensor.

    Args:
        Y (int, float, list): TT-tensor (or it may be int/float).

    Returns:
        list: TT-tensor, which is a copy of the given TT-tensor. If Y is a
        number, then result will be the same number. If Y is np.ndarray, then
        the result will the corresponding copy in numpy format.

    """
    if _is_num(Y):
        return Y
    elif isinstance(Y, np.ndarray):
        return Y.copy()
    else:
        return [G.copy() for G in Y]


def erank(Y):
    """Compute effective TT-rank of the given TT-tensor.

    Effective TT-rank r of a TT-tensor Y with shape [n_1, n_2, ..., n_d] and
    TT-ranks r_0, r_1, ..., r_d (r_0 = r_d = 1) is a solution of equation
    n_1 r + \sum_{\alpha=2}^{d-1} n_\alpha r^2 + n_d r =
    \sum_{\alpha=1}^{d} n_\alpha r_{\alpha-1} r_{\alpha}.

    The representation with a constant TT-rank r (r_0 = 1, r_1 = r_2 = ... =
    r_{d-1} = r, r_d = 1) yields the same total number of parameters as in the
    original decomposition of the tensor Y.

    Args:
        Y (list): TT-tensor.

    Returns:
        float: effective TT-rank.

    """
    d, n, r = len(Y), shape(Y), ranks(Y)
    sz = np.dot(n * r[0:d], r[1:])
    b = r[0] * n[0] + n[d-1] * r[d]
    a = np.sum(n[1:d-1])
    return (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)


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
    return Z[0, ..., 0]


def get(Y, k, to_item=True):
    """Compute the element of the TT-tensor.

    Args:
        Y (list): TT-tensor.
        k (list, np.ndarray): the multi-index for the tensor.
        to_item (bool): flag, if True, then the float will be returned, and if
            it is False, then the 1-element array will be returned. This option
            is usefull in some special cases, then Y is a subset of TT-cores.

    Returns:
        float: the element of the TT-tensor.

    """
    k = np.asanyarray(k, dtype=int)
    Q = Y[0][0, k[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, k[i], :])
    return Q[0] if to_item else Q


def getter(Y, compile=True):
    """Build the fast getter function to compute the element of the TT-tensor.

    Args:
        Y (list): TT-tensor.
        compile (bool): flag, if True, then the getter will be called one time
            with a random multi-index to compile its code.

    Returns:
        function: the function that computes the element of the TT-tensor. It
        has one argument "k" (list) which is the multi-index for the tensor.

    Note:
        Note that the gain from using this getter instead of the base function
        "get" appears only in the case of many requests for calculating the
        tensor value (otherwise, the time spent on compiling the getter may
        turn out to be significant).

    """
    Y_nb = tuple([np.array(G, order='C') for G in Y])

    @nb.jit(nopython=True)
    def get(k):
        Q = Y_nb[0]
        y = [Q[0, k[0], r2] for r2 in range(Q.shape[2])]
        for i in range(1, len(Y_nb)):
            Q = Y_nb[i]
            R = np.zeros(Q.shape[2])
            for r1 in range(Q.shape[0]):
                for r2 in range(Q.shape[2]):
                    R[r2] += y[r1] * Q[r1, k[i], r2]
            y = list(R)
        return y[0]

    if compile:
        y = get(np.zeros(len(Y), dtype=int))

    return get


def mean(Y, P=None, norm=True):
    """Compute mean value of the TT-tensor with the given inputs probability.

    Args:
        Y (list): TT-tensor.
        P (list): optional probabilities for each dimension. It is the list of
            length d (number of tensor dimensions), where each element is also
            a list with length equals to the number of tensor elements along the
            related dimension. Hence, P[m][i] relates to the probability of the
            i-th input for the m-th mode (dimension).
        norm (bool): service (inner) flag, should be True.

    Returns:
        float: the mean value of the TT-tensor.

    """
    R = np.ones((1, 1))
    for i in range(len(Y)):
        k = Y[i].shape[1]
        if P is not None:
            Q = P[i][:k]
        else:
            Q = np.ones(k) / k if norm else np.ones(k)
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


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


def mul_scalar(Y1, Y2):
    """Compute scalar product for Y1 and Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        list: TT-tensor, which represents the scalar product of Y1 and Y2.

    """
    v = None
    for i, (G1, G2) in enumerate(zip(Y1, Y2)):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)
        v = G.copy() if i == 0 else v @ G
    return v.item()


def norm(Y):
    """Compute Frobenius norm of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        float: Frobenius norm of the TT-tensor.

    """
    v = mul_scalar(Y, Y)
    return np.sqrt(v) if v > 0 else 0.


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


def ranks(Y):
    """Return the TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: TT-ranks in form of the 1D array of ints of the length d+1,
        where "d" is a number of tensor dimensions (the first and last elements
        are equal 1).

    """
    return np.array([1] + [G.shape[2] for G in Y], dtype=int)


def shape(Y):
    """Return the shape of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: shape of the tensor in form of the 1D array of ints of the
        length "d", where "d" is a number of tensor dimensions.

    """
    return np.array([G.shape[1] for G in Y], dtype=int)


def show(Y):
    """Display (print) mode sizes and TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    """
    n, r = shape(Y), ranks(Y)
    l = max(int(np.ceil(np.log10(max(r)+1))) + 1, 3)
    form_str = '{:^' + str(l) + '}'

    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(k) for k in n])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in n])
    s3 = ''.join([form_str.format(q) for q in r])

    print(f'{s1}\n{s2}\n{s3}\n')


def size(Y):
    """Return the size (number of parameters) of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        int: total number of parameters in the TT-representation (it is a sum
        of sizes of all TT-cores).

    """
    return np.sum([G.size for G in Y])


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
        Y2 = teneva.tensor_const(shape(Y1), -1.*Y2)
    else:
        Y2 = copy(Y2)
        Y2[0] *= -1.

    return add(Y1, Y2)


def sum(Y):
    """Compute sum of all tensor elements.

    Args:
        Y (list): TT-tensor.

    Returns:
        float: the sum of all tensor elements.

    """
    return mean(Y, norm=False)


def _is_num(A):
    return isinstance(A, (int, float))

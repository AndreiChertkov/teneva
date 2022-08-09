"""Package teneva, module core.tensor: basic operations with TT-tensors.

This module contains the basic operations and utilities for TT-tensors,
including "add", "get", "mul", etc.

"""
import numba as nb
import numpy as np


from .utils import _is_num
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
        Y1 = teneva.tensor_const(teneva.shape(Y2), Y1)
    elif _is_num(Y2):
        Y2 = teneva.tensor_const(teneva.shape(Y1), Y2)

    n, r1, r2, Y = teneva.shape(Y1), teneva.ranks(Y1), teneva.ranks(Y2), []
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


def get(Y, k, to_item=True):
    """Compute the element (or elements) of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (list, np.ndarray): the multi-index for the tensor or a batch of
            multi-indices in the form of a list of lists or array of the shape
            [samples, d].
        to_item (bool): flag, if True, then the float will be returned, and if
            it is False, then the 1-element array will be returned. This option
            is usefull in some special cases, then Y is a subset of TT-cores.

    Returns:
        float: the element of the TT-tensor. If argument "k" is a batch of
        multi-indices, then array of length "samples" will be returned.

    """
    k = np.asanyarray(k, dtype=int)
    if len(k.shape) == 2:
        return get_many(Y, k)

    Q = Y[0][0, k[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, k[i], :])

    return Q[0] if to_item else Q


def get_many(Y, K):
    """Compute the elements of the TT-tensor on many indices.

    Args:
        Y (list): d-dimensional TT-tensor.
        K (list of list, np.ndarray): the multi-indices for the tensor in the
            form of a list of lists or array of the shape [samples, d].

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices "K" (array
        of length "samples").

    """
    K = np.asanyarray(K, dtype=int)
    Q = Y[0][0, K[:, 0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('kq,qkp->kp', Q, Y[i][:, K[:, i], :])
    return Q[:, 0]


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
            v, p = stab(v, p)

    v = v.item()

    return (v, p) if use_stab else v


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


def show(Y):
    """Display (print) mode sizes and TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    """
    n, r = teneva.shape(Y), teneva.ranks(Y)
    l = max(int(np.ceil(np.log10(max(r)+1))) + 1, 3)
    form_str = '{:^' + str(l) + '}'

    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(k) for k in n])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in n])
    s3 = ''.join([form_str.format(q) for q in r])

    print(f'{s1}\n{s2}\n{s3}\n')


def stab(G, p0=0, thr=1.E-100):
    # TODO: add docstring and add into demo
    v_max = np.max(np.abs(G))

    if v_max <= thr:
        return G, p0

    p = int(np.floor(np.log2(v_max)))
    Q = G / 2**p

    return Q, p0 + p


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
        Y2 = teneva.tensor_const(teneva.shape(Y1), -1.*Y2)
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
    return teneva.mean(Y, norm=False)


def TT_core_to_QTT(core, e=0, r_max=int(1e12)):
    # TODO: add docstring and add into demo
    r1, n, r2 = core.shape
    d = int(np.log2(n))
    assert 2**d == n

    A = core.reshape(-1, r2, order='F')
    A, V0 = teneva.matrix_svd(A, e=e, r=r_max)

    res = []
    for i in range(d-1):
        As = A.shape[0] // 2
        r = A.shape[1]
        A = np.hstack([A[:As], A[As:]])
        A, V = teneva.matrix_svd(A, e=e, r=r_max)
        res.append(V.reshape(-1, 2, r, order='C'))


    res.append(A.reshape(r1, 2, -1, order='F'))
    res[0] = np.einsum("ijk,kl", res[0], V0)

    return res[::-1]

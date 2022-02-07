"""Package teneva, module core.cross: cross approximation in the TT-format.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally by multidimensional cross approximation
method in the TT-format (TT-CAM).

"""
import numpy as np


from .grid import ind2str
from .grid import str2ind
from .maxvol import maxvol
from .maxvol import maxvol_rect
from .tensor import accuracy
from .tensor import copy
from .tensor import truncate


def cross(f, Y0, e, evals=None, nswp=10, dr_min=1, dr_max=2, cache=None,
    info={}):
    """Compute the TT-approximation for implicit tensor given functionally.

    Args:
        f (function): function f(I) which computes tensor elements for a given
            set of multi-indices I, where I is 2D np.ndarray of the shape
            [samples x dimensions]. The function should return 1D np.ndarray of
            the length equals to samples.
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        e (float): accuracy (> 0) for truncation of the final result and
            algorithm convergence criterion. If between iterations the relative
            rate of solution change is less than this value, then the operation
            of the algorithm will be interrupted.
        evals (int): an optionally set limit on the maximum number of requests
            to the objective function. If specified, then the total number of
            requests will not exceed this value. Note that the actual number of
            requests may be less, since the values are requested in batches.
        nswp (int): maximum number of iterations (sweeps) of the algorithm.
        dr_min (int): minimum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            maxvol_rect (see teneva.core.maxvol.maxvol_rect for more details).
            Note that "dr_min" should be no bigger than "dr_max".
        dr_max (int): maximum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            maxvol_rect (see teneva.core.maxvol.maxvol_rect for more details).
            Note that "dr_max" should be no less than "dr_min".
        cache (dict): an optionally set dictionary, which will be filled with
            requested function values. Since the algorithm sometimes requests
            the same tensor indices, the use of such a cache in some cases can
            speed up the operation of the algorithm if the time to find a value
            in the cache is less than the time to calculate the function.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
            At the end of the function work, it will contain parameters:
            "k_evals" - total number of requests to target function "f";
            "k_cache" - total number of requests to cache;
            "e" - the final value of the convergence criterion;
            "nswp" - the real number of performed iterations (sweeps);
            "stop" - stop type of the algorithm (see note below).

    Returns:
        list: TT-Tensor which approximates the implicit tensor.

    Note:
        Note that the end of the algorithm operation occurs when one of the
        three criteria is reached: 1) the maximum number of iterations ("nswp")
        performed; 2) the maximum allowable number of the objective function
        calls ("evals") has been done (more precisely, if the next request will
        result in exceeding this value, then algorithm will not perform this
        new request); 3) the convergence criterion ("e") is reached. The
        corresponding stop type ("nswp", "evals" or "e") will be written into
        the item "stop" of the "info" dictionary.

    """
    Y = copy(Y0)
    d = len(Y)

    Il = [np.empty((1, 0), dtype=int)] + [None for i in range(d)]
    Ir = [None for i in range(d)] + [np.empty((1, 0), dtype=int)]
    Ig = [_reshape(np.arange(G.shape[1], dtype=int), (-1, 1)) for G in Y]

    R = np.ones((1, 1))
    for i in range(d-1):
        G = np.tensordot(R, Y[i], 1)
        Y[i], Il[i+1], R = cross_prep_l2r(G, Ig[i], Il[i])

    R = np.ones((1, 1))
    for i in range(d-1, 0, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], Ir[i], R = cross_prep_r2l(G, Ig[i], Ir[i+1])

    info['k_evals'] = 0
    info['k_cache'] = 0
    info['e'] = -1.
    info['nswp'] = 0
    info['stop'] = None

    def func(I):
        if cache is None:
            if evals is not None and info['k_evals'] + len(I) > evals:
                return None
            info['k_evals'] += len(I)
            return f(I)

        I_new = np.array([i for i in I if ind2str(i) not in cache])
        if len(I_new):
            if evals is not None and info['k_evals'] + len(I_new) > evals:
                return None
            Y_new = f(I_new)
            for k, i in enumerate(I_new):
                cache[ind2str(i)] = Y_new[k]

        info['k_cache'] += len(I) - len(I_new)
        info['k_evals'] += len(I_new)

        Y = np.array([cache[ind2str(i)] for i in I])
        return Y

    Yold = None
    for swp in range(nswp):
        R = np.ones((1, 1))
        for i in range(d):
            G = np.tensordot(R, Y[i], 1)
            y = func(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            if y is None:
                Y[i] = G
                info['stop'] = 'evals'
                return truncate(Y, e)
            Y[i], Il[i+1], R = cross_build_l2r(
                *G.shape, y, Ig[i], Il[i], dr_min, dr_max)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            G = np.tensordot(Y[i], R, 1)
            y = func(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            if y is None:
                Y[i] = G
                info['stop'] = 'evals'
                return truncate(Y, e)
            Y[i], Ir[i], R = cross_build_r2l(
                *G.shape, y, Ig[i], Ir[i+1], dr_min, dr_max)
        Y[0] = np.tensordot(R, Y[0], 1)

        info['nswp'] = swp + 1

        if Yold is not None:
            eps = accuracy(Y, Yold)
            info['e'] = eps

            if eps <= e:
                info['stop'] = 'e'
                return truncate(Y, e)

        Yold = copy(Y)

    info['stop'] = 'nswp'
    return truncate(Y, e)


def cross_build_l2r(r1, n, r2, y, Ig, I, dr_min, dr_max):
    G = _reshape(y, (-1, r2))
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, dr_min, dr_max)
    G = _reshape(B, (r1, n, -1))
    J = cross_index_stack_l2r(r1, Ig, I)[Ind, :]
    R = Q[Ind, :] @ np.diag(s) @ V
    return G, J, R


def cross_build_r2l(r1, n, r2, y, Ig, I, dr_min, dr_max):
    G = _reshape(y, (r1, -1)).T
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, dr_min, dr_max)
    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(r2, Ig, I)[Ind, :]
    R = (Q[Ind, :] @ np.diag(s) @ V).T
    return G, J, R


def cross_cache2data(cache):
    I = np.array([str2ind(s) for s in cache.keys()], dtype=int)
    Y = np.array([y for y in cache.values()])
    return I, Y


def cross_index_merge(i1, i2, i3):
    r1 = i1.shape[0] or 1
    r2 = i2.shape[0]
    r3 = i3.shape[0] or 1
    w1 = _kron(_ones(r3 * r2), i1)
    w2 = np.empty((w1.shape[0], 0))
    if i3.size and r2:
        w2 = _kron(i3, _ones(r1 * r2))
    w3 = _kron(_kron(_ones(r3), i2), _ones(r1))
    return np.hstack((w1, w3, w2))


def cross_index_stack_l2r(r, Ig, I):
    n = Ig.size
    J = _kron(Ig, _ones(r))
    if I.size:
        J = np.hstack((_kron(_ones(n), I), J))
    return _reshape(J, (n * r, -1))


def cross_index_stack_r2l(r, Ig, I):
    n = Ig.size
    J = _kron(_ones(r), Ig)
    if I.size:
        J = np.hstack((J, _kron(I, _ones(n))))
    return _reshape(J, (n * r, -1))


def cross_prep_l2r(G, Ig, I=None):
    r1, n, r2 = G.shape
    G = _reshape(G, (r1 * n, r2))

    Q, R = _qr(G)
    Ind, B = _maxvol(Q)

    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_l2r(r1, Ig, I)[Ind, :]
    R = Q[Ind, :] @ R
    return G, J, R


def cross_prep_r2l(G, Ig, I=None):
    r1, n, r2 = G.shape
    G = _reshape(G, (r1, n * r2)).T

    Q, R = _qr(G)
    Ind, B = _maxvol(Q)

    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(r2, Ig, I)[Ind, :]
    R = (Q[Ind, :] @ R).T
    return G, J, R


def _kron(a, b):
    return np.kron(a, b)


def _maxvol(a):
    return maxvol(a)


def _maxvol_rect(a, dr_min=1, dr_max=2, tau=1.1):
    N, u = a.shape

    if N <= u:
        I = np.arange(N, dtype=int)
        B = np.eye(N, dtype=a.dtype)
    else:
        I, B = maxvol_rect(a, tau, dr_min, dr_max)

    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _qr(a):
    return np.linalg.qr(a)


def _reshape(a, shape):
    return np.reshape(a, shape, order='F')


def _svd(a, full_matrices=False):
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except:
        b = a + 1.E-14 * np.max(np.abs(a)) * np.random.randn(*a.shape)
        return np.linalg.svd(b, full_matrices=full_matrices)

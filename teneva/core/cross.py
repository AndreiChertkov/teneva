"""Package teneva, module core.cross: cross approximation in the TT-format.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally by multidimensional cross approximation
method in the TT-format (TT-CAM).

"""
import numpy as np


from .grid import ind_to_str
from .grid import str_to_ind
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
    info['evals_max'] = evals
    info['k_evals'] = 0
    info['k_cache'] = 0
    info['e'] = -1.
    info['nswp'] = 0
    info['stop'] = None

    Y = copy(Y0)
    d = len(Y)

    Il = [np.empty((1, 0), dtype=int)] + [None for i in range(d)]
    Ir = [None for i in range(d)] + [np.empty((1, 0), dtype=int)]
    Ig = [_reshape(np.arange(G.shape[1], dtype=int), (-1, 1)) for G in Y]

    R = np.ones((1, 1))
    for i in range(d):
        G = np.tensordot(R, Y[i], 1)
        Y[i], R, Il[i+1] = _iter(G, Ig[i], Il[i], l2r=True)
    Y[d-1] = np.tensordot(Y[d-1], R, 1)

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ir[i] = _iter(G, Ig[i], Ir[i+1], l2r=False)
    Y[0] = np.tensordot(R, Y[0], 1)

    Yold = None
    for swp in range(nswp):
        R = np.ones((1, 1))
        for i in range(d):
            G = np.tensordot(R, Y[i], 1)

            I_curr = _index_merge(Il[i], Ig[i], Ir[i+1])
            y = _func(f, I_curr, cache, info)

            if y is None:
                Y[i] = G
                info['stop'] = 'evals'
                return truncate(Y, e)

            G = _reshape(y, G.shape)
            Y[i], R, Il[i+1] = _iter(G, Ig[i], Il[i], dr_min, dr_max, l2r=True)

        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            G = np.tensordot(Y[i], R, 1)

            I_curr = _index_merge(Il[i], Ig[i], Ir[i+1])
            y = _func(f, I_curr, cache, info)

            if y is None:
                Y[i] = G
                info['stop'] = 'evals'
                return truncate(Y, e)

            G = _reshape(y, G.shape)
            Y[i], R, Ir[i] = _iter(G, Ig[i], Ir[i+1], dr_min, dr_max, l2r=False)

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


def cross_cache2data(cache):
    I = np.array([str_to_ind(s) for s in cache.keys()], dtype=int)
    Y = np.array([y for y in cache.values()])
    return I, Y


def _func(f, I, cache=None, info={}):
    if cache is None:
        if info['evals_max'] and info['k_evals'] + len(I) > info['evals_max']:
            return None
        info['k_evals'] += len(I)
        return f(I)

    I_new = np.array([i for i in I if ind_to_str(i) not in cache])
    if len(I_new):
        if info['evals_max'] and info['k_evals']+len(I_new) > info['evals_max']:
            return None
        Y_new = f(I_new)
        for k, i in enumerate(I_new):
            cache[ind_to_str(i)] = Y_new[k]

    info['k_cache'] += len(I) - len(I_new)
    info['k_evals'] += len(I_new)

    Y = np.array([cache[ind_to_str(i)] for i in I])
    return Y


def _index_merge(i1, i2, i3):
    r1 = i1.shape[0] or 1
    r2 = i2.shape[0]
    r3 = i3.shape[0] or 1

    w1 = np.kron(_ones(r3 * r2), i1)

    w2 = np.kron(np.kron(_ones(r3), i2), _ones(r1))

    if i3.size and r2:
        w3 = np.kron(i3, _ones(r1 * r2))
    else:
        w3 = np.empty((w1.shape[0], 0))

    return np.hstack((w1, w2, w3))


def _index_stack(r, Ig, I, l2r=True):
    n = Ig.size
    I_new = np.kron(Ig, _ones(r)) if l2r else np.kron(_ones(r), Ig)
    if I.size and l2r:
        I_new = np.hstack((np.kron(_ones(n), I), I_new))
    if I.size and not l2r:
        I_new = np.hstack((I_new, np.kron(I, _ones(n))))
    return _reshape(I_new, (n * r, -1))


def _iter(G, Ig, I, dr_min=0, dr_max=0, l2r=True):
    """Find important rows in unfolding matrix from the given TT-core.

    Args:
        G (np.ndarray): TT-core of the shape r1 x n x r2 that relates to the
            k-th mode of the tensor (k = 1, 2, ..., d).
        Ig (np.ndarray): full range of grid indices for the k-th mode, i.e.
            it is just [0, 1, ..., n-1].
        I (np.ndarray): list of selected multiindices for the previous (k-1)-th
            mode. It has the shape r1 x (k-1), and each its row
            (i_1, ..., i_{k-1}) corresponds to the one selected row in the
            undolding matrix. For the 0-th mode it is should be empty array.
        dr_min (int): minimum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            maxvol_rect (see teneva.core.maxvol.maxvol_rect for more details).
            Note that "dr_min" should be no bigger than "dr_max".
        dr_max (int): maximum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            maxvol_rect (see teneva.core.maxvol.maxvol_rect for more details).
            Note that "dr_max" should be no less than "dr_min".
        l2r (bool): ...

    Returns:
        np.ndarray: updated TT-core.
        np.ndarray: list of selected multiindices for the current k-th mode.
            It has the shape r2 x k, and each its row (i_1, ..., i_k)
            corresponds to the one selected row in the undolding matrix.
        np.ndarray: the matrix of the shape r2 x r2 by which the next TT-core
            should be multiplied to keep the tensor unchanged.

    """
    r1, n, r2 = G.shape
    G = _reshape(G, (r1 * n, r2)) if l2r else _reshape(G, (r1, n * r2)).T

    if dr_max > 0:
        Q, s, V = np.linalg.svd(G, full_matrices=False)
        R = np.diag(s) @ V

        n_rows, n_cols = Q.shape
        if n_rows <= n_cols:
            ind, B = np.arange(n_rows, dtype=int), np.eye(n_rows, dtype=a.dtype)
        else:
            tau = 1.1
            ind, B = maxvol_rect(Q, 1.1, dr_min, dr_max)
    else:
        Q, R = np.linalg.qr(G)

        ind, B = maxvol(Q)

    G = B if l2r else B.T
    G = _reshape(G, (r1, n, -1)) if l2r else _reshape(G, (-1, n, r2))

    R = Q[ind, :] @ R
    R = R if l2r else R.T

    I_new = _index_stack(r1 if l2r else r2, Ig, I, l2r)[ind, :]

    return G, R, I_new


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _reshape(a, shape):
    return np.reshape(a, shape, order='F')

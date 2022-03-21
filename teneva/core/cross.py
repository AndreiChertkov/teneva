"""Package teneva, module core.cross: cross approximation in the TT-format.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally by multidimensional cross approximation
method in the TT-format (TT-CROSS).

"""
import numpy as np


from .maxvol import maxvol
from .maxvol import maxvol_rect
from .tensor import accuracy
from .tensor import copy
from .tensor import shape


def cross(f, Y0, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=2, tau0=1.05, k0=100, info={}, cache=None):
    """Compute the TT-approximation for implicit tensor given functionally.

    This function computes the TT-approximation for implicit tensor given
    functionally by multidimensional cross approximation method in the
    TT-format (TT-CROSS method).

    Args:
        f (function): function f(I) which computes tensor elements for the given
            set of multi-indices I, where I is a 2D np.ndarray of the shape
            [samples, dimensions]. The function should return 1D np.ndarray of
            the length equals to "samples" (the values of the target function
            for all provided samples).
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
            It may be, fo example, random TT-tensor, which can be built by the
            "rand" function from teneva: "Y0 = teneva.rand(n, r)", where "n"
            is a size of tensor modes (e.g., "n = [5, 6, 7, 8, 9]" for the
            5-dimensional tensor) and "r" is a TT-rank of this TT-tensor (e.g.,
            "r = 3").
        m (int): optional limit on the maximum number of requests to the
            objective function (> 0). If specified, then the total number of
            requests will not exceed this value. Note that the actual number of
            requests may be less, since the values are requested in batches.
        e (float): optional algorithm convergence criterion (> 0). If between
            iterations the relative rate of solution change is less than this
            value, then the operation of the algorithm will be interrupted.
        nswp (int): optional maximum number of iterations (sweeps) of the
            algorithm (>= 0). One sweep corresponds to a complete pass of all
            tensor TT-cores from left to right and then from right to left. If
            nswp = 0, then only "maxvol-preiteration" will be performed.
        tau (float): accuracy parameter (>= 1) for the algorithm "maxvol_rect"
            (see "maxvol_rect" function for more details).
        dr_min (int): minimum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            "maxvol_rect" (see "maxvol_rect" function for more details). Note
            that "dr_min" should be no bigger than "dr_max".
        dr_max (int): maximum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            "maxvol_rect" (see "maxvol_rect" function for more details). Note
            that "dr_max" should be no less than "dr_min".
        tau0 (float): accuracy parameter (>= 1) for the algorithm "maxvol" (see
            "maxvol" function for more details). It will be used while maxvol
            preiterations and while the calls of "maxvol" function from the
            "maxvol_rect" algorithm.
        k0 (int): maximum number of maxvol iterations (>= 1; see "maxvol"
            function for more details). It will be used while maxvol
            preiterations and while the calls of "maxvol" function from the
            "maxvol_rect" algorithm.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
            At the end of the function work, it will contain parameters: "m" -
            total number of requests to the target function; "e" - the final
            value of the convergence criterion; "nswp" - the real number of
            performed iterations (sweeps); "m_cache" - total number of requests
            to the cache; "stop" - stop type of the algorithm (see note below).
        cache (dict): an optionally set dictionary, which will be filled with
            requested function values. Since the algorithm sometimes requests
            the same tensor indices, the use of such a cache may speed up the
            operation of the algorithm if the time to find a value in the cache
            is less than the time to calculate the function.

    Returns:
        list: TT-Tensor which approximates the implicit tensor.

    Note:
        Note that the end of the algorithm operation occurs when one of the
        following criteria is reached (at list one of the arguments m / e /
        nswp should be set): 1) the maximum allowable number of the objective
        function calls ("m") has been done (more precisely, if the next request
        will result in exceeding this value, then algorithm will not perform
        this new request); 2) the convergence criterion ("e") is reached; 3)
        the maximum number of iterations ("nswp") is performed; 4) the
        algorithm is already converged (all requested values are in the cache
        already). The related stop type ("m", "e", "nswp" or "conv") will be
        written into the item "stop" of the "info" dictionary.

        The resulting TT-tensor usually has overestimated ranks, so you should
        truncate the result. Use for this "Y = truncate(Y, e)" (e.g.,
        "e = 1.E-8") after the call of this function.

    """
    if m is None and e is None and nswp is None:
        raise ValueError('One of the arguments m / e / nswp should be set')

    info['e'] = -1.
    info['m'] = 0
    info['m_cache'] = 0
    info['m_max'] = int(m) if m else None
    info['nswp'] = 0
    info['stop'] = None

    Y = copy(Y0)
    d = len(Y)
    n = shape(Y)

    Ig = [_reshape(np.arange(k, dtype=int), (-1, 1)) for k in n] # Grid indices
    Ir = [None for i in range(d+1)]                              # Row  indices
    Ic = [None for i in range(d+1)]                              # Col. indices

    R = np.ones((1, 1))
    for i in range(d):
        G = np.tensordot(R, Y[i], 1)
        Y[i], R, Ir[i+1] = _iter(G, Ig[i], Ir[i], tau0=tau0, k0=k0, l2r=True)
    Y[d-1] = np.tensordot(Y[d-1], R, 1)

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter(G, Ig[i], Ic[i+1], tau0=tau0, k0=k0, l2r=False)
    Y[0] = np.tensordot(R, Y[0], 1)

    while True:
        if nswp is not None and info['nswp'] >= nswp:
            info['stop'] = 'nswp'
            return Y

        if info['m_cache'] > 5 * info['m']:
            info['stop'] = 'conv'
            return Y

        Yold = copy(Y)

        R = np.ones((1, 1))
        for i in range(d):
            Z = _func(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                info['stop'] = 'm'
                Y[i] = np.tensordot(R, Y[i], 1)
                return Y
            Y[i], R, Ir[i+1] = _iter(Z, Ig[i], Ir[i],
                tau, dr_min, dr_max, tau0, k0, l2r=True)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            Z = _func(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                info['stop'] = 'm'
                Y[i] = np.tensordot(Y[i], R, 1)
                return Y
            Y[i], R, Ic[i] = _iter(Z, Ig[i], Ic[i+1],
                tau, dr_min, dr_max, tau0, k0, l2r=False)
        Y[0] = np.tensordot(R, Y[0], 1)

        info['nswp'] += 1

        info['e'] = accuracy(Y, Yold)
        if e is not None and info['e'] <= e:
            info['stop'] = 'e'
            return Y


def _func(f, Ig, Ir, Ic, info, cache=None):
    n = Ig.shape[0]
    r1 = Ir.shape[0] if Ir is not None else 1
    r2 = Ic.shape[0] if Ic is not None else 1

    I = np.kron(np.kron(_ones(r2), Ig), _ones(r1))
    if Ir is not None:
        Ir_ = np.kron(_ones(n * r2), Ir)
        I = np.hstack((Ir_, I))
    if Ic is not None:
        Ic_ = np.kron(Ic, _ones(r1 * n))
        I = np.hstack((I, Ic_))

    y = _func_eval(f, I, info, cache)
    if y is not None:
        return _reshape(y, (r1, n, r2))


def _func_eval(f, I, info, cache=None):
    if cache is None:
        if info['m_max'] is not None and info['m'] + len(I) > info['m_max']:
            return None
        info['m'] += len(I)
        return f(I)

    I_new = np.array([i for i in I if tuple(i) not in cache])
    if len(I_new):
        if info['m_max'] is not None and info['m'] + len(I_new) > info['m_max']:
            return None
        Y_new = f(I_new)
        for k, i in enumerate(I_new):
            cache[tuple(i)] = Y_new[k]

    info['m'] += len(I_new)
    info['m_cache'] += len(I) - len(I_new)

    return np.array([cache[tuple(i)] for i in I])


def _iter(Z, Ig, I, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100, l2r=True):
    r1, n, r2 = Z.shape
    Z = _reshape(Z, (r1 * n, r2)) if l2r else _reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)
    ind, B = _maxvol(Q, tau, dr_min, dr_max, tau0, k0)

    G = B if l2r else B.T
    G = _reshape(G, (r1, n, -1)) if l2r else _reshape(G, (-1, n, r2))

    R = Q[ind, :] @ R
    R = R if l2r else R.T

    I_new = np.kron(Ig, _ones(r1)) if l2r else np.kron(_ones(r2), Ig)
    if I is not None:
        I_old = np.kron(_ones(n), I) if l2r else np.kron(I, _ones(n))
        I_new = np.hstack((I_old, I_new)) if l2r else np.hstack((I_new, I_old))
    I_new = I_new[ind, :]

    return G, R, I_new


def _maxvol(A, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100):
    n, r = A.shape
    dr_max = min(dr_max, n - r)
    dr_min = min(dr_min, dr_max)

    if n <= r:
        I = np.arange(n, dtype=int)
        B = np.eye(n, dtype=float)
    elif dr_max == 0:
        I, B = maxvol(A, tau0, k0)
    else:
        I, B = maxvol_rect(A, tau, dr_min, dr_max, tau0, k0)

    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _reshape(A, n):
    return np.reshape(A, n, order='F')

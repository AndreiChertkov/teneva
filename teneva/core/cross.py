"""Package teneva, module core.cross: cross approximation in the TT-format.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally by multidimensional cross approximation
method in the TT-format (TT-CROSS).

"""
import numpy as np
import teneva
from time import perf_counter as tpc


def cross(f, Y0, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=1, tau0=1.05, k0=100, info={}, cache=None, I_vld=None, Y_vld=None, e_vld=None, log=False, func=None):
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
            "r = 1").
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
            that "dr_max" should be no less than "dr_min". If "dr_max = 0",
            then basic maxvol algorithm will be used (rank will be constant).
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
            value of the convergence criterion; "e_vld" - the final error on the
            validation dataset; "nswp" - the real number of performed
            iterations (sweeps); "m_cache" - total number of requests to the
            cache; "stop" - stop type of the algorithm (see note below).
        cache (dict): an optionally set dictionary, which will be filled with
            requested function values. Since the algorithm sometimes requests
            the same tensor indices, the use of such a cache may speed up the
            operation of the algorithm if the time to find a value in the cache
            is less than the time to calculate the target function.
        I_vld (np.ndarray): optional multi-indices for items of validation
            dataset in the form of array of the shape [samples, d].
        Y_vld (np.ndarray): optional values for items related to I_vld of
            validation dataset in the form of array of the shape [samples].
        e_vld (float): optional algorithm convergence criterion (> 0). If
            after sweep, the error on the validation dataset is less than this
            value, then the operation of the algorithm will be interrupted.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.
        func (function): if this function is set, then it will replace the inner
            function "_func", which deals with requests to the objective
            function "f". This argument is used for internal experiments.

    Returns:
        list: TT-Tensor which approximates the implicit tensor.

    Note:
        Note that the end of the algorithm operation occurs when one of the
        following criteria is reached (at list one of the arguments m / e /
        nswp / e_vld should be set): 1) the maximum allowable number of the
        objective function calls ("m") has been done (more precisely, if the
        next request will result in exceeding this value, then algorithm will
        not perform this new request); 2) the convergence criterion ("e") is
        reached; 3) the maximum number of iterations ("nswp") is performed; 4)
        the algorithm is already converged (all requested values are in the
        cache already) 5) the error on validation dataset "I_vld", "Y_vld" is
        less than "e_vld". The related stop type ("m", "e", "nswp", "conv" or
        "e_vld") will be written into the item "stop" of the "info" dictionary.

        The resulting TT-tensor usually has overestimated ranks, so you should
        truncate the result. Use for this "Y = truncate(Y, e)" (e.g.,
        "e = 1.E-8") after this function call.

    """
    if m is None and e is None and nswp is None:
        if I_vld is None or Y_vld is None:
            raise ValueError('One of arguments m/e/nswp should be set')
        elif e_vld is None:
            raise ValueError('One of arguments m/e/e_vld/nswp should be set')
    if e_vld is not None and (I_vld is None or Y_vld is None):
        raise ValueError('Validation dataset is not set')

    _time = tpc()

    info['r'] = teneva.erank(Y0)
    info['e'] = -1.
    info['e_vld'] = -1.
    info['m'] = 0
    info['m_cache'] = 0
    info['m_max'] = int(m) if m else None
    info['nswp'] = 0
    info['stop'] = None
    info['with_cache'] = cache is not None

    Y = teneva.copy(Y0)
    d = len(Y)
    n = teneva.shape(Y)

    Ig = [teneva._reshape(np.arange(k, dtype=int), (-1, 1)) for k in n] # Grid indices
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

    _info(Y, info, _time, I_vld, Y_vld, e_vld, log)

    while True:
        if nswp is not None and info['nswp'] >= nswp:
            _info(Y, info, _time, I_vld, Y_vld, e_vld, log, 'nswp')
            return Y

        if info['m_cache'] > 5 * info['m']:
            _info(Y, info, _time, I_vld, Y_vld, e_vld, log, 'conv')
            return Y

        Yold = teneva.copy(Y)

        R = np.ones((1, 1))
        for i in range(d):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                Y[i] = np.tensordot(R, Y[i], 1)
                _info(Y, info, _time, I_vld, Y_vld, e_vld, log, 'm')
                return Y
            Y[i], R, Ir[i+1] = _iter(Z, Ig[i], Ir[i],
                tau, dr_min, dr_max, tau0, k0, l2r=True)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                Y[i] = np.tensordot(Y[i], R, 1)
                _info(Y, info, _time, I_vld, Y_vld, e_vld, log, 'm')
                return Y
            Y[i], R, Ic[i] = _iter(Z, Ig[i], Ic[i+1],
                tau, dr_min, dr_max, tau0, k0, l2r=False)
        Y[0] = np.tensordot(R, Y[0], 1)

        info['nswp'] += 1

        info['e'] = teneva.accuracy(Y, Yold)
        if e is not None and info['e'] <= e:
            _info(Y, info, _time, I_vld, Y_vld, e_vld, log, 'e')
            return Y

        if _info(Y, info, _time, I_vld, Y_vld, e_vld, log):
            return Y


def _func(f, Ig, Ir, Ic, info, cache=None):
    n = Ig.shape[0]
    r1 = Ir.shape[0] if Ir is not None else 1
    r2 = Ic.shape[0] if Ic is not None else 1

    I = np.kron(np.kron(teneva._ones(r2), Ig), teneva._ones(r1))
    if Ir is not None:
        Ir_ = np.kron(teneva._ones(n * r2), Ir)
        I = np.hstack((Ir_, I))
    if Ic is not None:
        Ic_ = np.kron(Ic, teneva._ones(r1 * n))
        I = np.hstack((I, Ic_))

    y = _func_eval(f, I, info, cache)
    if y is not None:
        return teneva._reshape(y, (r1, n, r2))


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


def _info(Y, info, t, I_vld, Y_vld, e_vld, log=False, stop=None):
    info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, Y_vld)

    if stop is None and e_vld is not None and info['e_vld'] >= 0:
        if info['e_vld'] <= e_vld:
            stop = 'e_vld'
    info['stop'] = stop

    info['r'] = teneva.erank(Y)
    info['t'] = tpc() - t

    _log(Y, info, log)

    return info['stop']


def _iter(Z, Ig, I, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100, l2r=True):
    r1, n, r2 = Z.shape
    Z = teneva._reshape(Z, (r1 * n, r2)) if l2r else teneva._reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)
    ind, B = teneva._maxvol(Q, tau, dr_min, dr_max, tau0, k0)

    G = B if l2r else B.T
    G = teneva._reshape(G, (r1, n, -1)) if l2r else teneva._reshape(G, (-1, n, r2))

    R = Q[ind, :] @ R
    R = R if l2r else R.T

    I_new = np.kron(Ig, teneva._ones(r1)) if l2r else np.kron(teneva._ones(r2), Ig)
    if I is not None:
        I_old = np.kron(teneva._ones(n), I) if l2r else np.kron(I, teneva._ones(n))
        I_new = np.hstack((I_old, I_new)) if l2r else np.hstack((I_new, I_old))
    I_new = I_new[ind, :]

    return G, R, I_new


def _log(Y, info, log):
    if not log:
        return

    text = ''

    if info['nswp'] == 0:
        text += f'# pre | '
    else:
        text += f'# {info["nswp"]:-3d} | '

    text += f'time: {info["t"]:-10.3f} | '

    if info['with_cache']:
        text += f'evals: {info["m"]:-8.2e} (+ {info["m_cache"]:-8.2e}) | '
    else:
        text += f'evals: {info["m"]:-8.2e} | '

    text += f'rank: {info["r"]:-5.1f} | '

    if info['e_vld'] >= 0:
        text += f'err: {info["e_vld"]:-7.1e} | '

    if info['e'] >= 0:
        text += f'eps: {info["e"]:-7.1e} | '

    if info['stop']:
        text += f'stop: {info["stop"]} | '

    print(text)

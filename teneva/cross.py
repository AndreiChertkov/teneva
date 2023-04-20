"""Package teneva, module cross: construct TT-tensor, using TT-cross.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally  by the rank-adaptive multidimensional
cross approximation method in the TT-format (TT-cross).

"""
import numpy as np
import teneva
from time import perf_counter as tpc


def cross(f, Y0, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=1,
          tau0=1.05, k0=100, info={}, cache=None, I_vld=None, y_vld=None,
          e_vld=None, cb=None, func=None, log=False):
    """Compute the TT-approximation for implicit tensor given functionally.

    This function computes the TT-approximation for implicit tensor given
    functionally by the rank-adaptive multidimensional cross approximation
    method in the TT-format (TT-cross).

    Args:
        f (function): function f(I) which computes tensor elements for the
            given set of multi-indices I, where I is a 2D np.ndarray of the
            shape [samples, dimensions]. The function should return 1D
            np.ndarray of the length equals to samples, which relates to the
            values of the target function for all provided samples.
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
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
            that dr_min should be no bigger than dr_max.
        dr_max (int): maximum number of added rows in the process of adaptively
            increasing the TT-rank of the approximation using the algorithm
            "maxvol_rect" (see "maxvol_rect" function for more details). Note
            that dr_max should be no less than dr_min. If dr_max = 0,
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
            At the end of the function work, it will contain parameters: m -
            total number of requests to the target function; e - the final
            value of the convergence criterion; e_vld - the final error on the
            validation dataset; nswp - the real number of performed
            iterations (sweeps); m_cache - total number of requests to the
            cache; stop - stop type of the algorithm (see note below).
        cache (dict): an optionally set dictionary, which will be filled with
            requested function values. Since the algorithm sometimes requests
            the same tensor indices, the use of such a cache may speed up the
            operation of the algorithm if the time to find a value in the cache
            is less than the time to calculate the target function.
        I_vld (np.ndarray): optional multi-indices for items of validation
            dataset in the form of array of the shape [samples_vld, d], where
            samples_vld is a size of the validation dataset.
        y_vld (np.ndarray): optional values of the tensor for multi-indices
            I_vld of validation dataset in the form of array of the shape
            [samples_vld].
        e_vld (float): optional algorithm convergence criterion (> 0). If
            after sweep, the error on the validation dataset is less than this
            value, then the operation of the algorithm will be interrupted.
        cb (function): optional callback function. It will be called after
            every sweep and the accuracy check with the arguments: Y,
            info and opts, where Y is the current approximation (TT-tensor),
            info is the info dictionary and the dictionary opts contains fields
            Ir, Ic, cache and Yold. If the callback returns a true value, then
            the algorithm will be stopped (in the info dictionary, in this case,
            the stop type of the algorithm will be cb).
        func (function): if this function is set, then it will replace the inner
            function _func, which deals with requests to the objective
            function f. This argument is used for internal experiments.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-Tensor which approximates the implicit tensor.

    Note:
        Note that at list one of the arguments m / e / nswp / e_vld should be
        set by user. The end of the algorithm operation occurs when one of the
        following criteria is reached: 1) the maximum allowable number of the
        objective function calls (m) has been done (more precisely, if the
        next request will result in exceeding this value, then algorithm will
        not perform this new request); 2) the convergence criterion (e) is
        reached; 3) the maximum number of iterations (nswp) is performed; 4)
        the algorithm is already converged (all requested values are in the
        cache already); 5) the error on validation dataset I_vld, y_vld is
        less than e_vld; 6) the callback function returns true value. The
        related stop type (m, e, nswp, conv, e_vld or cb) will be written into
        the item stop of the info dictionary.

        The resulting TT-tensor usually has overestimated ranks, so you should
        truncate the result. Use for this Y = teneva.truncate(Y, e) (e.g.,
        e = 1.E-8) after this function call.

    """
    if m is None and e is None and nswp is None:
        if I_vld is None or y_vld is None:
            raise ValueError('One of arguments m/e/nswp should be set')
        elif e_vld is None:
            raise ValueError('One of arguments m/e/e_vld/nswp should be set')
    if e_vld is not None and (I_vld is None or y_vld is None):
        raise ValueError('Validation dataset is not set')

    _time = tpc()
    info.update({'r': teneva.erank(Y0), 'e': -1, 'e_vld': -1, 'nswp': 0,
        'stop': None, 'm': 0, 'm_cache': 0, 'm_max': int(m) if m else None,
        'with_cache': cache is not None})

    d = len(Y0)
    n = teneva.shape(Y0)
    Y = teneva.copy(Y0)

    Ig = [teneva._reshape(np.arange(k, dtype=int), (-1, 1)) for k in n]
    Ir = [None for i in range(d+1)]
    Ic = [None for i in range(d+1)]

    R = np.ones((1, 1))
    for i in range(d):
        G = np.tensordot(R, Y[i], 1)
        Y[i], R, Ir[i+1] = _iter(G, Ig[i], Ir[i], tau0=tau0, k0=k0, ltr=True)
    Y[d-1] = np.tensordot(Y[d-1], R, 1)

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter(G, Ig[i], Ic[i+1], tau0=tau0, k0=k0, ltr=False)
    Y[0] = np.tensordot(R, Y[0], 1)

    info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
    teneva._info_appr(info, _time, nswp, e, e_vld, log)

    while True:
        Yold = teneva.copy(Y)

        R = np.ones((1, 1))
        for i in range(d):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                Y[i] = np.tensordot(R, Y[i], 1)
                info['r'] = teneva.erank(Y)
                info['e'] = teneva.accuracy(Y, Yold)
                info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
                info['stop'] = 'm'
                teneva._info_appr(info, _time, nswp, e, e_vld, log)
                return Y
            Y[i], R, Ir[i+1] = _iter(Z, Ig[i], Ir[i],
                tau, dr_min, dr_max, tau0, k0, ltr=True)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                Y[i] = np.tensordot(Y[i], R, 1)
                info['r'] = teneva.erank(Y)
                info['e'] = teneva.accuracy(Y, Yold)
                info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
                info['stop'] = 'm'
                teneva._info_appr(info, _time, nswp, e, e_vld, log)
                return Y
            Y[i], R, Ic[i] = _iter(Z, Ig[i], Ic[i+1],
                tau, dr_min, dr_max, tau0, k0, ltr=False)
        Y[0] = np.tensordot(R, Y[0], 1)

        info['nswp'] += 1
        info['r'] = teneva.erank(Y)
        info['e'] = teneva.accuracy(Y, Yold)
        info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)

        if info['m_cache'] > 5 * info['m']:
            info['stop'] = 'conv'

        if cb:
            opts = {'Yold': Yold, 'Ir': Ir, 'Ic': Ic, 'cache': cache}
            if cb(Y, info, opts) is True:
                info['stop'] = info['stop'] or 'cb'

        if teneva._info_appr(info, _time, nswp, e, e_vld, log):
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


def _iter(Z, Ig, I, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100, ltr=True):
    r1, n, r2 = Z.shape

    if ltr:
        Z = teneva._reshape(Z, (r1 * n, r2))
    else:
        Z = teneva._reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)

    ind, B = teneva._maxvol(Q, tau, dr_min, dr_max, tau0, k0)

    if ltr:
        G = teneva._reshape(B, (r1, n, -1))
        R = Q[ind, :] @ R
        I_new = np.kron(Ig, teneva._ones(r1))
        if I is not None:
            I_old = np.kron(teneva._ones(n), I)
            I_new = np.hstack((I_old, I_new))

    else:
        G = teneva._reshape(B.T, (-1, n, r2))
        R = (Q[ind, :] @ R).T
        I_new = np.kron(teneva._ones(r2), Ig)
        if I is not None:
            I_old = np.kron(I, teneva._ones(n))
            I_new = np.hstack((I_new, I_old))

    return G, R, I_new[ind, :]

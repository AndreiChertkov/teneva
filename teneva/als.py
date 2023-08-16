"""Package teneva, module als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which computes the TT-approximation for
the tensor by TT-ALS algorithm, using given random samples (i.e., the set of
random tensor multi-indices and related tensor values).

"""
import numpy as np
from opt_einsum import contract
import scipy as sp
import teneva
from time import perf_counter as tpc


def als(I_trn, y_trn, Y0, nswp=50, e=1.E-16, info={}, I_vld=None, y_vld=None,
        e_vld=None, r=None, r_add=10000, e_adap=1.E-3, lamb=0.001, w=None,
        cb=None, swap_tol=3, allow_swap=False, allow_skip_cores=False,
        use_stab=False, log=False):
    """Build TT-tensor by TT-ALS method using given random tensor samples.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d], where d is a number of tensor's
            dimensions and samples is a size of the train dataset.
        y_trn (np.ndarray): values of the tensor for multi-indices I_trn in
            the form of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int): number of ALS iterations (sweeps). If e or e_vld
            parameter is set, then the real number of sweeps may be less (see
            info dict with the exact number of performed sweeps).
        e (float): optional algorithm convergence criterion (> 0). If between
            iterations (sweeps) the relative rate of solution change is less
            than this value, then the operation of the algorithm will be
            interrupted.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
            At the end of the function work, it will contain parameters: e -
            the final value of the convergence criterion; e_vld - the final
            error on the validation dataset; nswp - the real number of
            performed iterations (sweeps); stop - stop type of the algorithm
            (nswp, e, e_vld or cb).
        I_vld (np.ndarray): optional multi-indices for items of validation
            dataset in the form of array of the shape [samples_vld, d], where
            samples_vld is a size of the validation dataset.
        y_vld (np.ndarray): optional values of the tensor for multi-indices
            I_vld of validation dataset in the form of array of the shape
            [samples_vld].
        e_vld (float): optional algorithm convergence criterion. If after
            sweep, the error on the validation dataset is less than this value,
            then the operation of the algorithm will be interrupted.
        r (int): maximum TT-rank for rank-adaptive ALS algorithm. If is None,
            then the TT-ALS with constant rank will be used (in the case of the
            constant rank, its value will be the same as the rank of the
            initial approximation Y0).
        r_add (int): maximum rank grow on one iteration for the rank-adaptive
            ALS algorithm. It is used only if r argument is not None.
        e_adap (float): convergence criterion for rank-adaptive TT-ALS
            algorithm (> 0). It is used only if r argument is not None.
        lamb (float): regularization parameter for least squares.
        w (np.ndarray): optional vector for weights of the input data (it
            should have a length equal to the number of elements in the data
            set). If this vector is used, then lamb parameter should be None.
        cb (function): optional callback function. It will be called after each
            sweep and the accuracy check with the arguments: Y, info and opts,
            where Y is the current approximation (TT-tensor), info is the info
            dictionary and the dictionary opts contains fields Yl, Yr and Yold.
            If the callback returns a true value, then the algorithm will be
            stopped (in the info dictionary, in this case, the stop type of the
            algorithm will be cb).
        swap_tol (int): experimental option.
        allow_swap (bool): experimental flag.
        allow_skip_cores (bool): if there is no data to learn all slices of the,
            TT-cores still work, keeping these slices. If the flag is not
            enabled, then in case of insufficient size of the training dataset,
            an error will be generated.
        use_stab (bool): if the flag is set, then the rank-adaptive method will
            use additional stabilization of the cores.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep (and before the
            first sweep).

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    _time = tpc()

    info.update({'e': -1, 'e_vld': -1, 'nswp': 0, 'stop': None})
    info['r'] = teneva.erank(Y0)

    I_trn = np.asanyarray(I_trn, dtype=int)
    y_trn = np.asanyarray(y_trn, dtype=float)

    m = I_trn.shape[0]
    d = I_trn.shape[1]

    if allow_swap:
        msg = 'The option "allow_swap" works only with adaptive rank'
        assert r is not None, msg
        I_trn = I_trn.copy()
        rearrange = np.arange(d)
        info['rearrange'] = rearrange
        print('!!! Note that "allow_swap" is a VERY experimental option')

    Y = teneva.copy(Y0)
    if r is not None:
        Y = teneva.orthogonalize(Y, 0, use_stab)

    if not allow_skip_cores:
        for k in range(d):
            if np.unique(I_trn[:, k]).size != Y[k].shape[1]:
                msg = 'One groundtruth sample is needed for every slice'
                raise ValueError(msg)

    info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
    teneva._info_appr(info, _time, nswp, e, e_vld, log)

    Yl = [np.ones((m, Y[k].shape[0])) for k in range(d)]
    Yr = [np.ones((Y[k].shape[2], m)) for k in range(d)]

    for k in range(d-1, 0, -1):
        i = I_trn[:, k]
        Q = Y[k][:, i, :]
        contract('riq,qi->ri', Q, Yr[k], out=Yr[k-1])

    while True:
        Yold = teneva.copy(Y)
        was_swap = False

        idx_cache = dict()
        for k in range(0, d-1 if r is None else d-2, +1):
            i = I_trn[:, k]

            if r is None:
                Y[k] = _optimize_core(Y[k], i, y_trn, Yl[k], Yr[k],
                    lamb=lamb, w=w)
                contract('jk,kjl->jl', Yl[k], Y[k][:, i, :], out=Yl[k+1])
            else:
                swaped = {} if allow_swap else None
                r_max = min(r, Y[k].shape[-1] + r_add)
                Y[k], Y[k+1] = _optimize_core_adaptive(Y[k], Y[k+1],
                    i, I_trn[:, k+1], y_trn, Yl[k], Yr[k+1],
                    e_adap, r_max, lamb, w, ltr=True,
                    allow_swap=swaped, swap_tol=swap_tol, cache=idx_cache)
                idx_cache = dict(i1=idx_cache['i2'])
                if allow_swap and swaped.get('swapped', False):
                    print(f'DEBUG | idxs: {k} <-> {k+1}')
                    was_swap = True
                    I_trn[:, [k, k+1]] = I_trn[:, [k+1, k]]
                    i = I_trn[:, k]
                    swap_two = np.arange(len(rearrange))
                    swap_two[k], swap_two[k+1] = swap_two[k+1], swap_two[k]
                    rearrange[:] = swap_two[rearrange]
                    idx_cache = {}
                Yl[k+1] = contract('jk,kjl->jl', Yl[k], Y[k][:, i, :])

        idx_cache = dict()
        for k in range(d-1, 0 if r is None else 1, -1):
            i = I_trn[:, k]

            if r is None:
                Y[k] = _optimize_core(Y[k], i, y_trn, Yl[k], Yr[k],
                    lamb=lamb, w=w)
                contract('ijk,kj->ij', Y[k][:, i, :], Yr[k], out=Yr[k-1])
            else:
                swaped = {} if allow_swap else None
                r_max = min(r, Y[k-1].shape[-1] + r_add)
                Y[k-1], Y[k] = _optimize_core_adaptive(Y[k-1], Y[k],
                    I_trn[:, k-1], i, y_trn, Yl[k-1], Yr[k],
                    e_adap, r_max, lamb, w, ltr=False,
                    allow_swap=swaped, swap_tol=swap_tol, cache=idx_cache)
                idx_cache = dict(i2=idx_cache['i1'])
                if allow_swap and swaped.get('swapped', False):
                    print(f'DEBUG | idxs: {k} <-> {k-1}')
                    was_swap = True
                    I_trn[:, [k, k-1]] = I_trn[:, [k-1, k]]
                    i = I_trn[:, k]
                    swap_two = np.arange(len(rearrange))
                    swap_two[k], swap_two[k-1] = swap_two[k-1], swap_two[k]
                    rearrange[:] = swap_two[rearrange]
                    idx_cache = {}
                Yr[k-1] = contract('ijk,kj->ij', Y[k][:, i, :], Yr[k])

        info['nswp'] += 1
        info['r'] = teneva.erank(Y)
        info['e'] = 1.E+10 if was_swap else teneva.accuracy(Y, Yold)
        info['e_vld'] = teneva.accuracy_on_data(
            Y, I_vld[:, rearrange] if allow_swap else I_vld, y_vld)

        if cb:
            opts = {'Yold': Yold, 'Yl': Yl, 'Yr': Yr}
            if cb(Y, info, opts) is True:
                info['stop'] = info['stop'] or 'cb'

        if teneva._info_appr(info, _time, nswp, e, e_vld, log):
            return Y


def _lstsq(A, y, lamb, w):
    if lamb is not None:
        if w is not None:
            AW = w[:, None] * A
            AtA = A.T @ AW
            Aty = AW.T @ y
        else:
            AtA = A.T @ A
            Aty = A.T @ y
        return sp.linalg.lstsq(AtA + lamb * np.identity(A.shape[1]), Aty,
            overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')
    else:
        if w is not None:
            A = w[:, None] * A
            y = y * w
        return sp.linalg.lstsq(A, y,
            overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')


def _optimize_core(Q, i, y_trn, Yl, Yr, lamb, w):
    Q = Q.copy()

    for k in range(Q.shape[1]):
        idx = np.where(i == k)[0]
        if not idx.any():
            continue

        lhs = Yr[:, idx].T[:, np.newaxis, :]
        rhs = Yl[idx, :][:, :, np.newaxis]
        A = (lhs * rhs).reshape(len(idx), -1)
        b = y_trn[idx]

        sol, residuals, rank, s = _lstsq(A, b, lamb=lamb,
            w=w[idx] if w is not None else None)
        Q[:, k, :] = sol.reshape(Q[:, k, :].shape)

        if False and rank < A.shape[1]: # TODO: check
            print(f'ALS WRN | Bad cond in LSTSQ: {rank} < {A.shape[1]}')

    return Q


def _optimize_core_adaptive(Q1, Q2, i1, i2, y_trn, Yl, Yr, e, r, lamb, w,
                            ltr=True, allow_swap=None, swap_tol=3, cache=None):
    shape = Q1.shape[0], Q2.shape[2]
    shapeQ1 = Q1.shape[:2]
    shapeQ2 = Q2.shape[1:]

    Q = np.empty((Q1.shape[0], Q1.shape[1], Q2.shape[1], Q2.shape[2]))

    cache = {} if cache is None else cache

    try:
        i1_cache = cache['i1']
    except KeyError:
        cache['i1'] = i1_cache = dict()
        for k1 in range(Q1.shape[1]):
            i1_cache[k1] = i1 == k1

    try:
        i2_cache = cache['i2']
    except KeyError:
        cache['i2'] = i2_cache = dict()
        for k2 in range(Q2.shape[1]):
            i2_cache[k2] = i2 == k2

    for k1 in range(Q1.shape[1]):
        for k2 in range(Q2.shape[1]):
            idx = cache['i1'][k1] & cache['i2'][k2]
            if not idx.any():
                continue

            lhs = Yr[:, idx].T[:, np.newaxis, :]
            rhs = Yl[idx, :][:, :, np.newaxis]
            A = (lhs * rhs).reshape(idx.sum(), -1)
            b = y_trn[idx]

            sol, residuals, rank, s = _lstsq(A, b, lamb=lamb,
                w=w[idx] if w is not None else None)
            Q[:, k1, k2, :] = sol.reshape(shape)

            if False and rank < A.shape[1]: # TODO: check
                print(f'ALS WRN | Bad cond in LSTSQ: {rank} < {A.shape[1]}')

    Qs = Q.reshape(np.prod(Q.shape[:2]), -1)
    V1, V2 = teneva.matrix_skeleton(Qs, e, r,
        rel=True, give_to='r' if ltr else 'l')

    rank1 = V1.shape[-1]

    if allow_swap is not None:
        Q = np.transpose(Q, [0, 2, 1, 3])
        Qsr = Q.reshape(np.prod(Q.shape[:2]), -1)
        V1r, V2r = teneva.matrix_skeleton(Qsr, e, r,
            rel=True, give_to='r' if ltr else 'l')
        rank2 = V1r.shape[-1]

        qual1 = _quality_of_decomp(Qs, V1, V2)
        qual2 = _quality_of_decomp(Qsr, V1r, V2r) * swap_tol
        if rank2 < rank1 or qual1 > qual2:
            print(f'DEBUG | ranks: {rank2} < {rank1}, swapping', end=' ')
            allow_swap['swapped'] = True
            V1 = V1r
            V2 = V2r
            shapeQ1 = (Q1.shape[0], Q2.shape[1])
            shapeQ2 = (Q1.shape[1], Q2.shape[-1])

    return V1.reshape(*shapeQ1, -1), V2.reshape(-1, *shapeQ2)


def _quality_of_decomp(Q, V1, V2):
    return np.linalg.norm(V1 @ V2 - Q) / np.linalg.norm(Q)

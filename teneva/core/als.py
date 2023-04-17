"""Package teneva, module core.als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which computes the TT-approximation for
the tensor by TT-ALS algorithm, using given random samples (i.e., the set of
random tensor multi-indices and related tensor values).

"""
import numpy as np
from opt_einsum import contract
import scipy as sp
import teneva
from time import perf_counter as tpc


def als(I_trn, y_trn, Y0, nswp=50, e=1.E-16, info={}, I_vld=None, y_vld=None, e_vld=None, r=None, e_adap=1.E-3, lamb=0.001, W=None, cb=None, allow_skip_cores=False, log=False):
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
        e_vld (float): optional algorithm convergence criterion (> 0). If
            after sweep, the error on the validation dataset is less than this
            value, then the operation of the algorithm will be interrupted.
        r (int): maximum TT-rank for rank-adaptive ALS algorithm (> 0). If is
            None, then the TT-ALS with constant rank will be used (in the case
            of the constant rank, its value will be the same as the rank of the
            initial approximation Y0).
        e_adap (float): convergence criterion for rank-adaptive TT-ALS
            algorithm (> 0). It is used only if r argument is not None.
        lamb (float): regularization parameter for least squares.
        W (np.ndarray): optional matrix for least squares regularization.
        cb (function): optional callback function. It will be called after
            every sweep and the accuracy check with the arguments: Y,
            info and opts, where Y is the current approximation (TT-tensor),
            info is the info dictionary and the dictionary opts contains fields
            Yl, Yr and Yold. If the callback returns a true value, then the
            algorithm will be stopped (in the info dictionary, in this case,
            the stop type of the algorithm will be cb).
        allow_skip_cores (bool): if there is no data to learn all cores slices,
            still work, keeping these slices old.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    _time = tpc()
    info.update({'r': teneva.erank(Y0), 'e': -1, 'e_vld': -1, 'nswp': 0,
        'stop': None})

    if W is not None and not lamb:
        W = np.sqrt(W)

    I_trn = np.asanyarray(I_trn, dtype=int)
    y_trn = np.asanyarray(y_trn, dtype=float)

    m = I_trn.shape[0]
    d = I_trn.shape[1]

    Y = teneva.copy(Y0)

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

        for k in range(0, d-1 if r is None else d-2, +1):
            i = I_trn[:, k]
            if r is None:
                Y[k] = _optimize_core(Y[k], i, y_trn, Yl[k], Yr[k],
                    lamb=lamb, W=W)
                contract('jk,kjl->jl', Yl[k], Y[k][:, i, :], out=Yl[k+1])
            else:
                Y[k], Y[k+1] = _optimize_core_adaptive(Y[k], Y[k+1],
                    i, I_trn[:, k+1], y_trn, Yl[k], Yr[k+1],
                    e_adap, r, lamb=lamb, W=W, l2r=True)
                Yl[k+1] = contract('jk,kjl->jl', Yl[k], Y[k][:, i, :])

        for k in range(d-1, 0 if r is None else 1, -1):
            i = I_trn[:, k]
            if r is None:
                Y[k] = _optimize_core(Y[k], i, y_trn, Yl[k], Yr[k],
                    lamb=lamb, W=W)
                contract('ijk,kj->ij', Y[k][:, i, :], Yr[k], out=Yr[k-1])
            else:
                Y[k-1], Y[k] = _optimize_core_adaptive(Y[k-1], Y[k],
                    I_trn[:, k-1], i, y_trn, Yl[k-1], Yr[k],
                    e_adap, r, lamb=lamb, W=W, l2r=False)
                Yr[k-1] = contract('ijk,kj->ij', Y[k][:, i, :], Yr[k])

        info['nswp'] += 1
        info['r'] = teneva.erank(Y)
        info['e'] = teneva.accuracy(Y, Yold)
        info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)

        if cb:
            opts = {'Yold': Yold, 'Yl': Yl, 'Yr': Yr}
            if cb(Y, info, opts) is True:
                info['stop'] = info['stop'] or 'cb'

        if teneva._info_appr(info, _time, nswp, e, e_vld, log):
            return Y


def _lstsq(A, y, lamb=0.001, W=None):
    if lamb:
        if W is not None:
            AW = W[:, None] * A
            AtA = A.T @ AW
            Aty = AW.T @ y
        else:
            AtA = A.T @ A
            Aty = A.T @ y
        return sp.linalg.lstsq(AtA + lamb * np.identity(A.shape[1]), Aty,
            overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')
    else:
        if W is not None:
            A = W[:, None] * A
            y = y * W
        return sp.linalg.lstsq(A, y,
            overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')


def _optimize_core(Q, i, y_trn, Yl, Yr, lamb=0, W=None):
    Q = Q.copy()

    for k in range(Q.shape[1]):
        idx = np.where(i == k)[0]
        if not idx.any():
            continue

        lhs = Yr[:, idx].T[:, np.newaxis, :]
        rhs = Yl[idx, :][:, :, np.newaxis]
        A = (lhs * rhs).reshape(len(idx), -1)
        b = y_trn[idx]

        sol, residuals, rank, s = _lstsq(A, b, lamb=lamb, W=W)
        Q[:, k, :] = sol.reshape(Q[:, k, :].shape)

        if False and rank < A.shape[1]:
            print(f'Bad cond in LSTSQ: {rank} < {A.shape[1]}')

    return Q


def _optimize_core_adaptive(Q1, Q2, i1, i2, y_trn, Yl, Yr, e=1e-6, r=None, lamb=0, W=None, l2r=True):
    shape = Q1.shape[0], Q2.shape[2]
    Q = np.empty((Q1.shape[0], Q1.shape[1], Q2.shape[1], Q2.shape[2]))

    for k1 in range(Q1.shape[1]):
        for k2 in range(Q2.shape[1]):
            idx = (i1 == k1) & (i2 == k2)
            if not idx.any():
                continue

            lhs = Yr[:, idx].T[:, np.newaxis, :]
            rhs = Yl[idx, :][:, :, np.newaxis]
            A = (lhs * rhs).reshape(idx.sum(), -1)
            b = y_trn[idx]

            sol, residuals, rank, s = _lstsq(A, b, lamb=lamb, W=W)
            Q[:, k1, k2, :] = sol.reshape(shape)

            if False and rank < A.shape[1]:
                print(f'Bad cond in LSTSQ: {rank} < {A.shape[1]}')

    Q = Q.reshape(np.prod(Q.shape[:2]), -1)
    V1, V2 = teneva.matrix_skeleton(Q, e, r, rel=True, give_to='r' if l2r else 'l')
    return V1.reshape(*Q1.shape[:2], -1), V2.reshape(-1, *Q2.shape[1:])

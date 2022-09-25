"""Package teneva, module core.als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which compute the TT-approximation for
the tensor by TT-ALS algorithm, using given random samples.

"""
import numpy as np
from opt_einsum import contract
import scipy as sp
import teneva
from time import perf_counter as tpc


def als(I_trn, Y_trn, Y0, nswp=50, e=1.E-16, info={}, I_vld=None, Y_vld=None, e_vld=None, r=None, e_adap=1.E-3, log=False):
    """Build TT-tensor by TT-ALS from the given random tensor samples.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for multi-indices I in the form
            of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int): number of ALS iterations (sweeps). If "e" or "e_vld"
            parameter is set, then the real number of sweeps may be less (see
            "info" dict with the exact number of performed sweeps).
        e (float): optional algorithm convergence criterion (> 0). If between
            iterations (sweeps) the relative rate of solution change is less
            than this value, then the operation of the algorithm will be
            interrupted.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
            At the end of the function work, it will contain parameters: "e" -
            the final value of the convergence criterion; "e_vld" - the final
            error on the validation dataset; "nswp" - the real number of
            performed iterations (sweeps); "stop" - stop type of the algorithm
            ("nswp", "e" or "e_vld").
        I_vld (np.ndarray): optional multi-indices for items of validation
            dataset in the form of array of the shape [samples, d].
        Y_vld (np.ndarray): optional values for items related to "I_vld" of
            validation dataset in the form of array of the shape [samples].
        e_vld (float): optional algorithm convergence criterion (> 0). If
            after sweep, the error on the validation dataset is less than this
            value, then the operation of the algorithm will be interrupted.
        r (int): maximum TT-rank for rank-adaptive ALS algorithm (> 0). If is
            None, then the TT-ALS with constant rank will be used.
        e_adap (float): convergence criterion for rank-adaptive TT-ALS
            algorithm (> 0). It is used if "r" is not None.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    _time = tpc()

    info['r'] = teneva.erank(Y0)
    info['e'] = -1.
    info['e_vld'] = -1.
    info['nswp'] = 0
    info['stop'] = None

    if e_adap is None:
        e_adap = e

    I_trn = np.asanyarray(I_trn, dtype=int)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    Y = teneva.copy(Y0)

    m = I_trn.shape[0]
    d = I_trn.shape[1]

    for k in range(d):
        if np.unique(I_trn[:, k]).size != Y[k].shape[1]:
            raise ValueError('One groundtruth sample is needed for every slice')

    Yl = [np.ones((m, Y[k].shape[0])) for k in range(d)]
    Yr = [np.ones((Y[k].shape[2], m)) for k in range(d)]

    for k in range(d-1, 0, -1):
        i = I_trn[:, k]
        Q = Y[k][:, i, :]
        contract('riq,qi->ri', Q, Yr[k], out=Yr[k-1])

    _info(Y, info, _time, I_vld, Y_vld, e_vld, log)

    while True:
        Yold = teneva.copy(Y)

        for k in range(0, d-1 if r is None else d-2, +1):
            i = I_trn[:, k]
            if r is not None:
                Y[k], Y[k+1] = _optimize_core_adaptive(Y[k], Y[k+1],
                    i, I_trn[:, k+1], Y_trn, Yl[k], Yr[k+1], e_adap, r)
                Yl[k+1] = contract('jk,kjl->jl', Yl[k], Y[k][:, i, :])
            else:
                Y[k] = _optimize_core(Y[k], i, Y_trn, Yl[k], Yr[k])
                contract('jk,kjl->jl', Yl[k], Y[k][:, i, :], out=Yl[k+1])

        for k in range(d-1, 0 if r is None else 1, -1):
            i = I_trn[:, k]
            if r is not None:
                Y[k-1], Y[k] = _optimize_core_adaptive(Y[k-1], Y[k],
                    I_trn[:, k-1], i, Y_trn, Yl[k-1], Yr[k], e_adap, r)
                Yr[k-1] = contract('ijk,kj->ij', Y[k][:, i, :], Yr[k])
            else:
                Y[k] = _optimize_core(Y[k], i, Y_trn, Yl[k], Yr[k])
                contract('ijk,kj->ij', Y[k][:, i, :], Yr[k], out=Yr[k-1])

        stop = None

        info['e'] = teneva.accuracy(Y, Yold)
        if stop is None and info['e'] >= 0 and not np.isinf(info['e']):
            if e is not None and info['e'] <= e:
                stop = 'e'

        info['nswp'] += 1
        if stop is None:
            if nswp is not None and info['nswp'] >= nswp:
                stop = 'nswp'

        if _info(Y, info, _time, I_vld, Y_vld, e_vld, log, stop):
            return Y


def als2(I_trn, Y_trn, Y0, nswp=10, eps=None):
    """Build TT-tensor by TT-ALS from the given random tensor samples (OLD).

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for multi-indices I in the form
            of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int):  number of ALS iterations (sweeps).
        eps (float): desired accuracy of approximation.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    Note:
        This is the alternative realization of the TT-ALS algorithm. The
        version from "als" function in the most cases works better and much
        faster. Applications are not expected to use this function. Note also
        that the code of this function is not optimized.

    """
    I_trn = np.asanyarray(I_trn, dtype=int)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    P, d = I_trn.shape

    norm = np.linalg.norm(Y_trn)
    Y_trn = Y_trn.copy()
    Y_trn /= norm

    Y = [G.copy() for G in Y0]

    elist = []

    def getRow(leftU, rightV, jVec):
        jLeft = jVec[:len(leftU)] if len(leftU) > 0 else None
        jRight = jVec[-len(rightV):] if len(rightV) > 0 else None

        multU = np.ones([1, 1])
        for k in range(len(leftU)):
            multU = multU @ leftU[k][:, jLeft[k], :]

        multV= np.ones([1, 1])
        for k in range(len(rightV)-1, -1, -1):
            multV = rightV[k][:, jRight[k], :] @ multV

        return np.kron(multV.T, multU)

    for swp in range(nswp):

        for k in range(d):
            r1, n, r2 = Y[k].shape

            core = np.zeros([r1, n, r2])

            leftU = Y[:k] if k > 0 else []
            rightV = Y[k+1:] if k < d-1 else []

            for i in range(n):
                thetaI = np.where(I_trn[:, k] == i)[0]
                if len(thetaI) == 0:
                    continue

                A = np.zeros([len(thetaI), r1*r2])
                for j in range(len(thetaI)):
                    A[j:j+1, :] += getRow(leftU, rightV, I_trn[thetaI[j], :])

                vec_slice = np.linalg.lstsq(A, Y_trn[thetaI], rcond=-1)[:1]
                core[:, i, :] += teneva._reshape(vec_slice, [r1, r2])

            Y[k] = core.copy()

        if eps is not None:
            get = lambda x: teneva.get(Y, x)
            e = 0.5 * sum((get(I_trn[p, :]) - Y_trn[p])**2 for p in range(P))
            elist.append(e)
            if e < eps or swp > 0 and abs(e - elist[-2]) < eps:
                break

    Y[0] *= norm

    return Y


def _info(Y, info, t, I_vld=None, Y_vld=None, e_vld=None, log=False, stop=None):
    info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, Y_vld)

    if stop is None and info['e_vld'] >= 0 and not np.isinf(info['e_vld']):
        if e_vld is not None and info['e_vld'] <= e_vld:
            stop = 'e_vld'
    info['stop'] = stop

    info['r'] = teneva.erank(Y)
    info['t'] = tpc() - t

    _log(Y, info, log)

    return info['stop']


def _log(Y, info, log):
    if not log:
        return

    text = ''

    if info['nswp'] == 0:
        text += f'# pre | '
    else:
        text += f'# {info["nswp"]:-3d} | '

    text += f'time: {info["t"]:-10.3f} | '

    text += f'rank: {info["r"]:-5.1f} | '

    if info['e_vld'] >= 0:
        text += f'err: {info["e_vld"]:-7.1e} | '

    if info['e'] >= 0:
        text += f'eps: {info["e"]:-7.1e} | '

    if info['stop']:
        text += f'stop: {info["stop"]} | '

    print(text)


def _optimize_core(Q, i, Y_trn, Yl, Yr):
    Q = Q.copy()

    for k in range(Q.shape[1]):
        idx = np.where(i == k)[0]

        lhs = Yr[:, idx].T[:, np.newaxis, :]
        rhs = Yl[idx, :][:, :, np.newaxis]
        A = (lhs * rhs).reshape(len(idx), -1)
        Ar = A.shape[1]
        b = Y_trn[idx]

        sol, residuals, rank, s = sp.linalg.lstsq(A, b,
            overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')
        Q[:, k, :] = sol.reshape(Q[:, k, :].shape)

        if False and rank < Ar:
            print(f'Bad cond in LSTSQ: {rank} < {Ar}')

    return Q


def _optimize_core_adaptive(Q1, Q2, i1, i2, Y_trn, Yl, Yr, e=1e-6, r=None):
    shape = Q1.shape[0], Q2.shape[2]
    Q = np.empty((Q1.shape[0], Q1.shape[1], Q2.shape[1], Q2.shape[2]))

    for k1 in range(Q1.shape[1]):
        for k2 in range(Q2.shape[1]):
            idx = (i1 == k1) & (i2 == k2)

            # TODO: Add this check to the main func at the beginning:
            assert idx.any(), 'Not enough samples'

            lhs = Yr[:, idx].T[:, np.newaxis, :]
            rhs = Yl[idx, :][:, :, np.newaxis]
            A = (lhs * rhs).reshape(idx.sum(), -1)
            Ar = A.shape[1]
            b = Y_trn[idx]

            sol, residuals, rank, s = sp.linalg.lstsq(A, b,
                overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')
            Q[:, k1, k2, :] = sol.reshape(shape)

            if False and rank < Ar:
                print(f'Bad cond in LSTSQ: {rank} < {Ar}')

    Q = Q.reshape(np.prod(Q.shape[:2]), -1)
    V1, V2 = teneva.matrix_skeleton(Q, e, r, rel=True)
    return V1.reshape(*Q1.shape[:2], -1), V2.reshape(-1, *Q2.shape[1:])

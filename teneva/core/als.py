"""Package teneva, module core.als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which computes the TT-approximation
for the tensor by TT-ALS algorithm, using given random samples.

"""
import numpy as np
import scipy as sp
from time import perf_counter as tpc


from .act_one import copy
from .act_one import get
from .act_two import accuracy
from .data import accuracy_on_data
from .props import erank
from .utils import _reshape


def als(I_trn, Y_trn, Y0, nswp=50, e=1.E-16, info={}, I_vld=None, Y_vld=None, e_vld=None, log=False):
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
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    _time = tpc()

    info['r'] = erank(Y0)
    info['e'] = -1.
    info['e_vld'] = -1.
    info['nswp'] = 0
    info['stop'] = None

    I_trn = np.asanyarray(I_trn, dtype=int)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    Y = copy(Y0)

    m = I_trn.shape[0]
    d = I_trn.shape[1]

    for k in range(d):
        if np.unique(I_trn[:, k]).size != Y[k].shape[1]:
            raise ValueError('One groundtruth sample is needed for every slice')

    Yl = [np.ones((1, m, Y[k].shape[0])) for k in range(d)]

    Yr = [None for _ in range(d-1)] + [np.ones((1, m, 1))]
    for k in range(d-1, 0, -1):
        i = I_trn[:, k]
        Q = Y[k][:, i, :]
        Yr[k-1] = np.einsum('riq,qis->ris', Q, Yr[k])

    _info(Y, info, _time, I_vld, Y_vld, e_vld, log)

    while True:
        Yold = copy(Y)

        for k in range(0, d-1, +1):
            i = I_trn[:, k]
            Y[k] = _optimize_core(Y[k], i, Y_trn, Yl[k], Yr[k])
            Yl[k+1] = np.einsum('ijk,kjl->ijl', Yl[k], Y[k][:, i, :])

        for k in range(d-1, 0, -1):
            i = I_trn[:, k]
            Y[k] = _optimize_core(Y[k], i, Y_trn, Yl[k], Yr[k])
            Yr[k-1] = np.einsum('ijk,kjl->ijl', Y[k][:, i, :], Yr[k])

        stop = None

        info['e'] = accuracy(Y, Yold)
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
                core[:, i, :] += _reshape(vec_slice, [r1, r2])

            Y[k] = core.copy()

        if eps is not None:
            get = lambda x: get(Y, x)
            e = 0.5 * sum((get(I_trn[p, :]) - Y_trn[p])**2 for p in range(P))
            elist.append(e)
            if e < eps or swp > 0 and abs(e - elist[-2]) < eps:
                break

    Y[0] *= norm

    return Y


def _info(Y, info, t, I_vld, Y_vld, e_vld, log=False, stop=None):
    info['e_vld'] = accuracy_on_data(Y, I_vld, Y_vld)

    if stop is None and info['e_vld'] >= 0 and not np.isinf(info['e_vld']):
        if e_vld is not None and info['e_vld'] <= e_vld:
            stop = 'e_vld'
    info['stop'] = stop

    info['r'] = erank(Y)
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


def _optimize_core(G, i, Y_trn, Yl, Yr):
    Q = G.copy()
    for k in range(Q.shape[1]):
        idx = np.where(i == k)[0]
        lhs = np.transpose(Yr[:, idx, 0], [1, 0])[:, :, np.newaxis]
        rhs = Yl[0, idx, :][:, np.newaxis, :]
        A = _reshape(lhs * rhs, (len(idx), -1))
        b = Y_trn[idx]
        sol, residuals = sp.linalg.lstsq(A, b)[0:2]
        Q[:, k, :] = _reshape(sol, Q[:, k, :].shape, 'C')
    return Q

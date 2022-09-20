"""Package teneva, module core.als_spectral: build TT-tensor of coefficients.

This module contains the functions "als_spectral" and "als_cheb" which compute
the TT-approximation of tensor of spectral coefficients (i.e., the TT-Tucker
core tensor) by TT-ALS algorithm, using given random samples.

"""
import numpy as np
from opt_einsum import contract
import scipy as sp
import teneva
from time import perf_counter as tpc


from .als import _info


def als_cheb(X_trn, Y_trn, A0, a, b, nswp=50, e=1.E-16, info={}, log=False):
    """Draft of the function. TODO: add order-adaptive version."""
    _time = tpc()
    info = {}
    n = teneva.shape(A0)
    fh = lambda X: teneva.cheb_pol(X, a[0], b[0], n[0]).T
    A = als_spectral(X_trn, Y_trn, A0, fh, nswp, e, info, log)
    return A


def als_spectral(X_trn, Y_trn, Y0, fh, nswp=50, e=1.E-16, info={}, log=False):
    """Build TT-Tucker core tensor by TT-ALS from the given samples.

    Args:
        X_trn (np.ndarray): set of train spatial points in the form of array of
            the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for points X in the form of
            array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        fh (function): function, that generates a line in the H-matrices in
            the TT-Tucker.
        nswp (int): number of ALS iterations (sweeps). If "e" parameter is set,
            then the real number of sweeps may be less (see "info" dict with
            the exact number of performed sweeps).
        e (float): optional algorithm convergence criterion (> 0). If between
            iterations (sweeps) the relative rate of solution change is less
            than this value, then the operation of the algorithm will be
            interrupted.
        info (dict): an optionally set dictionary, which will be filled with
            reference information about the process of the algorithm operation.
            At the end of the function work, it will contain parameters: "e" -
            the final value of the convergence criterion; "nswp" - the real
            number of performed iterations (sweeps); "stop" - stop type of the
            algorithm ("nswp" or "e").
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the
        TT-Tucker core tensor.

    """
    _time = tpc()

    info['r'] = teneva.erank(Y0)
    info['e'] = -1.
    info['nswp'] = 0
    info['stop'] = None

    X_trn = np.asanyarray(X_trn, dtype=float)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    Y = teneva.copy(Y0)

    m = X_trn.shape[0]
    d = X_trn.shape[1]

    Yl = [np.ones((m, Y[k].shape[0])) for k in range(d)]
    Yr = [np.ones((Y[k].shape[2], m)) for k in range(d)]

    # Assuming they are all the same for now (TODO!):
    n = Y[0].shape[1]
    H = fh(X_trn.reshape(-1)).reshape((*X_trn.shape, n))
    del X_trn # For test and for memory

    for k in range(d-1, 0, -1):
        contract('ik,rkq,qi->ri', H[:, k, :], Y[k], Yr[k], out=Yr[k-1])

    _info(Y, info, _time, log=log)

    while True:
        Yold = teneva.copy(Y)

        for k in range(0, d-1, +1):
            Hk =  H[:, k, :]
            _optimize_core(Y[k], Y_trn, Yl[k], Yr[k], Hk)
            contract('jr,jk,krl->jl', Hk, Yl[k], Y[k], out=Yl[k+1])

        for k in range(d-1, 0, -1):
            Hk =  H[:, k, :]
            _optimize_core(Y[k], Y_trn, Yl[k], Yr[k], Hk)
            contract('jr,irk,kj->ij', Hk, Y[k], Yr[k], out=Yr[k-1])

        stop = None

        info['e'] = teneva.accuracy(Y, Yold)
        if stop is None and info['e'] >= 0 and not np.isinf(info['e']):
            if e is not None and info['e'] <= e:
                stop = 'e'

        info['nswp'] += 1
        if stop is None:
            if nswp is not None and info['nswp'] >= nswp:
                stop = 'nswp'

        if _info(Y, info, _time, log=log, stop=stop):
            return Y


def _optimize_core(Q, Y_trn, Yl, Yr, Hk):
    m = Yl.shape[0]
    A = contract('li,ik,ij->ikjl', Yr, Yl, Hk).reshape(m, -1)
    b = Y_trn
    Ar = A.shape[1]

    sol, residuals, rank, s = sp.linalg.lstsq(A, b,
        overwrite_a=True, overwrite_b=True, lapack_driver='gelsy')
    Q[...] = sol.reshape(Q.shape)

    if False and rank < Ar:
        print(f'Bad cond in LSTSQ: {rank} < {Ar}')

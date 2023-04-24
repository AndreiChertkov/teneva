"""Package teneva, module als_func: construct TT-tensor of coefficients.

This module contains the function "als_func" which computes the
TT-approximation of tensor of coefficients (i.e., the TT-Tucker core tensor) by
TT-ALS algorithm, using given random samples.

"""
import numpy as np
from opt_einsum import contract
import scipy as sp
import teneva
from time import perf_counter as tpc


def als_func(X_trn, y_trn, A0, a=-1., b=+1., nswp=50, e=1.E-16, info={},
             X_vld=None, y_vld=None, e_vld=None, fh=None, n_max=None,
             thr_pow=1.E-6, log=False):
    """Build TT-Tucker core tensor by TT-ALS from the given samples.

    Args:
        X_trn (np.ndarray): set of train spatial points in the form of array of
            the shape [samples, d], where d is a number of function's input
            dimensions and samples is a size of the train dataset.
        y_trn (np.ndarray): values of the function for inputs X_trn in the
            form of array of the shape [samples].
        A0 (list): TT-tensor, which is the initial approximation for algorithm.
            It should have the equal size for all modes.
        a (float): grid lower bounds for each dimension (should be the same for
            all dimensions in the current version). It is not used if fh
            argument is given.
        b (float): grid upper bounds for each dimension (should be the same for
            all dimensions in the current version). It is not used if fh
            argument is given.
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
            (nswp, e or e_vld).
        X_vld (np.ndarray): optional spatial points for items of validation
            dataset in the form of array of the shape [samples_vld, d], where
            samples_vld is a size of the validation dataset.
        y_vld (np.ndarray): optional values of the function for spatial points
            X_vld of validation dataset in the form of array of the shape
            [samples_vld].
        e_vld (float): optional algorithm convergence criterion (> 0). If
            after sweep, the error on the validation dataset is less than this
            value, then the operation of the algorithm will be interrupted.
        fh (function): optional function, that generates a line in the H
            matrices in the TT-Tucker. If it is not set, then a and b
            arguments should be provided (the Chebyshev interpolation will be
            used in this case).
        n_max (int): optional maximum mode size for coefficients' tensor. If
            the parameter is set, then a dynamic search for the optimal value
            will be carried out.
        thr_pow (float): optional parameter for dynamic search of the optimal
            value of the mode size for coefficients' tensor.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor
            of interpolation coefficients (i.e., TT-Tucker core tensor).

    """
    _time = tpc()
    info.update({'r': teneva.erank(A0), 'e': -1, 'e_vld': -1, 'nswp': 0,
        'stop': None})

    X_trn = np.asanyarray(X_trn, dtype=float)
    y_trn = np.asanyarray(y_trn, dtype=float)

    m = X_trn.shape[0]
    d = X_trn.shape[1]
    n = [G.shape[1] for G in A0]

    Y = teneva.copy(A0)
    if n_max:
        for k in range(d):
            G = np.zeros((Y[k].shape[0], n_max, Y[k].shape[2]))
            G[:, :Y[k].shape[1], :] = Y[k]
            Y[k] = G

    if fh is None:
        is_cheb = True
        fh_size = n_max or Y[0].shape[1]
        fh = lambda X: teneva.func_basis(
            teneva.poi_scale(X, a, b, kind='cheb'), fh_size).T
    else:
        is_cheb = False
        if X_vld is not None or y_vld is not None:
            raise NotImplementedError('Validation works only for "cheb" now.')

    if X_vld is not None and y_vld is not None:
        y_our = teneva.func_get(X_vld, Y, a, b)
        info['e_vld'] = np.linalg.norm(y_our - y_vld)
        info['e_vld'] /= np.linalg.norm(y_vld)
    teneva._info_appr(info, _time, nswp, e, e_vld, log)

    Yl = [np.ones((m, A0[k].shape[0])) for k in range(d)]
    Yr = [np.ones((A0[k].shape[2], m)) for k in range(d)]

    H = fh(X_trn.reshape(-1)).reshape((*X_trn.shape, -1))
    del X_trn # For test and for memory

    for k in range(d-1, 0, -1):
        contract('ik,rkq,qi->ri', H[:, k, :n[k]], Y[k][:, :n[k], :], Yr[k],
            out=Yr[k-1])

    while True:
        Yold = teneva.copy(Y)
        nold = list(n)

        for lr in [1, -1]:
            rng = range(0, d-1, +1) if lr == 1 else range(d-1, 0, -1)
            for k in rng:
                n_k = n[k]
                if n_max is not None: # Temporary increase max pow of poly
                    n_k = min(n_k + 1, n_max)

                n[k] = n_k =_optimize_core(Y[k][:, :n_k, :], y_trn,
                    Yl[k], Yr[k], H[:, k, :n_k], n_max, thr_pow)
                Hk =  H[:, k, :n_k]

                if lr == 1:
                    contract('jr,jk,krl->jl', Hk, Yl[k], Y[k][:, :n_k, :],
                        out=Yl[k+1])
                else:
                    contract('jr,irk,kj->ij', Hk, Y[k][:, :n_k, :], Yr[k],
                        out=Yr[k-1])

        for k, c in enumerate(Yold):
            c[:, nold[k]:, :] = 0.
        for k, c in enumerate(Y):
            c[:, n[k]:, :] = 0.

        info['nswp'] += 1
        info['r'] = teneva.erank(Y)
        info['e'] = teneva.accuracy(Y, Yold)
        if X_vld is not None and y_vld is not None:
            y_our = teneva.func_get(X_vld, Y, a, b)
            info['e_vld'] = np.linalg.norm(y_our - y_vld)
            info['e_vld'] /= np.linalg.norm(y_vld)

        if teneva._info_appr(info, _time, nswp, e, e_vld, log):
            if n_max is not None:
                Y = [c[:, :n[k], :].copy() for k, c in enumerate(Y)]
            return Y


def _optimize_core(Q, y_trn, Yl, Yr, Hk, n_max, thr_pow):
    A = contract('li,ik,ij->ikjl', Yr, Yl, Hk).reshape(Yl.shape[0], -1)
    sol, residuals, rank, s = sp.linalg.lstsq(A, y_trn,
        overwrite_a=True, overwrite_b=False, lapack_driver='gelsy')
    Q[...] = sol.reshape(Q.shape)

    n_k = Q.shape[1]
    if n_max is not None:
        if n_k > 1 and np.abs(Q[:, -1, :]).max() / np.abs(Q).max() < thr_pow:
            n_k = _optimize_core(Q[:, :-1, :], y_trn,
                Yl, Yr, Hk[:, :-1], n_max, thr_pow)

    return n_k

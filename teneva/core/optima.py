"""Package teneva, module core.optima: estimation of min and max of tensor.

This module contains the maxvol-based algorithm for computation of minimum and
maximum element of the given TT-tensor (function optima_tt).

"""
import numpy as np


from .maxvol import maxvol
from .tensor import copy
from .tensor import erank
from .tensor import get
from .tensor import getter
from .tensor import mul
from .tensor import shape
from .tensor import sub
from .transformation import truncate
from .transformation import orthogonalize


from teneva import tensor_delta


def optima_tt(Y, nswp=5, nswp_outer=1, r=70, e=1.E-10, log=False):
    """Find multi-indices which relate to min and max elements of TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        nswp (int): number of power-iterations (> 0).
        nswp_outer (int): number of repeats of power-iterations (> 0).
        r (int): maximum TT-rank while power-iterations (> 0).
        e (float): accuracy for intermediate truncations (> 0).
        log (bool): if flag is True, then the log for optimization process will
            be presented in console.

    Returns:
        [np.ndarray, np.ndarray]: multi-index (array of length d) which relates
        to minimum TT-tensor element and multi-index (array of length d) which
        relates to maximum TT-tensor element.

    Note:
        As it turns out empirically, this function often gives the same result
        as if we converted the TT-tensor to the full format (i.e., teneva.full)
        and explicitly found its minimum and maximum elements. However, this
        values will not always correspond to the minimum and maximum of the
        original tensor (for which the TT-tensor is an approximation). In the
        latter case, the accuracy of the min/max will depend on the accuracy of
        the TT-approximation.

    """
    i_min, y_min, i_max, y_max = optima_tt_simple(Y)
    is_min_better = y_min < 0 and -y_min > y_max
    is_max_better = not is_min_better

    _log(y_min, y_max, None, erank(Y), is_outer=True, with_log=log)

    for swp in range(nswp_outer):
        _log(None, None, swp, is_outer=True, with_log=log)

        i_min, y_min, i_max, y_max = _optima_tt_iter(Y,
            i_min, y_min, i_max, y_max, is_min_better, nswp, r, e, log)
        i_min, y_min, i_max, y_max = _optima_tt_iter(Y,
            i_min, y_min, i_max, y_max, is_max_better, nswp, r, e, log)

    return i_min, i_max


def optima_tt_simple(Y):
    """Helper function for TT-tensor optimization."""
    get = getter(Y)
    Y = copy(Y)
    d = len(Y)
    n = shape(Y)

    Ig = [_reshape(np.arange(k, dtype=int), (-1, 1)) for k in n]
    Ir = [None for i in range(d+1)]
    Ic = [None for i in range(d+1)]

    R = np.ones((1, 1))
    for i in range(d):
        G = np.tensordot(R, Y[i], 1)
        Y[i], R, Ir[i+1] = _iter(G, Ig[i], Ir[i], l2r=True)
    Y[d-1] = np.tensordot(Y[d-1], R, 1)

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter(G, Ig[i], Ic[i+1], l2r=False)
    Y[0] = np.tensordot(R, Y[0], 1)

    res = {'i_min': None, 'y_min': None, 'i_max': None, 'y_max': None}
    for i in range(d):
        Z = _func(get, Ig[i], Ir[i], Ic[i+1], res)

    return res['i_min'], res['y_min'], res['i_max'], res['y_max']


def _func(f, Ig, Ir, Ic, res):
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

    for i in I:
        y = f(i)
        if res['y_min'] is None or y < res['y_min']:
            res['i_min'] = i.copy()
            res['y_min'] = y
        if res['y_max'] is None or y > res['y_max']:
            res['i_max'] = i.copy()
            res['y_max'] = y


def _iter(Z, Ig, I=None, l2r=True):
    r1, n, r2 = Z.shape
    Z = _reshape(Z, (r1 * n, r2)) if l2r else _reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)
    ind, B = _maxvol(Q)

    i_max, j_max = np.divmod(np.abs(Z).argmax(), Z.shape[1])
    if not i_max in ind:
        # TODO: Maybe we can do it more accurate:
        ind[-1] = i_max

    G = B if l2r else B.T
    G = _reshape(G, (r1, n, -1)) if l2r else _reshape(G, (-1, n, r2))

    R = Q[ind, :] @ R
    R = R if l2r else R.T

    Ig = Ig.reshape((-1, 1))
    I_new = np.kron(Ig, _ones(r1)) if l2r else np.kron(_ones(r2), Ig)
    if I is not None:
        I_old = np.kron(_ones(n), I) if l2r else np.kron(I, _ones(n))
        I_new = np.hstack((I_old, I_new)) if l2r else np.hstack((I_new, I_old))
    I_new = I_new[ind, :]

    return G, R, I_new


def _log(y_min=None, y_max=None, swp=None, r=None, is_max=False, is_outer=False, y_eps=None, with_log=False):
    if not with_log:
        return

    text = ''

    if is_outer:
        text += f'outer : '
    else:
        text += f'inner : '

    if swp is not None:
        text += f'{swp+1:-3d} | '
    else:
        text += f'pre | '

    if is_outer:
        text += '... | '
    else:
        text += 'MAX | ' if is_max else 'MIN | '

    if r is not None:
        text += f'rank = {r:-5.1f} | '

    if y_min is not None:
        text += f'y_min = {y_min:-16.7e} | '

    if y_max is not None:
        text += f'y_max = {y_max:-16.7e} | '

    if y_eps is not None:
        text += f'y_eps = {y_eps:-16.7e} | '

    print(text)


def _maxvol(A, tau=1.1, tau0=1.05, k0=100):
    n, r = A.shape
    if n <= r:
        I = np.arange(n, dtype=int)
        B = np.eye(n, dtype=float)
    else:
        I, B = maxvol(A, tau0, k0)
    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _optima_tt_iter(Y, i_min, y_min, i_max, y_max, is_max, nswp, r, e, log=False):
    """Find max/min for TT-tensor using good approximation of min/max."""
    if is_max:
        i_opt, y_opt, y_ref = i_max, y_max, y_min
    else:
        i_opt, y_opt, y_ref = i_min, y_min, y_max

    def _check(i_opt, y_opt, i_opt_new, y_opt_new):
        if (is_max and y_opt_new > y_opt) or (not is_max and y_opt_new < y_opt):
            return i_opt_new.copy(), y_opt_new, np.abs(y_opt_new - y_opt)
        else:
            return i_opt, y_opt, 0.

    def _gouge(Z, i_opt):
        y = get(Z, i_opt)
        D = tensor_delta(shape(Z), i_opt, y)
        Z = sub(Z, D)
        Z = truncate(Z, e)
        return Z

    def _result():
        if is_max:
            return i_min, y_min, i_opt, y_opt
        else:
            return i_opt, y_opt, i_max, y_max

    Z = sub(Y, y_ref)
    scale = y_max - y_min

    if is_max:
        _, __, i_opt_new, ___ = optima_tt_simple(Z)
    else:
        i_opt_new, _, __, ___ = optima_tt_simple(Z)
    y_opt_new = get(Y, i_opt_new)
    i_opt, y_opt, y_eps = _check(i_opt, y_opt, i_opt_new, y_opt_new)

    _log(
        y_min if is_max else y_opt,
        y_opt if is_max else y_max,
        -1, erank(Z), is_max, y_eps=y_eps, with_log=log)

    Z = _gouge(Z, i_opt)

    for swp in range(nswp):
        if scale < 1.E-16:
            print('Warning! Almost zero scale. Break')
            return _result()
        if erank(Z) * 2 >= r:
            return _result()

        Z = mul(Z, Z)
        Z = mul(Z, 1./scale**2)
        Z = truncate(Z, e)

        _, __, i_opt_new, scale = optima_tt_simple(Z)
        y_opt_new = get(Y, i_opt_new)
        i_opt, y_opt, y_eps = _check(i_opt, y_opt, i_opt_new, y_opt_new)

        if y_eps > 0:
            Z = _gouge(Z, i_opt)

        _log(
            y_min if is_max else y_opt,
            y_opt if is_max else y_max,
            swp, erank(Z), is_max, y_eps=y_eps, with_log=log)

    return _result()


def _reshape(A, n):
    return np.reshape(A, n, order='F')



def TT_top_k_sq(cores, k=100, to_orth=True):

    if to_orth:
        cores = orthogonalize(cores, 0)


    cores0 = cores[0]
    cur_core = cores0.reshape(*cores0.shape[1:])
    cur_idx = np.arange(cores0.shape[1])[:, None]

    for Gc in cores[1:]:
        cur_core = np.einsum("ij,jkl->ikl", cur_core, Gc).reshape(-1, Gc.shape[-1])
        n = Gc.shape[1]
        cur_idx = np.hstack((
            np.kron(cur_idx, np.ones(n, dtype=int)[:, None]),
            np.kron(np.ones(cur_idx.shape[0], dtype=int)[:, None], np.arange(n)[:, None])
        ))


        idx_k = np.argsort(np.sum(cur_core**2, axis=1))[:-(k+1):-1]

        cur_idx = cur_idx[idx_k]
        cur_core = cur_core[idx_k]

    return cur_idx[0]


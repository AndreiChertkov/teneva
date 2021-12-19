import numpy as np


from .grid import ind2str
from .maxvol import maxvol
from .maxvol import maxvol_rect
from .tensor import truncate


def cross(f, Y0, e, evals=None, nswp=10, kr=2, rf=2, cache=None, info={}):
    d = len(Y0)
    Y = [Y0[i].copy() for i in range(d)]
    Il = [np.empty((1, 0), dtype=int)] + [None for i in range(d)]
    Ir = [None for i in range(d)] + [np.empty((1, 0), dtype=int)]
    Ig = [_reshape(np.arange(G.shape[1], dtype=int), (-1, 1)) for G in Y]

    R = np.ones((1, 1))
    for i in range(d-1):
        G = np.tensordot(R, Y[i], 1)
        Y[i], Il[i+1], R = cross_prep_l2r(G, Ig[i], Il[i])

    R = np.ones((1, 1))
    for i in range(d-1, 0, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], Ir[i], R = cross_prep_r2l(G, Ig[i], Ir[i+1])

    def func(I):
        if cache is None:
            info['k_cache'] = 0
            info['k_evals'] = info.get('k_evals', 0) + len(I)
            return f(I)

        I_new = np.array([i for i in I if ind2str(i) not in cache])
        if len(I_new):
            Y_new = f(I_new)
            for k, i in enumerate(I_new):
                cache[ind2str(i)] = Y_new[k]

        info['k_cache'] = info.get('k_cache', 0) + (len(I) - len(I_new))
        info['k_evals'] = info.get('k_evals', 0) + len(I_new)

        Y = np.array([cache[ind2str(i)] for i in I])
        return Y

    for _ in range(nswp):
        R = np.ones((1, 1))
        for i in range(d):
            G = np.tensordot(R, Y[i], 1)
            y = func(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            if y is None:
                return truncate(Y, e)
            Y[i], Il[i+1], R = cross_build_l2r(
                *G.shape, y, Ig[i], Il[i], kr, rf)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            G = np.tensordot(Y[i], R, 1)
            y = func(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            if y is None:
                return truncate(Y, e)
            Y[i], Ir[i], R = cross_build_r2l(
                *G.shape, y, Ig[i], Ir[i+1], kr, rf)
        Y[0] = np.tensordot(R, Y[0], 1)

    Y = truncate(Y, e)
    return Y


def cross_build_l2r(r1, n, r2, y, Ig, I, kr, rf):
    G = _reshape(y, (-1, r2))
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, kr, rf)
    G = _reshape(B, (r1, n, -1))
    J = cross_index_stack_l2r(r1, Ig, I)[Ind, :]
    R = Q[Ind, :] @ np.diag(s) @ V
    return G, J, R


def cross_build_r2l(r1, n, r2, y, Ig, I, kr, rf):
    G = _reshape(y, (r1, -1)).T
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, kr, rf)
    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(r2, Ig, I)[Ind, :]
    R = (Q[Ind, :] @ np.diag(s) @ V).T
    return G, J, R


def cross_index_merge(i1, i2, i3):
    r1 = i1.shape[0] or 1
    r2 = i2.shape[0]
    r3 = i3.shape[0] or 1
    w1 = _kron(_ones(r3 * r2), i1)
    w2 = np.empty((w1.shape[0], 0))
    if i3.size and r2:
        w2 = _kron(i3, _ones(r1 * r2))
    w3 = _kron(_kron(_ones(r3), i2), _ones(r1))
    return np.hstack((w1, w3, w2))


def cross_index_stack_l2r(r, Ig, I):
    n = Ig.size
    J = _kron(Ig, _ones(r))
    if I.size:
        J = np.hstack((_kron(_ones(n), I), J))
    return _reshape(J, (n * r, -1))


def cross_index_stack_r2l(r, Ig, I):
    n = Ig.size
    J = _kron(_ones(r), Ig)
    if I.size:
        J = np.hstack((J, _kron(I, _ones(n))))
    return _reshape(J, (n * r, -1))


def cross_prep_l2r(G, Ig, I=None):
    r1, n, r2 = G.shape
    G = _reshape(G, (-1, G.shape[-1]))

    Q, R = _qr(G)
    Ind, B = _maxvol(Q)

    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_l2r(r1, Ig, I)[Ind, :]
    R = Q[Ind, :] @ R
    return G, J, R


def cross_prep_r2l(G, Ig, I=None):
    r1, n, r2 = G.shape
    G = _reshape(G, (G.shape[0], -1)).T

    Q, R = _qr(G)
    Ind, B = _maxvol(Q)

    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(r2, Ig, I)[Ind, :]
    R = (Q[Ind, :] @ R).T
    return G, J, R


def _kron(a, b):
    return np.kron(a, b)


def _maxvol(a):
    return maxvol(a)


def _maxvol_rect(a, kr=1, rf=1, tau=1.1):
    N, u = a.shape
    N_min = min(N, u + kr)
    N_max = min(N, u + kr + rf)

    if N <= u:
        I = np.arange(N, dtype=int)
        B = np.eye(N, dtype=a.dtype)
    else:
        I, B = maxvol_rect(a, tau, N_min, N_max)

    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _qr(a):
    return np.linalg.qr(a)


def _reshape(a, shape):
    return np.reshape(a, shape, order='F')


def _svd(a, full_matrices=False):
    try:
        return np.linalg.svd(a, full_matrices=full_matrices)
    except:
        b = a + 1.E-14 * np.max(np.abs(a)) * np.random.randn(*a.shape)
        return np.linalg.svd(b, full_matrices=full_matrices)

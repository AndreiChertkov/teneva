import numpy as np


from .maxvol import maxvol
from .maxvol import rect_maxvol


def cross(f, Y0, nswp=10, kickrank=2, rf=2):
    d = len(Y0)
    Y = [Y0[i].copy() for i in range(d)]
    Il = [np.empty((1, 0), dtype=int)] + [None for i in range(d)]
    Ir = [None for i in range(d)] + [np.empty((1, 0), dtype=int)]
    Ig = [_reshape(np.arange(G.shape[1], dtype=int), (-1, 1)) for G in Y]

    R = np.ones((1, 1))
    for i in range(d-1):
        Y[i], Il[i+1], R = cross_prep_l2r(Y[i], R, Il[i])

    R = np.ones((1, 1))
    for i in range(d-1, 0, -1):
        Y[i], Ir[i], R = cross_prep_r2l(Y[i], R, Ir[i+1])

    for _ in range(nswp):
        R = np.ones((1, 1))
        for i in range(d):
            G = np.tensordot(R, Y[i], 1)
            y = f(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            Y[i], Il[i+1], R = cross_build_l2r(*G.shape, y, Il[i], kickrank, rf)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            G = np.tensordot(Y[i], R, 1)
            y = f(cross_index_merge(Il[i], Ig[i], Ir[i+1]))
            Y[i], Ir[i], R = cross_build_r2l(*G.shape, y, Ir[i+1], kickrank, rf)
        Y[0] = np.tensordot(R, Y[0], 1)

    return Y


def cross_build_l2r(r1, n, r2, y, I, kickrank, rf):
    G = _reshape(y, (-1, r2))
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, kickrank, rf)
    G = _reshape(B, (r1, n, -1))
    J = cross_index_stack_l2r(n, r1, I)[Ind, :]
    R = Q[Ind, :] @ np.diag(s) @ V
    return G, J, R


def cross_build_r2l(r1, n, r2, y, I, kickrank, rf):
    G = _reshape(y, (r1, -1)).T
    Q, s, V = _svd(G)
    Ind, B = _maxvol_rect(Q, kickrank, rf)
    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(n, r2, I)[Ind, :]
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


def cross_index_stack_l2r(n, r, I):
    e1 = _ones(r)
    e2 = _ones(n)
    e3 = _reshape(np.arange(n, dtype=int), (-1, 1))
    J = _kron(e3, e1)
    if I.size:
        J = np.hstack((_kron(e2, I), J))
    return _reshape(J, (n * r, -1))


def cross_index_stack_r2l(n, r, I):
    e1 = _ones(r)
    e2 = _ones(n)
    e3 = _reshape(np.arange(n, dtype=int), (-1, 1))
    J = _kron(e1, e3)
    if I.size:
        J = np.hstack((J, _kron(I, e2)))
    return _reshape(J, (n * r, -1))


def cross_prep_l2r(G, R0, I0=None):
    r1, n, r2 = G.shape
    G = np.tensordot(R0, G, 1)
    r = G.shape[0]
    G = _reshape(G, (-1, G.shape[-1]))

    Q, R = _qr(G)
    I, B = _maxvol(Q)
    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_l2r(n, r, I0)[I, :]
    R = Q[I, :] @ R
    return G, J, R


def cross_prep_r2l(G, R0, I0=None):
    r1, n, r2 = G.shape
    G = np.tensordot(G, R0, 1)
    r = G.shape[2]
    r2 = r
    G = _reshape(G, (G.shape[0], -1)).T

    Q, R = _qr(G)
    I, B = _maxvol(Q)
    G = _reshape(B.T, (-1, n, r2))
    J = cross_index_stack_r2l(n, r, I0)[I, :]
    R =(Q[I, :] @ R).T
    return G, J, R


def _kron(a, b):
    return np.kron(a, b)


def _maxvol(a):
    return maxvol(a)


def _maxvol_rect(a, kickrank=1, rf=1, tau=1.1):
    N, u = a.shape
    N_min = min(N, u + kickrank)
    N_max = min(N, u + kickrank + rf)

    if N <= u:
        I = np.arange(N, dtype=int)
        B = np.eye(N, dtype=a.dtype)
    else:
        I, B = rect_maxvol(a, tau, N_min, N_max)

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

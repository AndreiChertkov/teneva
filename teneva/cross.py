import numpy as np


from .maxvol import maxvol
from .maxvol import rect_maxvol
from .utils import kron
from .utils import reshape


def _index_merge(i1, i2, i3):
    r1 = i1.shape[0] or 1
    r2 = i2.shape[0]
    r3 = i3.shape[0] or 1
    w1 = kron(np.ones((r3 * r2, 1), dtype=np.int32), i1)
    w2 = np.empty((w1.shape[0], 0))
    if i3.size and r2:
        w2 = kron(i3, np.ones((r1 * r2, 1), dtype=np.int32))
    w3 = kron(kron(np.ones((r3, 1), dtype=np.int32), i2), np.ones((r1, 1), dtype=np.int32))
    return np.hstack((w1, w3, w2))


def _index_stack(n, r, Il, Ir, right=True):
    e1 = np.ones((r, 1), dtype=np.int32)
    e2 = np.ones((n, 1), dtype=np.int32)
    e3 = reshape(np.arange(n, dtype=np.int32), (-1, 1))
    if right:
        J = kron(e1, e3)
        if Ir.size:
            J = np.hstack((J, kron(Ir, e2)))
    else:
        J = kron(e3, e1)
        if Il.size:
            J = np.hstack((kron(e2, Il), J))
    return reshape(J, (n * r, -1))


def _qr_maxvol(G, R0, I0=None, right=True):
    r1, n, r2 = G.shape
    if right:
        G = np.tensordot(G, R0, 1)
        r = G.shape[2]
        r2 = r
        G = reshape(G, (G.shape[0], -1)).T
    else:
        G = np.tensordot(R0, G, 1)
        r = G.shape[0]
        G = reshape(G, (-1, G.shape[-1]))

    Q, R = np.linalg.qr(G)
    I, B = maxvol(Q)
    G = reshape(B.T, (-1, n, r2))
    J = _index_stack(n, r, I0, I0, right)[I, :]
    R = np.dot(Q[I, :], R)
    if right:
        R = R.T
    return G, J, R


def _update_core(G, Il, Ir, fun, kickrank, rf, tau=1.1, right=True):
    r1, n, r2 = G.shape
    p = reshape(np.arange(n, dtype=np.int32), (-1, 1))
    G = fun(_index_merge(Il, p, Ir))

    if right:
        r = r2
        G = reshape(G, (r1, -1)).T
    else:
        r = r1
        G = reshape(G, (-1, r2))

    q, s1, v1 = np.linalg.svd(G, full_matrices=False)
    R = np.diag(s1) @ v1

    N, u = q.shape
    N_min = min(N, u + kickrank)
    N_max = min(N, u + kickrank + rf)
    if N <= u:
        I = np.arange(q.shape[0], dtype=np.int32)
        B = np.eye(q.shape[0], dtype=q.dtype)
    else:
        I, B = rect_maxvol(q, tau, N_min, N_max)

    if right:
        G = reshape(B.T, (-1, n, r2))
    else:
        G = reshape(B, (r1, n, -1))

    R = q[I, :].dot(R)

    J = _index_stack(n, r, Il, Ir, right)[I, :]

    if right:
        R = R.T
    return G, J, R


def cross(f, Y0, nswp=10, kickrank=2, rf=2):
    d = len(Y0)
    Y = [Y0[i].copy() for i in range(d)]
    Il = [np.empty((1, 0), dtype=np.int32)] + [None for i in range(d)]
    Ir = [None for i in range(d)] + [np.empty((1, 0), dtype=np.int32)]

    R = np.ones((1, 1))
    for i in range(d-1):
        Y[i], Il[i+1], R = _qr_maxvol(Y[i], R, Il[i], right=False)

    R = np.ones((1, 1))
    for i in range(d-1, 0, -1):
        Y[i], Ir[i], R = _qr_maxvol(Y[i], R, Ir[i+1], right=True)

    for _ in range(nswp):
        R = np.ones((1, 1))
        for i in range(d):
            Y[i] = np.tensordot(R, Y[i], 1)
            Y[i], Il[i+1], R = _update_core(Y[i], Il[i], Ir[i+1], f, kickrank, rf, right=False)
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            Y[i] = np.tensordot(Y[i], R, 1)
            Y[i], Ir[i], R = _update_core(Y[i], Il[i], Ir[i+1], f, kickrank, rf, right=True)
        Y[0] = np.tensordot(R, Y[0], 1)

    return Y

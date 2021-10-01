import numba as nb
import numpy as np
from copy import deepcopy


from .utils import orthogonalize
from .utils import reshape
from .utils import svd_truncated


def add(Y1, Y2):
    R1 = [1] + [G.shape[2] for G in Y1]
    R2 = [1] + [G.shape[2] for G in Y2]
    N = [G.shape[1] for G in Y1]
    Y = []
    for i, (G1, G2, n) in enumerate(zip(Y1, Y2, N)):
        if i == 0:
            G = np.concatenate([G1, G2], axis=2)
        elif i == len(N) - 1:
            G = np.concatenate([G1, G2], axis=0)
        else:
            R1_l, R1_r = R1[i:i+2]
            R2_l, R2_r = R2[i:i+2]
            Z1 = np.zeros([R1_l, n, R2_r])
            Z2 = np.zeros([R2_l, n, R1_r])
            L1 = np.concatenate([G1, Z1], axis=2)
            L2 = np.concatenate([Z2, G2], axis=2)
            G = np.concatenate([L1, L2], axis=0)
        Y.append(G)
    return Y


def erank(Y):
    """Compute effective rank of the TT-tensor."""
    d = len(Y)
    N = np.array([G.shape[1] for G in Y])
    R = np.array([1] + [G.shape[-1] for G in Y[:-1]] + [1])

    sz = np.dot(N * R[0:d], R[1:])
    b = R[0] * N[0] + N[d - 1] * R[d]
    a = np.sum(N[1:d - 1])
    return (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)


def full(Y):
    A = Y[0].copy()
    for i in range(1, len(Y)):
        A = np.tensordot(A, Y[i], 1)
    return A[0, ..., 0]


def get(Y, n):
    Q = Y[0][0, n[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, n[i], :])
    return Q[0]


def getter(Y, compile=True):
    Y_nb = tuple([np.array(G, order='C') for G in Y])

    @nb.jit(nopython=True)
    def get(n):
        Q = Y_nb[0]
        y = [Q[0, n[0], r2] for r2 in range(Q.shape[2])]
        for i in range(1, len(Y_nb)):
            Q = Y_nb[i]
            R = np.zeros(Q.shape[2])
            for r1 in range(Q.shape[0]):
                for r2 in range(Q.shape[2]):
                    R[r2]+= y[r1] * Q[r1, n[i], r2]
            y = list(R)
        return y[0]

    if compile:
        y = get(np.zeros(len(Y), dtype=int))

    return get


def mean(Y, P=None, norm=True):
    """Compute mean value of the TT-tensor with the given probability."""
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        if P is not None:
            Q = P[i][:n]
        else:
            Q = np.ones(n) / n if norm else np.ones(n)
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def mul(Y1, Y2):
    Y = []
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        Y.append(G)
    return Y


def norm(Y):
    """Compute 2-norm of the given TT-tensor."""
    return np.sqrt(sum(mul(Y, Y)))


def rand(N, R, f=np.random.randn):
    N = np.asanyarray(N, dtype=np.int32)

    d = N.size

    if isinstance(R, (int, float)):
        R = [1] + [int(R)] * (d - 1) + [1]
    R = np.asanyarray(R, dtype=np.int32)

    ps = np.cumsum(np.concatenate(([1], N * R[0:d] * R[1:d+1])))
    ps = ps.astype(np.int32)
    core = f(ps[d] - 1)

    Y = []
    for i in range(d):
        G = core[ps[i]-1:ps[i+1]-1]
        Y.append(G.reshape((R[i], N[i], R[i+1]), order='F'))

    return Y


def show(Y):
    N = [G.shape[1] for G in Y]
    R = [G.shape[0] for G in Y] + [1]
    l = max(int(np.ceil(np.log10(np.max(R)+1))) + 1, 3)
    form_str = '{:^' + str(l) + '}'
    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(n) for n in N])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in N])
    s3 = ''.join([form_str.format(r) for r in R])
    print(f'{s1}\n{s2}\n{s3}\n')


def sum(Y):
    return mean(Y, norm=False)


def truncate(Y, e, rmax=np.iinfo(np.int32).max):
    d = len(Y)
    N = [G.shape[1] for G in Y]
    Z = orthogonalize(Y, d-1)
    delta = e / np.sqrt(d-1) * np.linalg.norm(Z[-1])
    for k in range(d-1, 0, -1):
        M = reshape(Z[k], [Z[k].shape[0], -1])
        L, M = svd_truncated(M, delta, rmax)
        Z[k] = reshape(M, [-1, N[k], Z[k].shape[2]])
        Z[k-1] = np.einsum('ijk,kl', Z[k-1], L, optimize=True)
    return Z

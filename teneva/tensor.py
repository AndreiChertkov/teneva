import numba as nb
import numpy as np


from .utils import orthogonalize
from .utils import reshape
from .utils import svd_truncated
from .utils import unfolding_right


def add(Y1, Y2):
    """Conpute sum of two TT-tensors of the same shape."""
    r1, d1 = ranks_and_dims(Y1)
    r2, d2 = ranks_and_dims(Y2)
    Y = []
    for i, (G1, G2, d) in enumerate(zip(Y1, Y2, d1)):
        if i == 0:
            G = np.concatenate([G1, G2], axis=2)
        elif i == len(d1) - 1:
            G = np.concatenate([G1, G2], axis=0)
        else:
            r1_l, r1_r = r1[i:i+2]
            r2_l, r2_r = r2[i:i+2]
            zeros1 = np.zeros([ r1_l, d, r2_r ])
            zeros2 = np.zeros([ r2_l, d, r1_r ])
            line1 = np.concatenate([G1, zeros1], axis=2)
            line2 = np.concatenate([zeros2, G2], axis=2)
            G = np.concatenate([line1, line2], axis=0)
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


def get(Y, x):
    """Evaluate TT-tensor in x item, i.e. compute Y[x]."""
    Q = Y[0][0, x[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, x[i], :])
    return Q[0]


def getter(Y, compile=True):
    """Return fast get function that evaluate TT-tensor in any x item."""
    Y_nb = tuple([np.array(G, order='F') for G in Y])

    @nb.jit(nopython=True)
    def get(x):
        Q = Y_nb[0]
        y = [Q[0, x[0], r2] for r2 in range(Q.shape[2])]
        for i in range(1, len(Y_nb)):
            Q = Y_nb[i]
            R = np.zeros(Q.shape[2])
            for r1 in range(Q.shape[0]):
                for r2 in range(Q.shape[2]):
                    R[r2]+= y[r1] * Q[r1, x[i], r2]
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
            Q = P[i, 0:n]
        else:
            Q = np.ones(n) / n if norm else np.ones(n)
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def mul(Y1, Y2):
    C = []
    for G1, G2 in zip(Y1, Y2):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        C.append(G)
    return C


def ranks_and_dims(Y):
    r = [1]
    d = []
    for G in Y:
        r += [ G.shape[2] ]
        d += [ G.shape[1] ]

    return np.array(r, dtype=int), np.array(d, dtype=int)


def norm(Y):
    """Compute 2-norm of the given TT-tensor."""
    return np.sqrt(sum(mul(Y, Y)))


def rand(N, R, f=np.random.randn):
    N = np.asanyarray(N, dtype=np.int32)
    d = N.size

    if isinstance(R, (int, float)):
        R = [1] + [int(R)] * (d - 1) + [1]
    R = np.asanyarray(R, dtype=np.int32)

    ps = np.cumsum(np.concatenate(([1], N * R[0:d] * R[1:d +1])))
    ps = ps.astype(np.int32)
    core = f(ps[d] - 1)

    Y = []
    for i in range(d):
        G = core[ps[i]-1:ps[i+1]-1]
        Y.append(G.reshape((R[i], N[i], R[i+1]), order='F'))
    return Y


def sum(Y):
    return mean(Y, norm=False)


def truncate(Y, e, rmax=np.iinfo(np.int32).max):
    d = len(Y)
    N = [G.shape[1] for G in Y]
    orthogonalize(Y, d-1)
    delta = e / np.sqrt(d-1) * np.linalg.norm(Y[-1])
    for k in range(d-1, 0, -1):
        M = reshape(Y[k], [Y[k].shape[0], -1])
        L, M = svd_truncated(M, delta, rmax)
        Y[k] = reshape(M, [-1, N[k], Y[k].shape[2]])
        Y[k-1] = np.einsum('ijk,kl', Y[k-1], L, optimize=True)
    return Y


def repr_tt(Y):
    dims  = [i.shape[1] for i in Y]
    ranks = [i.shape[0] for i in Y] + [1]

    max_rank = np.max(ranks)
    max_len = int(np.ceil(np.log10(max_rank))) + 1
    max_len = max(max_len, 3)
    #form_str = "{:^" + str(max_len) + "d}"
    form_str = "{:^" + str(max_len) + "}"

    r0 = ' '*(max_len//2)
    r1 = r0 + ''.join([form_str.format(i) for i in dims])
    r2 = r0 + ''.join([form_str.format('/ \\') for i in dims])
    r3 = ''.join([form_str.format(i) for i in ranks])

    print(f"{r1}\n{r2}\n{r3}\n")

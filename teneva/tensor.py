import numba as nb
import numpy as np


from .utils import orthogonalize
from .utils import svd_truncated
from .utils import unfolding_right


def erank(Y):
    d = len(Y)
    N = np.array([G.shape[1] for G in Y])
    R = np.array([1] + [G.shape[-1] for G in Y[:-1]] + [1])

    sz = np.dot(N * R[0:d], R[1:])
    b = R[0] * N[0] + N[d - 1] * R[d]
    if d is 2:
        er = sz * 1. / b
    else:
        a = np.sum(N[1:d - 1])
        er = (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)
    return er


def get(Y, x):
    Q = Y[0][0, x[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, x[i], :])
    return Q[0]


def getter(Y, compile=True):
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
        y = get(np.zeros(len(Y_nb), dtype=int)) # Compile

    return get


def mean(Y, P=None, norm=True):
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        if P is not None:
            Q = P[i, 0:n]
        else:
            Q = np.ones(n) / n if norm else np.ones(n)
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def mul(A, B):
    C = []
    for G1, G2 in zip(A, B):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        C.append(G)
    return C


def ranks_and_dims(cores):
    r = [1]
    d = []
    for c in cores:
        r += [ c.shape[2] ]
        d += [ c.shape[1] ]

    return np.array(r, dtype=int), np.array(d, dtype=int)



def sum(cores1, cores2):
    """
    поэлементная сумма тензоров в ТТ
    не оптимизирована, но и вызывается редко
    """
    r1, d1 = ranks_and_dims(cores1)
    r2, d2 = ranks_and_dims(cores2)
    assert (d1 == d2).all(), "Wrong dimensions"
    cores = []
    n_1 = len(d1) - 1
    for i, (c1, c2, d) in enumerate(zip(cores1, cores2, d1)):
        if i==0:
            new_core = np.concatenate([c1, c2], axis=2)
            cores.append(new_core)
            continue

        if i==n_1:
            new_core = np.concatenate([c1, c2], axis=0)
            cores.append(new_core)
            continue

        r1_l, r1_r = r1[i:i+2]
        r2_l, r2_r = r2[i:i+2]

        zeros1 = np.zeros([ r1_l, d, r2_r ])
        zeros2 = np.zeros([ r2_l, d, r1_r ])
        line1 = np.concatenate([c1, zeros1], axis=2)
        line2 = np.concatenate([zeros2, c2], axis=2)
        new_core = np.concatenate([line1, line2], axis=0)
        cores.append(new_core)

    return cores


def norm(Y):
    return np.sqrt(recap(mul(Y, Y)))


def rand(N, R, f=np.random.randn):
    N = np.asanyarray(N, dtype=np.int32)
    d = N.size
    if d < 3:
        raise ValueError('Dimension should be at least 3.')

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


def recap(Y):
    return mean(Y, norm=False)


def truncate(Y, e, rmax=np.iinfo(np.int32).max):
    d = len(Y)
    N = [G.shape[1] for G in Y]
    orthogonalize(Y, d-1)
    delta = e / np.sqrt(d-1) * np.linalg.norm(Y[-1])
    for mu in range(d-1, 0, -1):
        M = unfolding_right(Y[mu])
        L, M = svd_truncated(M, delta=delta, rmax=rmax, left_ortho=False)
        Y[mu] = np.reshape(M, [-1, N[mu], Y[mu].shape[2]], order='F')
        Y[mu-1] = np.einsum('ijk,kl', Y[mu-1], L, optimize=True)
    return Y



def repr_tt(cores):
    dims  = [i.shape[1] for i in cores]
    ranks = [i.shape[0] for i in cores] + [1]

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

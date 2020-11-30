import numba as nb
import numpy as np


def erank(Y):
    d = len(Y)
    n = np.array([G.shape[1] for G in Y])
    r = np.array([1] + [G.shape[-1] for G in Y[:-1]] + [1])

    sz = np.dot(n * r[0:d], r[1:])
    if sz == 0:
        er = 0e0
    else:
        b = r[0] * n[0] + n[d - 1] * r[d]
        if d is 2:
            er = sz * 1.0 / b
        else:
            a = np.sum(n[1:d - 1])
            er = (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)
    return er


def get(Y, x):
    Q = Y[0][:, x[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('rq,qp->rp', Q, Y[i][:, x[i], :])
    return Q[0, 0]


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


def mean(Y, P=None):
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        Q = P[i, 0:n] if P is not None else np.ones(n) / n
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def mul(A, B):
    C = []
    for G1, G2 in zip(A, B):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        C.append(G)
    return C


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

    ps = np.cumsum(np.concatenate(([1], N * R[0:d] * R[1:d +1]))).astype(np.int32)
    core = f(ps[d] - 1)

    Y = []
    for i in range(d):
        G = core[ps[i] - 1:ps[i + 1] - 1]
        Y.append(G.reshape((R[i], N[i], R[i + 1]), order='F'))
    return Y


def recap(Y):
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        R = R @ np.einsum('rmq,m->rq', Y[i], np.ones(n))
    return R[0, 0]


def truncate(Y, e, rmax=np.iinfo(np.int32).max):

    def reshape(a, sz):
        return np.reshape(a, sz, order="F")

    def left_unfolding(core):  # rs[mu] ns[mu] x rs[mu+1]
        return reshape(core, [-1, core.shape[2]])

    def right_unfolding(core):  # rs[mu] x ns[mu] rs[mu+1]
        return reshape(core, [core.shape[0], -1])

    def left_orthogonalize(cores, mu, recursive=False):
        assert 0 <= mu < len(cores)-1
        coreL = left_unfolding(cores[mu])
        Q, R = np.linalg.qr(coreL, mode='reduced')
        cores[mu] = reshape(Q, cores[mu].shape[:-1] + (Q.shape[1], ))
        rightcoreR = right_unfolding(cores[mu+1])
        cores[mu+1] = reshape(np.dot(R, rightcoreR), (R.shape[0], ) + cores[mu+1].shape[1:])
        if recursive and mu < len(cores)-2:
            left_orthogonalize(cores, mu+1)
        return R

    def right_orthogonalize(cores, mu, recursive=False):
        assert 1 <= mu < len(cores)
        coreR = right_unfolding(cores[mu])
        L, Q = scipy.linalg.rq(coreR, mode='economic', check_finite=False)
        cores[mu] = reshape(Q, (Q.shape[0], ) + cores[mu].shape[1:])
        leftcoreL = left_unfolding(cores[mu-1])
        cores[mu-1] = reshape(np.dot(leftcoreL, L), cores[mu-1].shape[:-1] + (L.shape[1], ))
        # cores[mu-1] = reshape(np.dot(leftcoreL, R), cores[mu-1].shape)
        if recursive and mu > 1:
            right_orthogonalize(cores, mu-1)
        return L

    def orthogonalize(cores, mu):
        L = np.array([[1]])
        R = np.array([[1]])
        for i in range(0, mu):
            R = left_orthogonalize(cores, i)
        for i in range(len(cores)-1, mu, -1):
            L = right_orthogonalize(cores, i)
        return R, L

    def truncated_svd(M, delta=None, eps=None, rmax=None, left_ortho=True, verbose=False):
        if delta is not None and eps is not None:
            raise ValueError('Provide either `delta` or `eps`')
        if delta is None and eps is not None:
            delta = eps*np.linalg.norm(M)
        if delta is None and eps is None:
            delta = 0
        if rmax is None:
            rmax = np.iinfo(np.int32).max
        assert rmax >= 1

        if M.shape[0] <= M.shape[1]:
            cov = M.dot(M.T)
            singular_vectors = 'left'
        else:
            cov = M.T.dot(M)
            singular_vectors = 'right'

        if np.linalg.norm(cov) < 1e-14:
            return np.zeros([M.shape[0], 1]), np.zeros([1, M.shape[1]])

        w, v = np.linalg.eigh(cov)
        w[w < 0] = 0
        w = np.sqrt(w)
        svd = [v, w]
        # Sort eigenvalues and eigenvectors in decreasing importance
        idx = np.argsort(svd[1])[::-1]
        svd[0] = svd[0][:, idx]
        svd[1] = svd[1][idx]

        S = svd[1]**2
        where = np.where(np.cumsum(S[::-1]) <= delta**2)[0]
        if len(where) == 0:
            rank = max(1, int(np.min([rmax, len(S)])))
        else:
            rank = max(1, int(np.min([rmax, len(S) - 1 - where[-1]])))
        left = svd[0]
        left = left[:, :rank]

        if singular_vectors == 'left':
            if left_ortho:
                M2 = left.T.dot(M)
            else:
                M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
                left = left*svd[1][:rank]
        else:
            if left_ortho:
                M2 = M.dot(left * (1. / svd[1][:rank])[np.newaxis, :])
                left, M2 = M2, left.dot(np.diag(svd[1][:rank])).T
            else:
                M2 = M.dot(left)
                left, M2 = M2, left.T

        return left, M2

    N = len(Y)
    shape = [G.shape[1] for G in Y]
    cores = Y
    orthogonalize(cores, N-1)
    delta = e / np.sqrt(N - 1) * np.linalg.norm(cores[-1])
    for mu in range(N-1, 0, -1):
        M = right_unfolding(cores[mu])
        left, M = truncated_svd(M, delta=delta, rmax=rmax, left_ortho=False)
        cores[mu] = np.reshape(M, [-1, shape[mu], cores[mu].shape[2]], order='F')
        cores[mu-1] = np.einsum('ijk,kl', cores[mu-1], left, optimize=True)

    return Y

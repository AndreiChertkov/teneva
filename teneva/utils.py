import itertools
import numpy as np


def confidence(F, alpha=.05):
    r"""
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array_like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| > \epsilon \right) \leq 2e^{-2n\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.

    """
    eps = np.sqrt(np.log(2./alpha) / (2 * len(F)))
    return np.clip(F - eps, 0, 1), np.clip(F + eps, 0, 1)


def core_one(n, r):
    return np.kron(np.ones([1, n, 1]), np.eye(r)[:, None, :])


def get_cdf(x):
    _x = np.array(x, copy=True)
    _x.sort()
    _y = np.linspace(1./len(_x), 1, len(_x))

    _x = np.r_[-np.inf, _x]
    _y = np.r_[0, _y]

    def cdf(z):
        return _y[np.searchsorted(_x, z, 'right') - 1]

    return cdf


def get_partial(Y, n):
    Q = Y[0][0, n[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, n[i], :])
    return Q


def ind2poi(I, a, b, n):
    """Transforms multiindices (samples) into points of the uniform grid."""
    if isinstance(a, list): a = np.array(a)
    if isinstance(b, list): b = np.array(b)
    if isinstance(n, list): n = np.array(n)

    if len(I.shape) == 1:
        # If we have only one multiindex
        t = I * 1. / (n - 1)
        x = t * (b - a) + a
        return x

    A = np.repeat(a.reshape((1, -1)), I.shape[0], axis=0)
    B = np.repeat(b.reshape((1, -1)), I.shape[0], axis=0)
    N = np.repeat(n.reshape((1, -1)), I.shape[0], axis=0)
    T = I * 1. / (N - 1)
    X = T * (B - A) + A
    return X


def ind2str(i):
    """Transforms array of int like [1, 2, 3] into string like '1-2-3'."""
    return '-'.join([str(int(v)) for v in i])


def kron(a, b):
    return np.kron(a, b)


def lhs(shape, samples):
    d = len(shape)
    indices = np.empty((samples, d), dtype=int)
    for i, sh in enumerate(shape):
        part1 = np.repeat(np.arange(sh), samples // sh)
        part2 = np.random.choice(sh, samples-len(part1), replace=False)
        indices[:, i] = np.concatenate([part1, part2])
        np.random.shuffle(indices[:, i])
    return indices


def orthogonalize(Z, k):
    # Z = [G.copy() for G in Y]
    L = np.array([[1.]])
    R = np.array([[1.]])
    for i in range(0, k):
        G = reshape(Z[i], [-1, Z[i].shape[2]])
        Q, R = np.linalg.qr(G, mode='reduced')
        Z[i] = reshape(Q, Z[i].shape[:-1] + (Q.shape[1], ))
        G = reshape(Z[i+1], [Z[i+1].shape[0], -1])
        Z[i+1] = reshape(np.dot(R, G), (R.shape[0], ) + Z[i+1].shape[1:])
    for i in range(len(Z)-1, k, -1):
        G = reshape(Z[i], [Z[i].shape[0], -1])
        L, Q = scipy.linalg.rq(G, mode='economic', check_finite=False)
        Z[i] = reshape(Q, (Q.shape[0], ) + Z[i].shape[1:])
        G = reshape(Z[i-1], [-1, Z[i-1].shape[2]])
        Z[i-1] = reshape(np.dot(G, L), Z[i-1].shape[:-1] + (L.shape[1], ))
    return Z


def reshape(a, sz):
    return np.reshape(a, sz, order='F')


def second_order_2_TT(A, i, j, shapes):
    if i > j: # Так не должно быть
        j, i = i, j
        A = A.T

    U, V = skeleton(A)
    r = U.shape[1]

    core1 = U.reshape(1, U.shape[0], r)
    core2 = V.reshape(r, V.shape[1], 1)

    cores = []
    for num, n in enumerate(shapes):
        if num < i or num > j:
            cores.append(np.ones([1, n, 1]))
        if num == i:
            cores.append(core1)
        if num == j:
            cores.append(core2)
        if i < num < j:
            cores.append(core_one(n, r))

    return cores


def skeleton(a, eps=1.E-10, r=int(1e12), hermitian=False):
    u, s, v = np.linalg.svd(a, full_matrices=False,
        compute_uv=True, hermitian=hermitian)
    r = min(r, sum(s>eps))
    un = u[:, :r]
    sn = np.diag(np.sqrt(s[:r]))
    vn = v[:r]
    return un @ sn, sn @ vn


def sum_many(tensors, e=1.E-10, rmax=None, freq_trunc=15):
    cores = tensors[0]
    for i, t in enumerate(tensors[1:]):
        cores = teneva.add(cores, t)
        if (i+1) % freq_trunc == 0:
            cores = teneva.truncate(cores, e=e)
    return teneva.truncate(cores, e=e, rmax=rmax)


def svd_truncated(M, delta, rmax=None):
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
        M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
        left = left*svd[1][:rank]
    else:
        M2 = M.dot(left)
        left, M2 = M2, left.T

    return left, M2


def str2ind(s):
    """Transforms string like '1-2-3' into array of int like [1, 2, 3]."""
    return np.array([int(v) for v in s.split('-')], dtype=int)


def tt_sample(shape, k):
    def one_mode(sh1, sh2, rng):
        res = []
        if len(sh2) == 0:
            lhs_1 = lhs(sh1, k)
            for n in range(rng):
                for i in lhs_1:
                    res.append(np.concatenate([i, [n]]))
            len_1, len_2 = len(lhs_1), 1
        elif len(sh1) == 0:
            lhs_2 = lhs(sh2, k)
            for n in range(rng):
                for j in lhs_2:
                    res.append(np.concatenate([[n], j]))
            len_1, len_2 = 1, len(lhs_2)
        else:
            lhs_1 = lhs(sh1, k)
            lhs_2 = lhs(sh2, k)
            for n in range(rng):
                for i, j in itertools.product(lhs_1,  lhs_2):
                    res.append(np.concatenate([i, [n], j]))
            len_1, len_2 = len(lhs_1), len(lhs_2)
        return res, len_1, len_2

    idx = [0]
    idx_many = []
    pnts_many = []

    for i in range(len(shape)):
        pnts, len_1, len_2 = one_mode(shape[:i], shape[i+1:], shape[i])
        pnts_many.append(pnts)
        idx.append(idx[-1] + len(pnts))
        idx_many.append(len_2)

    return np.vstack(pnts_many), np.array(idx), np.array(idx_many)


def tt_svd(t, eps=1E-10, max_r=int(1e12)):
    A = t = np.asanyarray(t)
    r = 1
    res = []
    for sh in t.shape[:-1]:
        A = A.reshape(r*sh, -1)
        G, A = skeleton(A, eps=eps, r=max_r)
        G = G.reshape(r, sh, -1)
        res.append(G)
        r = G.shape[-1]
    res.append(A.reshape(r, t.shape[-1], 1))
    return res


def tt_svd_incomplete(I, Y, idx, idx_many, rank, eps_skel=1e-10):
    shapes = np.max(I, axis=0) + 1
    d = len(shapes)

    Y_curr = Y[idx[0]:idx[1]]
    Y_curr = Y_curr.reshape(shapes[0], -1, order='C')
    Y_curr, _ = skeleton(Y_curr, r=rank, eps=eps_skel)
    cores = [Y_curr[None, ...]]

    for mode in range(1, d):
        # The mode-th TT-core will have the shape r0 x n x r1
        r0 = cores[-1].shape[-1]
        r1 = rank if mode < d-1 else 1
        n = shapes[mode]

        I_curr = I[idx[mode]:idx[mode+1]]
        M = np.array([get_partial(cores[:mode], i) for i in I_curr[::idx_many[mode], :mode]])

        Y_curr = Y[idx[mode]:idx[mode+1]].reshape(-1, idx_many[mode], order='C')
        if Y_curr.shape[1] > r1:
            Y_curr, _ = skeleton(Y_curr, r=r1)
        r1 = Y_curr.shape[1]

        core = np.empty([r0, n, r1])
        step = Y_curr.shape[0] // n
        for i in range(n):
            A = M[i*step:(i+1)*step]
            b = Y_curr[i*step:(i+1)*step]
            core[:, i, :] = np.linalg.lstsq(A, b, rcond=-1)[0]
        cores.append(core)

    return cores

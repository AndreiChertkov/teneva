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


def get_cdf(x):
    _x = np.array(x, copy=True)
    _x.sort()
    _y = np.linspace(1./len(_x), 1, len(_x))

    _x = np.r_[-np.inf, _x]
    _y = np.r_[0, _y]

    def cdf(z):
        return _y[np.searchsorted(_x, z, 'right') - 1]

    return cdf


def kron(a, b):
    return np.kron(a, b)


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

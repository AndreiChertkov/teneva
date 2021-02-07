import numpy as np
from scipy.interpolate import interp1d


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
    nobs = len(F)
    epsilon = np.sqrt(np.log(2./alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)

    return lower, upper


def get_cdf(x):
    _x = np.array(x, copy=True)
    _x.sort()
    _y = np.linspace(1./len(_x), 1, len(_x))

    _x = np.r_[-np.inf, _x]
    _y = np.r_[0, _y]

    def cdf(z):
        tind = np.searchsorted(_x, z, 'right') - 1
        return _y[tind]

    return cdf


def kron(a, b):
    return np.kron(a, b)


def orthogonalize(Y, mu):
    L = np.array([[1]])
    R = np.array([[1]])
    for i in range(0, mu):
        R = orthogonalize_left(Y, i)
    for i in range(len(Y)-1, mu, -1):
        L = orthogonalize_right(Y, i)
    return R, L


def orthogonalize_left(Y, mu):
    assert 0 <= mu < len(Y)-1
    G = unfolding_left(Y[mu])
    Q, R = np.linalg.qr(G, mode='reduced')
    Y[mu] = reshape(Q, Y[mu].shape[:-1] + (Q.shape[1], ))
    G = unfolding_right(Y[mu+1])
    Y[mu+1] = reshape(np.dot(R, G), (R.shape[0], ) + Y[mu+1].shape[1:])
    return R


def orthogonalize_right(Y, mu):
    assert 1 <= mu < len(Y)
    G = unfolding_right(Y[mu])
    L, Q = scipy.linalg.rq(G, mode='economic', check_finite=False)
    Y[mu] = reshape(Q, (Q.shape[0], ) + Y[mu].shape[1:])
    G = unfolding_left(Y[mu-1])
    Y[mu-1] = reshape(np.dot(G, L), Y[mu-1].shape[:-1] + (L.shape[1], ))
    return L


def reshape(a, sz):
    return np.reshape(a, sz, order = 'F')


def svd_truncated(M, delta=None, eps=None, rmax=None, left_ortho=True):
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


def unfolding_left(G):
    return reshape(G, [-1, G.shape[2]])


def unfolding_right(G):
    return reshape(G, [G.shape[0], -1])

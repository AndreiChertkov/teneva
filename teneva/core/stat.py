"""Package teneva, module core.stat: helper functions for processing statistics.

This module contains the helper functions for processing statistics, including
computation of the CDF function and its confidence bounds, as well as sampling
from the TT-tensor.

"""
import numpy as np
import teneva


def cdf_confidence(x, alpha=0.05):
    """Construct a Dvoretzky-Kiefer-Wolfowitz confidence band for the CDF.

    Args:
        x (np.ndarray): the empirical distribution in the form of 1D np.ndarray
            of length "m".
        alpha (float): "alpha" for the "(1 - alpha)" confidence band.

    Returns:
        [np.ndarray, np.ndarray]: CDF lower and upper bounds in the form of 1D
        np.ndarray of the length "m".

    """
    eps = np.sqrt(np.log(2. / alpha) / (2 * len(x)))
    return np.clip(x - eps, 0, 1), np.clip(x + eps, 0, 1)


def cdf_getter(x):
    """Build the getter for CDF.

    Args:
        x (list or np.ndarray): one-dimensional points.

    Returns:
        function: the function that computes CDF values. Its input may be one
        point (float) or a set of points (1D np.ndarray). The output
        (corresponding CDF value/values) will have the same type.

    """
    x = np.array(x, copy=True)
    x.sort()
    y = np.linspace(1./len(x), 1, len(x))

    x = np.r_[-np.inf, x]
    y = np.r_[0, y]

    def cdf(z):
        return y[np.searchsorted(x, z, 'right') - 1]

    return cdf


def sample_ind_rand(Y, m=1):
    """Sample random multi-indices according to given probability TT-tensor.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int, float): number of samples.
    Returns:
        np.ndarray: generated multi-indices for the tensor in the form
        of array of the shape [m, d], where "d" is the dimension of the tensor.

    """
    d = len(Y)
    res = np.zeros((m, d), dtype=np.int32)
    phi = [None]*(d+1)
    phi[-1] = np.ones(1)
    for i in range(d-1, 0, -1):
        phi[i] = np.sum(Y[i], axis=1) @ phi[i+1]


    p = Y[0] @ phi[1]
    p = p.flatten()
    p = np.maximum(p, 0)
    p = p/p.sum()
    ind = np.random.choice(Y[0].shape[1], m, p=p)
    phi[0] = Y[0][0, ind, :] # ind here is an array even if m=1
    res[:, 0] = ind
    for i, c in enumerate(Y[1:], start=1):
        p = np.einsum('ma,aib,b->mi', phi[i-1], Y[i], phi[i+1])
        p = np.maximum(p, 0)
        ind = np.array([np.random.choice(c.shape[1], p=pi/pi.sum()) for pi in p])
        res[:, i] = ind
        phi[i] = np.einsum("il,lij->ij", phi[i-1], c[:, ind])

    return res


def sample_ind_rand_square(Y, m=1, unique=True, m_fact=5, max_rep=100, float_cf=None):
    """Sample random multi-indices according to given probability TT-tensor squaring it.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int, float): number of samples.
        unique (bool): if True, then unique multi-indices will be generated.
        m_fact (int): scale factor to find enough unique samples.
        max_rep (int): number of restarts to find enough unique samples.
        float_cf

    Returns:
        np.ndarray: generated multi-indices for the tensor in the form
        of array of the shape [m, d], where "d" is the dimension of the tensor.

    """
    err_msg = 'Can not generate the required number of samples'

    d = len(Y)
    Z, p = teneva.orthogonalize(Y, 0, use_stab=True)

    G = Z[0]
    r1, n, r2 = G.shape

    if float_cf is not None:
        n, nold = n * float_cf, n
        G = _extend_core(G, n)

    Q = G.reshape(n, r2)

    Q, I1 = _sample_core_first(Q, teneva._range(n), m_fact*m if unique else m)
    I = np.empty([I1.shape[0], d])
    I[:, 0] = I1[:, 0]

    for di, G in enumerate(Z[1:], start=1):
        r1, n, r2 = G.shape

        if float_cf is not None:
            n, nold = n * float_cf, n
            G = _extend_core(G, n)

        Qtens = np.einsum('kr,riq->kiq', Q, G, optimize='optimal')
        Q = np.empty([Q.shape[0], r2])

        for im, qm, qnew in zip(I, Qtens, Q):
            norms = np.sum(qm**2, axis=1)
            norms /= norms.sum()

            i_cur = im[di] = np.random.choice(n, size=1, p=norms)
            qnew[:] = qm[i_cur]

    if unique:
        I = np.unique(I, axis=0)
        if I.shape[0] < m:
            if max_rep < 0 or m_fact > 1000000:
                raise ValueError(err_msg)
            return sample_ind_rand(Y, m, True, 2*m_fact, max_rep-1,
                float_cf=float_cf)
        else:
            np.random.shuffle(I)

    I = I[:m]

    if I.shape[0] != m:
        raise ValueError(err_msg)

    if float_cf is not None:
        I = I / float_cf
    else:
        I = I.astype(int)

    return I


def _extend_core(G, n):
    r1, nold, r2 = G.shape
    Gn = np.empty([r1, n, r2])
    for i1 in range(r1):
        for i2 in range(r2):
            Gn[i1, :, i2] = np.interp(np.arange(n)*(nold - 1)/(n - 1),
                range(nold), G[i1, :, i2])
    return Gn


def _sample_core_first(Q, I, m):
    n = Q.shape[0]

    norms = np.sum(Q**2, axis=1)
    norms /= norms.sum()

    ind = np.random.choice(n, size=m, p=norms, replace=True)

    return Q[ind, :], I[ind, :]

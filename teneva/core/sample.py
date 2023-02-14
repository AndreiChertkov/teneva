"""Package teneva, module core.sample: random sampling for/from the TT-tensor.

This module contains functions for sampling from the TT-tensor.

"""
import numpy as np
import teneva


def sample_ind_rand(Y, m=1, unsert=1e-10):
    """Sample random multi-indices according to given probability TT-tensor.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int, float): number of samples.
    Returns:
        np.ndarray: generated multi-indices for the tensor in the form
        of array of the shape [m, d], where d is the dimension of the tensor.

    """
    d = len(Y)
    res = np.zeros((m, d), dtype=np.int32)
    phi = [None]*(d+1)
    phi[-1] = np.ones(1)
    for i in range(d-1, 0, -1):
        phi[i] = np.sum(Y[i], axis=1) @ phi[i+1]


    p = Y[0] @ phi[1]
    p = p.flatten()
    p += unsert
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
        of array of the shape [m, d], where d is the dimension of the tensor.

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

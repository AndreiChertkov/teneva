"""Package teneva, module core.sample: random sampling for/from the TT-tensor.

This module contains functions for sampling from the TT-tensor and for
generation of random multi-indices and points for learning.

"""
import itertools
import numpy as np
import teneva


def sample(Y, m=1, unsert=1e-10):
    """Sample according to given probability TT-tensor.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int): number of samples.
        unsert (float): noise parameter.
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


def sample_square(Y, m=1, unique=True, m_fact=5, max_rep=100, float_cf=None):
    """Sample according to given probability TT-tensor (squaring it).

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        m (int): number of samples.
        unique (bool): if True, then unique multi-indices will be generated.
        m_fact (int): scale factor to find enough unique samples.
        max_rep (int): number of restarts to find enough unique samples.
        float_cf (float): special parameter (TODO: check).

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
            return sample(Y, m, True, 2*m_fact, max_rep-1, float_cf=float_cf)
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


def sample_lhs(n, m):
    """Generate LHS samples (multi-indices) for the tensor of the given shape.

    Args:
        n (list, np.ndarray): tensor size for each dimension (list or
            np.ndarray of int/float of the length d, where d is the
            dimension of the tensor).
        m (int, float): number of samples.

    Returns:
        np.ndarray: generated multi-indices for the tensor in the form of array
        of the shape [m, d], where d is the dimension of the tensor.

    """
    n = np.asanyarray(n, dtype=int)
    m = int(m)
    d = n.shape[0]

    I = np.empty((m, d), dtype=int)
    for i, k in enumerate(n):
        I1 = np.repeat(np.arange(k), m // k)
        I2 = np.random.choice(k, m-len(I1), replace=False)
        I[:, i] = np.concatenate([I1, I2])
        np.random.shuffle(I[:, i])

    return I


def sample_tt(n, r=4):
    """Generate special samples for the tensor of the shape n.

    Generate special samples (multi-indices) for the tensor, which are the best
    (in many cases) for the subsequent construction of the TT-tensor.

    Args:
        n (list, np.ndarray): tensor size for each dimension (list or
            np.ndarray of int/float of the length d).
        r (int): expected TT-rank of the tensor. The number of generated
            samples will be selected according to this value.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): generated multi-indices for the
        tensor in the form of array of the shape [samples, d], starting
        poisitions in generated samples for the corresponding dimensions in the
        form of array of the shape [d+1] and numbers of points for the right
        unfoldings in generated samples in the form of array of the shape [d].

    Note:
        The resulting number of samples will be chosen adaptively based on the
        specified expected TT-rank (r).

    """
    def one_mode(sh1, sh2, rng):
        res = []
        if len(sh2) == 0:
            lhs_1 = sample_lhs(sh1, r)
            for n in range(rng):
                for i in lhs_1:
                    res.append(np.concatenate([i, [n]]))
            len_1, len_2 = len(lhs_1), 1
        elif len(sh1) == 0:
            lhs_2 = sample_lhs(sh2, r)
            for n in range(rng):
                for j in lhs_2:
                    res.append(np.concatenate([[n], j]))
            len_1, len_2 = 1, len(lhs_2)
        else:
            lhs_1 = sample_lhs(sh1, r)
            lhs_2 = sample_lhs(sh2, r)
            for n in range(rng):
                for i, j in itertools.product(lhs_1, lhs_2):
                    res.append(np.concatenate([i, [n], j]))
            len_1, len_2 = len(lhs_1), len(lhs_2)
        return res, len_1, len_2

    I, idx, idx_many = [], [0], []
    for i in range(len(n)):
        pnts, len_1, len_2 = one_mode(n[:i], n[i+1:], n[i])
        I.append(pnts)
        idx.append(idx[-1] + len(pnts))
        idx_many.append(len_2)

    return np.vstack(I), np.array(idx), np.array(idx_many)


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

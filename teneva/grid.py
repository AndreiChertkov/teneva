import itertools
import numpy as np


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


def ind2poi_cheb(I, a, b, n):
    T = np.cos(np.pi * I / (n - 1))
    X = T * (b - a) / 2 + (b + a) / 2
    return X


def ind2str(i):
    """Transforms array of int like [1, 2, 3] into string like '1-2-3'."""
    return '-'.join([str(int(v)) for v in i])


def sample_lhs(n, m):
    """Build m LHS samples (indices) for the tensor of the shape n."""
    if isinstance(n, list): n = np.array(n)

    d = len(n)
    I = np.empty((m, d), dtype=int)
    for i, sh in enumerate(n):
        I1 = np.repeat(np.arange(sh), m // sh)
        I2 = np.random.choice(sh, m-len(I1), replace=False)
        I[:, i] = np.concatenate([I1, I2])
        np.random.shuffle(I[:, i])
    return I


def sample_tt(n, m):
    """Generate special m samples (indices) for the tensor of the shape n."""
    def one_mode(sh1, sh2, rng):
        res = []
        if len(sh2) == 0:
            lhs_1 = lhs(sh1, m)
            for n in range(rng):
                for i in lhs_1:
                    res.append(np.concatenate([i, [n]]))
            len_1, len_2 = len(lhs_1), 1
        elif len(sh1) == 0:
            lhs_2 = lhs(sh2, m)
            for n in range(rng):
                for j in lhs_2:
                    res.append(np.concatenate([[n], j]))
            len_1, len_2 = 1, len(lhs_2)
        else:
            lhs_1 = lhs(sh1, m)
            lhs_2 = lhs(sh2, m)
            for n in range(rng):
                for i, j in itertools.product(lhs_1,  lhs_2):
                    res.append(np.concatenate([i, [n], j]))
            len_1, len_2 = len(lhs_1), len(lhs_2)
        return res, len_1, len_2

    idx = [0]
    idx_many = []
    I = []

    for i in range(len(n)):
        pnts, len_1, len_2 = one_mode(n[:i], n[i+1:], n[i])
        I.append(pnts)
        idx.append(idx[-1] + len(pnts))
        idx_many.append(len_2)

    return np.vstack(I), np.array(idx), np.array(idx_many)


def str2ind(s):
    """Transforms string like '1-2-3' into array of int like [1, 2, 3]."""
    return np.array([int(v) for v in s.split('-')], dtype=int)

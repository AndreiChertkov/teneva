"""Package teneva, module core.grid: functions to construct the grid.

This module contains a set of functions for creating and transforming
multidimensional grids for discretizing functions on uniform and Chebyshev
grids, as well as a number of methods for tensor sampling.

"""
import itertools
import numpy as np


def grid_flat(n):
    """Compute the multiindices for the full multidimensional grid.

    Args:
        n (list): number of grid points for each dimension (list or np.ndarray
            of length "d", where "d" is a number of dimensions).

    Returns:
        np.ndarray: multiindices for the full (flatten) grid (it is 2D array of
            the shape d x n^d).

    """
    d = len(n)
    I = [np.arange(k).reshape(1, -1) for k in n]
    I = np.meshgrid(*I, indexing='ij')
    I = np.array(I).reshape((d, -1), order='F')
    return I


def grid_prep_opts(a, b, n=None, d=None, reps=None):
    """Helper function that prepare grid parameters."""
    for item in [a, b, n]:
        if isinstance(item, (list, np.ndarray)):
            if d is None:
                d = len(item)
            elif d != len(item):
                raise ValueError('Invalid grid options a/b/n')

    if not d or d <= 0:
        raise ValueError('Invalid grid options a/b/n (can not recover d)')

    if a is not None:
        if isinstance(a, (int, float)):
            a = [a] * d
        if isinstance(a, list):
            a = np.array(a, dtype=float)
        if reps is not None:
            a = np.repeat(a.reshape((1, -1)), reps, axis=0)


    if b is not None:
        if isinstance(b, (int, float)):
            b = [b] * d
        if isinstance(b, list):
            b = np.array(b, dtype=float)
        if reps is not None:
            b = np.repeat(b.reshape((1, -1)), reps, axis=0)

    if n is not None:
        if isinstance(n, (int, float)):
            n = [n] * d
        if isinstance(n, list):
            n = np.array(n, dtype=int)
        if reps is not None:
            n = np.repeat(n.reshape((1, -1)), reps, axis=0)

    if n is None:
        return a, b
    else:
        return a, b, n


def ind2poi(I, a, b, n, kind='uni'):
    """Transforms multiindices (samples) into points of the spatial grid.

    This function may be used for function approximation by TT-ALS, TT-ANOVA or
    TT-CAM (requested tensor indices should be transformed into spatial grid
    points and then used as an argument for the target function).

    Args:
        I (np.ndarray): multiindices for the tensor in the form of array of the
            shape [samples, d], where "samples" is the number of samples and
            "d" is the dimension of the tensor. For the case of only one
            sample, it may be 1D array or list of length "d".
        a (list): grid lower bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the lower bounds for each
            dimension will be the same.
        b (list): grid upper bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the upper bounds for each
            dimension will be the same.
        n (list): tensor size for each dimension (list or np.ndarray of length
            "d"). It may be also float, then the size for each dimension will be
            the same.
        kind (str): the grid type, it may be "uni" (uniform grid) and "cheb"
            (Chebyshev grid).

    Returns:
        np.ndarray: points of the grid in the form of array of the shape
            [samples, d]. If input "I" is 1D array or list (the case of only
            one sample), then function will also return 1D array of length "d".

    """
    if isinstance(I, list):
        I = np.array(I, dtype=int)

    d = I.shape[-1]
    m = I.shape[0] if len(I.shape) > 1 else None
    a, b, n = grid_prep_opts(a, b, n, d, m)

    if kind == 'uni':
        X = I / (n - 1) * (b - a) + a
    elif kind == 'cheb':
        X = np.cos(np.pi * I / (n - 1)) * (b - a) / 2 + (b + a) / 2
    else:
        raise ValueError(f'Unknown grid type "{kind}"')

    return X


def ind2str(i):
    """Transforms array of int like [1, 2, 3] into string like '1-2-3'.

    Simple function that may be used for the cache of the TT-CAM.

    Args:
        i (list): multiindex in the form of array of the shape [d] or list of
            length "d".

    Returns:
        str: multiindex converted to string, where indexes are separated by
            hyphens.

    """
    return '-'.join([str(int(v)) for v in i])


def sample_lhs(n, m):
    """Build m LHS samples (indices) for the tensor of the shape n.

    Args:
        n (list): tensor size for each dimension (list or np.ndarray of the
            length "d").
        m (int): number of samples.

    Returns:
        np.ndarray: generated multiindices for the tensor in the form of array
            of the shape [m, d].

    """
    if isinstance(n, list):
        n = np.array(n, dtype=int)

    d = len(n)

    I = np.empty((m, d), dtype=int)
    for i, k in enumerate(n):
        I1 = np.repeat(np.arange(k), m // k)
        I2 = np.random.choice(k, m-len(I1), replace=False)
        I[:, i] = np.concatenate([I1, I2])
        np.random.shuffle(I[:, i])

    return I


def sample_tt(n, m=4):
    """Generate special samples (indices) for the tensor of the shape n.

    Generate special samples (indices) for the tensor of the shape n. The
    generated samples are the best (in many cases) for the subsequent
    construction of the TT-tensor.

    Args:
        n (list): tensor size for each dimension (list or np.ndarray of the
            length "d").
        m (int): expected TT-rank of the tensor. The number of samples will be
            selected from this value.

    Returns:
        np.ndarray: generated multiindices for the tensor in the form of array
            of the shape [samples, d].
        np.ndarray: starting poisitions in generated samples for the
            corresponding dimensions in the form of array of the shape [d+1].
        np.ndarray: numbers of points for the right unfoldings in generated
            samples in the form of array of the shape [d].

    Note:
        The resulting number of samples will be chosen adaptively based on the
        specified expected TT-rank (m).

    """
    def one_mode(sh1, sh2, rng):
        res = []
        if len(sh2) == 0:
            lhs_1 = sample_lhs(sh1, m)
            for n in range(rng):
                for i in lhs_1:
                    res.append(np.concatenate([i, [n]]))
            len_1, len_2 = len(lhs_1), 1
        elif len(sh1) == 0:
            lhs_2 = sample_lhs(sh2, m)
            for n in range(rng):
                for j in lhs_2:
                    res.append(np.concatenate([[n], j]))
            len_1, len_2 = 1, len(lhs_2)
        else:
            lhs_1 = sample_lhs(sh1, m)
            lhs_2 = sample_lhs(sh2, m)
            for n in range(rng):
                for i, j in itertools.product(lhs_1, lhs_2):
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
    """Transforms string like '1-2-3' into array of int like [1, 2, 3].

    Simple function that transforms string like `1-2-3` into array of int like `[1, 2, 3]` (it is used for the cache of the TT-cross).

    Args:
        s (str): d-dimensional multiindex in the form of the string where
            indexes are separated by hyphens.

    Returns:
        np.ndarray: multiindex in the form of the 1D array of size "d".

    """
    return np.array([int(v) for v in s.split('-')], dtype=int)

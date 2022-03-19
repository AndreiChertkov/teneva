"""Package teneva, module core.grid: functions to construct the grid.

This module contains a set of functions for creating and transforming
multidimensional grids for discretizing functions on uniform and Chebyshev
grids, as well as a number of methods for tensor sampling.

"""
import itertools
import numpy as np


def cache_to_data(cache={}):
    """Transform cache of the TT-CROSS into I, Y data arrays.

    Args:
        cache (dict): cache of the TT-CROSS (see "cross" function), that
            contains the requested function values and related tensor
            multi-indices.

    Returns:
        [np.ndarray, np.ndarray]: tensor multi-indices (I; in the form of array
        of the shape [samples, dimension]) and related function values (Y; in
        the form of array of the shape [samples]).

    """
    I = np.array([i for i in cache.keys()], dtype=int)
    Y = np.array([y for y in cache.values()])
    return I, Y


def grid_flat(n):
    """Compute the multi-indices for the full multidimensional grid.

    Args:
        n (list, np.ndarray): number of grid points for each dimension (list or
            np.ndarray of length "d", where "d" is a number of dimensions).

    Returns:
        np.ndarray: multi-indices for the full (flatten) grid (it is 2D array of
        the shape [d, n^d]).

    """
    d = len(n)
    I = [np.arange(k).reshape(1, -1) for k in n]
    I = np.meshgrid(*I, indexing='ij')
    I = np.array(I, dtype=int).reshape((d, -1), order='F').T
    return I


def grid_prep_opt(opt, d=None, kind=float, reps=None):
    """Helper function that prepare grid parameter.

    Args:
        opt (int, float, list, np.ndarray): grid parameter for each dimension.
        d (int): the dimension of the grid. If should be set if "opt" is scalar.
        kind (class 'int', class 'float'): data type for option (int or float).
        reps (int): optional number of repetitions for option.

    Returns:
        np.ndarray: prepared option. It is 1D np.ndarray of length "d" if reps
        is None, or np.ndarray of the shape [reps, d] (values repeated along
        the 1th axis) otherwise.

    """
    if opt is None:
        return None

    if isinstance(opt, (int, float)):
        if d is None or d <= 0:
            raise ValueError('Invalid grid option')
        opt = np.ones(d, dtype=kind) * kind(opt)
    opt = np.asanyarray(opt, dtype=kind)

    if reps is not None:
        opt = np.repeat(opt.reshape((1, -1)), reps, axis=0)

    return opt


def grid_prep_opts(a=None, b=None, n=None, d=None, reps=None):
    """Helper function that prepare grid parameters a, b, n.

    Args:
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d" or float).
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d" or float).
        n (int, float, list, np.ndarray): grid size for each dimension (list or
            np.ndarray of length "d" or int/float).
        d (int): the dimension of the grid. If all "a", "b", "n" are numbers,
            then it should be set, otherwise it is optional ("d" will be
            recoverd fromv the given "a" / "b" / "n").
        reps (int): optional number of repetitions for a, b, n.

    Returns:
        [np.ndarray, np.ndarray, np.ndarray]: grid lower bounds "a", grid upper
        bounds "b" and grid size "n" for each dimension. All opts will be 1D
        arrays of length "d" if reps is None, or np.ndarray of the shape
        [reps, d] (values repeated along the 1th axis) otherwise.

    Note:
        In case of a mismatch in the size of the arrays "a" / "b" / "n" or the
        impossibility of restoring the dimension "d", the function will
        generate the error (ValueError).

    """
    for item in [a, b, n]:
        if isinstance(item, (list, np.ndarray)):
            if d is None:
                d = len(item)
            elif d != len(item):
                raise ValueError('Invalid grid option')

    a = grid_prep_opt(a, d, float, reps)
    b = grid_prep_opt(b, d, float, reps)
    n = grid_prep_opt(n, d, int, reps)

    return a, b, n


def ind_to_poi(I, a, b, n, kind='uni'):
    """Transform multi-indices (samples) into points of the spatial grid.

    Args:
        I (list, np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d], where "samples" is the number of samples
            and "d" is the dimension of the tensor. For the case of only one
            sample, it may be 1D array or list of length "d".
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        n (int, float, list, np.ndarray): tensor size for each dimension (list
            or np.ndarray of length "d"). It may be also int/float, then the
            size for each dimension will be the same.
        kind (str): the grid type, it may be "uni" (uniform grid) and "cheb"
            (Chebyshev grid). In case of the uniform grid, index "0" relates to
            the spatial point "a" and index "n-1" relates to the spatial point
            "b". In case of the Chebyshev grid, index "0" relates to the
            spatial point "b" and index "n-1" relates to the spatial point "a".

    Returns:
        np.ndarray: points of the grid in the form of array of the shape
        [samples, d]. If input "I" is 1D list or np.ndarray (the case of only
        one sample), then function will also return 1D np.ndarray of length "d".

    Note:
        This function may be used for function approximation by low rank tensor
        methods (requested tensor indices should be transformed into spatial
        grid points and then used as an argument for the target function).

    """
    I = np.asanyarray(I, dtype=int)
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


def sample_lhs(n, m):
    """Generate LHS samples (multi-indices) for the tensor of the given shape.

    Args:
        n (list, np.ndarray): tensor size for each dimension (list or
            np.ndarray of int/float of the length "d", where "d" is the
            dimension of the tensor).
        m (int, float): number of samples.

    Returns:
        np.ndarray: generated multi-indices for the tensor in the form of array
        of the shape [m, d], where "d" is the dimension of the tensor.

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
            np.ndarray of int/float of the length "d").
        r (int): expected TT-rank of the tensor. The number of generated
            samples will be selected according to this value.

    Returns:
        [np.ndarray, np.ndarray, np.ndarray]: generated multi-indices for the
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

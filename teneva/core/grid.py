"""Package teneva, module core.grid: functions to operate with grids.

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
        n (int, float, list, np.ndarray): number of grid points for each
            dimension (list or np.ndarray of length "d", where "d" is a number
            of dimensions). It may be also a number, then the 1D grid will be
            returned.

    Returns:
        np.ndarray: multi-indices for the full (flatten) grid (it is 2D array of
        the shape [d, n^d]).

    """
    if isinstance(n, (int, float, np.int32, np.float32, np.int64, np.float64)):
        return np.arange(int(n))

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


def ind_qtt_to_tt(I_qtt, q):
    """Transform tensor multi-indices from QTT (long) to base TT (short) format.

    Args:
        I_qtt (list, np.ndarray): QTT multi-indices for the tensor in the form
            of array of the shape [samples, d*q], where "samples" is the number
            of samples, "d" is the dimension of the TT-tensor and "q" is a
            quantization value. For the case of only one sample, it may be 1D
            array or list of the length "d*q".
        q (int): quantization value (TT-tensor mode size will be n=2^q).

    Returns:
        np.ndarray: TT multi-indices, which relates to the given QTT
        multi-indices in the form of array of the shape [samples, d]. If input
        "I_qtt" is 1D list or np.ndarray (the case of only one sample), then
        function will also return 1D np.ndarray of the length "d".

    """
    I_qtt = grid_prep_opt(I_qtt, kind=int)

    if len(I_qtt.shape) == 1:
        is_many = False
        I_qtt = I_qtt.reshape(1, -1)
    else:
        is_many = True

    d = int(I_qtt.shape[1] / q)
    m = I_qtt.shape[0]
    n = [2]*q

    I = np.zeros((m, d), dtype=int)
    for i in range(d):
        I_qtt_curr = I_qtt[:, q*i:q*(i+1)].T
        I[:, i] = np.ravel_multi_index(I_qtt_curr, n, order='F')

    return I if is_many else I[0, :]


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
    I = np.asanyarray(I)
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


def ind_tt_to_qtt(I, n):
    """Transform tensor multi-indices from base TT (short) to QTT (long) format.

    Args:
        I (list, np.ndarray): TT multi-indices for the tensor in the form of
            array of the shape [samples, d], where "samples" is the number of
            samples and "d" is the dimension of the TT-tensor. For the case of
            only one sample, it may be 1D array or list of the length "d".
        n (int): TT-tensor mode size. It should be like "n=2^q", where "q" is a
            quantization value.

    Returns:
        np.ndarray: QTT multi-indices, which relates to the given TT
        multi-indices in the form of array of the shape [samples, d*q]. If input
        "I" is 1D list or np.ndarray (the case of only one sample), then
        function will also return 1D np.ndarray of the length "d*q".

    """
    I = grid_prep_opt(I, kind=int)

    if len(I.shape) == 1:
        is_many = False
        I = I.reshape(1, -1)
    else:
        is_many = True

    d = int(I.shape[1])
    m = I.shape[0]
    q = int(np.log2(n))
    n_qtt = [2]*q

    if 2**q != n:
        raise ValueError('Invalid mode size (it should be a power of two)')

    I_qtt = np.zeros((m, d*q), dtype=int)
    for i in range(d):
        I_curr = I[:, i]
        I_qtt_curr = np.unravel_index(I_curr, n_qtt, order='F')
        I_qtt_curr = np.array(I_qtt_curr).T
        I_qtt[:, q*i:q*(i+1)] = I_qtt_curr

    return I_qtt if is_many else I_qtt[0, :]


def poi_scale(X, a, b, kind='uni'):
    """Scale points from [a, b] into unit interval.

    Args:
        X (list, np.ndarray): points of the spatial grid in the form of array of
            the shape [samples, d], where "samples" is the number of samples and
            "d" is the dimension of the tensor. For the case of only one
            sample, it may be 1D array or list of length "d".
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        kind (str): the grid type, it may be "uni" (uniform grid) and "cheb"
            (Chebyshev grid).

    Returns:
        np.ndarray: scaled points of the spatial grid. It has the same shape as
        input array "X". The interval will be [0, 1] in case of the uniform
        grid, [-1, 1] in the case of the Chebyshev grid.

    """
    X = np.asanyarray(X, dtype=float)
    d = X.shape[-1]
    m = X.shape[0] if len(X.shape) > 1 else None

    a, b, _ = grid_prep_opts(a, b, None, d, m)

    if kind == 'uni':
        X_sc = (X - a) / (b - a)
        X_sc[X_sc < 0.] = 0.
        X_sc[X_sc > 1.] = 1.
    elif kind == 'cheb':
        X_sc = (X - (b + a) / 2) * (2 / (b - a))
        X_sc[X_sc < -1.] = -1.
        X_sc[X_sc > +1.] = +1.
    else:
        raise ValueError(f'Unknown grid type "{kind}"')

    return X_sc


def poi_to_ind(X, a, b, n, kind='uni'):
    """Transform points of the spatial grid (samples) into multi-indices.

    Args:
        X (list, np.ndarray): points of the spatial grid in the form of array of
            the shape [samples, d], where "samples" is the number of samples and
            "d" is the dimension of the tensor. For the case of only one
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
        np.ndarray: multi-indices for the tensor in the form of array of the
        shape [samples, d]. If input "X" is 1D list or np.ndarray (the case of
        only one sample), then function will also return 1D np.ndarray of the
        length "d".

    Note:
        Points that are outside the domain ("a" and "b") will be transformed to
        the nearest grid indexes (i.e., "0" or "n-1").

    """
    X_sc = poi_scale(X, a, b, kind)
    d = X_sc.shape[-1]
    m = X_sc.shape[0] if len(X_sc.shape) > 1 else None
    n = grid_prep_opt(n, d, kind=int, reps=m)

    if kind == 'uni':
        I = X_sc * (n - 1)
    elif kind == 'cheb':
        I = np.arccos(X_sc) / np.pi * (n - 1)
    else:
        raise ValueError(f'Unknown grid type "{kind}"')

    I = np.rint(I)
    I = np.array(I, dtype=int)

    I[I < 0] = 0
    I[I > n-1] = n[I > n-1] - 1

    return I


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

"""Package teneva, module core.cheb: Chebyshev interpolation in the TT-format.

This module contains the functions for construction of the Chebyshev
interpolation in the TT-format as well as calculating the values of the
function using the constructed interpolation coefficients.

"""
import numpy as np


from .cross import cross
from .grid import grid_prep_opts
from .grid import ind2poi
from .tensor import copy
from .tensor import shape
from .tensor import truncate


def cheb_bld(f, a, b, n, **args):
    """Compute the function values on the Chebyshev grid.

    Args:
        f (function): function f(X) for interpolation, where X should be 2D
            np.ndarray of the shape [samples, dimensions]. The function should
            return 1D np.ndarray of the length equals to samples.
        a (list): grid lower bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the lower bounds for each
            dimension will be the same.
        b (list): grid upper bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the upper bounds for each
            dimension will be the same.
        n (list): tensor size for each dimension (list or np.ndarray of length
            "d"). It may be also float, then the size for each dimension will be
            the same.
        args (dict): named arguments for TT-CAM ("cross") function except the
            target function "f", i.e. "(Y0, e, evals, nswp, dr_min, dr_max,
            cache, info)". Note that initial approximation "Y0" and accuracy
            "e" are required.

    Returns:
        Y (list): TT-Tensor with function values on the Chebyshev grid.

    Note:
        At least one of the variables "a", "b", "n" must be a list (to be able
        to automatically determine the dimension).

    """
    a, b, n = grid_prep_opts(a, b, n)
    Y = cross(lambda I: f(ind2poi(I.astype(int), a, b, n, 'cheb')), **args)
    return Y


def cheb_get(X, A, a, b, z=0.):
    """Compute the Chebyshev approximation in given points (approx. f(X)).

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where "d" is the number of dimensions).
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (list): grid lower bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the lower bounds for each
            dimension will be the same.
        b (list): grid upper bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the upper bounds for each
            dimension will be the same.
        z (float): the value for points, which are outside the spatial grid.

    Returns:
        np.ndarray: approximated function values in X points (it is 1D array of
            the shape [samples]).

    """
    d = len(A)
    n = shape(A)
    m = X.shape[0]
    a, b, n = grid_prep_opts(a, b, n, d)

    # TODO: check if this operation is effective. It may be more profitable to
    # generate polynomials for each tensor mode separately:
    T = cheb_pol(X, a, b, max(n))

    Y = np.ones(m) * z
    for i in range(m):
        if np.max(a - X[i, :]) > 1.E-16 or np.max(X[i, :] - b) > 1.E-16:
            # We skip the points outside the grid bounds:
            continue

        Q = np.einsum('rkq,k->rq', A[0], T[:n[0], i, 0])
        for j in range(1, d):
            Q = Q @ np.einsum('rjq,j->rq', A[j], T[:n[j], i, j])
        Y[i] = Q[0, 0]

    return Y


def cheb_get_full(A, a, b, m=None, e=1.E-6):
    """Compute the Chebyshev approximation (TT-tensor) on the full given grid.

    Args:
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (float): grid lower bounds for each dimension (the lower bounds for
            each dimension should be the same).
        b (float): grid upper bounds for each dimension (the upper bounds for
            each dimension should be the same).
        m (int): number of grid points for each dimension (>= 2). If is not set,
            then original grid size (from the interpolation) will be used.
        e (float): accuracy for truncation of the result (> 0).

    Returns:
        list: TT-tensor of the approximated function values on the full grid. (m x m x ... x m).

    Note:
        This function works correctly only for grids with an equal number of
        points for each mode!

    """
    d = len(A)
    n = A[0].shape[1]
    m = m or n
    I = np.arange(m).reshape((1, -1))
    X = ind2poi(I, a, b, m, 'cheb').reshape(-1)
    T = cheb_pol(X, a, b, n)
    Q = []
    for i in range(d):
        Q.append(np.einsum('riq,ij->rjq', A[i], T))

    return truncate(Q, e)


def cheb_int(Y, e=1.E-6):
    """Compute the TT-tensor for Chebyshev interpolation coefficients.

    Args:
        Y (list): TT-tensor with function values on the Chebyshev grid.
        e (float): accuracy for truncation of the result (> 0).

    Returns:
        list: TT-tensor that collects interpolation coefficients. It has the
            same shape as the given tensor Y.

    """
    A = copy(Y)
    for k in range(len(A)):
        r, m, q = A[k].shape
        A[k] = np.swapaxes(A[k], 0, 1)
        A[k] = A[k].reshape((m, -1))
        A[k] = np.vstack([A[k], A[k][m-2 : 0 : -1, :]])
        A[k] = np.fft.fft(A[k], axis=0).real
        A[k] = A[k][:m, :] / (m - 1)
        A[k][0, :] /= 2.
        A[k][m-1, :] /= 2.
        A[k] = A[k].reshape((m, r, q))
        A[k] = np.swapaxes(A[k], 0, 1)
    return truncate(A, e)


def cheb_pol(X, a, b, m):
    """Compute the Chebyshev polynomials in the given points.

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where d is a number of dimensions).
        a (list): grid lower bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the lower bounds for each
            dimension will be the same.
        b (list): grid upper bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the upper bounds for each
            dimension will be the same.
        m (int): maximum order for Chebyshev polynomial (>= 1). The polynomials
            of the order 0,1,...,m-1 will be computed.

    Returns:
        np.ndarray: values of the Chebyshev polynomials of the order 0,1,...,m-1
            in X points (it is 3D array of the shape [m x samples x d]).

    Note:
        Before calculating polynomials, the points are scaled from [a, b] to standard [-1, 1] limits.

    """
    d = X.shape[-1]
    reps = X.shape[0] if len(X.shape) > 1 else None
    a, b = grid_prep_opts(a, b, None, d, reps)
    X = (2. * X - b - a) / (b - a)

    T = np.ones([m] + list(X.shape))

    if m < 2:
        return T

    T[1, ] = X.copy()
    for k in range(2, m):
        T[k, ] = 2. * X * T[k - 1, ] - T[k - 2, ]

    return T


def cheb_sum(A, a, b):
    """Integrate the function from its Chebyshev approximation in the TT-format.

    Args:
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (list): grid lower bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the lower bounds for each
            dimension will be the same.
        b (list): grid upper bounds for each dimension (list or np.ndarray of
            length "d"). It may be also float, then the upper bounds for each
            dimension will be the same.

    Returns:
        float: the value of the integral.

    Note:
        This function works only for symmetric grids!

    """
    d = len(A)
    a, b = grid_prep_opts(a, b, None, d)

    Y = copy(A)
    v = np.array([[1.]])
    for k in range(len(Y)):
        r, m, q = Y[k].shape
        Y[k] = np.swapaxes(Y[k], 0, 1)
        Y[k] = Y[k].reshape(m, -1)
        p = np.arange(Y[k].shape[0])[::2]
        p = np.repeat(p.reshape(-1, 1), Y[k].shape[1], axis=1)
        Y[k] = np.sum(Y[k][::2, :] * 2. / (1. - p**2), axis=0)
        Y[k] = Y[k].reshape(r, q)
        v = (v @ Y[k]) * (b[k] - a[k]) / 2.

    return v[0, 0]

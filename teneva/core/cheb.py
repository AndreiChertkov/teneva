"""Package teneva, module core.cheb: Chebyshev interpolation in the TT-format.

This module contains the functions for construction of the Chebyshev
interpolation in the TT-format as well as calculating the values of the
function using the constructed interpolation coefficients.

Note:
    See module "core.cheb_full" with the same functions in the full format.

"""
import numpy as np


from .cross import cross
from .grid import grid_prep_opt
from .grid import grid_prep_opts
from .grid import ind_to_poi
from .tensor import copy
from .tensor import shape
from .transformation import truncate


def cheb_bld(f, a, b, n, eps, Y0, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=2, tau0=1.05, k0=100, info={}, cache=None):
    """Compute the function values on the Chebyshev grid.

    Args:
        f (function): function f(X) for interpolation, where X should be 2D
            np.ndarray of the shape [samples, dimensions]. The function should
            return 1D np.ndarray of the length equals to "samples".
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        n (int, float, list, np.ndarray): tensor size for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the size
            for each dimension will be the same.
        eps (float): accuracy of truncation of the TT-CROSS result (> 0).
        Y0 (list): TT-tensor, which is the initial approximation for TT-CROSS
            algorithm. It may be, fo example, random TT-tensor, which can be
            built by the "rand" function from teneva: "Y0 = teneva.rand(n, r)",
            where "n" is a size of tensor modes (e.g., "n = [5, 6, 7, 8, 9]"
            for the 5-dimensional tensor) and "r" is a TT-rank of this
            TT-tensor (e.g., "r = 3").

    Returns:
        list: TT-Tensor with function values on the Chebyshev grid.

    Note:
        The arguments "m", "e", "nswp", "tau", "dr_min", "dr_max", "tau0",
        "k0", "info" and "cache" are relate to TT-CROSS algorithm (see "cross"
        function for more details). Note that at list one of the arguments m /
        e / nswp should be set.

        At least one of the variables "a", "b", "n" must be a list or
        np.ndarray (to be able to automatically determine the dimension).

        See also the same function ("cheb_bld_full") in the full format.

    """
    a, b, n = grid_prep_opts(a, b, n)
    Y = cross(lambda I: f(ind_to_poi(I, a, b, n, 'cheb')),
        Y0, m, e, nswp, tau, dr_min, dr_max, tau0, k0, info, cache)
    return truncate(Y, eps)


def cheb_get(X, A, a, b, z=0.):
    """Compute the Chebyshev approximation in given points (approx. f(X)).

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where "d" is the number of dimensions).
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        z (float): the value for points, which are outside the spatial grid.

    Returns:
        np.ndarray: approximated function values in X points (it is 1D array of
        the shape [samples]).

    Note:
        See also the same function ("cheb_get_full") in the full format.

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

        Q = np.einsum('rjq,j->rq', A[0], T[:n[0], i, 0])
        for j in range(1, d):
            Q = Q @ np.einsum('rjq,j->rq', A[j], T[:n[j], i, j])
        Y[i] = Q[0, 0]

    return Y


def cheb_gets(A, a, b, m=None):
    """Compute the Chebyshev approximation (TT-tensor) all over the new grid.

    Args:
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        m (int, float, list, np.ndarray): tensor size for each dimension of the
            new grid (list or np.ndarray of length "d"). It may be also
            int/float, then the size for each dimension will be the same. If it
            is not set, then original grid size (from the interpolation) will be
            used.

    Returns:
        list: TT-tensor of the approximated function values on the full new
        grid. This relates to the d-dimensional array of the shape "m".

    Note:
        Sometimes additional rounding of the result is relevant. Use for this
        "Y = truncate(Y, e)" (e.g., "e = 1.E-8") after the function call.

        See also the same function ("cheb_gets_full") in the full format.

    """
    d = len(A)
    n = shape(A)
    a, b, n = grid_prep_opts(a, b, n, d)
    m = n if m is None else grid_prep_opt(m, d, int)

    Q = []
    for k in range(d):
        I = np.arange(m[k], dtype=int).reshape((-1, 1))
        X = ind_to_poi(I, a[k], b[k], m[k], 'cheb').reshape(-1)
        T = cheb_pol(X, a[k], b[k], n[k])
        Q.append(np.einsum('riq,ij->rjq', A[k], T))

    return Q


def cheb_int(Y):
    """Compute the TT-tensor for Chebyshev interpolation coefficients.

    Args:
        Y (list): TT-tensor with function values on the Chebyshev grid.

    Returns:
        list: TT-tensor that collects interpolation coefficients. It has the
        same shape as the given tensor Y.

    Note:
        Sometimes additional rounding of the result is relevant. Use for this
        "A = truncate(A, e)" (e.g., "e = 1.E-8") after the function call.

        See also the same function ("cheb_int_full") in the full format.

    """
    d = len(Y)
    A = copy(Y)
    for k in range(d):
        r, m, q = A[k].shape
        A[k] = np.swapaxes(A[k], 0, 1)
        A[k] = A[k].reshape((m, r * q))
        A[k] = np.vstack([A[k], A[k][m-2 : 0 : -1, :]])
        A[k] = np.fft.fft(A[k], axis=0).real
        A[k] = A[k][:m, :] / (m - 1)
        A[k][0, :] /= 2.
        A[k][m-1, :] /= 2.
        A[k] = A[k].reshape((m, r, q))
        A[k] = np.swapaxes(A[k], 0, 1)
    return A


def cheb_pol(X, a, b, m):
    """Compute the Chebyshev polynomials in the given points.

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where "d" is a number of dimensions).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.
        m (int): maximum order for Chebyshev polynomial (>= 1). The first "m"
            polynomials (of the order 0, 1, ..., m-1) will be computed.

    Returns:
        np.ndarray: values of the Chebyshev polynomials of the order 0,1,...,m-1
        in X points (it is 3D array of the shape [m, samples, d]).

    Note:
        Before calculating polynomials, the points are scaled from [a, b] to
        standard [-1, 1] limits.

    """
    d = X.shape[-1]
    reps = X.shape[0] if len(X.shape) > 1 else None
    a, b, _ = grid_prep_opts(a, b, None, d, reps)
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
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length "d"). It may be also float, then the upper
            bounds for each dimension will be the same.

    Returns:
        float: the value of the integral.

    Note:
        This function works only for symmetric grids!

        See also the same function ("cheb_sum_full") in the full format.

    """
    d = len(A)
    n = shape(A)
    a, b, n = grid_prep_opts(a, b, n, d)

    for k in range(d):
        if abs(abs(b[k]) - abs(a[k])) > 1.E-16:
            raise ValueError('This function works only for symmetric grids')

    Y = copy(A)
    v = np.array([[1.]])
    for k in range(d):
        r, m, q = Y[k].shape
        Y[k] = np.swapaxes(Y[k], 0, 1)
        Y[k] = Y[k].reshape(m, r * q)
        p = np.arange(Y[k].shape[0])[::2]
        p = np.repeat(p.reshape(-1, 1), Y[k].shape[1], axis=1)
        Y[k] = np.sum(Y[k][::2, :] * 2. / (1. - p**2), axis=0)
        Y[k] = Y[k].reshape(r, q)
        v = v @ Y[k]
        v *= (b[k] - a[k]) / 2.

    return v[0, 0]

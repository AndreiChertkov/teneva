"""Package teneva, module core.cheb: Chebyshev interpolation in the TT-format.

This module contains the functions for construction of the Chebyshev
interpolation in the TT-format as well as calculating the values of the
function using the constructed interpolation coefficients.

Note:
    See module "core.cheb_full" with the same functions in the full format.

"""
import numpy as np
import scipy as sp
import teneva
from scipy.fftpack import  dct

def cheb_bld(f, a, b, n, eps=1.E-10, Y0=None, m=None, e=1.E-10, nswp=None, tau=1.1, dr_min=1, dr_max=2, tau0=1.05, k0=100, info={}, cache=None, I_vld=None, Y_vld=None, e_vld=None, log=False, func=None):
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

    Returns:
        list: TT-Tensor with function values on the Chebyshev grid.

    Note:
        The arguments "Y0" "m", etc. are relate to TT-CROSS algorithm (see
        "cross" function for more details). Note that at list one of the
        arguments m / e / nswp should be set. If "Y0" is not provided, then
        random rank-1 TT-tensor will be used.

        At least one of the variables "a", "b", "n" must be a list or
        np.ndarray (to be able to automatically determine the dimension).

        See also the same function ("cheb_bld_full") in the full format.

    """
    a, b, n = teneva.grid_prep_opts(a, b, n)
    if Y0 is None:
        Y0 = teneva.tensor_rand(n, r=1)
    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, m, e, nswp, tau, dr_min, dr_max, tau0, k0, info, cache,
        I_vld, Y_vld, e_vld, log, func)
    return teneva.truncate(Y, eps)


def cheb_diff_matrix(a, b, n, m=1):
    """Construct the Chebyshev differential matrix of any order.

    The function returns the matrix D (if "m=1"), which, for the known vector
    "y" of values of a one-dimensional function on the Chebyshev grid, gives
    its first derivative, i.e., y' = D y. If the argument "m" is greater than
    1, then the function returns a list of matrices corresponding to the first
    derivative, the second derivative, and so on. Note that the derivative
    error can be significant near the boundary of the region.

    Args:
        a (float): grid lower bound.
        b (float): grid upper bound.
        n (int, float): grid size.
        m (int): the maximum order of derivative.

    Returns:
        list of np.ndarray or np.ndarray: the Chebyshev differential matrices
        of order 1, 2, ..., m if m > 1, or only one matrix corresponding to the
        first derivative if m = 1.

    """
    n = int(n)
    k = np.arange(n)
    n1 = np.int(np.floor(n / 2))
    n2 = np.int(np.ceil(n / 2))
    th = k * np.pi / (n - 1)

    T = np.tile(th/2, (n, 1))
    DX = 2. * np.sin(T.T + T) * np.sin(T.T - T)
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))
    DX[range(n), range(n)] = 1.
    DX = DX.T

    Z = 1. / DX
    Z[range(n), range(n)] = 0.

    C = sp.linalg.toeplitz((-1.)**k)
    C[+0, :] *= 2.
    C[-1, :] *= 2.
    C[:, +0] *= 0.5
    C[:, -1] *= 0.5

    D_list = []
    D = np.eye(n)
    for i in range(m):
        D = (i+1) * Z * (C * np.tile(np.diag(D), (n, 1)).T - D)
        D[range(n), range(n)] = -np.sum(D, axis=1)
        l = (2. / (b - a))**(i+1)
        D_list.append(D * l)

    return D_list[0] if m == 1 else D_list


def cheb_diff_matrix_spectral(n, m=1):
    """Construct the Chebyshev differential matrix of any order in spectral representation

    The function returns the matrix D (if "m=1"), which, for the known vector
    "y" of values of a coefficient on Chebyshev polynomial one-dimensional function, gives
    its coefficients of first derivative. If the argument "m" is greater than
    1, then the function returns D^m
    Args:
        n (int, float): max poly power.
        m (int): the maximum order of derivative.

    Returns:
        np.ndarray: the Chebyshev differential matrices

    """
    res = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i):
            res[j, i] = 2*i*((i + j) % 2)- i*(i % 2)*(0 + (j==0))

    D = res
    for _ in range(m-1):
        res = res @ D

    return res


def cheb_get(X, A, a, b, z=0., skip_out=True):
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
    n = teneva.shape(A)
    m = X.shape[0]
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    # TODO: check if this operation is effective. It may be more profitable to
    # generate polynomials for each tensor mode separately:
    T = cheb_pol(teneva.poi_scale(X, a, b, 'cheb'), max(n))

    Y = np.ones(m) * z
    for i in range(m):
        if skip_out:
            if np.max(a - X[i, :]) > 1.E-99 or np.max(X[i, :] - b) > 1.E-99:
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
        "Z = truncate(Z, e)" (e.g., "e = 1.E-8") after the function call.

        See also the same function ("cheb_gets_full") in the full format.

    """
    d = len(A)
    n = teneva.shape(A)
    a, b, n = teneva.grid_prep_opts(a, b, n, d)
    m = n if m is None else teneva.grid_prep_opt(m, d, int)

    Z = []
    for k in range(d):
        I = np.arange(m[k], dtype=int).reshape((-1, 1))
        X = teneva.ind_to_poi(I, -1., +1., m[k], 'cheb').reshape(-1)
        T = cheb_pol(X, n[k])
        Z.append(np.einsum('riq,ij->rjq', A[k], T))

    return Z

def cheb_int_old(Y):
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
    A = teneva.copy(Y)
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
    A = [None]*len(Y)
    for k, y in enumerate(Y):
        A[k] = dct(y, 1, axis=1) / (y.shape[1] - 1)
        A[k][:, 0, :] /= 2.
        A[k][:, -1, :] /= 2.
    return A


def cheb_pol(X, m=10):
    """Compute the Chebyshev polynomials in the given points.

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where "d" is a number of dimensions). All points
            should be from the interval [-1, 1].
        m (int): maximum order for Chebyshev polynomial (>= 1). The first "m"
            polynomials (of the order 0, 1, ..., m-1) will be computed.

    Returns:
        np.ndarray: values of the Chebyshev polynomials of the order 0,1,...,m-1
        in X points (it is 3D array of the shape [m, samples, d]).

    """
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
    n = teneva.shape(A)
    n_max = max(n)
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    # TODO make universal function with arbitrary p
    p = 2. / (1 - np.arange(0, n_max, 2)**2)
    v = np.array([[1.]])
    for ak, bk, y, nk in zip(a, b, A, n):
        v = v @ (p[:(nk + 1)//2] @ y[:, ::2])
        v *= (bk - ak) / 2.

    return v.item()

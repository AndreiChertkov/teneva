"""Package teneva, module core.cheb_full: Chebyshev interpolation.

This module contains the functions for construction of the Chebyshev
interpolation in the full format as well as calculating the values of the
function using the constructed interpolation coefficients.

Note:
    See module "core.cheb" with the same functions in the TT-format.

"""
import numpy as np
import teneva


def cheb_bld_full(f, a, b, n, **args):
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
        args (dict): named arguments, which are not used in this function
            (added for compatibility with "cheb_bld" function).

    Returns:
        np.ndarray: d-dimensional array with function values on the Chebyshev
        grid.

    Note:
        At least one of the variables "a", "b", "n" must be a list or
        np.ndarray (to be able to automatically determine the dimension).

        See also the same function ("cheb_bld") in the TT-format.

    """
    a, b, n = teneva.grid_prep_opts(a, b, n)
    I = teneva.grid_flat(n)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    Y = f(X).reshape(n, order='F')
    return Y


def cheb_get_full(X, A, a, b, z=0.):
    """Compute the Chebyshev approximation in given points (approx. f(X)).

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where "d" is the number of dimensions).
        A (np.ndarray): d-dimensional tensor of the interpolation coefficients.
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
        See also the same function ("cheb_get") in the TT-format.

    """
    d = len(A.shape)
    n = A.shape
    m = X.shape[0]
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    # TODO: check if this operation is effective. It may be more profitable to
    # generate polynomials for each tensor mode separately:
    T = teneva.cheb_pol(teneva.poi_scale(X, a, b, 'cheb'), max(n))

    Y = np.ones(m) * z
    for i in range(m):
        if np.max(a - X[i, :]) > 1.E-99 or np.max(X[i, :] - b) > 1.E-99:
            # We skip the points outside the grid bounds:
            continue

        Q = A.copy()
        for j in range(d):
            Q = np.tensordot(Q, T[:n[j], i, j], axes=([0], [0]))
        Y[i] = Q

    return Y


def cheb_gets_full(A, a, b, m=None):
    """Compute the Chebyshev approximation all over the new grid.

    Args:
        A (np.ndarray): d-dimensional tensor of the interpolation coefficients.
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
        np.ndarray: array of the approximated function values on the full new
        grid. This is the d-dimensional array of the shape "m".

    Note:
        This function is not efficient in the full format, and corresponds to a
        simple explicit calculation of all values at the nodes of the new grid.
        At the same time, in the TT-format ("cheb_gets"), the corresponding
        operation is carried out in an implicit efficient form. Accordingly,
        this function is provided only for uniformity of presentation.

    """
    d = len(A.shape)
    n = A.shape
    a, b, n = teneva.grid_prep_opts(a, b, n, d)
    m = n if m is None else teneva.grid_prep_opt(m, d, int)

    I = teneva.grid_flat(m)
    X = teneva.ind_to_poi(I, a, b, m, 'cheb')
    Z = cheb_get_full(X, A, a, b)
    Z = Z.reshape(m, order='F')

    return Z


def cheb_int_full(Y):
    """Compute the tensor for Chebyshev interpolation coefficients.

    Args:
        Y (np.ndarray): d-dimensional array with function values on the
            Chebyshev grid.

    Returns:
        np.ndarray: array that collects interpolation coefficients. It has the
        same shape as the given tensor Y.

    Note:
        See also the same function ("cheb_int") in the TT-format.

    """
    d = len(Y.shape)
    A = Y.copy()
    for k in range(d):
        n = np.array(Y.shape, dtype=int)
        m = n[k]
        n[[0, k]] = n[[k, 0]]
        A = np.swapaxes(A, 0, k)
        A = A.reshape((m, -1), order='F')
        A = np.vstack([A, A[m-2 : 0 : -1, :]])
        A = np.fft.fft(A, axis=0).real
        A = A[:m, :] / (m - 1)
        A[0, :] /= 2.
        A[m-1, :] /= 2.
        A = A.reshape(n, order='F')
        A = np.swapaxes(A, 0, k)
    return A


def cheb_sum_full(A, a, b):
    """Integrate the function from its Chebyshev approximation.

    Args:
        A (np.ndarray): d-dimensional tensor of the interpolation coefficients.
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

        See also the same function ("cheb_sum") in the TT-format.

    """
    d = len(A.shape)
    n = A.shape
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    for k in range(d):
        if abs(abs(b[k]) - abs(a[k])) > 1.E-16:
            raise ValueError('This function works only for symmetric grids')

    v = A.copy()
    for k in range(d):
        v = v.reshape(n[k], -1)
        p = np.arange(n[k])[::2]
        p = np.repeat(p.reshape(-1, 1), v.shape[1], axis=1)
        v = np.sum(v[::2, :] * 2. / (1. - p**2), axis=0)
        v *= (b[k] - a[k]) / 2.

    return v[0]

"""Package teneva, module func: functional TT-format with interpolation.

This module contains the functions for construction of the functional
TT-representation, including Chebyshev interpolation in the TT-format, as well
as calculating the values of the function using the constructed interpolation
coefficients.

"""
from functools import reduce
import numpy as np
import scipy as sp
from scipy.fftpack import dct
from scipy.fftpack import dst
import teneva


def func_basis(X, m=10, kind='cheb'):
    """Compute the basis functions in the given points.

    The function computes values of the first m basis functions (Chebyshev
    polynomials in the current version) in the given points X.

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where d is a number of dimensions). All points
            should be from the interval [-1, 1] for "cheb" kind and from the
            interval [0, pi] for "sin" kind.
        m (int): maximum order of the basis function (>= 1). The first m
            functions (of the order 0, 1, ..., m-1) will be computed.
        kind (str): kind of the basis ("cheb").

    Returns:
        np.ndarray: values of the basis functions of the order 0, 1, ..., m-1
        in X points (it is 3D array of the shape [m, samples, d]).

    """
    if kind != 'cheb':
        raise NotImplementedError(f'The kind "{kind}" is not supported')

    T = np.ones([m] + list(X.shape))

    if m < 2:
        return T

    T[1, ] = X.copy()
    for k in range(2, m):
        T[k, ] = 2. * X * T[k - 1, ] - T[k - 2, ]

    return T


def func_diff_matrix(a, b, n, m=1, kind='cheb'):
    """Construct the differential matrix (Chebyshev or Sin) of any order.

    The function returns the matrix D (if m=1), which, for the known vector
    y of values of a one-dimensional function on the related grid ("Chebyshev"
    grid if kind is "cheb" or uniform grid if kind is "sin"), gives its first
    derivative, i.e., y' = D y. If the argument m is greater than 1, then the
    function returns a list of matrices corresponding to the first derivative,
    the second derivative, and so on. Note that the derivative error can be
    significant near the boundary of the region.

    Args:
        a (float): grid lower bound.
        b (float): grid upper bound.
        n (int): grid size.
        m (int): the maximum order of derivative.
        kind (str): kind of the basis ("cheb" or "sin").

    Returns:
        list of np.ndarray or np.ndarray: the differential matrices of order 1,
        2, ..., m if m > 1, or only one matrix corresponding to the first
        derivative if m = 1.

    """
    n = int(n)
    k = np.arange(n)

    if kind == 'sin':
        D = np.diag((b - a) / np.pi * np.arange(1, n+1))
        D_list = [D]

        for i in range(m):
            D_list.append(D_list[-1] * D)

    elif kind == 'cheb':
        n1 = int(np.floor(n / 2))
        n2 = int(np.ceil(n / 2))
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

    else:
        raise ValueError('Invalid "kind"')

    return D_list[0] if m == 1 else D_list


def func_diff_matrix_apply(A, D, kind='cheb'):
    """Draft. TODO"""
    if kind == 'sin':
        d = np.diag(D)
        return [d[:G.shape[1]] @ G for G in A]

    elif kind == 'cheb':
        raise NotImplementedError()

    else:
        raise ValueError('Invalid "kind"')


def func_get(X, A, a, b, z=0., kind='cheb', skip_out=True):
    """Compute the functional TT-approximation in given points (approx. f(X)).

    Args:
        X (np.ndarray): spatial points of interest (it is 2D array of the shape
            [samples, d], where d is the number of dimensions).
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length d). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length d). It may be also float, then the upper
            bounds for each dimension will be the same.
        z (float): the value for points, which are outside the spatial grid.
        kind (str): kind of the basis ("cheb").
        skip_out (bool): if flag is set, then the values outside the spatial
            grid will be set to z values.

    Returns:
        np.ndarray: approximated function values in X points (it is 1D array of
        the shape [samples]).

    """
    if kind != 'cheb':
        raise NotImplementedError(f'The kind "{kind}" is not supported')

    d = len(A)
    n = teneva.shape(A)
    m = X.shape[0]
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    # TODO: check if this operation is effective. It may be more profitable to
    # generate polynomials for each tensor mode separately:
    T = func_basis(teneva.poi_scale(X, a, b, 'cheb'), max(n))

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


def func_get_spectral(Y, X, H):
    """Draft. TODO"""
    X = np.asanyarray(X)

    assert X.ndim == 1 and len(X) == len(Y)

    def f(v, arg):
        c, y = arg
        return np.einsum("i,j,ijk->k", v, y[:c.shape[1]], c)

    return reduce(f, zip(Y, H(X)), np.array([1]))[0]


def func_gets(A, m=None, kind='cheb'):
    """Compute the functional approximation (TT-tensor) all over the new grid.

    Args:
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        m (int, float, list, np.ndarray): tensor size for each dimension of the
            new grid (list or np.ndarray of length d). It may be also
            int/float, then the size for each dimension will be the same. If it
            is not set, then original grid size (from the interpolation) will be
            used.
        kind (str): kind of the basis ("cheb" or "sin").

    Returns:
        list: TT-tensor of the approximated function values on the full new
        grid. This relates to the d-dimensional array of the shape m.

    Note:
        Sometimes additional rounding of the result is relevant. Use for this
        Z = teneva.truncate(Z, e) (e.g., e = 1.E-8) after the function call.

    """
    assert kind in ['cheb', 'sin']

    d = len(A)
    n = teneva.shape(A)
    m = n if m is None else teneva.grid_prep_opt(m, d, int)

    Z = []
    for k in range(d):
        if kind == 'cheb':
            I = np.arange(m[k], dtype=int).reshape((-1, 1))
            X = teneva.ind_to_poi(I, -1., +1., m[k], 'cheb').reshape(-1)
            T = func_basis(X, n[k])
        elif kind == 'sin':
            X = np.linspace(0, np.pi, m[k] + 2)[1:-1]
            T = np.sin(np.outer(X, np.arange(1, n[k] + 1))).T
        Z.append(np.einsum('riq,ij->rjq', A[k], T))

    return Z


def func_int(Y, kind='cheb'):
    """Compute the TT-tensor for functional interpolation coefficients.

    Args:
        Y (list): TT-tensor with function values on the corresponding grid.

    Returns:
        list: TT-tensor that collects interpolation coefficients. It has the
        same shape as the given tensor Y.

    Note:
        Sometimes additional rounding of the result is relevant. Use for this
        A = teneva.truncate(A, e) (e.g., e = 1.E-8) after the function call.

    """
    assert kind in ['cheb', 'sin']

    A = [None] * len(Y)

    for k, y in enumerate(Y):
        if kind == 'cheb':
            A[k] = dct(y, 1, axis=1) / (y.shape[1] - 1)
            A[k][:, 0, :] /= 2.
            A[k][:, -1, :] /= 2.
        elif kind == 'sin':
            A[k] = dst(y, 1, axis=1) / (y.shape[1] + 1)

    return A


def func_int_general(Y, X, basis_func, rcond=1.E-6):
    """Compute the TT-tensor for functional interpolation coefficients.

    Args:
        Y (list): TT-tensor.
        X (list, np.ndarray): values of continuous argument for each TT-core.
            It should be 2-dim array or 1-dim array if the values are the same
            for all TT-cores.
        basis_func (function): function, which corresponds to the values of
            the basis functions in the TT-format. It should return np.ndarray
            of the size n x m, where n is a number of one-dimensional points
            and m is a number of basis functions.

    Returns:
        list: TT-tensor, which represents the interpolation coefficients in
        terms of the functional TT-format.

    """
    d = len(Y)
    X = np.asarray(X)

    if X.ndim == 1:
        H_mats = [basis_func(X).T] * d
        X = [X] * d
    else:
        H_mats = [None] * d

    A = []
    for G, X_curr, H_mat in zip(Y, X, H_mats):
        if H_mat is None:
            H_mat = basis_func(X_curr).T

        r1, n, r2 = G.shape
        M = np.transpose(G, [1, 0, 2]).reshape(n, -1)

        Q = sp.linalg.lstsq(H_mat, M, overwrite_a=False, overwrite_b=True,
            rcond=rcond)[0]
        Q = np.transpose(Q.reshape(n, r1, r2), [1, 0, 2])
        A.append(Q)

    return A


def func_sum(A, a, b, kind='cheb'):
    """Integrate a function from its functional approximation in the TT-format.

    Args:
        A (list): TT-tensor of the interpolation coefficients (it has d
            dimensions).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length d). It may be also float, then the lower
            bounds for each dimension will be the same.
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length d). It may be also float, then the upper
            bounds for each dimension will be the same.
        kind (str): kind of the basis ("cheb" or "sin").

    Returns:
        float: the value of the integral.

    Note:
        This function works only for symmetric grids!

    """
    assert kind in ['cheb', 'sin']

    d = len(A)
    n = teneva.shape(A)
    n_max = max(n)
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    if kind == 'cheb':
        p = 2. / (1 - np.arange(0, n_max, 2)**2)
    elif kind == 'sin':
        p = 2. / np.arange(1, n_max + 1, 2)

    v = np.array([[1.]])
    for ak, bk, y, nk in zip(a, b, A, n):
        v = v @ (p[:(nk + 1)//2] @ y[:, ::2])
        v *= (bk - ak) / 2.

    return v.item()

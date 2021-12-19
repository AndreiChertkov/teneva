import numpy as np


from .cross import cross
from .grid import ind2poi_cheb
from .tensor import copy
from .tensor import rand
from .tensor import truncate


def cheb_bld(f, a, b, n, **args):
    """Compute the function values on the Chebyshev grid.

    Args:
        f (function): Function f(X) for interpolation, where X should be 2D
            np.array of the shape [samples x dimensions].
        a (float): Grid lower bound.
        b (float): Grid upper bound.
        n (int): Number of grid points for each dimension (>= 2).
        args: Arguments for cross function except the target function (Y0, e,
            nswp, kr, rf, cache, info). Note that initial approximation Y0 is
            required.

    Returns:
        Y (list): TT-Tensor with function values on the Chebyshev grid.

    """
    Y = cross(lambda I: f(ind2poi_cheb(I.T.astype(int), a, b, n)), **args)
    return Y


def cheb_get(X, A, a, b, z=0.):
    """Compute the Chebyshev approximation in given points (approx. f(X)).

    Args:
        X (np.array): Spatial points of interest (it is 2D array of the shape
            [dimensions x samples]).
        A (list): Tensor of the interpolation coefficients in the TT-format
            (it is the list of length [dimensions], where elements are
            TT-cores, which look like 3D np.array).
        a (float): Grid lower bound.
        b (float): Grid upper bound.
        z (float): The value for points, which are outside the spatial grid.

    Returns:
        np.array: Approximated function values in X points (it is 1D array of
            the shape [samples]).

    """
    d = len(A)
    n = A[0].shape[1]
    T = cheb_pol(X, a, b, n)
    Y = np.ones(X.shape[1]) * z
    l1 = np.ones(d) * a
    l2 = np.ones(d) * b
    for j in range(X.shape[1]):
        if np.max(l1 - X[:, j]) > 1.E-16 or np.max(X[:, j] - l2) > 1.E-16:
            continue
        Q = np.einsum('riq,i->rq', A[0], T[:, 0, j])
        for i in range(1, d):
            Q = Q @ np.einsum('riq,i->rq', A[i], T[:, i, j])
        Y[j] = Q[0, 0]
    return Y


def cheb_get_full(A, a, b, m=None, e=1.E-6):
    """Compute the Chebyshev approximation (TT-tensor) on the full given grid.

    Args:
        A (list): Tensor of the interpolation coefficients in the TT-format
            (it is the list of length [dimensions], where elements are
            TT-cores, which look like 3D np.array).
        a (float): Grid lower bound.
        b (float): Grid upper bound.
        m (int): Number of grid points for each dimension (>= 2). If is not set,
            then original grid size (from the interpolation) will be used.
        e (float): Accuracy for truncation of the result (> 0).

    Returns:
        list: Tensor in the TT-format of the approximated function values on
            the full grid (m x m x ... x m).

    """
    d = len(A)
    n = A[0].shape[1]
    m = m or n
    I = np.arange(m).reshape((1, -1))
    X = ind2poi_cheb(I, a, b, m).reshape(-1)
    T = cheb_pol(X, a, b, n)
    Q = []
    for i in range(d):
        Q.append(np.einsum('riq,ij->rjq', A[i], T))
    Q = truncate(Q, e)
    return Q


def cheb_ind(d, n):
    """Compute the multi indices for the full Chebyshev grid.

    Args:
        d (int): Dimension (>= 1).
        n (int): Number of grid points for each dimension (>= 2).

    Returns:
        np.array: Multi indices for the full (flatten) Chebyshev grid (it is 2D
            array  of the shape d x n^d).

    """
    I = []
    for k in range(d):
        I.append(np.arange(n).reshape(1, -1))
    I = np.meshgrid(*I, indexing='ij')
    I = np.array(I).reshape((d, -1), order='F')
    return I


def cheb_int(Y, e=1.E-6):
    """Compute the TT-tensor for Chebyshev interpolation coefficients.

    Args:
        Y (list): Tensor of the function values in the Chebyshev grid in the
            TT-format (it is the list of length [dimensions], where elements are
            TT-cores, which look like 3D np.array).
        e (float): Accuracy for truncation of the result (> 0).

    Returns:
        list: Tensor of the interpolation coefficients in the TT-format (it is
            the list of length [dimensions], where elements are TT-cores, which
            look like 3D np.array).

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
    A = truncate(A, e)
    return A


def cheb_pol(X, a, b, m):
    """Compute the Chebyshev Polynomial in the given points.

    Args:
        X (np.array): Spatial points of interest (it is 2D array of the shape
            [dimensions x samples]).
        a (float): Grid lower bound.
        b (float): Grid upper bound.
        m (int): maximum order for Chebyshev Polynomial (>= 1).

    Returns:
        np.array: Values of the Chebyshev Polynomials in X points (it is 2D
            array of the shape [m x samples]).

    """
    X = (2. * X - b - a) / (b - a)
    T = np.ones([m] + list(X.shape))
    if m < 2:
        return T

    T[1, ] = X.copy()
    for k in range(2, m):
        T[k, ] = 2. * X * T[k - 1, ] - T[k - 2, ]

    return T


def cheb_sum(A, a, b):
    """Integrate the function from its Chebyshev approximation.

    Args:
        A (list): Tensor of the interpolation coefficients in the TT-format
            (it is the list of length [dimensions], where elements are
            TT-cores, which look like 3D np.array).
        a (float): Grid lower bound.
        b (float): Grid upper bound.

    Returns:
        float: The value of the integral.

    """
    d = len(A)
    Y = copy(A)
    v = np.array([[1.]])
    for k in range(d):
        r, m, q = Y[k].shape
        Y[k] = np.swapaxes(Y[k], 0, 1)
        Y[k] = Y[k].reshape(m, -1)
        p = np.arange(Y[k].shape[0])[::2]
        p = np.repeat(p.reshape(-1, 1), Y[k].shape[1], axis=1)
        Y[k] = np.sum(Y[k][::2, :] * 2. / (1. - p**2), axis=0)
        Y[k] = Y[k].reshape(r, q)
        v = v @ Y[k]
        v*= (b - a) / 2.
    return v[0, 0]

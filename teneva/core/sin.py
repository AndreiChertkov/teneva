import numpy as np
from scipy.fftpack import  dst

import teneva


def sin_diff_matrix_spectral(n, a=0, b=np.pi, m=1):
    """Construct the sin differential matrix of any order in spectral representation 

    The function returns the matrix D (if "m=1"), which, for the known vector
    "y" of values of a coefficient on Chebyshev polynomial one-dimensional function, gives
    its coefficients of first derivative. If the argument "m" is greater than
    1, then the function returns D^m
    Args:
        n (int, float): max poly power.
        m (int): order of derivative.

    Returns:
        np.ndarray: the Chebyshev differential matrices

    """
    cf = (b - a)/np.pi
    return np.diag( (cf*np.arange(1, n+1))**m )


def sin_apply_diff_matrix(A, a=0, b=np.pi, m=1):

    n_max = max(teneva.shape(A))
    cf = (b - a)/np.pi
    k = (cf*np.arange(1, n_max+1))**m

    return [k[:c.shape[1]] @ c for c in A]


def sin_int(Y):
    """Compute the TT-tensor for sin interpolation coefficients.

    Args:
        Y (list): TT-tensor with function values on the uniform grid.

    Returns:
        list: TT-tensor that collects interpolation coefficients. It has the
        same shape as the given tensor Y.

    Note:
        Sometimes additional rounding of the result is relevant. Use for this
        "A = truncate(A, e)" (e.g., "e = 1.E-8") after the function call.

    """
    A = [None]*len(Y)
    for k, y in enumerate(Y):
        A[k] = dst(y, 1, axis=1) / (y.shape[1] + 1)
    return A


def sin_gets(A):
    """Compute the sin approximation (TT-tensor) all over the new grid.

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

    """
    d = len(A)
    n = teneva.shape(A)

    Z = []
    for n_k, A_k in zip(n, A):
        X = np.linspace(0, np.pi, n_k + 2)[1:-1]
        T = np.sin(np.outer(X,  np.arange(1, n_k + 1) ))
        # Z.append(np.einsum('riq,ij->rjq', A_k, T))
        Z.append(T @ A_k)

    return Z


def sin_sum(A, a, b):
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

    """
    d = len(A)
    n = teneva.shape(A)
    n_max = max(n)
    a, b, n = teneva.grid_prep_opts(a, b, n, d)

    # TODO make universal function with arbitrary p
    p = 2./np.arange(1, n_max + 1, 2)
    v = np.array([1.])
    for ak, bk, y, nk in zip(a, b, A, n):
        v = v @ (p[:(nk + 1)//2] @ y[:, ::2])
        v *= (bk - ak) / np.pi

    return v.item()


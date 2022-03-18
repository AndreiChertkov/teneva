"""Package teneva, module core.maxvol: maxvol-like algorithms.

This module contains the implementation of the maxvol algorithm (function
maxvol) and rect_maxvol algorithm (function maxvol_rect) for matrices. The
corresponding functions find in a given matrix square and rectangular
maximal-volume submatrix, respectively (for the case of square submatrix, it
has approximately the maximum value of the modulus of the determinant).

"""
import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve_triangular


def maxvol(A, e=1.05, k=100):
    """Compute the maximal-volume submatrix for given tall matrix.

    Args:
        A (np.ndarray): tall matrix of the shape [n, r] (n > r).
        e (float): accuracy parameter (should be >= 1). If the parameter is
            equal to 1, then the maximum number of iterations will be performed
            until true convergence is achieved. If the value is greater than
            one, the algorithm will complete its work faster, but the accuracy
            will be slightly lower (in most cases, the optimal value is within
            the range of 1.01 - 1.1).
        k (int): maximum number of iterations (should be >= 1).

    Returns:
        [np.ndarray, np.ndarray]: the row numbers "I" containing the
        maximal-volume submatrix in the form of 1D array of length "r" and
        coefficient matrix "B" in the form of 2D array of shape [n, r], such
        that A = B A[I, :] and A (A[I, :])^{-1} = B.

    Note:
        The description of the basic implementation of this algorithm is
        presented in the work: Goreinov S., Oseledets, I., Savostyanov, D.,
        Tyrtyshnikov, E., Zamarashkin, N. "How to find a good submatrix".
        Matrix Methods: Theory, Algorithms And Applications: Dedicated to the Memory of Gene Golub (2010): 247-256.

    """
    n, r = A.shape

    if n <= r:
        raise ValueError('Input matrix should be "tall"')

    P, L, U = lu(A, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(L[:r, :], Q, trans=1, check_finite=False,
        unit_diagonal=True, lower=True).T

    for _ in range(k):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= e:
            break

        I[j] = i

        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.

        B -= np.outer(bj, bi / B[i, j])

    return I, B


def maxvol_rect(A, e=1.1, dr_min=0, dr_max=None, e0=1.05, k0=10):
    """Compute the maximal-volume rectangular submatrix for given tall matrix.

    Within the framework of this function, the original maxvol algorithm is
    first called (see function "maxvol"). Then additional rows of the matrix
    are greedily added until the maximum allowed number is reached or until
    convergence.

    Args:
        A (np.ndarray): tall matrix of the shape [n, r] (n > r).
        e (float): accuracy parameter.
        dr_min (int): minimum number of added rows (should be >= 0 and <= n-r).
        dr_max (int): maximum number of added rows (should be >= 0). If the
            value is not specified, then the number of added rows will be
            determined by the precision parameter "e", while the resulting
            submatrix can even has the same size as the original matrix "A".
            If r + dr_max is greater than n, then dr_max will be set such that
            r + dr_max = n.
        e0 (float): accuracy parameter for the original maxvol algorithm
            (should be >= 1). See function "maxvol" for details.
        k0 (int): maximum number of iterations for the original maxvol algorithm
            (should be >= 1). See function "maxvol" for details.

    Returns:
        [np.ndarray, np.ndarray]: the row numbers "I" containing the rectangular
        maximal-volume submatrix in the form of 1D array of length r + dr,
        where "dr" is a number of additional selected rows (dr >= dr_min and
        dr <= dr_max) and coefficient matrix "B" in the form of 2D array of
        the shape [n, r+dr], such that A = B A[I, :].

    Note:
        The description of the basic implementation of this algorithm is
        presented in the work: Mikhalev A, Oseledets I., "Rectangular
        maximum-volume submatrices and their applications." Linear Algebra and
        its Applications 538 (2018): 187-211.

    """
    n, r = A.shape
    r_min = r + dr_min
    r_max = r + dr_max if dr_max is not None else n
    r_max = min(r_max, n)

    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError('Invalid minimum/maximum number of added rows')

    I0, B = maxvol(A, e0, k0)

    I = np.hstack([I0, np.zeros(r_max-r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B, axis=1)**2

    for k in range(r, r_max):
        i = np.argmax(F)

        if k >= r_min and F[i] <= e*e:
            break

        I[k] = i
        S[i] = 0

        v = B.dot(B[i])
        l = 1. / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)

    I = I[:B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)

    return I, B

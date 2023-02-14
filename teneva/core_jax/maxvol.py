"""Package teneva, module core_jax.maxvol: maxvol-like algorithms.

This module contains the implementation of the maxvol algorithm (function
"maxvol") and rect_maxvol algorithm (function "maxvol_rect") for matrices.
The corresponding functions find in a given matrix square and rectangular
maximal-volume submatrix, respectively (for the case of square submatrix, it
has approximately the maximum value of the modulus of the determinant).

"""
import jax
import jax.numpy as np
from jax.scipy.linalg import lu as jlu
from jax.scipy.linalg import solve_triangular as jsolve_triangular
import teneva.core_jax as teneva


@jax.jit
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
        (np.ndarray, np.ndarray): the row numbers I containing the maximal
        volume submatrix in the form of 1D array of length r and coefficient
        matrix B in the form of 2D array of shape [n, r], such that
        A = B A[I, :] and A (A[I, :])^{-1} = B.

    Note:
        The description of the basic implementation of this algorithm is
        presented in the work: Goreinov S., Oseledets, I., Savostyanov, D.,
        Tyrtyshnikov, E., Zamarashkin, N. "How to find a good submatrix".
        Matrix Methods: Theory, Algorithms And Applications: Dedicated to the Memory of Gene Golub (2010): 247-256.

    """
    n, r = A.shape

    P, L, U = jlu(A)
    I = P[:, :r].argmax(axis=0)
    Q = jsolve_triangular(U, A.T, trans=1, lower=False)
    B = jsolve_triangular(L[:r, :], Q, trans=1, unit_diagonal=True, lower=True)

    @jax.jit
    def step(args):
        I, B, i, j = args
        x = B[i, :]
        y = B[:, j]
        y = y.at[i].set(y[i] - 1)
        I = I.at[i].set(j)
        B -= np.outer(y / B[i, j], x)
        return I, B, i, j

    # @jax.jit
    def cond(data):
        I, B, b, k_cur = data
        return np.logical_and(k_cur < k, np.abs(b) > e)

    # @jax.jit
    def body(data):
        I, B, b, k_cur = data
        i, j = np.divmod(np.abs(B).argmax(), n)
        b = B[i, j]
        I, B, i, j = jax.lax.cond(np.abs(b) > e, step,
            lambda args: args, operand=(I, B, i, j))
        return I, B, b, k_cur+1

    I, B, b, k = jax.lax.while_loop(cond, body, (I, B, 2 * e, 0))
    return I[:r], B.T


def maxvol_rect(A, e=1.1, dr_min=0, dr_max=None, e0=1.05, k0=10):
    """Compute the maximal-volume rectangular submatrix for given tall matrix.

    DRAFT (works with error now) !!!

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
            determined by the precision parameter e, while the resulting
            submatrix can even has the same size as the original matrix A.
            If r + dr_max is greater than n, then dr_max will be set such
            that r + dr_max = n.
        e0 (float): accuracy parameter for the original maxvol algorithm
            (should be >= 1). See function "maxvol" for details.
        k0 (int): maximum number of iterations for the original maxvol algorithm
            (should be >= 1). See function "maxvol" for details.

    Returns:
        (np.ndarray, np.ndarray): the row numbers I containing the rectangular
        maximal-volume submatrix in the form of 1D array of length r + dr,
        where dr is a number of additional selected rows (dr >= dr_min and
        dr <= dr_max) and coefficient matrix B in the form of 2D array of shape
        [n, r+dr], such that A = B A[I, :].

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

    I0, B = maxvol(A, e0, k0)

    I = np.hstack([I0, np.zeros(r_max-r, dtype=I0.dtype)])
    S = np.ones(n, dtype=I0.dtype)
    S.at[I0].set(0.)
    F = S * np.linalg.norm(B, axis=1)**2

    # @jax.jit
    def cond(data):
        I, B, F, S, f, k_cur = data

        return np.logical_and(k_cur < r_max, f > e*e)

    # @jax.jit
    def body(data):
        I, B, F, S, f, k_cur = data
        i = np.argmax(F)
        # if k >= N_min and F[i] <= e*e: break
        I.at[k_cur].set(i)
        S.at[i].set(0.)

        v = B.dot(B[i])
        l = 1. / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)

        return I, B, F, S, F[i], k_cur+1

    I, B, F, S, f, k = jax.lax.while_loop(cond, body, (I, B, F, S, 2 * e*e, r))
    I = I[:B.shape[1]]
    #B[I] = np.eye(B.shape[1], dtype=B.dtype)

    return I, B

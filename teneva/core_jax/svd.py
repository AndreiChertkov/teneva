"""Package teneva, module core_jax.svd: SVD-based algorithms.

This module contains the basic implementation of the TT-SVD algorithm (function
svd) as well as functions for constructing the SVD decomposition (function
matrix_svd) and skeleton decomposition (matrix_skeleton) for the matrices.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def matrix_skeleton(A, r):
    """Construct truncated skeleton decomposition A = U V for the given matrix.

    Args:
        A (np.ndarray): matrix of the shape "[m, n]".
        r (int): rank for the truncated SVD decomposition.

    Returns:
        [np.ndarray, np.ndarray]: factor matrix "U" of the shape "[m, r]" and
        factor matrix "V" of the shape "[r, n]".

    """
    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=False)

    S = np.diag(np.sqrt(s[:r]))
    return U[:, :r] @ S, S @ V[:r, :]

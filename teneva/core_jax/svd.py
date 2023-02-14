"""Package teneva, module core_jax.svd: SVD-based algorithms.

This module contains the basic implementation of the TT-SVD algorithm (function
svd) as well as functions for constructing the skeleton decomposition
(matrix_skeleton) for the matrices.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def matrix_skeleton(A, r):
    """Construct truncated skeleton decomposition A = U V for the given matrix.

    Args:
        A (np.ndarray): matrix of the shape [m, n].
        r (int): rank for the truncated SVD decomposition.

    Returns:
        (np.ndarray, np.ndarray): factor matrix U of the shape [m, r] and
        factor matrix V of the shape [r, n].

    """
    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=False)

    S = np.diag(np.sqrt(s[:r]))
    return U[:, :r] @ S, S @ V[:r, :]


def svd(Y_full, r):
    """Construct TT-tensor from the given full tensor using TT-SVD algorithm.

    Args:
        Y_full (np.ndarray): tensor (multidimensional array) in the full format.
        r (int): rank of the constructed TT-tensor.

    Returns:
        list: TT-tensor, which represents the rank-r TT-approximation.

    Note:
        This function does not take advantage of jax's ability to speed up the
        code and can be slow, but it should only be meaningfully used for
        tensors of small dimensions.

    """
    d = len(Y_full.shape)
    n = Y_full.shape

    if len(set(n)) > 1:
        raise ValueError('Invalid tensor')

    if r > n[0]:
        raise ValueError('Rank can not be greater than mode size')

    Z = Y_full.copy()
    Y = []
    q = 1
    for k in n[:-1]:
        Z = Z.reshape(q * k, -1)
        G, Z = matrix_skeleton(Z, r)
        G = G.reshape(q, k, -1)
        q = G.shape[-1]
        Y.append(G)
    Y.append(Z.reshape(q, n[-1], 1))

    Yl = Y[0]
    Ym = np.array(Y[1:d-1])
    Yr = Y[-1]

    return [Yl, Ym, Yr]

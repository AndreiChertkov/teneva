"""Package teneva, module core.svd: SVD-based algorithms.

This module contains the basic implementation of the TT-SVD algorithm (function
svd) as well as new original TT-SVD-incomplete algorithm (function
svd_incomplete), which implements efficient construction of the TT-tensor based
on specially selected elements. This module also contains functions for
constructing the SVD decomposition (function matrix_svd) and skeleton
decomposition (matrix_skeleton) for the matrices.

"""
import numpy as np


from .act_one import get


def matrix_skeleton(A, e=1.E-10, r=1.E+12, hermitian=False, rel=False, give_to='m'):
    """Construct truncated skeleton decomposition A = U V for the given matrix.

    Args:
        A (np.ndarray): matrix of the shape [m, n].
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank for the SVD decomposition (> 0).
        hermitian (flag): if True, then "hermitian" SVD will be used.

    Returns:
        [np.ndarray, np.ndarray]: factor matrix U of the shape [m, q] and factor
        matrix V of the shape [q, n], where "q" is selected rank (q <= r).

    """
    if r is None:
        r = 1E+12

    U, s, V = np.linalg.svd(A, full_matrices=False, hermitian=hermitian)

    ss = s/s[0] if rel else s
    where = np.where(np.cumsum(ss[::-1]**2) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    r = max(1, min(int(r), len(s) - dlen))

    #if rel:
    #    r = max(1, min(int(r), sum(s/s[0] > e)))
    #else:
    #    r = max(1, min(int(r), sum(s > e)))

    if give_to == 'm':
        S = np.diag(np.sqrt(s[:r]))
        return U[:, :r] @ S, S @ V[:r, :]

    S = np.diag(s[:r])
    if give_to == 'l':
        return U[:, :r] @ S, V[:r, :]

    if give_to == 'r':
        return U[:, :r], S @ V[:r, :]


def matrix_svd(A, e=1.E-10, r=1.E+12, e0=1.E-14):
    """Construct truncated SVD decomposition A = U V for the given matrix.

    Args:
        A (np.ndarray): matrix of the shape [m, n].
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank for the SVD decomposition (> 0).
        e0 (float): minimum norm of the A.T A matrix (> 0). If norm less than
            this value, then rank-1 zero factors will be returned.

    Returns:
        [np.ndarray, np.ndarray]: factor matrix U of the shape [m, q] and factor
        matrix V of the shape [q, n], where "q" is selected rank (q <= r).

    """
    m, n = A.shape
    C = A @ A.T if m <= n else A.T @ A

    #if np.linalg.norm(C) < e0:
    #    return np.zeros([m, 1]), np.zeros([1, n])

    w, U = np.linalg.eigh(C)

    w[w < 0] = 0.
    w = np.sqrt(w)

    idx = np.argsort(w)[::-1]
    w = w[idx]
    U = U[:, idx]

    s = w**2
    where = np.where(np.cumsum(s[::-1]) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    rank = max(1, min(int(r), len(s) - dlen))
    w = w[:rank]
    U = U[:, :rank]

    V = ((1. / w)[:, np.newaxis] * U.T) @ A if m <= n else U.T
    U = U * w if m <= n else A @ U

    return U, V


def svd(Y_full, e=1E-10, r=1.E+12):
    """Construct TT-tensor from the given full tensor using TT-SVD algorithm.

    Args:
        Y_full (np.ndarray): tensor (multidimensional array) in the full format.
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank of the constructed TT-tensor (> 0).

    Returns:
        list: TT-tensor, which represents an approximation with a given
        accuracy (e) and a TT-rank constraint (r) for the given full tensor.

    """
    n = Y_full.shape
    Z = Y_full.copy()
    Y = []
    q = 1
    for k in n[:-1]:
        Z = Z.reshape(q * k, -1)
        G, Z = matrix_skeleton(Z, e, r)
        G = G.reshape(q, k, -1)
        q = G.shape[-1]
        Y.append(G)
    Y.append(Z.reshape(q, n[-1], 1))
    return Y


def svd_matrix(Y_full, e=1E-10, r=1.E+12):
    """Construct QTT-matrix from the given full matrix using TT-SVD algorithm.

    Args:
        Y_full (np.ndarray): matrix of the shape "2^q x 2^q" in the full format.
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank of the constructed TT-tensor (> 0).

    Returns:
        list: TT-tensor / QTT-matrix, which represents an approximation with a
            given accuracy (e) and a TT-rank constraint (r) for the given full
            matrix (it has "q" dimensions and mode equals "4").

    Note:
        The matrix size should be the power of 2, and only square matrices are
        supported.

    """
    q = int(np.log2(Y_full.shape[0]))

    Z_full = Y_full.reshape([2]*(2*q), order='F')

    ind1 = np.arange(0, q).reshape(-1, 1)
    ind2 = np.arange(q, 2*q).reshape(-1, 1)
    prm = np.hstack((ind1, ind2)).reshape(-1)
    Z_full = Z_full.transpose(prm)

    Z_full = Z_full.reshape([4]*q, order='F')

    return svd(Z_full, e, r)


def svd_incomplete(I, Y, idx, idx_many, e=1.E-10, r=1.E+12):
    """Construct TT-tensor from the given specially selected samples.

    Args:
        I (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        Y (np.ndarray): values of the tensor for multi-indices I in the form of
            array of the shape [samples].
        idx (np.ndarray): starting poisitions in generated samples for the
            corresponding dimensions in the form of array of the shape [d+1].
        idx_many (np.ndarray): numbers of points for the right unfoldings in
            generated samples in the form of array of the shape [d].
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank of the constructed TT-tensor (> 0).

    Returns:
        list: TT-tensor, which represents an approximation with a given
        accuracy (e) and a TT-rank constraint (r) for the full tensor.

    Note:
        The samples I and opts idx and idx_many should be generated by the
        function "sample_tt".

    """
    shapes = np.max(I, axis=0) + 1
    d = len(shapes)

    Y_curr = Y[idx[0]:idx[1]]
    Y_curr = Y_curr.reshape(shapes[0], -1, order='C')
    Y_curr, _ = matrix_skeleton(Y_curr, e, r)
    Y_res = [Y_curr[None, ...]]

    for mode in range(1, d):
        # The mode-th TT-core will have the shape r0 x n x r1:
        r0 = Y_res[-1].shape[-1]
        r1 = r if mode < d-1 else 1
        n = shapes[mode]

        I_curr = I[idx[mode]:idx[mode+1], :]
        M = np.array([get(Y_res[:mode], i, to_item=False)
            for i in I_curr[::idx_many[mode], :mode]])

        Y_curr = Y[idx[mode]:idx[mode+1]].reshape(-1, idx_many[mode], order='C')
        if Y_curr.shape[1] > r1:
            Y_curr, _ = matrix_skeleton(Y_curr, e, r1)
        r1 = Y_curr.shape[1]

        G = np.empty([r0, n, r1])
        step = Y_curr.shape[0] // n
        for i in range(n):
            A = M[i*step:(i+1)*step]
            b = Y_curr[i*step:(i+1)*step]
            G[:, i, :] = np.linalg.lstsq(A, b, rcond=-1)[0]
        Y_res.append(G)

    return Y_res

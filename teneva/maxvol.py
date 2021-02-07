import numpy as np
import scipy as sp
from scipy.linalg import get_lapack_funcs, get_blas_funcs


def maxvol(A, e=1.05, K=100):
    N, r = A.shape
    assert e >= 1 and N > r and K > 0
    P, L, U = lu(A)
    I = P.argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, lower=False)
    B = solve_triangular(L[:r, :], Q, trans=1, unit_diagonal=True, lower=True)
    for _ in range(K):
        i, j = np.divmod(np.abs(B).argmax(), N)
        if np.abs(B[i, j]) <= e: break
        x = B[i, :]
        y = B[:, j].copy()
        y[i]-= 1.
        I[i] = j
        B-= np.outer(y / B[i, j], x)
    return I[:r], B.T


def rect_maxvol(A, tol=1., maxK=None, min_add_K=None, minK=None,
        start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1):
    tol2 = tol**2
    N, r = A.shape
    if N <= r:
        return np.arange(N, dtype=np.int32), np.eye(N, dtype=A.dtype)
    if maxK is None or maxK > N:
        maxK = N
    if maxK < r:
        maxK = r
    if minK is None or minK < r:
        minK = r
    if minK > N:
        minK = N
    if min_add_K is not None:
        minK = max(minK, r + min_add_K)
    if minK > maxK:
        minK = maxK
    if top_k_index == -1 or top_k_index > N:
        top_k_index = N
    if top_k_index < r:
        top_k_index = r

    index = np.zeros(N, dtype=np.int32)
    chosen = np.ones(top_k_index)
    tmp_index, C = maxvol(A, 1.05, start_maxvol_iters, top_k_index)
    index[:r] = tmp_index
    chosen[tmp_index] = 0
    C = np.asfortranarray(C)
    row_norm_sqr = np.array([chosen[i]*np.linalg.norm(C[i], 2)**2 for
        i in range(top_k_index)])
    i = np.argmax(row_norm_sqr)
    K = r

    while (row_norm_sqr[i] > tol2 and K < maxK) or K < minK:
        index[K] = i
        chosen[i] = 0
        c = C[i].copy()
        v = C.dot(c.conj())
        l = 1.0/(1+v[i])
        C += -l*v[:, None]@c[None, :]
        C = np.hstack([C, l*v.reshape(-1,1)])
        row_norm_sqr -= (l*v[:top_k_index]*v[:top_k_index].conj()).real
        row_norm_sqr *= chosen
        i = row_norm_sqr.argmax()
        K += 1

    if identity_submatrix:
        C[index[:K]] = np.eye(K, dtype=C.dtype)

    return index[:K].copy(), C

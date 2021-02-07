import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve_triangular


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


def rect_maxvol(A, e, maxK, min_add_K=0, start_maxvol_iters=10):
    N, r = A.shape
    assert e >= 1 and N > r and maxK >= r and maxK <= N
    minK = min(maxK, r + min_add_K)
    I_tmp, B = maxvol(A, 1.05, start_maxvol_iters)
    I = np.zeros(N, dtype=np.int32)
    I[:r] = I_tmp
    C = np.ones(N, dtype=np.int32)
    C[I_tmp] = 0
    F = C * np.linalg.norm(B, axis=1)**2
    for k in range(r, maxK):
        i = np.argmax(F)
        if k >= minK and F[i] <= e*e: break
        I[k] = i
        C[i] = 0
        c = B[i].copy()
        v = B.dot(c)
        l = 1. / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, c), l * v.reshape(-1, 1)])
        F = C * (F - l * v[:N] * v[:N])
    I = I[:B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)
    return I, B

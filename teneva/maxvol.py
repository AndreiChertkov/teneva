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


def rect_maxvol(A, e, N_min, N_max, e0=1.05, K0=10):
    N, r = A.shape
    assert e >= 1 and N > r and N_min >= r and N_min <= N_max and N_max <= N
    I_tmp, B = maxvol(A, e0, K0)
    I = np.hstack([I_tmp, np.zeros(N_max-r, dtype=I_tmp.dtype)])
    S = np.ones(N, dtype=np.int32)
    S[I_tmp] = 0
    F = S * np.linalg.norm(B, axis=1)**2
    for k in range(r, N_max):
        i = np.argmax(F)
        if k >= N_min and F[i] <= e*e: break
        I[k] = i
        S[i] = 0
        v = B.dot(B[i])
        l = 1. / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
    I = I[:B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)
    return I, B

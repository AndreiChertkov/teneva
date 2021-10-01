import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve_triangular


def maxvol(A, e=1.05, k=100):
    n, r = A.shape
    assert e >= 1 and n > r and k > 0
    P, L, U = lu(A, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(L[:r, :], Q, trans=1, check_finite=False,
                         unit_diagonal=True, lower=True).T
    for _ in range(k):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= e: break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.
        B -= np.outer(bj, bi / B[i, j])
    return I, B


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

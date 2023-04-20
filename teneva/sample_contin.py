"""Package teneva, module sample_contin: sampling from func. TT-tensor.

This module contains function for sampling from the functiona TT-tensor (i.e.,
the tensor of interpolation coeeficients).

"""
import itertools
import numpy as np
import teneva


def sample_contin(A, cores_are_prepared=False):
    """
    to test this code, run:

        np.random.seed(42)
        Y1 = teneva.rand([4]*2, 2)


        np.random.seed(42)
        pnts = np.array([sample_contin(Y1) for _ in range(10000)])

        plt.figure(figsize=(8, 8))
        plt.scatter(pnts[:, 0], pnts[:, 1], s=[.5]*pnts.shape[0]);

        N = 512
        X_m = np.linspace(-1, 1, N)
        x, y = np.meshgrid(X_m, X_m)
        x = x.reshape(-1)
        y = y.reshape(-1)
        X = np.array([x, y]).T

        p_vals = teneva.cheb_get(X, Y1, -1, 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(p_vals.reshape(N, -1)**2, origin='lower');
        plt.colorbar();
    """


    if not cores_are_prepared:
        A = [np.copy(i) for i in A]
        for G in A:
            G[:, 0, :] *= np.sqrt(2.)

        A = teneva.orthogonalize(A, 0)


    X_prev = None
    G_prev = np.array([1.])

    for G in A:
        X_prev, G_prev = _step_sample_contin(X_prev, G_prev, G)

    return X_prev


def _step_sample_contin(x_prev, G_prev, G):
    """
    G_prev -- matix of vector of prev cores (1 by H matrix of basis functions values)
    G -- current core

    returns next G_prev as well as Xs'
    """
    G_cur = np.einsum("i,ijk->jk", G_prev, G)

    G0 = np.copy(G_cur)
    G0[0, :] /= np.sqrt(2)

    x_new = np.empty(1 if x_prev is None else x_prev.shape[0] + 1)
    if x_prev is not None:
        x_new[:-1] = x_prev


    p = sum([np.polynomial.chebyshev.Chebyshev(cf)**2 for cf in G0.T])
    x_cur = x_new[-1] = _sample_poly_1(p)

    H_new = _cheb_my_poly(np.array(x_cur), G.shape[1])
    G_new = np.einsum("i,ik", H_new, G_cur)

    return x_new, G_new


# we can take this func from optima_cont
def _cheb_my_poly(X, n):
    X = np.asarray(X)
    to_red = False
    if X.ndim == 0:
        X = X[None]
        to_red = True

    res = np.ones([X.shape[0], n])
    if n > 1:
        res[:, 1] = X

    for i in range(2, n):
        res[:, i] = 2*X*res[:, i - 1] - res[:, i - 2]

    res[:, 0] = np.sqrt(0.5) # Нормированные на одно и то же полиномы берём

    if to_red:
        res = res[0]

    return res


def _sample_poly_1(p2):
    p2_int = p2.integ(lbnd=-1)

    xi = np.random.uniform()

    p2_int_sh = p2_int - xi*(p2_int(1))
    roots = [np.real(i) for i in p2_int_sh.roots() if np.imag(i) == 0]
    assert len(roots) == 1
    return roots[0]

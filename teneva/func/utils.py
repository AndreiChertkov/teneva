"""Package teneva, module func.utils: helpers for Function class."""
import numpy as np
import pickle
from time import perf_counter as tpc


def _cores_addition(X, a0=0):
    Y = []
    for x in X:
        G = np.ones([2, len(x), 2])
        G[1, :, 0] = 0.
        G[0, :, 1] = x
        Y.append(G)

    Y[0] = Y[0][0:1, ...].copy()
    Y[-1] = Y[-1][..., 1:2].copy()
    Y[-1][0, :, 0] += a0
    return Y


def _cores_mults(X):
    return [x[None, :, None] for x in X]


def _data_check(X, Y_real, func):
    if X is None:
        return -1., 0.
    t = tpc()
    Y_appr = func(X)
    e = np.linalg.norm(Y_appr - Y_real) / np.linalg.norm(Y_real)
    t = tpc() - t
    return e, t


def _data_load(fpath, m=None, n_ref=None, kind_ref=None):
    data = np.load(fpath, allow_pickle=True)
    I = data.get('I')
    X = data.get('X')
    Y = data.get('Y')
    t = data.get('t').item()
    n = data.get('n')
    kind = data.get('kind').item()

    if n_ref is not None:
        if len(n) != len(n_ref):
            raise ValueError('Invalid dimension for the loaded data')

        for [n_, n_ref_] in zip(n, n_ref):
            if n_ != n_ref_:
                raise ValueError('Invalid grid size for the loaded data')

    if kind_ref is not None:
        if kind != kind_ref:
            raise ValueError('Invalid grid kind for the loaded data')

    if I is not None and len(I.shape) < 2:
        I = None
    if X is not None and len(X.shape) < 2:
        X = None
    if Y is not None and len(Y.shape) < 1:
        Y = None

    if m is not None:
        m = int(m)
        if m > Y.size:
            raise ValueError('Invalid subset of data')

        I = I[:m, :] if I is not None else None
        X = X[:m, :] if X is not None else None
        Y = Y[:m] if Y is not None else None

    return I, X, Y, t


def _data_prep(I=None, X=None, Y=None, perm=None):
    I_new = None
    if I is not None:
        I_new = np.asanyarray(I, dtype=int).copy()
        if perm is not None:
            I_new = I_new[:, perm]

    X_new = None
    if X is not None:
        X_new = np.asanyarray(X, dtype=float).copy()
        if perm is not None:
            X_new = X_new[:, perm]

    Y_new = None
    m_new = 0
    if Y is not None:
        Y_new = np.asanyarray(Y, dtype=float).copy()
        m_new = len(Y_new)

    return I_new, X_new, Y_new, m_new


def _data_save(fpath, I=None, X=None, Y=None, t=None, n=None, kind=None):
    np.savez_compressed(fpath, I=I, X=X, Y=Y, t=t, n=n, kind=kind)


def _noise(y, noise_add=None, noise_mul=None):
    if y is not None and noise_mul is not None:
        y *= noise_mul * np.random.randn(y.size) + 1

    if y is not None and noise_add is not None:
        y += noise_add * np.random.randn(y.size)

    return y

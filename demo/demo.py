"""Approximation of the multivariate Rosenbrock function with noise."""
import numpy as np
from scipy.optimize import rosen
from time import perf_counter as tpc


import teneva


np.random.seed(1234567890)


# Parameters:
d         = 100         # Dimension of the function
A         = [-5.] * d   # Lower bound for spatial grid
B         = [+5.] * d   # Upper bound for spatial grid
N         = [100] * d   # Shape of the tensor (it may be non-uniform)
M_trn     = 100000      # Number of train points (for ANOVA and ALS)
M_tst     = 100000      # Number of test points
nswp      = 1           # Sweep number for cross
nswp_als  = 20          # Sweep number for ALS
eps       = 1.E-8       # Desired accuracy
kr        = 1           # Cross parameter (kickrank)
rf        = 1           # Cross parameter
order     = 1           # ANOVA order (1 or 2)
r0        = 3           # TT-rank for Cross initial guess
r         = 3           # TT-rank for ALS and ANOVA


# Target function:
def func(I):
    X = teneva.ind2poi(I, A, B, N)
    return rosen(X.T)


# Train data (for ALS and ANOVA):
I_trn = teneva.lhs(N, M_trn)
Y_trn = func(I_trn)


# Test data:
I_tst = np.vstack([np.random.choice(N[i], M_tst) for i in range(d)]).T
Y_tst = func(I_tst)


def als():
    Y = teneva.rand(N, r)

    Y = teneva.als(I_trn, Y_trn, Y, nswp_als)

    return Y, Y_trn.size, 0


def anova():
    anova = teneva.ANOVA(order, I_trn, Y_trn)
    Y = anova.cores(r)

    return Y, Y_trn.size, 0


def anova_als():
    anova = teneva.ANOVA(order, I_trn, Y_trn)
    Y = anova.cores(r)

    Y = teneva.als(I_trn, Y_trn, Y, nswp_als)

    return Y, Y_trn.size, 0


def anova_cross_cache():
    cache, info = {}, {}

    anova = teneva.ANOVA(order, I_trn, Y_trn)
    Y = anova.cores(r)

    Y = teneva.cross(func, Y, nswp, kr, rf, cache, info=info)
    Y = teneva.truncate(Y, eps)

    return Y, info['k_evals'], info['k_cache']


def anova_cross_cache_als():
    cache, info = {}, {}

    anova = teneva.ANOVA(order, I_trn, Y_trn)
    Y = anova.cores(r)

    Y = teneva.cross(func, Y, nswp, kr, rf, cache, info=info)
    Y = teneva.truncate(Y, eps)

    I_trn_new = np.array([teneva.str2ind(s) for s in cache.keys()], dtype=int)
    Y_trn_new = np.array([y for y in cache.values()])
    Y = teneva.als(I_trn_new, Y_trn_new, Y, nswp_als)

    return Y, Y_trn.size + info['k_evals'], info['k_cache']


def cross():
    info = {}

    Y = teneva.rand(N, r0)

    Y = teneva.cross(func, Y, nswp, kr, rf, info=info)
    Y = teneva.truncate(Y, eps)

    return Y, info['k_evals'], 0


def cross_cache():
    cache, info = {}, {}

    Y = teneva.rand(N, r0)

    Y = teneva.cross(func, Y, nswp, kr, rf, cache, info=info)
    Y = teneva.truncate(Y, eps)

    return Y, info['k_evals'], info['k_cache']


def cross_cache_als():
    cache, info = {}, {}

    Y = teneva.rand(N, r0)

    Y = teneva.cross(func, Y, nswp, kr, rf, cache, info=info)
    Y = teneva.truncate(Y, eps)

    I_trn_new = np.array([teneva.str2ind(s) for s in cache.keys()], dtype=int)
    Y_trn_new = np.array([y for y in cache.values()])
    Y = teneva.als(I_trn_new, Y_trn_new, Y, nswp_als)

    return Y, Y_trn.size + info['k_evals'], info['k_cache']


def proc(Y, k, k_cache, t_build, name):
    get = teneva.getter(Y)

    Z = np.array([get(i) for i in I_trn])
    e_trn = np.linalg.norm(Z - Y_trn) / np.linalg.norm(Y_trn)

    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)

    text = '\n'
    text += f'-------------- > {name}\n'
    text += f'Error on train : {e_trn:-10.2e}\n'
    text += f'Error on test  : {e_tst:-10.2e}\n'
    text += f'Tensor TT-rank : {teneva.erank(Y):-10.1f}\n'
    text += f'Build time     : {t_build:-10.2f}\n'
    text += f'Function evals : {k:-10d}'
    if k_cache:
        text += f' (+ cache {k_cache})'

    print(text)


def run():
    text = '\n'
    text += f'> Approximation of {d}-dim Rosenbrock function on uniform grid'
    print(text)

    t = tpc()
    Y, k, k_cache = anova()
    proc(Y, k, k_cache, tpc() - t, 'TT-ANOVA')

    t = tpc()
    Y, k, k_cache = als()
    proc(Y, k, k_cache, tpc() - t, 'TT-ALS')

    t = tpc()
    Y, k, k_cache = anova_als()
    proc(Y, k, k_cache, tpc() - t, 'TT-ANOVA + TT-ALS')

    t = tpc()
    Y, k, k_cache = cross()
    proc(Y, k, k_cache, tpc() - t, 'TT-Cross')

    t = tpc()
    Y, k, k_cache = cross_cache()
    proc(Y, k, k_cache, tpc() - t, 'TT-Cross-cache')

    t = tpc()
    Y, k, k_cache = anova_cross_cache()
    proc(Y, k, k_cache, tpc() - t, 'TT-ANOVA + TT-Cross-cache')

    t = tpc()
    Y, k, k_cache = cross_cache_als()
    proc(Y, k, k_cache, tpc() - t, 'TT-Cross-cache + TT-ALS')

    # t = tpc()
    # Y, k, k_cache = anova_cross_cache_als()
    # proc(Y, k, k_cache, tpc() - t, 'TT-ANOVA + TT-Cross-cache + TT-ALS')


if __name__ == '__main__':
    run()

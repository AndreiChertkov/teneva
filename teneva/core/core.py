"""Package teneva, module core.core: operations with individual TT-cores.

This module contains functions to work directly with individual TT-cores.

"""
import numpy as np


from .svd import matrix_svd


def core_stab(G, p0=0, thr=1.E-100):
    # TODO: add docstring and add into demo
    v_max = np.max(np.abs(G))

    if v_max <= thr:
        return G, p0

    p = int(np.floor(np.log2(v_max)))
    Q = G / 2**p

    return Q, p0 + p


def core_tt_to_qtt(core, e=0, r_max=int(1e12)):
    # TODO: add docstring and add into demo
    r1, n, r2 = core.shape
    d = int(np.log2(n))
    assert 2**d == n

    A = core.reshape(-1, r2, order='F')
    A, V0 = matrix_svd(A, e=e, r=r_max)

    res = []
    for i in range(d-1):
        As = A.shape[0] // 2
        r = A.shape[1]
        A = np.hstack([A[:As], A[As:]])
        A, V = matrix_svd(A, e=e, r=r_max)
        res.append(V.reshape(-1, 2, r, order='C'))


    res.append(A.reshape(r1, 2, -1, order='F'))
    res[0] = np.einsum("ijk,kl", res[0], V0)

    return res[::-1]

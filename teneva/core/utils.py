"""Package teneva, module core.utils: various (inner) helper functions.

This module contains some helper functions which used in other "core" modules.

"""
import numpy as np


from .maxvol import maxvol
from .maxvol import maxvol_rect


def _is_num(A):
    return isinstance(A, (int, float))


def _maxvol(A, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100):
    n, r = A.shape
    dr_max = min(dr_max, n - r)
    dr_min = min(dr_min, dr_max)

    if n <= r:
        I = np.arange(n, dtype=int)
        B = np.eye(n, dtype=float)
    elif dr_max == 0:
        I, B = maxvol(A, tau0, k0)
    else:
        I, B = maxvol_rect(A, tau, dr_min, dr_max, tau0, k0)

    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _range(n):
    return np.arange(n).reshape(-1, 1)


def _reshape(A, n, order='F'):
    return np.reshape(A, n, order=order)

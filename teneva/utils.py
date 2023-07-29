"""Package teneva, module utils: various (inner) helper functions.

This module contains some helper functions which used in other modules.

"""
import numpy as np
import teneva
from time import perf_counter as tpc


def _info_appr(info, t, nswp, e, e_vld, log=False):
    info['t'] = tpc() - t

    if info['stop'] is None:
        if e_vld is not None and info['e_vld'] >= 0:
            if info['e_vld'] <= e_vld and not np.isinf(info['e_vld']):
                info['stop'] = 'e_vld'

    if info['stop'] is None:
        if e is not None and info['e'] >= 0:
            if info['e'] <= e and not np.isinf(info['e']):
                info['stop'] = 'e'

    if info['stop'] is None:
        if nswp is not None:
            if info['nswp'] >= nswp:
                info['stop'] = 'nswp'

    if log:
        text = ''

        if info['nswp'] == 0:
            text += f'# pre | '
        else:
            text += f'# {info["nswp"]:-3d} | '

        text += f'time: {info["t"]:-10.3f} | '

        if 'm' in info:
            if 'with_cache' in info and info['with_cache']:
                text += f'evals: {info["m"]:-8.2e} '
                text += f'(+ {info["m_cache"]:-8.2e}) | '
            else:
                text += f'evals: {info["m"]:-8.2e} | '

        text += f'rank: {info["r"]:-5.1f} | '

        if info['e_vld'] >= 0:
            text += f'e_vld: {info["e_vld"]:-7.1e} | '

        if info['e'] >= 0:
            text += f'e: {info["e"]:-7.1e} | '

        if info['stop']:
            text += f'stop: {info["stop"]} | '

        print(text)

    return info['stop']


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
        I, B = teneva.maxvol(A, tau0, k0)
    else:
        I, B = teneva.maxvol_rect(A, tau, dr_min, dr_max, tau0, k0)

    return I, B


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def _rand(seed=None):
    if seed is None or isinstance(seed, int):
        return np.random.default_rng(seed)
    else:
        return seed


def _range(n):
    return np.arange(n).reshape(-1, 1)


def _reshape(A, n, order='F'):
    return np.reshape(A, n, order=order)


def _vector_index_expand(q, i):
    if i < 0:
        if i == -1:
            ind = [1] * q
        else:
            raise ValueError('Only "-1" is supported for negative indices.')
    else:
        ind = []
        for _ in range(q):
            ind.append(i % 2)
            i = int(i / 2)
        if i > 0:
            raise ValueError('Index is out of range.')

    return ind


def _vector_index_prepare(q, i):
    n = 1 << q
    if i >= n or i < -n:
        raise ValueError('Incorrect index.')
    return i if i >= 0 else n + i

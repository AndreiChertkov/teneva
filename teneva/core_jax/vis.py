"""Package teneva, module core_jax.vis: visualization methods for tensors.

This module contains the functions for visualization of TT-tensors.

"""
import jax.numpy as np


def show(Y):
    """Check and display mode size and TT-rank of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    """
    if not isinstance(Y, list) or len(Y) != 3:
        raise ValueError('Invalid TT-tensor')

    Yl, Ym, Yr = Y

    if not isinstance(Yl, np.ndarray) or len(Yl.shape) != 3:
        raise ValueError('Invalid left core of TT-tensor')

    if not isinstance(Ym, np.ndarray) or len(Ym.shape) != 4:
        raise ValueError('Invalid middle cores of TT-tensor')

    if not isinstance(Yr, np.ndarray) or len(Yr.shape) != 3:
        raise ValueError('Invalid right core of TT-tensor')

    if Ym.shape[1] != Ym.shape[3]:
        raise ValueError('Invalid shape of middle cores for TT-tensor')

    d = Ym.shape[0] + 2
    n = Ym.shape[2]
    r = Ym.shape[3]

    if r > n:
        raise ValueError('TT-rank should be no greater than mode size')

    if Yl.shape[0] != 1 or Yl.shape[1] != n or Yl.shape[2] != r:
        raise ValueError('Invalid shape of left core for TT-tensor')

    if Yr.shape[0] != r or Yr.shape[1] != n or Yr.shape[2] != 1:
        raise ValueError('Invalid shape of right core for TT-tensor')

    text = f'TT-tensor-jax | d = {d:-5d} | n = {n:-5d} | r = {r:-5d} |'
    print(text)

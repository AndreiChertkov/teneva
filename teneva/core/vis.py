"""Package teneva, module core.vis: visualization methods for tensors.

This module contains the functions for visualization of TT-tensors.

"""
import numpy as np
import teneva


def show(Y):
    """Display (print) mode sizes and TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    """
    d = len(Y)
    n = teneva.shape(Y)
    r = teneva.ranks(Y)

    text = f'TT-{d}D : '
    for k in range(d):
        text += f'|{n[k]}|'
        if k < d-1:
            text += f'- {r[k+1]} -'
    print(text)

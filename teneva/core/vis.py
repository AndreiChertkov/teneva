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

    text1 = f'TT-tensor {d:-5d}D : '
    text2 = f'<rank>  = {teneva.erank(Y):-6.1f} : '

    for k in range(d):
        text1 += ' ' * max(0, len(text2)-len(text1)-1)
        text1 += f'|{n[k]}|'

        if k < d-1:
            text2 += ' ' * (len(text1)-len(text2)-1)
            text2 += f'\\{r[k+1]}/'

    print(text1 + '\n' + text2)

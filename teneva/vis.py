"""Package teneva, module vis: visualization methods for tensors.

This module contains the functions for visualization of TT-tensors.

"""
import numpy as np
import teneva


def show(Y):
    """Check and display mode sizes and TT-ranks of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    """
    if not isinstance(Y, list) or len(Y) == 0:
        raise ValueError('Invalid TT-tensor')

    d = len(Y)
    n = []
    r = [1]

    for G in Y:
        if not isinstance(G, np.ndarray) or len(G.shape) != 3:
            raise ValueError('Invalid core for TT-tensor')

        if G.shape[0] != r[-1]:
            raise ValueError('Invalid shape of core for TT-tensor')

        n.append(G.shape[1])
        r.append(G.shape[2])

    if r[-1] != 1:
        raise ValueError('Invalid shape of core for TT-tensor')

    text1 = f'TT-tensor {d:-5d}D : '
    text2 = f'<rank>  = {teneva.erank(Y):-6.1f} : '

    for k in range(d):
        text1 += ' ' * max(0, len(text2)-len(text1)-1)
        text1 += f'|{n[k]}|'

        if k < d-1:
            text2 += ' ' * (len(text1)-len(text2)-1)
            text2 += f'\\{r[k+1]}/'

    print(text1 + '\n' + text2)

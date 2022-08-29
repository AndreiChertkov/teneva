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
    n = teneva.shape(Y)
    r = teneva.ranks(Y)
    k = int(np.ceil(np.log10(max(n)+1))) + 1
    l = max(int(np.ceil(np.log10(max(r)+1))) + 1, k)
    form_str = '{:^' + str(l) + '}'

    s0 = ' '*(l//2)
    s1 = s0 + ''.join([form_str.format(k) for k in n])
    s2 = s0 + ''.join([form_str.format('/ \\') for _ in n])
    s3 = ''.join([form_str.format(q) for q in r])

    print(f'{s1}\n{s2}\n{s3}\n')

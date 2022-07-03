"""Package teneva, module collection.vectors: various useful QTT-vectors.

This module contains the collection of functions for explicit construction of
various useful QTT-vectors (delta function and others).

"""
import numpy as np


def vector_delta(q, i, v=1.):
    """Build QTT-vector that is zero everywhere except for a given index.

    Construct a QTT-vector of the length "2^q" with only one nonzero element in
    position "i", that is equal to a given value "v".

    Args:
        q (int): quantization level. The resulting vector will have the total
            length 2^q (> 0).
        i (int): the index for nonzero element (< 2^q). Note that "negative
            index notation" is supported.
        v (float): the value of the vector at index "i".

    Returns:
        list: TT-tensor representing the QTT-vector.

    """
    i = _index_prepare(q, i)
    ind = _index_expand(q, i)
    Y = []
    for k in range(q):
        G = np.zeros((1, 2, 1))
        G[0, ind[k], 0] = 1.
        Y.append(G)
    Y[-1][0, ind[-1], 0] = v
    return Y


def _index_expand(q, i):
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


def _index_prepare(q, i):
    n = 1 << q
    if i >= n or i < -n:
        raise ValueError('Incorrect index.')
    return i if i >= 0 else n + i

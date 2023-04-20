"""Package teneva, module vectors: various useful QTT-vectors.

This module contains the collection of functions for explicit construction of
various useful QTT-vectors (delta function and others).

"""
import numpy as np
import teneva


def vector_delta(q, i, v=1.):
    """Build QTT-vector that is zero everywhere except for a given index.

    Construct a QTT-vector of the length 2^q with only one nonzero element in
    position i, that is equal to a given value v.

    Args:
        q (int): quantization level. The resulting vector will have the total
            length 2^q.
        i (int): the index for nonzero element (< 2^q). Note that "negative
            index notation" is supported.
        v (float): the value of the vector at index i.

    Returns:
        list: TT-tensor representing the QTT-vector.

    """
    i = teneva._vector_index_prepare(q, i)
    ind = teneva._vector_index_expand(q, i)
    Y = []
    for k in range(q):
        G = np.zeros((1, 2, 1))
        G[0, ind[k], 0] = 1.
        Y.append(G)
    Y[-1][0, ind[-1], 0] = v
    return Y

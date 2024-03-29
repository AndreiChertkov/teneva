"""Package teneva, module matrices: various useful QTT-matrices.

This module contains the collection of functions for explicit construction of
various useful QTT-matrices (delta function and others).

"""
import numpy as np
import teneva


def matrix_delta(q, i, j, v=1.):
    """Build QTT-matrix that is zero everywhere except for a given 2D index.

    Args:
        q (int): quantization level. The resulting matrix will have the shape
            2^q x 2^q.
        i (int): the col index for nonzero element (< 2^q). Note that "negative
            index notation" is supported.
        j (int): the row index for nonzero element (< 2^q). Note that "negative
            index notation" is supported.
        v (float): the value of the matrix at index i, j.

    Returns:
        list: TT-tensor with 4D TT-cores representing the QTT-matrix.

    """
    i = teneva._vector_index_prepare(q, i)
    j = teneva._vector_index_prepare(q, j)
    ind_col = teneva._vector_index_expand(q, i)
    ind_row = teneva._vector_index_expand(q, j)
    Y = []
    for k in range(q):
        G = np.zeros((1, 2, 2, 1))
        G[0, ind_col[k], ind_row[k], 0] = 1.
        Y.append(G)
    Y[-1][0, ind_col[-1], ind_row[-1], 0] = v
    return Y

"""Package teneva, module core_jax.transformation: transformation of TT-tensors.

This module contains the function for transformation of the TT-tensor into full
(numpy) format.

"""
import jax.numpy as np


def full(Y):
    """Export TT-tensor to the full (numpy) format.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: multidimensional array related to the given TT-tensor.

    Note:
         This function can only be used for relatively small tensors, because
         the resulting tensor will have "n^d" elements and may not fit in memory
         for large dimensions.

    """
    Z = Y[0][0, :, :]
    for i in range(len(Y[1])):
        Z = np.tensordot(Z, Y[1][i], 1)
    Y_full = np.tensordot(Z, Y[2][:, :, 0], 1)
    return Y_full

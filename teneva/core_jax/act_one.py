"""Package teneva, module core_jax.act_one: single TT-tensor operations.

This module contains the basic operations with one TT-tensor (Y), including
"copy", "get", "sum", etc.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def copy(Y):
    """Return a copy of the given TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        list: TT-tensor, which is a copy of the given TT-tensor.

    """
    return [Y[0].copy(), Y[1].copy(), Y[2].copy()]


def get(Y, k):
    """Compute the element of the TT-tensor.

    Args:
        Y (list): "d"-dimensional TT-tensor.
        k (np.ndarray): the multi-index for the tensor of the length "d".

    Returns:
        np.ndarray of size 1: the element of the TT-tensor.

    """
    def body(q, data):
        i, G = data
        q = np.einsum('q,qr->r', q, G[:, i, :])
        return q, None

    Yl, Ym, Yr = Y

    q = Yl[0, k[0], :]
    q, _ = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, _ = body(q, (k[-1], Yr))

    return q[0]


def get_many(Y, K):
    """Compute the elements of the TT-tensor on many multi-indices.

    Args:
        Y (list): "d"-dimensional TT-tensor.
        K (np.ndarray): the multi-indices for the tensor in the of the shape
            "[samples, d]".

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices "K" (array
        of the length "samples").

    """
    def body(Q, data):
        i, G = data
        Q = np.einsum('kq,qkr->kr', Q, G[:, i, :])
        return Q, None

    Yl, Ym, Yr = Y

    Q = Yl[0, K[:, 0], :]
    Q, _ = jax.lax.scan(body, Q, (K[:, 1:-1].T, Ym))
    Q, _ = body(Q, (K[:, -1], Yr))

    return Q[:, 0]

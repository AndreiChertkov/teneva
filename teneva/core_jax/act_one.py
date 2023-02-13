"""Package teneva, module core_jax.act_one: single TT-tensor operations.

This module contains the basic operations with one TT-tensor (Y), including
"copy", "get", "sum", etc.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def get(Y, k):
    """Compute the element of the TT-tensor.

    Args:
        Y (list): "d"-dimensional TT-tensor.
        k (list, jax.numpy.array): the multi-index for the tensor in the form
            of a list of "d" tensor indices for each tensor mode.

    Returns:
        float: the element of the TT-tensor.

    """
    def body(q, data):
        i, G = data
        q = np.einsum('q,qr->r', q, G[:, i, :])
        return q, None

    Yl, Ym, Yr = Y

    q, _ = body(np.ones(1), (k[0], Yl))
    q, _ = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, _ = body(q, (k[-1], Yr))

    return q[0]


def get_many(Y, K):
    """Compute the elements of the TT-tensor on many multi-indices.

    Args:
        Y (list): "d"-dimensional TT-tensor.
        K (list of list, jax.numpy.array): the multi-indices for the tensor in
            the form of a list of lists or array of the shape "[samples, d]".

    Returns:
        jax.numpy.array: the elements of the TT-tensor for multi-indices "K"
        (array of the length "samples").

    """
    def body(Q, data):
        i, G = data
        Q = np.einsum('kq,qkr->kr', Q, G[:, i, :])
        return Q, None

    Yl, Ym, Yr = Y

    Q, _ = body(np.ones(1), (K[:, 0], Yl))
    Q, _ = jax.lax.scan(body, Q, (k[:, 1:-1], Ym))
    Q, _ = body(Q, (K[:, -1], Yr))

    return Q[:, 0]

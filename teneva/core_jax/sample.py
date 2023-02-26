"""Package teneva, module core_jax.sample: random sampling for/from TT-tensor.

This module contains functions for sampling from the TT-tensor and for
generation of random multi-indices and points for learning.

"""
import jax
import jax.numpy as np


def sample(Y, zm, key):
    """Sample according to given probability TT-tensor.

    Args:
        Y (list): TT-tensor, which represents the discrete probability
            distribution.
        zm (list): list of middle interface vectors for tensor Y. Run function
            "zl, zm = interface_rtl(Y)" to generate it and then use zm vector.
        key (jax.random.PRNGKey): jax random key.
    Returns:
        np.ndarray: generated multi-index for the tensor.

    """
    def body(q, data):
        key, z, G = data

        p = np.einsum('r,riq,q->i', q, G, z)
        p = np.abs(p)
        p /= np.sum(p)

        i = jax.random.choice(key, np.arange(G.shape[1]), p=p)

        q = np.einsum('r,rq->q', q, G[:, i, :])
        q /= np.linalg.norm(q)

        return q, i

    Yl, Ym, Yr = Y

    keys = jax.random.split(key, len(Ym) + 2)

    q, il = body(np.ones(1), (keys[0], zm[0], Yl))
    q, im = jax.lax.scan(body, q, (keys[1:-1], zm, Ym))
    q, ir = body(q, (keys[-1], np.ones(1), Yr))

    return np.hstack((il, im, ir))


def sample_lhs(d, n, m, key):
    """Generate LHS multi-indices for the tensor of the given shape.

    Args:
        d (int): number of tensor dimensions.
        n (int): mode size of the tensor.
        m (int): number of samples.
        key (jax.random.PRNGKey): jax random key.

    Returns:
        np.ndarray: generated multi-indices for the tensor in the form of array
        of the shape [m, d].

    """
    I = np.empty((m, d), dtype=np.int32)

    I = []
    for _ in range(d):
        m1 = m // n
        i1 = np.repeat(np.arange(n), m1)

        key, key_cur = jax.random.split(key)
        m2 = m - len(i1)
        i2 = jax.random.choice(key_cur, np.arange(n), (m2,), replace=False)

        i = np.concatenate([i1, i2])

        key, key_cur = jax.random.split(key)
        i = jax.random.permutation(key_cur, i)

        I.append(i)

    return np.array(I).T


def sample_rand(d, n, m, key):
    """Generate random multi-indices for the tensor of the given shape.

    Args:
        d (int): number of tensor dimensions.
        n (int): mode size of the tensor.
        m (int): number of samples.
        key (jax.random.PRNGKey): jax random key.

    Returns:
        np.ndarray: generated multi-indices for the tensor in the form of array
        of the shape [m, d].

    """
    I = jax.random.choice(key, np.arange(n), (m, d), replace=True)
    return I

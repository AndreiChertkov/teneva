"""Package teneva, module core_jax.tensors: various useful TT-tensors.

This module contains the collection of functions for explicit construction of
various useful TT-tensors (only random tensor for now).

"""
import jax
import jax.numpy as np


def rand(d, n, r, key, a=-1., b=1.):
    """Construct a random TT-tensor from the uniform distribution.

    Args:
        d (int): number of tensor dimensions.
        n (int): mode size of the tensor.
        r (int): TT-rank of the tensor.
        key (jax.random.PRNGKey): jax random key.
        a (float): minimum value for random items of the TT-cores.
        b (float): maximum value for random items of the TT-cores.

    Returns:
        list: TT-tensor.

    """
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r), minval=a, maxval=b)
    Ym = jax.random.uniform(keym, (d-2, r, n, r), minval=a, maxval=b)
    Yr = jax.random.uniform(keyr, (r, n, 1), minval=a, maxval=b)

    return [Yl, Ym, Yr]


def rand_norm(d, n, r, key, m=0., s=1.):
    """Construct a random TT-tensor from the normal distribution.

    Args:
        d (int): number of tensor dimensions.
        n (int): mode size of the tensor.
        r (int): TT-rank of the tensor.
        key (jax.random.PRNGKey): jax random key.
        m (float): mean ("centre") of the distribution.
        s (float): standard deviation of the distribution (>0).

    Returns:
        list: TT-tensor.

    """
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = m + s * jax.random.normal(keyl, (1, n, r))
    Ym = m + s * jax.random.normal(keym, (d-2, r, n, r))
    Yr = m + s * jax.random.normal(keyr, (r, n, 1))

    return [Yl, Ym, Yr]

"""Package teneva, module core.act_two: operations with a pair of TT-tensors.

This module contains the basic operations with a pair of TT-tensors (Y1, Y2),
including "add", "mul", "sub", etc.

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def accuracy(Y1, Y2):
    """Compute || Y1 - Y2 || / || Y2 || for tensors in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        np.ndarray of size 1: the relative difference between two tensors.

    """
    z1, p1 = teneva.norm_stab(sub(Y1, Y2))
    z2, p2 = teneva.norm_stab(Y2)

    if (p1 - p2).sum() > +500:
        return 1.E+299
    if (p1 - p2).sum() < -500:
        return 0.

    c = 2.**(p1 - p2).sum()

    if np.isinf(c) or np.isinf(z1) or np.isinf(z2) or abs(z2) < 1.E-100:
        return -1 # TODO: check

    return c * z1 / z2


def add(Y1, Y2):
    """Compute Y1 + Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        list: TT-tensor, which represents the element wise sum of Y1 and Y2.

    """
    def body(q, data):
        G1, G2 = data

        r1_l, n, r1_r = G1.shape
        r2_l, n, r2_r = G2.shape

        Z1 = np.zeros([r1_l, n, r2_r])
        Z2 = np.zeros([r2_l, n, r1_r])

        L1 = np.concatenate([G1, Z1], axis=2)
        L2 = np.concatenate([Z2, G2], axis=2)

        G = np.concatenate([L1, L2], axis=0)

        return None, G

    Yl1, Ym1, Yr1 = Y1
    Yl2, Ym2, Yr2 = Y2

    Yl = np.concatenate([Yl1, Yl2], axis=2)
    _, Ym = jax.lax.scan(body, None, (Ym1, Ym2))
    Yr = np.concatenate([Yr1, Yr2], axis=0)

    return [Yl, Ym, Yr]


def mul(Y1, Y2):
    """Compute element wise product Y1 * Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        list: TT-tensor, which represents the element wise product of Y1 and Y2.

    """
    def body(q, data):
        G1, G2 = data

        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])

        return None, G

    Yl1, Ym1, Yr1 = Y1
    Yl2, Ym2, Yr2 = Y2

    _, Yl = body(None, (Yl1, Yl2))
    _, Ym = jax.lax.scan(body, None, (Ym1, Ym2))
    _, Yr = body(None, (Yr1, Yr2))

    return [Yl, Ym, Yr]


def mul_scalar(Y1, Y2):
    """Compute scalar product for Y1 and Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        np.ndarray of size 1: the scalar product.

    """
    def body(q, data):
        G1, G2 = data

        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)

        q = q @ G

        return q, G

    Yl1, Ym1, Yr1 = Y1
    Yl2, Ym2, Yr2 = Y2

    q, _ = body(np.ones(1), (Yl1, Yl2))
    q, _ = jax.lax.scan(body, q, (Ym1, Ym2))
    q, _ = body(q, (Yr1, Yr2))

    return q


def mul_scalar_stab(Y1, Y2):
    """Compute scalar product for Y1 and Y2 in the TT-format with stab. factor.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        (np.ndarray of size 1, np.ndarray): the scaled value of the scalar
        product and stabilization factor p for each TT-core (array of the
        length d). The resulting value is v * 2^{sum(p)}.

    """
    def body(q, data):
        G1, G2 = data

        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        G = np.sum(G, axis=1)

        q = q @ G

        q_max = np.max(np.abs(q))
        p = (np.floor(np.log2(q_max)))
        q = q / 2.**p

        return q, p

    Yl1, Ym1, Yr1 = Y1
    Yl2, Ym2, Yr2 = Y2

    q, pl = body(np.ones(1), (Yl1, Yl2))
    q, pm = jax.lax.scan(body, q, (Ym1, Ym2))
    q, pr = body(q, (Yr1, Yr2))

    return q, np.hstack((pl, pm, pr))


def sub(Y1, Y2):
    """Compute Y1 - Y2 in the TT-format.

    Args:
        Y1 (list): TT-tensor.
        Y2 (list): TT-tensor.

    Returns:
        list: TT-tensor, which represents the result of the operation Y1-Y2.

    """
    Y2 = teneva.copy(Y2)
    Y2[0] *= -1.

    return add(Y1, Y2)

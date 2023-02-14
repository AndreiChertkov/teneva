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
        Y (list): d-dimensional TT-tensor.
        k (np.ndarray): the multi-index for the tensor of the length d.

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
        Y (list): d-dimensional TT-tensor.
        K (np.ndarray): the multi-indices for the tensor in the of the shape
            [samples, d].

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices K (array
        of the length samples).

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


def get_stab(Y, k):
    """Compute the element of the TT-tensor with stabilization factor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (np.ndarray): the multi-index for the tensor of the length d.

    Returns:
        tuple: the scaled value of the TT-tensor v (np.ndarray of size 1) and
        stabilization factor p for each TT-core (np.ndarray of length
        d). The resulting value is v * 2^{sum(p)}.

    """
    def body(q, data):
        i, G = data
        q = np.einsum('q,qr->r', q, G[:, i, :])

        v_max = np.max(np.abs(q))
        p = (np.floor(np.log2(v_max))).astype(int)
        q = q / 2.**p

        return q, p

    Yl, Ym, Yr = Y

    q, pl = Yl[0, k[0], :], 0
    q, pm = jax.lax.scan(body, q, (k[1:-1], Ym))
    q, pr = body(q, (k[-1], Yr))

    pl = np.array(pl, dtype=np.int32)
    pr = np.array(pr, dtype=np.int32)

    return q[0], np.hstack((pl, pm, pr))


def mean(Y):
    """Compute mean value of the TT-tensor.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray of size 1: the mean value of the TT-tensor.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        q = np.ones(k) / k
        R = R @ np.einsum('rmq,m->rq', Y_cur, q)
        return R, None

    Yl, Ym, Yr = Y
    R, _ = scan(np.ones((1, 1)), Yl)
    R, _ = jax.lax.scan(scan, R, Ym)
    R, _ = scan(R, Yr)

    return R[0, 0]


def mean_stab(Y):
    """Compute mean value of the TT-tensor with stabilization factor.

    Args:
        Y (list): TT-tensor with d dimensions.

    Returns:
        tuple: the scaled mean value of the TT-tensor m (np.ndarray of size
        1) and stabilization factor p for each TT-core (np.ndarray of length
        d). The resulting value is m * 2^{sum(p)}.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        Q = np.ones(k) / k
        R = R @ np.einsum('rmq,m->rq', Y_cur, Q)

        v_max = np.max(np.abs(R))
        p = (np.floor(np.log2(v_max))).astype(int)
        R = R / 2.**p

        return R, p

    Yl, Ym, Yr = Y
    R, pl = scan(np.ones((1, 1)), Yl)
    R, pm = jax.lax.scan(scan, R, Ym)
    R, pr = scan(R, Yr)

    pl = np.array(pl, dtype=np.int32)
    pr = np.array(pr, dtype=np.int32)

    return R[0, 0], np.hstack((pl, pm, pr))


def sum(Y):
    """Compute sum of all tensor elements.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray of size 1: the sum of all tensor elements.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        q = np.ones(k)
        R = R @ np.einsum('rmq,m->rq', Y_cur, q)
        return R, None

    Yl, Ym, Yr = Y
    R, _ = scan(np.ones((1, 1)), Yl)
    R, _ = jax.lax.scan(scan, R, Ym)
    R, _ = scan(R, Yr)

    return R[0, 0]


def sum_stab(Y):
    """Compute sum of all tensor elements with stabilization factor.

    Args:
        Y (list): TT-tensor with d dimensions.

    Returns:
        tuple: the scaled sum of all TT-tensor elements m (np.ndarray of size
        1) and stabilization factor p for each TT-core (np.ndarray of length
        d). The resulting value is m * 2^{sum(p)}.

    """
    def scan(R, Y_cur):
        k = Y_cur.shape[1]
        Q = np.ones(k)
        R = R @ np.einsum('rmq,m->rq', Y_cur, Q)

        v_max = np.max(np.abs(R))
        p = (np.floor(np.log2(v_max))).astype(int)
        R = R / 2.**p

        return R, p

    Yl, Ym, Yr = Y
    R, pl = scan(np.ones((1, 1)), Yl)
    R, pm = jax.lax.scan(scan, R, Ym)
    R, pr = scan(R, Yr)

    pl = np.array(pl, dtype=np.int32)
    pr = np.array(pr, dtype=np.int32)

    return R[0, 0], np.hstack((pl, pm, pr))

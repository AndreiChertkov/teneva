"""Package teneva, module core_jax.transformation: transformation of TT-tensors.

This module contains the function for transformation of the TT-tensor into full
(numpy) format.

"""
import jax
import jax.numpy as np


def full(Y):
    """Export TT-tensor to the full (numpy) format.

    Args:
        Y (list): TT-tensor.

    Returns:
        np.ndarray: multidimensional array related to the given TT-tensor.

    Note:
         This function can only be used for relatively small tensors, because
         the resulting tensor will have n^d elements and may not fit in memory
         for large dimensions.

    """
    Z = Y[0][0, :, :]
    for i in range(len(Y[1])):
        Z = np.tensordot(Z, Y[1][i], 1)
    Y_full = np.tensordot(Z, Y[2][:, :, 0], 1)
    return Y_full


def orthogonalize_rtl(Y):
    """Orthogonalization for TT-tensor from right to left.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        list: TT-tensor with right orthogonalized modes.

    Note:
        It works now only for TT-tensors with mode size greater than TT-rank.

    """
    def body(R, G):
        r, n = G.shape[:2]

        G = np.reshape(G, (r*n, -1), order='F')
        G = np.reshape(G @ R, (r, n, -1), order='F')

        G = np.reshape(G, (r, -1), order='F')
        Q, R = np.linalg.qr(G.T, mode='reduced')
        G = np.reshape(Q.T, (r, n, -1), order='F')

        return R.T, G

    Yl, Ym, Yr = Y
    r, n = Yr.shape[:2]

    R, Yr = body(np.ones((1, 1)), Yr)
    R, Ym = jax.lax.scan(body, R, Ym, reverse=True)


    Yl = np.reshape(Yl, (n, r), order='F')
    Yl = Yl @ R
    Yl = np.reshape(Yl, (1, n, r), order='F')

    return [Yl, Ym, Yr]


def orthogonalize_rtl_stab(Y):
    """Orthogonalization for TT-tensor from right to left with stab. factor.

    Args:
        Y (list): d-dimensional TT-tensor.

    Returns:
        tuple: the scaled TT-tensor Y with right orthogonalized modes and
        stabilization factor p for each TT-core (np.ndarray of length
        d). The resulting tensor is Y * 2^{sum(p)}.

    Note:
        It works now only for TT-tensors with mode size greater than TT-rank.

    """
    def body(R, G):
        r, n = G.shape[:2]

        G = np.reshape(G, (r*n, -1), order='F')
        G = np.reshape(G @ R, (r, n, -1), order='F')

        G = np.reshape(G, (r, -1), order='F')
        Q, R = np.linalg.qr(G.T, mode='reduced')
        G = np.reshape(Q.T, (r, n, -1), order='F')

        v_max = np.max(np.abs(R))
        p = (np.floor(np.log2(v_max))).astype(int)
        R = R / 2.**p

        return R.T, (G, p)

    Yl, Ym, Yr = Y
    r, n = Yr.shape[:2]

    R, (Yr, pr) = body(np.ones((1, 1)), Yr)
    R, (Ym, pm) = jax.lax.scan(body, R, Ym, reverse=True)

    Yl = np.reshape(Yl, (n, r), order='F')
    Yl = Yl @ R
    Yl = np.reshape(Yl, (1, n, r), order='F')
    v_max = np.max(np.abs(Yl))
    pl = (np.floor(np.log2(v_max))).astype(int)
    Yl = Yl / 2.**pl

    pl = np.array(pl, dtype=np.int32)
    pr = np.array(pr, dtype=np.int32)

    return [Yl, Ym, Yr], np.hstack((pl, pm, pr))

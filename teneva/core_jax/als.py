"""Package teneva, module core_jax.als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which computes the TT-approximation for
the tensor by TT-ALS algorithm, using given random samples (i.e., the set of
random tensor multi-indices and related tensor values).

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


def als(I_trn, y_trn, Y0, nswp=50):
    """Build TT-tensor by TT-ALS method using given random tensor samples.

    Note that this function uses inner jax.jit calls. It is not recommended to
    "jax.jit" this function while calls.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d], where d is a number of tensor's
            dimensions and samples is a size of the train dataset.
        y_trn (np.ndarray): values of the tensor for multi-indices I_trn in
            the form of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int): number of ALS iterations (sweeps).

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    n = Y0[0].shape[1]
    m, d = I_trn.shape
    I_trn = I_trn.T
    for k in range(d):
        if np.unique(I_trn[k, :]).size != n:
            raise ValueError('One groundtruth sample is needed for every slice')
    inds = _build_indices(I_trn, n)
    Y = teneva.copy(Y0)
    Z_rtl = _iter_rtl_pre(Y[1], Y[2], I_trn)
    return jax.lax.fori_loop(0, nswp, _iter, (Y, Z_rtl, I_trn, y_trn, inds))[0]


def _build_indices(I_trn, n):
    # Precompute the data indices for each dimension and mode index:
    d = I_trn.shape[0]
    inds = []
    lens = []
    for k in range(d):
        inds.append([])
        for j in range(n):
            inds[-1].append(np.where(I_trn[k, :] == j)[0])
            lens.append(len(inds[-1][-1]))
    l = min(lens)
    for k in range(d):
        for j in range(n):
            inds[k][j] = inds[k][j][:l]
    return np.array(inds)


@jax.jit
def _iter(swp, data):
    Y, Z_rtl, I_trn, y_trn, inds = data
    Y[:2], Z_ltr = _iter_ltr(Y[0], Y[1], Z_rtl, I_trn, y_trn, inds)
    Y[1:], Z_rtl = _iter_rtl(Y[1], Y[2], Z_ltr, I_trn, y_trn, inds)
    return Y, Z_rtl, I_trn, y_trn, inds


@jax.jit
def _iter_ltr(Yl, Ym, Z_rtl, I_trn, y_trn, inds):
    d, m = I_trn.shape

    Il_trn, Im_trn = I_trn[0], I_trn[1:-1]
    yl_trn, ym_trn = y_trn, np.repeat(y_trn[:, None], d-2, axis=1).T
    indsl, indsm = inds[0], inds[1:-1]

    _, (Yl, Zl_ltr) = _body_ltr(
        np.ones((m, 1)), (Yl, Z_rtl[0], Il_trn, yl_trn, indsl))

    _, (Ym, Zm_ltr) = jax.lax.scan(_body_ltr,
        Zl_ltr, (Ym, Z_rtl[1], Im_trn, ym_trn, indsm))

    return (Yl, Ym), _shift_z_ltr(Zl_ltr, Zm_ltr)


@jax.jit
def _iter_rtl(Ym, Yr, Z_ltr, I_trn, y_trn, inds):
    d, m = I_trn.shape

    Im_trn, Ir_trn = I_trn[1:-1], I_trn[-1]
    ym_trn, yr_trn = np.repeat(y_trn[:, None], d-2, axis=1).T, y_trn
    indsm, indsr = inds[1:-1], inds[-1]

    _, (Yr, Zr_rtl) = _body_rtl(
        np.ones((1, m)), (Yr, Z_ltr[1], Ir_trn, yr_trn, indsr))

    _, (Ym, Zm_rtl) = jax.lax.scan(_body_rtl,
        Zr_rtl, (Ym, Z_ltr[0], Im_trn, ym_trn, indsm), reverse=True)

    return (Ym, Yr), _shift_z_rtl(Zm_rtl, Zr_rtl)


@jax.jit
def _iter_rtl_pre(Ym, Yr, I_trn):
    d, m = I_trn.shape

    Im_trn, Ir_trn = I_trn[1:-1], I_trn[-1]

    _, Zr_rtl = _body_rtl_pre(
        np.ones((1, m)), (Yr, Ir_trn))

    _, Zm_rtl = jax.lax.scan(_body_rtl_pre,
        Zr_rtl, (Ym, Im_trn), reverse=True)

    Zl_rtl, Zm_rtl = _shift_z_rtl(Zm_rtl, Zr_rtl)

    return (Zl_rtl, Zm_rtl)


@jax.jit
def _body_ltr(Z_ltr, data):
    G, Z_rtl, i, y, inds = data
    G = G.swapaxes(0, 1)
    _, G = jax.lax.scan(_optimize, (Z_ltr, Z_rtl, y), (G, inds))
    G = G.swapaxes(0, 1)
    Z_ltr = np.einsum('mq,qmr->mr', Z_ltr, G[:, i, :])
    return Z_ltr, (G, Z_ltr)


@jax.jit
def _body_rtl(Z_rtl, data):
    G, Z_ltr, i, y, inds = data
    G = G.swapaxes(0, 1)
    _, G = jax.lax.scan(_optimize, (Z_ltr, Z_rtl, y), (G, inds))
    G = G.swapaxes(0, 1)
    Z_rtl = np.einsum('rmq,qm->rm', G[:, i, :], Z_rtl)
    return Z_rtl, (G, Z_rtl)


@jax.jit
def _body_rtl_pre(Z_rtl, data):
    G, i = data
    Z_rtl = np.einsum('rmq,qm->rm', G[:, i, :], Z_rtl)
    return Z_rtl, Z_rtl


@jax.jit
def _optimize(args, data):
    Z_ltr, Z_rtl, y = args
    Q, idx = data

    lhs = Z_rtl[:, idx].T[:, np.newaxis, :]
    rhs = Z_ltr[idx, :][:, :, np.newaxis]

    A = (lhs * rhs).reshape(len(idx), -1)
    b = y[idx]

    sol = np.linalg.lstsq(A, b)[0]
    Q = sol.reshape(Q.shape)

    return (Z_ltr, Z_rtl, y), Q


@jax.jit
def _shift_z_ltr(Zl_ltr, Zm_ltr):
    return np.vstack((Zl_ltr[None], Zm_ltr[:-1])), Zm_ltr[-1]


@jax.jit
def _shift_z_rtl(Zm_rtl, Zr_rtl):
    return Zm_rtl[0], np.vstack((Zm_rtl[1:], Zr_rtl[None]))

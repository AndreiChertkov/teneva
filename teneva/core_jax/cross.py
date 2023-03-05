"""Package teneva, module core_jax.cross: construct TT-tensor, using TT-cross.

This module contains the function "cross" which computes the TT-approximation
for implicit tensor given functionally  by the rank-adaptive multidimensional
cross approximation method in the TT-format (TT-cross).

"""
import jax
import jax.numpy as np
import teneva.core_jax as teneva


from functools import partial


def cross(f, Y0, nswp=10):
    """Compute the TT-approximation for implicit tensor given functionally.

    This function computes the TT-approximation for implicit tensor given
    functionally by the multidimensional cross approximation method in the
    TT-format (TT-cross). Note that the "f" function is expected to be jitted.

    Args:
        f (function): function f(I) which computes tensor elements for the
            given set of multi-indices I, where I is a 2D np.ndarray of the
            shape [samples, dimensions]. The function should return 1D
            np.ndarray of the length equals to samples, which relates to the
            values of the target function for all provided samples.
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int): maximum number of iterations (sweeps) of the algorithm. One
            sweep corresponds to a complete pass of all tensor TT-cores from
            left to right and then from right to left.

    Returns:
        list: TT-Tensor which approximates the implicit tensor.

    """
    Yl, Ym, Yr = teneva.copy(Y0)
    d = len(Ym) + 2
    n = Yl.shape[1]

    #Ir = [np.zeros((1, 0)) for i in range(d+1)]
    #Ic = [np.zeros((1, 0)) for i in range(d+1)]

    #Y = teneva.convert(Y)

    @jax.jit
    def _iter_rtl_body_pre(args, G):
        R, Ic, d, k = args
        r = Ic.shape[0]
        Ic = Ic[:, :k]
        G = np.tensordot(G, R, 1)
        G, R, Ic = _iter_rtl(G, Ic)
        Ic = np.hstack(Ic, np.zeros((r, d-k-1)))
        return (R, Ic, d, k+1), (G, Ic)

    R = np.ones((1, 1))
    Icr = np.zeros((1, 0))
    (R, _, _, _), (Yr, Icr) = _iter_rtl_body_pre(
        (R, Icr, d, 0), Yr)
    (R, _, _, _), (Ym, Icm) = jax.lax.scan(_iter_rtl_body_pre,
        (R, Icr, d, 1), Ym, reverse=True)
    Icl, Icm = _shift_rtl(Icm, Icr)
    (R, _, _, _), (Yl, _) = _iter_rtl_body_pre(
        (R, Icl, d, d-1), Yl)
    Yl = np.tensordot(R, Yl, 1)

    for fff in Ic:
        print(fff)

    return Yl, Ym, Yr

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter_rtl(G, Ic[i+1])
    Y[0] = np.tensordot(R, Y[0], 1)


    Icl = Ic[1]
    Icm = np.vstack(Ic[2:-1])

    @partial(jax.jit, static_argnums=[2])
    def _func(Ir, Ic, ig):
        n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
        I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
        I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
        return np.reshape(f(I), (r1, n, r2), order='F')

    @jax.jit
    def _iter_ltr_body(args, Ic):
        R, Ir, ig = args
        #Z = _func(Ir, Ic, ig)

        n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
        I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
        I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
        Z = np.reshape(f(I), (r1, n, r2), order='F')

        G, R, Ir = _iter_ltr(Z, Ir)
        return (R, Ir, ig), (G, Ir)

    @jax.jit
    def _iter_rtl_body(args, Ir):
        R, Ic, ig = args
        #Z = _func(Ir, Ic, ig)

        n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
        I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
        I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
        Z = np.reshape(f(I), (r1, n, r2), order='F')

        G, R, Ic = _iter_rtl(Z, Ic)
        return (R, Ic, ig), (G, Ic)

    ig = np.arange(n)

    for _ in range(nswp):
        (R, _, _), (Yl, Irl) = _iter_ltr_body(
            (None, np.zeros((1, 0)), ig), Icl)
        (R, _, _), (Ym, Irm) = jax.lax.scan(_iter_ltr_body,
            (R, Irl, ig), Icm)
        Irm, Irr = _shift_ltr(Irl, Irm)
        (R, _, _), (Yr, _) = _iter_ltr_body(
            (R, Irr, ig), np.zeros((1, 0)))
        Yr = np.tensordot(Yr, R, 1)

        (R, _, _), (Yr, Icr) = _iter_rtl_body(
            (None, Irr, ig), np.zeros((1, 0)))
        (R, _, _), (Ym, Icm) = jax.lax.scan(_iter_rtl_body,
            (R, Icr, ig), Irm, reverse=True)
        Icl, Icm = _shift_rtl(Icm, Icr)
        (R, _, _), (Yl, _) = _iter_rtl_body(
            (R, Icl, ig), np.zeros((1, 0)))
        Yl = np.tensordot(R, Yl, 1)

    import numpy as onp
    Y = [onp.array(G) for G in Y]
    return teneva.convert(Y)


def _iter_ltr(Z, Ir):
    r1, n, r2 = Z.shape

    I = np.kron(np.arange(n), np.ones(r1)).reshape((-1,1))
    I = np.hstack((np.kron(np.ones((n, 1)), Ir), I))

    Q, R = np.linalg.qr(np.reshape(Z, (r1 * n, r2), order='F'))
    ind, B = teneva.maxvol(Q)
    G = np.reshape(B, (r1, n, -1), order='F')
    R = Q[ind, :] @ R

    return G, R, I[ind, :]


def _iter_rtl(Z, Il):
    r1, n, r2 = Z.shape

    I = np.kron(np.ones(r2), np.arange(n)).reshape((-1,1))
    I = np.hstack((I, np.kron(Il, np.ones((n, 1)))))

    Q, R = np.linalg.qr(np.reshape(Z, (r1, n * r2), order='F').T)
    ind, B = teneva.maxvol(Q)
    G = np.reshape(B.T, (-1, n, r2), order='F')
    R = (Q[ind, :] @ R).T

    return G, R, I[ind, :]


@jax.jit
def _shift_ltr(Zl_ltr, Zm_ltr):
    return np.vstack((Zl_ltr[None], Zm_ltr[:-1])), Zm_ltr[-1]


@jax.jit
def _shift_rtl(Zm_rtl, Zr_rtl):
    return Zm_rtl[0], np.vstack((Zm_rtl[1:], Zr_rtl[None]))















def cross_1(f, Y0, nswp=10):
    Y = teneva.copy(Y0)
    d = len(Y[1]) + 2
    n = Y[0].shape[1]

    Ir = [np.zeros((1, 0)) for i in range(d+1)]
    Ic = [np.zeros((1, 0)) for i in range(d+1)]

    Y = teneva.convert(Y)

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter_rtl(G, Ic[i+1])
    Y[0] = np.tensordot(R, Y[0], 1)

    def _func(n, Ir, Ic):
        r1, r2 = Ir.shape[0], Ic.shape[0]
        I = np.kron(np.kron(np.ones(r2), np.arange(n)), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
        I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
        return np.reshape(f(I), (r1, n, r2), order='F')

    def _iter_ltr_body(Ir, Ic):
        Z = _func(n, Ir, Ic)
        G, R, Ir = _iter_ltr(Z, Ir)
        return G, R, Ir

    def _iter_rtl_body(Ir, Ic):
        Z = _func(n, Ir, Ic)
        G, R, Ic = _iter_rtl(Z, Ic)
        return G, R, Ic

    Icl = Ic[1]
    Icm = Ic[2:]

    for _ in range(nswp):
        Y[0], R, Ir[1] = _iter_ltr_body(np.zeros((1, 0)), Ic[1])

        for i in range(1, d-1):
            Y[i], R, Ir[i+1] = _iter_ltr_body(Ir[i], Ic[i+1])

        Y[d-1], R, Ir[d] = _iter_ltr_body(Ir[d-1], np.zeros((1, 0)))
        Y[d-1] = np.tensordot(Y[d-1], R, 1)

        Y[d-1], R, Ic[d-1] = _iter_rtl_body(Ir[d-1], np.zeros((1, 0)))

        for i in range(d-2, 0, -1):
            Y[i], R, Ic[i] = _iter_rtl_body(Ir[i], Ic[i+1])

        Y[0], R, Ic[0] = _iter_rtl_body(np.zeros((1, 0)), Ic[1])
        Y[0] = np.tensordot(R, Y[0], 1)

    import numpy as onp
    Y = [onp.array(G) for G in Y]
    return teneva.convert(Y)


    def _iter_ltr(Z, Ir):
        r1, n, r2 = Z.shape

        I = np.kron(np.arange(n), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n, 1)), Ir), I))

        Q, R = np.linalg.qr(np.reshape(Z, (r1 * n, r2), order='F'))
        ind, B = teneva.maxvol(Q)
        G = np.reshape(B, (r1, n, -1), order='F')
        R = Q[ind, :] @ R

        return G, R, I[ind, :]


    def _iter_rtl(Z, Il):
        r1, n, r2 = Z.shape

        I = np.kron(np.ones(r2), np.arange(n)).reshape((-1,1))
        I = np.hstack((I, np.kron(Il, np.ones((n, 1)))))

        Q, R = np.linalg.qr(np.reshape(Z, (r1, n * r2), order='F').T)
        ind, B = teneva.maxvol(Q)
        G = np.reshape(B.T, (-1, n, r2), order='F')
        R = (Q[ind, :] @ R).T

        return G, R, I[ind, :]


if __name__ == '__main__':

    import jax
    import jax.numpy as np
    import teneva as teneva_base
    import teneva.core_jax as teneva
    from time import perf_counter as tpc
    rng = jax.random.PRNGKey(42)

    from jax.config import config
    config.update('jax_enable_x64', True)

    d = 10             # Dimension of the function
    n = 5              # Shape of the tensor
    r = 3              # TT-rank of the initial random tensor
    nswp = 5           # Sweep number for TT-cross iterations
    m_tst = int(1.E+4) # Number of test points

    a = -2.048 # Lower bound for the spatial grid
    b = +2.048 # Upper bound for the spatial grid

    def func_base(i):
        """Michalewicz function."""
        x = i / n * (b - a) + a
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return np.sum(y1 + y2)

        y1 = np.sin(((np.arange(d) + 1) * x**2 / np.pi))
        return -np.sum(np.sin(x) * y1**(2 * 10))

    func = jax.jit(jax.vmap(func_base))

    rng, key = jax.random.split(rng)
    I_tst = teneva.sample_rand(d, n, m_tst, key)
    y_tst = func(I_tst)

    rng, key = jax.random.split(rng)
    Y0 = teneva.rand(d, n, r, key)

    t = tpc()
    Y = cross(func, Y0, nswp)
    t = tpc() - t

    print(f'Build time           : {t:-10.2f}')

    # Compute approximation in test points:
    y_our = teneva.get_many(Y, I_tst)

    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)

    print(f'Error on test        : {e_tst:-10.2e}')

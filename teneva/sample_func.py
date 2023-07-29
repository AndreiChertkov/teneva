"""Package teneva, module sample_func: sampling from functional TT-tensor.

This module contains the function "sample_func" for sampling from the
functional TT-tensor (i.e., the tensor of interpolation coefficients).

"""
import numpy as np
import teneva


def sample_func(A, seed=None, cores_are_prepared=False):
    """Sample random points according to given functional probability TT-tensor.

    Args:
        A (list): TT-tensor, which represents the interpolation coeeficients
            for the probability distribution.
        cores_are_prepared (bool): special flag for inner usage.
        seed (int): random seed. It should be an integer number or a numpy
            Generator class instance.

    Returns:
        np.ndarray: generated point for the tensor in the form of array of the
        shape [d], where d is the dimension.

    """
    rand = teneva._rand(seed)

    if not cores_are_prepared:
        A = teneva.copy(A)
        for G in A:
            G[:, 0, :] *= np.sqrt(2.)
        A = teneva.orthogonalize(A, 0)

    x_prev = None
    G_prev = np.array([1.])
    for G in A:
        G_cur = np.einsum('i,ijk->jk', G_prev, G)

        G0 = np.copy(G_cur)
        G0[0, :] /= np.sqrt(2)

        x_new = np.empty(1 if x_prev is None else x_prev.shape[0] + 1)
        if x_prev is not None:
            x_new[:-1] = x_prev

        p = sum([np.polynomial.chebyshev.Chebyshev(cf)**2 for cf in G0.T])
        x_cur = x_new[-1] = _sample_poly_1(p, rand)

        H_new = _cheb_my_poly(np.array(x_cur), G.shape[1])
        G_prev = np.einsum('i,ik', H_new, G_cur)
        x_prev = x_new

    return x_prev


def _cheb_my_poly(X, n):
    X = np.asarray(X)
    to_red = False
    if X.ndim == 0:
        X = X[None]
        to_red = True

    res = np.ones([X.shape[0], n])
    if n > 1:
        res[:, 1] = X

    for i in range(2, n):
        res[:, i] = 2*X*res[:, i - 1] - res[:, i - 2]

    res[:, 0] = np.sqrt(0.5)

    if to_red:
        res = res[0]

    return res


def _sample_poly_1(p2, rand):
    p2_int = p2.integ(lbnd=-1)
    xi = rand.uniform() # np.random.uniform() #
    p2_int_sh = p2_int - xi*(p2_int(1))
    roots = [np.real(i) for i in p2_int_sh.roots() if np.imag(i) == 0]
    assert len(roots) == 1
    return roots[0]

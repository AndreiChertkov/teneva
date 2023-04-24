"""Package teneva, module optima_func: estimate max for function.

This module contains the novel algorithm for computation of minimum and
maximum element of the multivariate function presented as the TT-tensor
of Chebyshev coefficients.

"""
import numpy as np
import teneva


def optima_func_tt_beam(A, k, k_loc=None):
    """Find maximum modulo points in the functional TT-tensor.

    Args:
        A (list): d-dimensional TT-tensor of interpolation coefficients.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        k_loc (int): optional number of local maximum to take.

    Returns:
        np.ndarray: the set of k_loc best multidimensional points (array of the
        shape [k, d]).

    """
    if k_loc is None:
        k_loc = k

    A = teneva.copy(A)
    for G in A:
        G[:, 0, :] *= np.sqrt(2.)
    A = teneva.orthogonalize(A, 0)

    X_prev = None # x by H matrix of basis functions values
    G_prev = np.array([[1.]])
    for G in A:
        X_prev, G_prev = _step_top_k(X_prev, G_prev, G, k, k_loc)

    return X_prev


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

    return res[0] if to_red else res


def _find_poly_max(p, clip=[-1, 1], cheb=True, take_abs=True, k_max=None,
                   ret_vals=True):
    if cheb:
        # in p low power first!!!
        p = np.polynomial.chebyshev.cheb2poly(p)

    # polyder have high power first
    dp = np.polyder(p[::-1])[::-1]

    x0 = np.polynomial.polynomial.polyroots(dp)
    x0 = x0[np.abs(np.imag(x0)) < 1e-4].real # Ok if we add unnecessary points

    x0 = [i for i in x0 if (i >= clip[0]) and i <= clip[1] ]

    if clip[0] > -np.inf and clip[0] not in x0:
        x0.append(clip[0])

    if clip[1] < +np.inf and clip[1] not in x0:
        x0.append(clip[1])

    vals = np.polynomial.polynomial.polyval(x0, p)
    if take_abs:
        vals = np.abs(vals)

    if k_max is None:
        idx = np.argmax(vals)
    else:
        idx = np.argsort(vals)[::-1][:k_max]

    x_ret = np.asarray(x0)[idx]

    if ret_vals:
        return x_ret, vals[idx]
    else:
        return x_ret


def _step_top_k(X_prev, G_prev, G, k, k_loc):
    # G_prev = num_points x rank
    num_gets = np.empty(G_prev.shape[0], dtype=int)

    all_x = []
    all_y = []

    for i, v in enumerate(G_prev):
        G_cur = np.einsum('i,ijk->jk', v, G)

        G0 = np.copy(G_cur)
        G0[0, :] /= 2**0.5

        p = sum([np.polynomial.chebyshev.Chebyshev(cf)**2 for cf in G0.T])
        p_poly = p.convert(kind=np.polynomial.Polynomial)

        x_max, y_max = _find_poly_max(list(p_poly), cheb=False,
            take_abs=True, k_max=k_loc, ret_vals=True)

        all_x.extend(x_max)
        all_y.extend(y_max)

        num_gets[i] = len(x_max)

    idx_maxx = np.argsort(all_y)[::-1][:k]
    k_real = len(idx_maxx)

    # trick as we flatten all y_max and (theoretically) we can get less points
    num_gets_cs = np.cumsum(num_gets)

    H_new_all = _cheb_my_poly(np.array(all_x)[idx_maxx], G.shape[1])

    G_new = np.empty([k_real, G.shape[2]])
    X_new = np.empty([k_real, 1 if X_prev is None else X_prev.shape[1] + 1 ])

    for idx, Hi, g_new, x_new in zip(idx_maxx, H_new_all, G_new, X_new):
        # idx_prev -- to what pull of prev points this max belongs
        idx_prev = np.searchsorted(num_gets_cs, idx, side='right')

        if X_prev is not None:
            x_new[:-1] = X_prev[idx_prev]

        x_new[-1] = all_x[idx]

        g_new[:] = np.einsum('i,j,ijk', G_prev[idx_prev], Hi, G)

    # ???  to_norm G_new???
    return X_new , G_new

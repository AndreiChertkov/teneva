from functools import reduce


def TT_to_Tucker(Y, X, H, rcond=1e-6):
    """Construct the TT-Tucker approximation for the given TT-tensor.

    Args:
        Y (list): TT-tensor.
        X (list, np.ndarray): values of argument in each TT-core (2-dim array or
            list of lists) or the same for all (1-dim array or list).
        H (function): function, which corresponds to the values of "H-matrix"
            in the TT-format, it returns np.ndarray of the size "n x m", where
            n is a number of points and m is a number of basis functions.

    Returns:
        list: TT-tensor, which represents the interpolation coefficients.

    """
    d = len(Y)
    X = np.asarray(X)

    if X.ndim == 1:
        H_mats = [H(X).T] * d
        X = [X] * d
    else:
        H_mats = [None] * d

    res = []
    for G, X_curr, H_mat in zip(Y, X, H_mats):
        if H_mat is None:
            H_mat = H(X_curr).T

        r1, n, r2 = G.shape
        M = np.transpose(G, [1, 0, 2]).reshape(n, -1)

        Q = sp.linalg.lstsq(H_mat, M, overwrite_a=False, overwrite_b=True,
            rcond=rcond)[0]
        Q = np.transpose(Q.reshape(n, r1, r2), [1, 0, 2])
        res.append(Q)

    return res


def get_spectral(Y, X, H):
    """??? TODO.

    Args:
        Y (list): TT-tensor.
        X (list, np.ndarray): values of argument in each TT-core (2-dim array or
            list of lists) or the same for all (1-dim array or list).
        H (function): function, which corresponds to the values of "H-matrix"
            in the TT-format, it returns np.ndarray of the size "n x m", where
            n is a number of points and m is a number of basis functions.

    Returns:
        ????: ??? TODO.

    """
    X = np.asanyarray(X)
    
    assert X.ndim == 1 and len(X) == len(Y)

    def f(v, arg):
        c, y = arg
        return np.einsum("i,j,ijk->k", v, y[:c.shape[1]], c)

    return reduce(f, zip(Y, H(X)), np.array([1]))[0]

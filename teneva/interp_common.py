from functools import reduce


def get_spectral(Y, X, H):
    """??? TODO.

    Args:
        Y (list): TT-tensor.
        X (list, np.ndarray): values of argument in each TT-core (2-dim array or
            list of lists) or the same for all (1-dim array or list).
        H (function): function, which corresponds to the values of "H-matrix"
            in the TT-format, it returns np.ndarray of the size n x m, where
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

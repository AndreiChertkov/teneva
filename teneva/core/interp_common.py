from functools import reduce


def TT_to_Tucker(Y, X, H, rcond=1e-6):
    """
    Y - list of cores
    X - vaues of X in each core (or the same for all)
    H -- function, which corresponds to the values of H-matrix in TT-format and its d, 
        returns np.array of size N x m, N -- number of points, m -- number of basis functions 
    """
    d = len(Y)
    X = np.asarray(X)
    print(X.shape)
    if X.ndim == 1:
        H_mats = [H(X).T]*d
        X = [X]*d
    else:
        H_mats = [None]*d

    res = []
    for c, X_i, H_mat in zip(Y, X, H_mats):
        if H_mat is None:
            H_mat = H(X_i).T

        r1, n, r2 = c.shape
        M = np.transpose(c, [1, 0, 2]).reshape(n, -1)
        c = sp.linalg.lstsq(H_mat, M, overwrite_a=False, overwrite_b=True,  rcond=rcond)[0]
        c = np.transpose(c.reshape(n, r1, r2), [1, 0, 2])
        res.append(c)

    return res


def get_spectral(cores, X, H):
    """
    cores - list of cores
    X - vaues of X, 1D array of size len(cores)
    H -- function, which corresponds to the values of H-matrix in TT-format and its d, 
        returns np.array of size N x m, N -- number of points, m -- number of basis functions 
    """
    X = np.asanyarray(X)
    assert X.ndim == 1 and len(X) == len(cores)

    def f(v, arg):
        c, y = arg
        return np.einsum("i,j,ijk->k", v, y[:c.shape[1]], c)

    return reduce(f, zip(cores, H(X)), np.array([1]))[0]



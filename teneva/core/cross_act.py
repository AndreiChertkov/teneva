"""Package teneva, module core.cross_act: compute function of TT-tensors.

This module contains the function "cross_act" which approximates the output of
the given function in the TT-format with input parameters also specified in the
TT-format. Modification of the cross approximation method in the TT-format
(TT-cross) is used.

"""
import numpy as np
import teneva


def cross_act(f, X_list, Y0, e=1.E-6, nswp=10, r=9999, dr=5, dr2=0, log=False):
    """Compute the output in the TT-format for the function of TT-tensors.

    This function computes the TT-approximation for the output of the given
    function in the TT-format with input parameters also specified in the
    TT-format. Modification of the cross approximation method in the TT-format
    (TT-cross) is used.

    Args:
        f (function): function f(X) which computes the output elements for the
            given set of input points X, where X is a 2D np.ndarray of the shape
            "[samples, D]", where "D" is a number of function inputs. The
            function should return 1D np.ndarray of the length equals to
            "samples" (i.e., it should be the values of the target function for
            all provided samples).
        X_list (list of lists): several ("D") TT-tensors, which are the input
            for the target function (f). Each TT-tensor should be represented
            as a list of its TT-cores. The dimension ("d") and mode sizes for
            all tensors must match.
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
            It may be, for example, random TT-tensor, which can be built by the
            "tensor_rand" function from teneva: "Y0 = teneva.tensor_rand(n,
            r)", where "n" is a size of tensor modes (e.g., "n = [5, 6, 7, 8,
            9]" for the 5-dimensional tensor) and "r" is related TT-rank (e.g.,
            "r = 1"). Note that the shape of this tensor should be same as for
            input tensors from "X_list".
        e (float): accuracy for SVD truncation and convergence criterion for
            algorithm (> 0). If between iterations the relative rate of
            solution change is less than this value, then the operation of the
            algorithm will be interrupted.
        nswp (int): maximum number of iterations (sweeps) of the algorithm
            (>= 0). One sweep corresponds to a complete pass of all tensor
            TT-cores from right to left and then from left to right.
        r (int): maximum rank for SVD operation (> 0).
        dr (int): rank for AMEN iterations ("kickrank"; >= 0).
        dr2 (int): additional rank for AMEN iterations ("kickrank2"; >= 0).
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each sweep.

    Returns:
        list: TT-Tensor which approximates the output of the function.

    """
    D = len(X_list)               # Number of function inputs
    d = len(X_list[0])            # Dimension of the (any) input tensor
    n = teneva.shape(X_list[0])   # Shape of the (any) input tensor

    # TT-cores of all input tensors (array of shape [d, D] of 3D arrays):
    X = np.array([[G.copy() for G in X] for X in X_list], dtype=np.object).T

    # TT-tensor for solution:
    Y = teneva.orthogonalize(Y0, d-1)

    # TT-tensor for error:
    Z = teneva.tensor_rand(n, dr) if dr > 0 else [None for _ in range(d)]
    Z = teneva.orthogonalize(Z, d-1) if dr > 0 else Z

    # Interface matrices:
    Rx  = _inter_build(d, D)
    Ry  = _inter_build(d)
    Rz  = _inter_build(d)
    Rxz = _inter_build(d, D)
    Ryz = _inter_build(d)

    # Initialization of interface matrices:
    for i in range(d-1):
        Rx[i+1, :], Ry[i+1], Rz[i+1], Rxz[i+1, :], Ryz[i+1] = _inter_update(
            X[i, :], Y[i], Z[i],
            Rx[i, :], Ry[i], Rz[i], Rxz[i, :], Ryz[i], z_rand=True, ltr=True)

    i, ltr, swp, e_curr = d, False, 0, 0.
    while swp <= nswp:
        i += 1 if ltr else -1

        G = _func(f, X[i, :], Rx[i, :], Rx[i+1, :])
        G = teneva.core_dot_inv(G, Ry[i], ltr=False)
        G = teneva.core_dot_inv(G, Ry[i+1], ltr=True)
        e_curr = max(e_curr, teneva.accuracy(Y[i], G))
        Y[i], U, V = _svd(G, d, e, r, ltr=ltr)
        U1 = U if ltr else V
        U2 = V if ltr else U

        if not ltr and i == 0:  # We reached the first mode (half of sweep)
            i, ltr = -1, True
            continue
        if ltr and i == d-1:    # We reached the last mode (end of sweep)
            _log(Y, swp, e_curr, log)
            if e_curr < e:
                break
            i, ltr, swp, e_curr = d, False, swp+1, 0,
            continue

        if dr > 0:              # "AMEN-like" operations:
            Gzy = _func(f, X[i, :],
                Rx[i] if ltr else Rxz[i, :],
                Rxz[i+1, :] if ltr else Rx[i+1])
            Gdz = teneva.core_dot(Y[i],
                Ryz[i+1 if ltr else i],
                ltr)
            Gzy = _amen_z(Gzy, Gdz,
                Ry[i if ltr else i+1],
                Rz[i+1 if ltr else i],
                dr, None, True, ltr)

            Gz = _func(f, X[i], Rxz[i], Rxz[i+1])
            Gdz = teneva.core_dot(Gdz,
                Ryz[i if ltr else i+1],
                not ltr)
            Z[i] = _amen_z(Gz, Gdz,
                Rz[i],
                Rz[i+1],
                dr, dr2, False, ltr)

            U1, U2 = _amen(Gzy, U1, U2, ltr)

        # Temporary notation for adjacent indices for the compactness:
        jn = i+1 if ltr else i
        jo = i if ltr else i+1
        ju = i+1 if ltr else i-1

        Y[i] = _matrix_to_core(U1, *Y[i].shape, ltr)

        Y[ju] = teneva.core_dot(Y[ju], U2, not ltr)

        Rx[jn, :], Ry[jn], Rz[jn], Rxz[jn, :], Ryz[jn] = _inter_update(
            X[i, :], Y[i], Z[i],
            Rx[jo, :], Ry[jo], Rz[jo], Rxz[jo, :], Ryz[jo],
            False, ltr)

    return Y


def _amen(G, U1, U2, ltr=True):
    r1, n, r2 = G.shape

    G = teneva._reshape(G, (r1*n, r2) if ltr else (r1, n*r2))
    G = G if ltr else G.T

    U1, U2_add = np.linalg.qr(np.hstack((U1, G)))

    U2 = np.hstack((U2, np.zeros((U2.shape[0], G.shape[1]), dtype=float)))
    U2 = U2_add @ U2.T if ltr else U2 @ U2_add.T

    return U1, U2


def _amen_z(G, dG, R1, R2, dr, dr2=None, is_dz=False, ltr=True):
    r1, n, r2 = G.shape

    if not is_dz:
        G = G - dG
    G = teneva.core_dot_inv(G, R1, not ltr if is_dz else False)

    if is_dz:
        G = G - dG
    G = teneva.core_dot_inv(G, R2, ltr if is_dz else True)

    _, U, V = _svd(G, eps=None, rmax=dr, ltr=ltr)
    G = U if ltr else V.T

    G = teneva._reshape(G, (r1, n, G.shape[1]) if ltr else (G.shape[0], n, r2))

    if not is_dz:
        G = teneva.core_qr_rand(G, dr2, ltr)

    return G


def _func(f, G, R1, R2):
    args = []
    for (G_, R1_, R2_) in zip(G, R1, R2):
        Q_ = teneva.core_dot(G_, R1_, ltr=False)
        Q_ = teneva.core_dot(Q_, R2_, ltr=True)
        args.append(teneva._reshape(Q_, -1))
    res = f(np.array(args).T)

    # Note, all R1/R2 have the same ranks, due to usage of "core_dot_maxvol"
    # in "_inter_update" function, hence we know the shape of result:
    n = G[0].shape[1]
    r1 = R1[0].shape[0] if isinstance(R1[0], np.ndarray) else 1
    r2 = R2[0].shape[-1] if isinstance(R2[0], np.ndarray) else 1
    return teneva._reshape(res, (r1, n, r2))


def _inter_build(d, D=None):
    s = d+1 if D is None else (d+1, D)
    R = np.zeros(s, dtype=np.object)
    R[0] = np.ones((1, 1) if D is None else D, dtype=float)
    R[d] = np.ones((1, 1) if D is None else D, dtype=float)
    return R


def _inter_update(Gx, Gy, Gz, Rx, Ry, Rz, Rxz, Ryz, z_rand=False, ltr=True):
    Ry, ind = teneva.core_dot_maxvol(Gy, Ry, None, not ltr)
    Rx = [teneva.core_dot_maxvol(G, R, ind, not ltr)[0]
        for (G, R) in zip(Gx, Rx)]
    if Gz is not None:
        r1, n, r2 = Gz.shape
        ind = None
        if z_rand:
            perm = np.random.permutation(r1*n if ltr else n*r2)
            ind = perm[:(r2 if ltr else r1)]
        Rz, ind = teneva.core_dot_maxvol(Gz, Rz, ind, not ltr)
        Ryz, _ = teneva.core_dot_maxvol(Gy, Ryz, ind, not ltr)
        Rxz = [teneva.core_dot_maxvol(G, R, ind, not ltr)[0]
            for (G, R) in zip(Gx, Rxz)]
    return Rx, Ry, Rz, Rxz, Ryz


def _log(Y, swp, e, log=False):
    if log:
        text = f'== cross-act # {swp+1:-4d} | '
        text += f'e: {e:-8.1e} | r: {teneva.erank(Y):-5.1f}'
        print(text)


def _matrix_to_core(M, r1, n, r2, ltr=True):
    if ltr:
        r2 = M.shape[1]
    else:
        r1 = M.shape[1]
    return teneva._reshape(M if ltr else M.T, (r1, n, r2))


def _rank_trunc(s, eps):
    if eps <= 0.:
        return len(s)
    s = np.cumsum(abs(s[::-1])**2)[::-1]
    r = [i for i in range(len(s)) if s[i] < eps**2]
    return len(s) if len(r) == 0 else np.amin(r)


def _svd(G, d=1, eps=1.E-6, rmax=9999999, is_qr=False, ltr=True):
    r1, n, r2 = G.shape
    G = teneva._reshape(G, (r1*n, r2) if ltr else (r1, n*r2))

    if is_qr: # Note: this is never used in the current code.
        U, V = np.linalg.qr(G if ltr else G.T)
        U = U if ltr else U.T
        V = V.T if ltr else V
        r = U.shape[1]
        s = np.ones(r)
    else:
        U, s, V = np.linalg.svd(G, full_matrices=False)
        V = V.T
        r = _rank_trunc(s, eps/np.sqrt(d) * np.linalg.norm(s)) if eps else rmax
        r = min(r, rmax, len(s))

    s = np.diag(s[:r])
    U = U[:, :r] if ltr else U[:, :r] @ s
    V = V[:, :r] @ s if ltr else V[:, :r]

    G = teneva._reshape(U @ V.T, (r1, n, r2))

    return G, U, V

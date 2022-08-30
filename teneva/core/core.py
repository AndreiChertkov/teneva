"""Package teneva, module core.core: operations with individual TT-cores.

This module contains functions to work directly with individual TT-cores.

"""
import numpy as np


from .svd import matrix_svd
from .utils import _reshape
import teneva


def core_dot(G, R, ltr=True):
    """Multiply TT-core G with matrix R."""
    # TODO: Add docs and demo.
    r1, n, r2 = G.shape
    G = teneva._reshape(G, (r1*n, r2) if ltr else (r1, n*r2))
    R = np.array([[R]]) if isinstance(R, (int, float)) else R
    G = G @ R if ltr else R @ G
    G = teneva._reshape(G, (r1, n, G.shape[1]) if ltr else (G.shape[0], n, r2))
    return G


def core_dot_inv(G, R, ltr=True):
    """Multiply TT-core G with inverted matrix R."""
    # TODO: Add docs and demo.
    r1, n, r2 = G.shape
    G = teneva._reshape(G, (r1*n, r2) if ltr else (r1, n*r2))
    G = np.linalg.solve(R.T, G.T).T if ltr else np.linalg.solve(R, G)
    return teneva._reshape(G, (r1, n, r2))


def core_dot_maxvol(G, R, ind=None, ltr=True):
    """Multiply TT-core G with matrix R and leaves the most important rows."""
    # TODO: Add docs and demo.
    r1, n, r2 = G.shape
    G = core_dot(G, R, ltr)
    G = teneva._reshape(G, (r1, n*G.shape[-1]) if ltr else (G.shape[0]*n, r2))
    ind = teneva._maxvol(G.T if ltr else G)[0] if ind is None else ind
    G = G[:, ind] if ltr else G[ind, :]
    return G, ind


def core_qr_rand(G, m, ltr=True):
    """Add random rows to TT-core G and return the new TT-core from Q-factor."""
    # TODO: Add docs and demo.
    r1, n, r2 = G.shape
    rnd = np.random.randn(r1*n if ltr else n*r2, m)
    G = teneva._reshape(G, (r1*n, r2) if ltr else (r1, n*r2))
    G = G if ltr else G.T
    G = np.hstack((G, rnd))
    G, _ = np.linalg.qr(G)
    G = G if ltr else G.T
    G = teneva._reshape(G, (r1, n, G.shape[1]) if ltr else (G.shape[0], n, r2))
    return G


def core_qtt_to_tt(Q_list):
    """Transform the list of QTT-cores into a TT-core.

    Args:
        Q_list (list of  np.ndarray): list of QTT-cores of the shapes
            [[q_0, 2, q_1], [q_1, 2, q_2], ...[q_(q-1), 2, q_q]] and length "q".

    Returns:
        np.ndarray: TT-core in the form of 3-dimensional array of the shape
            q_0 x 2^q x q_q.

    """
    G = Q_list[0].copy()

    for Q in Q_list[1:]:
        r1 = G.shape[0]
        r2 = Q.shape[-1]
        G = np.tensordot(G, Q, 1)
        G = _reshape(G, (r1, -1, r2))

    return G


def core_stab(G, p0=0, thr=1.E-100):
    """Scaling for the passed TT-core, i.e., G -> (Q, p), G = 2^p * Q.

    Args:
        G (np.ndarray): TT-core in the form of 3-dimensional array.
        p0 (int): optional initial value of the power-factor (it will be added
            to returned value "p").
        thr (float): threshold value for applying scaling (if the maximum
            modulo element in the TT-core is less than this value, then scaling
            will not be performed).

    Returns:
        (np.ndarray, int): scaled TT-core (Q) and power-factor (p), such that
            G = 2^p * Q.

    """
    v_max = np.max(np.abs(G))

    if v_max <= thr:
        return G, p0

    p = int(np.floor(np.log2(v_max)))
    Q = G / 2.**p

    return Q, p0 + p


def core_tt_to_qtt(G, e=0., r=1.E+12):
    """Transform the TT-core into a list of QTT-cores.

    Args:
        G (np.ndarray): TT-core in the form of 3-dimensional array of the shape
            r1 x n x r2. The mode size should be a power of two, i.e., n=2^d.
        e (float): desired approximation accuracy (> 0).
        r (int, float): maximum rank for the SVD decomposition (> 0).

    Returns:
        (list of np.ndarray): list of QTT-cores of the shape [q1, 2, q2], which
            approximates the given TT-core "G".

    """
    r1, n, r2 = G.shape
    d = int(np.log2(n))
    if 2**d != n:
        raise ValueError('Invalid mode size (it should be a power of two)')

    A = _reshape(G, (-1, r2))
    A, V0 = matrix_svd(A, e, r)

    Y = []
    for i in range(d-1):
        As = A.shape[0] // 2
        q = A.shape[1]
        A = np.hstack([A[:As], A[As:]])
        A, V = matrix_svd(A, e, r)
        Y.append(_reshape(V, (-1, 2, q), order='C'))

    Y.append(_reshape(A, (r1, 2, -1)))
    Y[0] = np.einsum("ijk,kl", Y[0], V0)

    return Y[::-1]

import numpy as np


def matrix_skeleton(a, eps=1.E-10, r=int(1e12), hermitian=False):
    u, s, v = np.linalg.svd(a, full_matrices=False,
        compute_uv=True, hermitian=hermitian)
    r = min(r, sum(s>eps))
    un = u[:, :r]
    sn = np.diag(np.sqrt(s[:r]))
    vn = v[:r]
    return un @ sn, sn @ vn


def matrix_svd(M, delta, rmax=None):
    if M.shape[0] <= M.shape[1]:
        cov = M.dot(M.T)
        singular_vectors = 'left'
    else:
        cov = M.T.dot(M)
        singular_vectors = 'right'

    if np.linalg.norm(cov) < 1e-14:
        return np.zeros([M.shape[0], 1]), np.zeros([1, M.shape[1]])

    w, v = np.linalg.eigh(cov)
    w[w < 0] = 0
    w = np.sqrt(w)
    svd = [v, w]
    idx = np.argsort(svd[1])[::-1]
    svd[0] = svd[0][:, idx]
    svd[1] = svd[1][idx]
    S = svd[1]**2
    where = np.where(np.cumsum(S[::-1]) <= delta**2)[0]
    if len(where) == 0:
        rank = max(1, int(np.min([rmax, len(S)])))
    else:
        rank = max(1, int(np.min([rmax, len(S) - 1 - where[-1]])))
    left = svd[0]
    left = left[:, :rank]

    if singular_vectors == 'left':
        M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
        left = left*svd[1][:rank]
    else:
        M2 = M.dot(left)
        left, M2 = M2, left.T

    return left, M2


def svd(t, eps=1E-10, max_r=int(1e12)):
    A = t = np.asanyarray(t)
    r = 1
    res = []
    for sh in t.shape[:-1]:
        A = A.reshape(r*sh, -1)
        G, A = matrix_skeleton(A, eps=eps, r=max_r)
        G = G.reshape(r, sh, -1)
        res.append(G)
        r = G.shape[-1]
    res.append(A.reshape(r, t.shape[-1], 1))
    return res


def svd_incomplete(I, Y, idx, idx_many, rank, eps_skel=1e-10):
    shapes = np.max(I, axis=0) + 1
    d = len(shapes)

    Y_curr = Y[idx[0]:idx[1]]
    Y_curr = Y_curr.reshape(shapes[0], -1, order='C')
    Y_curr, _ = matrix_skeleton(Y_curr, r=rank, eps=eps_skel)
    cores = [Y_curr[None, ...]]

    for mode in range(1, d):
        # The mode-th TT-core will have the shape r0 x n x r1
        r0 = cores[-1].shape[-1]
        r1 = rank if mode < d-1 else 1
        n = shapes[mode]

        I_curr = I[idx[mode]:idx[mode+1]]
        M = np.array([get_partial(cores[:mode], i) for i in I_curr[::idx_many[mode], :mode]])

        Y_curr = Y[idx[mode]:idx[mode+1]].reshape(-1, idx_many[mode], order='C')
        if Y_curr.shape[1] > r1:
            Y_curr, _ = matrix_skeleton(Y_curr, r=r1)
        r1 = Y_curr.shape[1]

        core = np.empty([r0, n, r1])
        step = Y_curr.shape[0] // n
        for i in range(n):
            A = M[i*step:(i+1)*step]
            b = Y_curr[i*step:(i+1)*step]
            core[:, i, :] = np.linalg.lstsq(A, b, rcond=-1)[0]
        cores.append(core)

    return cores


def get_partial(Y, n):
    Q = Y[0][0, n[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('q,qp->p', Q, Y[i][:, n[i], :])
    return Q

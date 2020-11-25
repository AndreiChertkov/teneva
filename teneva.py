import numba as nb
import numpy as np


def cross(func, Y0, nswp=10, kickrank=2, rf=2.0, mv=None, rmv=None):
    c = _init_alg(func, Y0)

    # setup_indices :
    d = c.d
    for i in range(d-1):
        nd = c.nodes[i]
        _left_qr_maxvol(nd, mv)
    for i in range(d-1, 0, -1):
        nd = c.nodes[i]
        _right_qr_maxvol(nd, mv)

    # xold = [G.copy() for G in Y0]
    for s in range(nswp):
        _iterate(c, kickrank=kickrank, rf=rf, mv=mv, rmv=rmv)
        x1 = [c.nodes[i].core for i in range(c.d)]

        # nrm = norm(x1)
        # er = (tt.tensor.from_list(x1) - tt.tensor.from_list(xold)).norm()
        # er_rel = er / nrm
        # print('swp: %d/%d er_rel = %3.1e er_abs = %3.1e erank = %3.1f fun_eval: %d' % (s, nswp-1, er_rel, er, erank(x1), c.fun_eval))
        # if er < eps * nrm: break
        # xold = [G.copy() for G in x1]

    return x1


def erank(Y):
    d = len(Y)
    n = np.array([G.shape[1] for G in Y])
    r = np.array([1] + [G.shape[-1] for G in Y[:-1]] + [1])

    sz = np.dot(n * r[0:d], r[1:])
    if sz == 0:
        er = 0e0
    else:
        b = r[0] * n[0] + n[d - 1] * r[d]
        if d is 2:
            er = sz * 1.0 / b
        else:
            a = np.sum(n[1:d - 1])
            er = (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)
    return er


def rand(N, R, f=np.random.randn):
    N = np.asanyarray(N, dtype=np.int32)
    d = N.size
    if d < 3:
        raise ValueError('Dimension should be at least 3.')

    if isinstance(R, (int, float)):
        R = [1] + [int(R)] * (d - 1) + [1]
    R = np.asanyarray(R, dtype=np.int32)

    ps = np.cumsum(np.concatenate(([1], N * R[0:d] * R[1:d +1]))).astype(np.int32)
    core = f(ps[d] - 1)

    Y = []
    for i in range(d):
        G = core[ps[i] - 1:ps[i + 1] - 1]
        Y.append(G.reshape((R[i], N[i], R[i + 1]), order='F'))
    return Y


def mul(A, B):
    C = []
    for G1, G2 in zip(A, B):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        C.append(G)
    return C


def get(Y, x):
    Q = Y[0][:, x[0], :]
    for i in range(1, len(Y)):
        Q = np.einsum('rq,qp->rp', Q, Y[i][:, x[i], :])
    return Q[0, 0]


def getter(Y, compile=True):
    Y_nb = tuple([np.array(G, order='F') for G in Y])

    @nb.jit(nopython=True)
    def get(x):
        Q = Y_nb[0]
        y = [Q[0, x[0], r2] for r2 in range(Q.shape[2])]
        for i in range(1, len(Y_nb)):
            Q = Y_nb[i]
            R = np.zeros(Q.shape[2])
            for r1 in range(Q.shape[0]):
                for r2 in range(Q.shape[2]):
                    R[r2]+= y[r1] * Q[r1, x[i], r2]
            y = list(R)
        return y[0]

    if compile:
        y = get(np.zeros(len(Y_nb), dtype=int)) # Compile

    return get


def sum(Y):
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        R = R @ np.einsum('rmq,m->rq', Y[i], np.ones(n))
    return R[0, 0]


def mean(Y, P=None):
    R = np.ones((1, 1))
    for i in range(len(Y)):
        n = Y[i].shape[1]
        Q = P[i, 0:n] if P is not None else np.ones(n) / n
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def norm(Y):
    return np.sqrt(sum(mul(Y, Y)))


def reshape(a, sz):
    return np.reshape(a, sz, order = 'F')


def mkron(a, b):
    return np.kron(a, b)


class node:
    def __init__(self):
        self.edges = [None for i in range(2)]


class edge:
    def __init__(self):
        self.nodes = [None for i in range(2)]


class _TtTree:
    def __init__(self, d, node_type, edge_type, init_boundary = None):
        self.d = d
        self.nodes = [node_type() for i in range(d)]
        self.edges = [edge_type() for i in range(d + 1)]
        #  None - N - N - N - None
        for i in range(d):
            self.nodes[i].edges[0] = self.edges[i]
            self.nodes[i].edges[1] = self.edges[i + 1]

        if init_boundary is not None:
            init_boundary(self.edges[0])
            init_boundary(self.edges[d])

        for i in range(d - 1):
            self.edges[i].nodes[1] = self.nodes[i]
            self.edges[i + 1].nodes[0] = self.nodes[i]

        self.edges[0].nodes[0] = None
        self.edges[d].nodes[1] = None


def _init_alg(fun, x0):
    d = len(x0)
    c = _TtTree(d, node, edge)
    c.fun = fun
    c.fun_eval = 0
    for i in range(d):
        c.nodes[i].core = x0[i].copy()
    for i in range(d+1):
        c.edges[i].Ru = []
        c.edges[i].Rv = []

    c.edges[0].Ru = np.ones((1, 1))
    c.edges[d].Rv = np.ones((1, 1))
    c.edges[0].ind_left = np.empty((1, 0), dtype=np.int32)
    c.edges[d].ind_right = np.empty((1, 0), dtype=np.int32)

    return c


def _index_merge(i1, i2, i3):
    if i1 is not []:
        r1 = i1.shape[0]
    else:
        r1 = 1
    r2 = i2.shape[0]
    if i1 is not []:
        r3 = i3.shape[0]
    else:
        r3 = 1
    w1 = mkron(np.ones((r3 * r2, 1), dtype=np.int32), i1)
    try:
        w2 = mkron(i3, np.ones((r1 * r2, 1), dtype=np.int32))
    except:
        w2 = np.empty((w1.shape[0], 0))
    w3 = mkron(mkron(np.ones((r3, 1), dtype=np.int32), i2), np.ones((r1, 1), dtype=np.int32))
    J = np.hstack((w1, w3, w2))
    return J


def _left_qr_maxvol(nd, mv=None):
    cr = nd.core.copy()
    r1, n1, r2 = cr.shape
    cr = np.tensordot(nd.edges[0].Ru, cr, 1)
    #nd.edges[0].Ru = np.ones((1, 1))
    r1 = cr.shape[0]
    cr = reshape(cr, (r1 * n1, r2))
    q, Ru = np.linalg.qr(cr)
    ind, c = mv(q)
    Ru = np.dot(q[ind, :], Ru)
    q = c.copy()
    nd.core = reshape(q, (r1, n1, r2)).copy()
    nd.edges[1].Ru = Ru.copy()
    nd.maxvol_left = np.unravel_index(ind, (r1, n1), order='F')
    i_left = nd.edges[0].ind_left
    #Update index
    w1 = mkron(np.ones((n1, 1), dtype=np.int32), i_left)
    w2 = mkron(reshape(np.arange(n1, dtype=np.int32),(-1, 1)), np.ones((r1, 1), dtype=np.int32))
    i_next = np.hstack((w1, w2))
    i_next = reshape(i_next, (r1 * n1, -1))
    i_next = i_next[ind, :]

    nd.edges[1].ind_left = i_next.copy()
    nd.edges[1].ind_left_add = i_next.copy()


def _right_qr_maxvol(nd, mv=None):
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    Rv = nd.edges[1].Rv.copy()
    cr = np.tensordot(cr, Rv, 1)
    r2 = cr.shape[2]
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, Rv = np.linalg.qr(cr)
    ind, c = mv(q)
    Rv = np.dot(q[ind, :], Rv)
    q = c.copy()
    nd.edges[0].Rv = Rv.T.copy()
    q = reshape(q.T, (-1, n, r2))
    nd.core = q.copy()
    nd.maxvol_right = np.unravel_index(ind, (n, r2), order='F')
    i_right = nd.edges[1].ind_right
    w1 = mkron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = mkron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1

    i_next = reshape(i_next, (n * r2, -1))
    i_next = i_next[ind, :]
    nd.edges[0].ind_right = i_next.copy()
    nd.edges[0].ind_right_add = i_next.copy()


def _apply_Ru(nd):
    cr = nd.core.copy()
    try:
        Ru = nd.edges[0].Ru.copy()
        cr = np.tensordot(Ru, cr, 1)
    except:
        print('Failed')
        pass
    nd.core = cr
    nd.edges[0].Ru = []


def _apply_Rv(nd):
    cr = nd.core.copy()
    try:
        Rv = nd.edges[1].Rv.copy()
        cr = np.tensordot(cr, Rv, 1)
    except:
        pass
    nd.core = cr
    nd.edges[1].Rv = []


def _update_core_left(nd, fun, kickrank=3, rf=2, tau=1.1, mv=None, rmv=None):
    fun_ev = 0
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right
    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))
    J = _index_merge(i_left, p, i_right)
    cr = fun(J)
    fun_ev += cr.size
    cr = reshape(cr, (-1, r2))

    q, s1, v1 = np.linalg.svd(cr, full_matrices=False)
    R = np.diag(s1).dot(v1)

    ind_new, C = rmv(q, tau, maxK = q.shape[1] + kickrank + rf, min_add_K=kickrank)
    nd.core = reshape(C, (r1, n, -1))
    Ru = q[ind_new, :].dot(R)
    i_left = nd.edges[0].ind_left
    w1 = mkron(np.ones((n, 1), dtype=np.int32), i_left)
    w2 = mkron(reshape(np.arange(n, dtype=np.int32), (-1, 1)), np.ones((r1, 1), dtype=np.int32))

    i_next = np.hstack((w1, w2))

    i_next = reshape(i_next, (r1 * n, -1))

    nd.edges[1].ind_left = i_next[ind_new, :].copy()
    try:
        nd.edges[1].Ru = np.dot(Ru, nd.edges[1].Ru)
    except:
        nd.edges[1].Ru = Ru.copy()

    return fun_ev


def _update_core_right(nd, fun, kickrank=1, rf=2, tau=1.1, mv=None, rmv=None):
    fun_ev = 0
    cr = nd.core
    r1, n, r2 = cr.shape

    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right

    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))

    J = _index_merge(i_left, p, i_right)

    cr = fun(J)
    fun_ev += cr.size
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, s1, v1 = np.linalg.svd(cr, full_matrices=False)
    R = np.diag(s1).dot(v1)
    ind_new, C = rmv(q, tau, q.shape[1] + kickrank + rf, min_add_K = kickrank)
    nd.core = C

    nd.core = reshape(nd.core.T, (-1, n, r2))
    Rv = q[ind_new, :].dot(R)

    try:
        nd.edges[0].Rv = np.dot(nd.edges[0].Rv, Rv.T)
    except:
        nd.edges[0].Rv = Rv.T.copy()

    i_right = nd.edges[1].ind_right
    w1 = mkron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = mkron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1

    i_next = reshape(i_next, (n * r2, -1))
    nd.edges[0].ind_right = i_next[ind_new, :].copy()
    return fun_ev


def _iterate(c, kickrank=1, rf=2, mv=None, rmv=None):
    d = c.d
    for i in range(c.d):
        c.nodes[i].edges[0].Rv = np.ones((1, 1))
        c.nodes[i].edges[1].Rv = np.ones((1, 1))
        c.nodes[i].edges[0].Ru = np.ones((1, 1))
        c.nodes[i].edges[1].Ru = np.ones((1, 1))

    for i in range(c.d):
        _apply_Ru(c.nodes[i])
        c.fun_eval += _update_core_left(c.nodes[i], c.fun, kickrank, rf, mv=mv, rmv=rmv)

    c.nodes[d-1].core = np.tensordot(c.nodes[d-1].core, c.nodes[d-1].edges[1].Ru, 1)

    for i in range(c.d-1, -1, -1):
        _apply_Rv(c.nodes[i])
        c.fun_eval += _update_core_right(c.nodes[i], c.fun, kickrank, rf, mv=mv, rmv=rmv)

    c.nodes[0].core = np.tensordot(c.nodes[0].edges[0].Rv, c.nodes[0].core, 1)

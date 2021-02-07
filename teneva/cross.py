import numpy as np


from .maxvol import maxvol
from .maxvol import rect_maxvol
from .tensor import getter
from .tensor import erank
from .utils import kron
from .utils import reshape


class Node:
    def __init__(self, core=None, edge1=None, edge2=None):
        self.core = core
        self.edges = [edge1, edge2]


class Edge:
    def __init__(self, node1=None, node2=None):
        self.nodes = [node1, node2]
        self.Ru = []
        self.Rv = []


class Tree:
    def __init__(self, d, Y, fun=None):
        self.d = d
        self.fun = fun
        self.edges = [Edge() for i in range(d + 1)]
        self.nodes = [Node(Y[i].copy(), self.edges[i], self.edges[i+1]) for i in range(d)]
        for i in range(d - 1):
            self.edges[i].nodes[1] = self.nodes[i]
            self.edges[i + 1].nodes[0] = self.nodes[i]


def cross_tmp(func, Y0, nswp=10, kickrank=2, rf=2.0, eps=None, val_size=10000, with_wrap=False, verbose=False):

    def func_wrapper(J):
        if with_wrap:
            Y = [c.nodes[i].core.copy() for i in range(c.d)]
            return func(J, Y)
        else:
            return func(J)

    d = len(Y0)
    c = Tree(d, Y0, func_wrapper)
    c.edges[0].Ru = np.ones((1, 1))
    c.edges[d].Rv = np.ones((1, 1))
    c.edges[0].ind_left = np.empty((1, 0), dtype=np.int32)
    c.edges[d].ind_right = np.empty((1, 0), dtype=np.int32)

    # setup_indices :
    for i in range(d-1):
        _left_qr_maxvol(c.nodes[i])
    for i in range(d-1, 0, -1):
        _right_qr_maxvol(c.nodes[i])

    #xold = [G.copy() for G in Y0]
    if verbose or eps is not None:
        # Create a validation set
        Xs_val = np.array([np.random.choice(I.shape[1], int(val_size)) for I in Y0], dtype=int).T
        ys_val = func(Xs_val)
        norm_ys_val = np.linalg.norm(ys_val)

    for s in range(nswp):
        for i in range(d):
            c.nodes[i].edges[0].Rv = np.ones((1, 1))
            c.nodes[i].edges[1].Rv = np.ones((1, 1))
            c.nodes[i].edges[0].Ru = np.ones((1, 1))
            c.nodes[i].edges[1].Ru = np.ones((1, 1))
        for i in range(d):
            _apply_Ru(c.nodes[i])
            c.fun_eval += _update_core_left(c.nodes[i], c.fun, kickrank, rf)

        c.nodes[d-1].core = np.tensordot(
            c.nodes[d-1].core, c.nodes[d-1].edges[1].Ru, 1)

        for i in range(c.d-1, -1, -1):
            _apply_Rv(c.nodes[i])
            c.fun_eval += _update_core_right(c.nodes[i], c.fun, kickrank, rf)

        c.nodes[0].core = np.tensordot(
            c.nodes[0].edges[0].Rv, c.nodes[0].core, 1)

        x1 = [c.nodes[i].core for i in range(c.d)]


        if verbose or eps is not None:
            get = getter(x1)
            y = np.array([get(x) for x in Xs_val])
            er = np.linalg.norm(ys_val - y)
            er_rel = er / norm_ys_val
        if verbose:
            # nrm = norm(x1)
            # er = (tt.tensor.from_list(x1) - tt.tensor.from_list(xold)).norm()
            # er_rel = er / nrm
            # print('swp: %d/%d er_rel = %3.1e er_abs = %3.1e erank = %3.1f fun_eval: %d' % (s, nswp-1, er_rel, er, erank(x1), c.fun_eval))
            print(f'swp: {s}/{nswp-1}, er_rel = {er_rel:3.1e}, er_abs = {er:3.1e}, erank = {erank(x1):3.1f}, fun_eval: {c.fun_eval}')
        if eps is not None and er_rel < eps:
            break
        # xold = [G.copy() for G in x1]

    return x1


def cross(func, Y0, nswp=10, kickrank=2, rf=2, wrap=False):

    def func_wrapper(J):
        if wrap:
            return func(J, [c.nodes[i].core.copy() for i in range(c.d)])
        else:
            return func(J)

    d = len(Y0)
    c = Tree(d, Y0, func_wrapper)
    _init(c)
    for _ in range(nswp):
        _prep(c)

        for i in range(d):
            _apply_Ru(c.nodes[i])
            _update_core_left(c.nodes[i], c.fun, kickrank, rf)

        c.nodes[d-1].core = np.tensordot(c.nodes[d-1].core, c.nodes[d-1].edges[1].Ru, 1)

        for i in range(c.d-1, -1, -1):
            _apply_Rv(c.nodes[i])
            _update_core_right(c.nodes[i], c.fun, kickrank, rf)

        c.nodes[0].core = np.tensordot(c.nodes[0].edges[0].Rv,c.nodes[0].core,1)

    return [c.nodes[i].core.copy() for i in range(c.d)]


def _init(c):
    c.edges[0].Ru = np.ones((1, 1))
    c.edges[c.d].Rv = np.ones((1, 1))
    c.edges[0].ind_left = np.empty((1, 0), dtype=np.int32)
    c.edges[c.d].ind_right = np.empty((1, 0), dtype=np.int32)
    for i in range(c.d - 1):
        _left_qr_maxvol(c.nodes[i])
    for i in range(c.d - 1, 0, -1):
        _right_qr_maxvol(c.nodes[i])


def _prep(c):
    for i in range(c.d):
        c.nodes[i].edges[0].Rv = np.ones((1, 1))
        c.nodes[i].edges[1].Rv = np.ones((1, 1))
        c.nodes[i].edges[0].Ru = np.ones((1, 1))
        c.nodes[i].edges[1].Ru = np.ones((1, 1))


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
    w1 = kron(np.ones((r3 * r2, 1), dtype=np.int32), i1)
    try:
        w2 = kron(i3, np.ones((r1 * r2, 1), dtype=np.int32))
    except:
        w2 = np.empty((w1.shape[0], 0))
    w3 = kron(kron(np.ones((r3, 1), dtype=np.int32), i2), np.ones((r1, 1), dtype=np.int32))
    return np.hstack((w1, w3, w2))


def _left_qr_maxvol(nd):
    cr = nd.core.copy()
    r1, n1, r2 = cr.shape
    cr = np.tensordot(nd.edges[0].Ru, cr, 1)
    r1 = cr.shape[0]
    cr = reshape(cr, (r1 * n1, r2))
    q, Ru = np.linalg.qr(cr)
    ind, c = maxvol(q)
    Ru = np.dot(q[ind, :], Ru)
    q = c.copy()
    nd.core = reshape(q, (r1, n1, r2)).copy()
    nd.edges[1].Ru = Ru.copy()
    nd.maxvol_left = np.unravel_index(ind, (r1, n1), order='F')
    i_left = nd.edges[0].ind_left
    w1 = kron(np.ones((n1, 1), dtype=np.int32), i_left)
    w2 = kron(reshape(np.arange(n1, dtype=np.int32),(-1, 1)), np.ones((r1, 1), dtype=np.int32))
    i_next = np.hstack((w1, w2))
    i_next = reshape(i_next, (r1 * n1, -1))
    i_next = i_next[ind, :]

    nd.edges[1].ind_left = i_next.copy()
    nd.edges[1].ind_left_add = i_next.copy()


def _right_qr_maxvol(nd):
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    Rv = nd.edges[1].Rv.copy()
    cr = np.tensordot(cr, Rv, 1)
    r2 = cr.shape[2]
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, Rv = np.linalg.qr(cr)
    ind, c = maxvol(q)
    Rv = np.dot(q[ind, :], Rv)
    q = c.copy()
    nd.edges[0].Rv = Rv.T.copy()
    q = reshape(q.T, (-1, n, r2))
    nd.core = q.copy()
    nd.maxvol_right = np.unravel_index(ind, (n, r2), order='F')
    i_right = nd.edges[1].ind_right
    w1 = kron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = kron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1

    i_next = reshape(i_next, (n * r2, -1))
    i_next = i_next[ind, :]
    nd.edges[0].ind_right = i_next.copy()
    nd.edges[0].ind_right_add = i_next.copy()


def _apply_Ru(nd):
    cr = nd.core.copy()
    Ru = nd.edges[0].Ru.copy()
    cr = np.tensordot(Ru, cr, 1)
    nd.core = cr
    nd.edges[0].Ru = []


def _apply_Rv(nd):
    cr = nd.core.copy()
    Rv = nd.edges[1].Rv.copy()
    cr = np.tensordot(cr, Rv, 1)
    nd.core = cr
    nd.edges[1].Rv = []


def _update_core_left(nd, fun, kickrank, rf, tau=1.1):
    cr = nd.core.copy()
    r1, n, r2 = cr.shape
    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right
    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))
    J = _index_merge(i_left, p, i_right)
    cr = fun(J)
    cr = reshape(cr, (-1, r2))

    q, s1, v1 = np.linalg.svd(cr, full_matrices=False)
    R = np.diag(s1).dot(v1)

    N, r = q.shape
    N_min = min(N, r + kickrank)
    N_max = min(N, r + kickrank + rf)
    if N <= r:
        ind_new = np.arange(q.shape[0], dtype=np.int32)
        C = np.eye(q.shape[0], dtype=q.dtype)
    else:
        ind_new, C = rect_maxvol(q, tau, N_min, N_max)
    nd.core = C
    nd.core = reshape(nd.core, (r1, n, -1))
    Ru = q[ind_new, :].dot(R)
    i_left = nd.edges[0].ind_left
    w1 = kron(np.ones((n, 1), dtype=np.int32), i_left)
    w2 = kron(reshape(np.arange(n, dtype=np.int32), (-1, 1)), np.ones((r1, 1), dtype=np.int32))

    i_next = np.hstack((w1, w2))

    i_next = reshape(i_next, (r1 * n, -1))

    nd.edges[1].ind_left = i_next[ind_new, :].copy()
    try:
        nd.edges[1].Ru = np.dot(Ru, nd.edges[1].Ru)
    except:
        nd.edges[1].Ru = Ru.copy()


def _update_core_right(nd, fun, kickrank, rf, tau=1.1):
    cr = nd.core
    r1, n, r2 = cr.shape

    i_left = nd.edges[0].ind_left
    i_right = nd.edges[1].ind_right

    p = np.arange(n, dtype=np.int32); p = reshape(p, (-1, 1))

    J = _index_merge(i_left, p, i_right)

    cr = fun(J)
    cr = reshape(cr, (r1, -1))
    cr = cr.T
    q, s1, v1 = np.linalg.svd(cr, full_matrices=False)
    R = np.diag(s1).dot(v1)

    N, r = q.shape
    N_min = min(N, r + kickrank)
    N_max = min(N, r + kickrank + rf)
    if N <= r:
        ind_new = np.arange(q.shape[0], dtype=np.int32)
        C = np.eye(q.shape[0], dtype=q.dtype)
    else:
        ind_new, C = rect_maxvol(q, tau, N_min, N_max)
    nd.core = C
    nd.core = reshape(nd.core.T, (-1, n, r2))
    Rv = q[ind_new, :].dot(R)

    try:
        nd.edges[0].Rv = np.dot(nd.edges[0].Rv, Rv.T)
    except:
        nd.edges[0].Rv = Rv.T.copy()

    i_right = nd.edges[1].ind_right
    w1 = kron(np.ones((r2, 1), dtype=np.int32), reshape(np.arange(n, dtype=np.int32),(-1, 1)))
    try:
        w2 = kron(i_right, np.ones((n, 1), dtype=np.int32))
        i_next = np.hstack((w1, w2))
    except:
        i_next = w1

    i_next = reshape(i_next, (n * r2, -1))
    nd.edges[0].ind_right = i_next[ind_new, :].copy()

"""Package teneva, module core.optima: estimate min and max value of the tensor.

This module contains the novel algorithm for computation of minimum and
maximum element of the given TT-tensor (function optima_tt).

"""
import numpy as np
import teneva


def optima_qtt(Y, k=100, e=1.E-12, r=100):
    """Find items which relate to min and max elements of the given TT-tensor.

    The provided TT-tensor "Y" is transformed into the QTT-format and then
    "optima_tt" method is applied to this QTT-tensor. Note that this method
    support only the tensors with constant mode size, which is a power of two,
    i.e., the shape should be "[2^q, 2^q, ..., 2^q]".

    Args:
        Y (list): d-dimensional TT-tensor of the shape "[2^q, 2^q, ..., 2^q]".
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        e (float): desired approximation accuracy for the QTT-tensor (> 0).
        r (int, float): maximum rank for the SVD decompositions while QTT-tensor
            construction (> 0).

    Returns:
        [np.ndarray, float, np.ndarray, float]: multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like "i_min, y_min, i_max, y_max".

    """
    n = teneva.shape(Y)

    for n_ in n[1:]:
        if n[0] != n_:
            msg = 'Invalid mode size (it should be equal for all modes)'
            raise ValueError(msg)

    n = n[0]
    q = int(np.log2(n))

    if 2**q != n:
        msg = 'Invalid mode size (it should be a power of two)'
        raise ValueError(msg)

    Z = teneva.tt_to_qtt(Y, e, r)

    i_min, y_min, i_max, y_max = optima_tt(Z, k)

    i_min = teneva.ind_qtt_to_tt(i_min, q)
    i_max = teneva.ind_qtt_to_tt(i_max, q)

    # We do this just in case to reduce the magnitude of numerical errors:
    y_min = teneva.get(Y, i_min)
    y_max = teneva.get(Y, i_max)

    return i_min, y_min, i_max, y_max


def optima_tt(Y, k=100):
    """Find items which relate to min and max elements of the given TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.

    Returns:
        [np.ndarray, float, np.ndarray, float]: multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like "i_min, y_min, i_max, y_max".

    Note:
        This function runs the "optima_tt_max" twice: first for the original
        tensor, and then for the tensor shifted by the value of the found
        maximum.

    """
    i1, y1 = optima_tt_max(Y, k)

    D = teneva.tensor_const(teneva.shape(Y), y1)
    Z = teneva.sub(Y, D)
    Z = teneva.mul(Z, Z)

    i2, _ = optima_tt_max(Z, k)
    y2 = teneva.get(Y, i2)

    if y2 > y1:
        return i1, y1, i2, y2
    else:
        return i2, y2, i1, y1


def optima_tt_beam(Y, k=100, l2r=True, ret_all=False):
    """Find multi-index of the maximum modulo item in the given TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        l2r (bool): if flag is set, hen the TT-cores are passed from left to
            right (that is, from the first to the last TT-core). Otherwise, the
            TT-cores are passed from right to left.
        ret_all (bool): if flag is set, then all "k" multi-indices will be
            returned. Otherwise, only best found multi-index will be returned.

    Returns:
        np.ndarray: multi-index (array of length d) which relates to maximum
        modulo TT-tensor element if "ret_all" flag is not set. If "ret_all" flag
        is set, then it will be the set of "k" best multi-indices (array of the
        shape "[k, d]").

    Note:
        This is an internal utility function. To find the optimum in the
        TT-tensor tensor, use the functions "optima_qtt", "optima_tt" or
        "optima_tt_max".

    """
    Z, p = teneva.orthogonalize(Y, 0 if l2r else len(Y)-1, use_stab=True)
    p0 = p / len(Z) # Scale factor (2^p0) for each TT-core

    G = Z[0 if l2r else -1]
    r1, n, r2 = G.shape

    I = teneva._range(n)
    Q = G.reshape(n, r2) if l2r else G.reshape(r1, n)

    Q *= 2**p0

    for G in (Z[1:] if l2r else Z[:-1][::-1]):
        r1, n, r2 = G.shape

        if l2r:
            Q = np.einsum('kr,riq->kiq', Q, G, optimize='optimal')
            Q = Q.reshape(-1, r2)
        else:
            Q = np.einsum('qir,rk->qik', G, Q, optimize='optimal')
            Q = Q.reshape(r1, -1)

        if l2r:
            I_l = np.kron(I, teneva._ones(n))
            I_r = np.kron(teneva._ones(I.shape[0]), teneva._range(n))
        else:
            I_l = np.kron(teneva._range(n), teneva._ones(I.shape[0]))
            I_r = np.kron(teneva._ones(n), I)
        I = np.hstack((I_l, I_r))

        q_max = np.max(np.abs(Q))
        norms = np.sum((Q/q_max)**2, axis=1 if l2r else 0)
        ind = np.argsort(norms)[:-(k+1):-1]

        I = I[ind, :]
        Q = Q[ind, :] if l2r else Q[:, ind]

        Q *= 2**p0

    return I if ret_all else I[0]


def optima_tt_max(Y, k=100):
    """Find the maximum modulo item in the given TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.

    Returns:
        [np.ndarray, float]: multi-index (array of length d) which relates to
        maximum modulo TT-tensor element and the related value of the tensor
        item (float).

    Note:
        This function runs the "optima_tt_beam" first from left to right, then
        from right to left, and returns the best result.

    """
    i_max_list = [
        optima_tt_beam(Y, k, l2r=True),
        optima_tt_beam(Y, k, l2r=False)]
    y_max_list = [teneva.get(Y, i) for i in i_max_list]

    index_best = np.argmax([abs(y) for y in y_max_list])
    return i_max_list[index_best], y_max_list[index_best]

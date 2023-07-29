"""Package teneva, module act_one: single TT-tensor operations.

This module contains the basic operations with one TT-tensor (Y), including
"copy", "get", "sum", etc.

"""
try:
    import numba as nb
    WITH_NUMBA = True
except Exception as e:
    WITH_NUMBA = False
import numpy as np
import teneva


def copy(Y):
    """Return a copy of the given TT-tensor.

    Args:
        Y (int, float, list): TT-tensor (or it may be int/float).

    Returns:
        list: TT-tensor, which is a copy of the given TT-tensor. If Y is a
        number, then result will be the same number. If Y is np.ndarray, then
        the result will the corresponding copy in numpy format.

    """
    if teneva._is_num(Y):
        return Y
    elif isinstance(Y, np.ndarray):
        return Y.copy()
    else:
        return [G.copy() for G in Y]


def get(Y, k, _to_item=True):
    """Compute the element (or elements) of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (list, np.ndarray): the multi-index for the tensor (list or 1D array
            of the length d) or a batch of multi-indices in the form of a list
            of lists or array of the shape [samples, d].

    Returns:
        float: the element of the TT-tensor. If argument k is a batch of
        multi-indices, then array of the length samples will be returned (the
        get_many function is called in this case).

    """
    d = len(Y)
    k = np.asanyarray(k, dtype=int)

    if k.ndim == 2:
        return get_many(Y, k)

    Q = Y[0][0, k[0], :]
    for i in range(1, d):
        Q = Q @ Y[i][:, k[i], :]

    return Q[0] if _to_item else Q


def get_and_grad(Y, idx):
    """Compute the element of the TT-tensor and gradients of its TT-cores.

    Args:
        Y (list): d-dimensional TT-tensor.
        idx (list, np.ndarray): the multi-index for the tensor.

    Returns:
        (float, list): the element y of the TT-tensor at provided multi-index
        idx and the TT-tensor which collects the gradients for all TT-cores.

    """
    phi_r = interface(Y, idx=idx, norm=None, ltr=False)
    phi_l = interface(Y, idx=idx, norm=None, ltr=True)

    # We check the correctness of the interfaces:
    val = phi_r[0].item()
    err = abs(val - phi_l[-1].item())
    flag = (abs(val) < 1e-8 and err < 1e-8) or err / abs(val) < 1e-6
    text = f'Something unexpected, {val}, {phi_l[-1].item()}, {err/abs(val)}'
    assert flag, text

    grad = [np.zeros(G.shape) for G in Y]
    for Gg, i, p_l, p_r in zip(grad, idx, phi_l[:-1], phi_r[1:]):
        Gg[:, i, :] = np.outer(p_l, p_r)

    return val, grad


def get_many(Y, K, _to_item=True):
    """Compute the elements of the TT-tensor on many indices (batch).

    Args:
        Y (list): d-dimensional TT-tensor.
        K (list of list, np.ndarray): the multi-indices for the tensor in the
            form of a list of lists or array of the shape [samples, d].

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices K (array
        of the length samples).

    """
    d = len(Y)
    K = np.asanyarray(K, dtype=int)

    Q = Y[0][0, K[:, 0], :]
    for i in range(1, d):
        Q = np.einsum('kq, qkr -> kr', Q, Y[i][:, K[:, i], :])

    return Q[:, 0] if _to_item else Q


def getter(Y, compile=True):
    """Build the fast getter function to compute the element of the TT-tensor.

    Args:
        Y (list): TT-tensor.
        compile (bool): flag, if True, then the getter will be called one time
            with a random multi-index to compile its code.

    Returns:
        function: the function that computes the element of the TT-tensor. It
        has one argument k (list or np.ndarray of the length d) which is
        the multi-index for the tensor.

    Note:
        Note that the gain from using this getter instead of the base function
        "get" appears only in the case of many requests for calculating the
        tensor value (otherwise, the time spent on compiling the getter may
        turn out to be significant). Also note that this function requires
        "numba" package to be installed.

        Attention: this function will be removed in the future! Use the
        "get_many" function instead (it's faster in most cases).

    """
    if not WITH_NUMBA:
        raise ValueError('Numba is required for this function')

    Y_nb = tuple([np.array(G, order='C') for G in Y])

    @nb.jit(nopython=True)
    def get(k):
        Q = Y_nb[0]
        y = [Q[0, k[0], r2] for r2 in range(Q.shape[2])]
        for i in range(1, len(Y_nb)):
            Q = Y_nb[i]
            R = np.zeros(Q.shape[2])
            for r1 in range(Q.shape[0]):
                for r2 in range(Q.shape[2]):
                    R[r2] += y[r1] * Q[r1, k[i], r2]
            y = list(R)
        return y[0]

    if compile:
        y = get(np.zeros(len(Y), dtype=int))

    return get


def interface(Y, p=None, idx=None, norm='linalg', ltr=False):
    """Generate interface vectors for provided TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        p (list, np.ndarray): optional weights for mode indices (list of the
            length d; the same for all modes).
        idx (list, np.ndarray): the multi-index for the tensor.
        norm (str): norm function to use.
        ltr (bool): the direction ("ltr" if True and "rtl" if False).

    Returns:
        list: interface vectors (d+1 vectors).

    """
    d = len(Y)
    phi = [None] * (d+1)
    phi[-1] = np.ones(1)

    if ltr:
        Y = Y[::-1]
        if idx is not None:
            idx = idx[::-1]

    for i in range(d-1, -1, -1):
        if idx is None:
            if p is None:
                mat = np.sum(Y[i], axis=1)
            else:
                mat = np.einsum('ijk,j->ik', Y[i], p)
        else:
            if p is None:
                mat = Y[i][:, idx[i], :]
            else:
                mat = Y[i][:, idx[i], :] * p[idx[i]]

        if ltr:
            mat = mat.T

        phi[i] = mat @ phi[i+1]

        if norm is not None:
            if norm.startswith('l'): # linalg
                phi[i] /= np.linalg.norm(phi[i])

            if norm.startswith('n'): # natural
                phi[i] /= Y[i].shape[1]

    if ltr:
        phi = phi[::-1]

    return phi


def mean(Y, P=None, norm=True):
    """Compute mean value of the TT-tensor with the given inputs probability.

    Args:
        Y (list): TT-tensor.
        P (list): optional probabilities for each dimension. It is the list of
            length d (number of tensor dimensions), where each element is
            also a list with length equals to the number of tensor elements
            along the related dimension. Hence, P[m][i] relates to the
            probability of the i-th input for the m-th mode (dimension).
        norm (bool): service (inner) flag, should be True.

    Returns:
        float: the mean value of the TT-tensor.

    """
    R = np.ones((1, 1))
    for i in range(len(Y)):
        k = Y[i].shape[1]
        if P is not None:
            Q = P[i][:k]
        else:
            Q = np.ones(k) / k if norm else np.ones(k)
        R = R @ np.einsum('rmq,m->rq', Y[i], Q)
    return R[0, 0]


def norm(Y, use_stab=False):
    """Compute Frobenius norm of the given TT-tensor.

    Args:
        Y (list): TT-tensor.
        use_stab (bool): if flag is set, then function will also return the
            second argument p, which is the factor of 2-power.

    Returns:
        float: Frobenius norm of the TT-tensor.

    """
    if use_stab:
        v, p = teneva.mul_scalar(Y, Y, use_stab=True)
        return np.sqrt(v) if v > 0 else 0., p/2
    else:
        v = teneva.mul_scalar(Y, Y)
        return np.sqrt(v) if v > 0 else 0.


def qtt_to_tt(Y, q):
    """Transform the QTT-tensor into a TT-tensor.

    Args:
        Y (list): QTT-tensor. It is d*q-dimensional tensor with mode size 2.
        q (int): quantization factor, i.e., the mode size of the TT-tensor will
            be n = 2^q.
    Returns:
        list: TT-tensor. It is d-dimensional tensor with mode size 2^q.

    """
    d = int(len(Y) / q)
    Z = []
    for k in range(d):
        G_list = Y[k*q:(k+1)*q]
        Z.append(teneva.core_qtt_to_tt(G_list))
    return Z


def sum(Y):
    """Compute sum of all tensor elements.

    Args:
        Y (list): TT-tensor.

    Returns:
        float: the sum of all tensor elements.

    """
    return mean(Y, norm=False)


def tt_to_qtt(Y, e=1.E-12, r=100):
    """Transform the TT-tensor into a QTT-tensor.

    Args:
        Y (list): TT-tensor. It is d-dimensional tensor with mode size n,
            which is a power of two, i.e., n=2^q.
        e (float): desired approximation accuracy.
        r (int): maximum rank for the SVD decomposition.

    Returns:
        list: QTT-tensor. It is d * q-dimensional tensor with mode size 2.

    """
    Z = []
    for G in Y:
        Z.extend(teneva.core_tt_to_qtt(G, e, r))
    return Z

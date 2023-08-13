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
        Y (int, float, list): TT-tensor (or it may be int, float and numpy
            array for convenience).

    Returns:
        list: TT-tensor, which is a copy of the given TT-tensor. If Y is a
        number, then result will be the same number. If Y is np.ndarray, then
        the result will the corresponding copy in numpy format. If the
        function's argument is None, then it will also return None.

    """
    if Y is None or teneva._is_num(Y):
        return Y
    elif isinstance(Y, np.ndarray):
        return Y.copy()
    else:
        return [G.copy() for G in Y]


def get(Y, i, _to_item=True):
    """Compute the element (or elements) of the TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        i (list, np.ndarray): the multi-index for the tensor (list or 1D array
            of the length d) or a batch of multi-indices in the form of a list
            of lists or array of the shape [samples, d].

    Returns:
        float: the element of the TT-tensor. If argument i is a batch of
        multi-indices, then array of the length samples will be returned (the
        get_many function is called in this case).

    """
    d = len(Y)
    i = np.asanyarray(i, dtype=int)

    if i.ndim == 2:
        return get_many(Y, i)

    Q = Y[0][0, i[0], :]
    for k in range(1, d):
        Q = Q @ Y[k][:, i[k], :]

    return Q[0] if _to_item else Q


def get_and_grad(Y, i, check_phi=False):
    """Compute the element of the TT-tensor and gradients of its TT-cores.

    Args:
        Y (list): d-dimensional TT-tensor.
        i (list, np.ndarray): the multi-index for the tensor.
        check_phi (bool): service flag, should be False.

    Returns:
        (float, list): the element y of the TT-tensor at provided multi-index
        and the TT-tensor of related gradients for all TT-cores.

    """
    phi_r = interface(Y, i=i, norm=None, ltr=False)
    phi_l = interface(Y, i=i, norm=None, ltr=True)
    value = phi_r[0].item()

    if check_phi:
        # We check the correctness of the interfaces:
        p1 = phi_r[0].item()
        p2 = phi_l[-1].item()
        err = abs(val - p2)
        flag = (abs(p1) < 1e-8 and err < 1e-8) or err / abs(p1) < 1e-6
        text = f'Something unexpected, {p1}, {p2}, {err/abs(p1)}'
        assert flag, text

    grad = [np.zeros(G.shape) for G in Y]
    for Q, k, p_l, p_r in zip(grad, i, phi_l[:-1], phi_r[1:]):
        Q[:, k, :] = np.outer(p_l, p_r)

    return value, grad


def get_many(Y, I, _to_item=True):
    """Compute the elements of the TT-tensor on many indices (batch).

    Args:
        Y (list): d-dimensional TT-tensor.
        I (list of list, np.ndarray): the multi-indices for the tensor in the
            form of a list of lists or array of the shape [samples, d].

    Returns:
        np.ndarray: the elements of the TT-tensor for multi-indices I (array
        of the length samples).

    """
    d = len(Y)
    I = np.asanyarray(I, dtype=int)

    Q = Y[0][0, I[:, 0], :]
    for k in range(1, d):
        Q = np.einsum('iq, qir -> ir', Q, Y[k][:, I[:, k], :])

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


def interface(Y, P=None, i=None, norm='linalg', ltr=False):
    """Generate interface vectors for provided TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        P (list, np.ndarray): optional weights for mode indices from left to
            right (list of lists of the length d; or just one list if the
            weights are the same for all modes and all modes are equal).
        i (list, np.ndarray): optional multi-index for the tensor.
        norm (str): optional norm function to use (it may be 'linalg' ['l'] for
            the usage of the np.linalg.norm or 'natural' ['n'] for usage of the
            natural norm, i.e., the related mode size; or it may be None).
        ltr (bool): the direction of computation of the interface vectors
            ("left to right" if True and "right to left" if False).

    Returns:
        list: list of d+1 interface vectors. Note that the first and last
        vectors always have length 1.

    """
    d = len(Y)
    phi = [None] * (d+1)
    phi[-1] = np.ones(1)

    if ltr:
        Y = Y[::-1]
        if i is not None:
            i = i[::-1]
        if P is not None and not isinstance(P[0], (int, float)):
            P = P[::-1]

    for k in range(d-1, -1, -1):
        if i is None:
            if P is None:
                Q = np.sum(Y[k], axis=1)
            else:
                p = P if isinstance(P[0], (int, float)) else P[k]
                Q = np.einsum('rmq,m->rq', Y[k], p)
        else:
            if P is None:
                Q = Y[k][:, i[k], :]
            else:
                p = P if isinstance(P[0], (int, float)) else P[k]
                Q = Y[k][:, i[k], :] * p[i[k]]

        if ltr:
            Q = Q.T

        phi[k] = Q @ phi[k+1]

        if norm is not None:
            if norm.startswith('l'): # linalg
                phi[k] /= np.linalg.norm(phi[k])

            if norm.startswith('n'): # natural
                phi[k] /= Y[k].shape[1]

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
    Z = np.ones((1, 1))
    for i in range(len(Y)):
        k = Y[i].shape[1]
        if P is None:
            p = np.ones(k) / k if norm else np.ones(k)
        else:
            p = P[i][:k]
        Z = Z @ np.einsum('rmq,m->rq', Y[i], p)
    return Z[0, 0]


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

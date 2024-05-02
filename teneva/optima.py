"""Package teneva, module optima: estimate min and max value of the tensor.

This module contains the novel algorithm for computation of minimum and
maximum element of the given TT-tensor (function optima_tt).

"""
import numpy as np
import teneva


def optima_qtt(Y, k=100, e=1.E-12, r=100):
    """Find items which relate to min and max elements of the given TT-tensor.

    The provided TT-tensor Y is transformed into the QTT-format and then
    "optima_tt" method is applied to this QTT-tensor. Note that this method
    support only the tensors with constant mode size, which is a power of two,
    i.e., the shape should be [2^q, 2^q, ..., 2^q].

    Args:
        Y (list): d-dimensional TT-tensor of the shape [2^q, 2^q, ..., 2^q].
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        e (float): desired approximation accuracy for the QTT-tensor (> 0).
        r (int, float): maximum rank for the SVD decompositions while QTT-tensor
            construction (> 0).

    Returns:
        (np.ndarray, float, np.ndarray, float): multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like i_min, y_min, i_max, y_max.

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
        (np.ndarray, float, np.ndarray, float): multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like i_min, y_min, i_max, y_max.

    Note:
        This function runs the "optima_tt_max" twice: first for the original
        tensor, and then for the tensor shifted by the value of the found
        maximum.

    """
    i1, y1 = optima_tt_max(Y, k)

    D = teneva.const(teneva.shape(Y), y1)
    Z = teneva.sub(Y, D)
    Z = teneva.mul(Z, Z)

    i2, _ = optima_tt_max(Z, k)
    y2 = teneva.get(Y, i2)

    if y2 > y1:
        return i1, y1, i2, y2
    else:
        return i2, y2, i1, y1


def optima_tt_beam(Y, k=100, l2r=True, ret_all=False, to_orth=True, p=None):
    """Find multi-index of the maximum modulo item in the given TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        l2r (bool): if flag is set, hen the TT-cores are passed from left to
            right (that is, from the first to the last TT-core). Otherwise, the
            TT-cores are passed from right to left.
        ret_all (bool): if flag is set, then all k multi-indices will be
            returned. Otherwise, only best found multi-index will be returned.

    Returns:
        np.ndarray: multi-index (array of length d) which relates to maximum
        modulo TT-tensor element if ret_all flag is not set. If ret_all flag
        is set, then it will be the set of k best multi-indices (array of the
        shape [k, d]).

    Note:
        This is an internal utility function. To find the optimum in the
        TT-tensor tensor, use the functions "optima_qtt", "optima_tt" or
        "optima_tt_max".

    """
    d = len(Y)
    if  to_orth:
        Z, p = teneva.orthogonalize(Y, 0 if l2r else len(Y)-1, use_stab=True)
    else:
        Z = Y
        if p is None:
            p = 1

    p0 = p / d # Scale factor (2^p0) for each TT-core

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
        (np.ndarray, float): multi-index (array of length d) which relates to
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



def optima_tt_maxvol(Y, k=10, how='smart', use='mv'):
    """Find items which relate to min and max elements of the given TT-tensor.

    Args:
        Y (list): d-dimensional TT-tensor.
        k (int): number of selected items (candidates for the optimum) for each
            tensor mode.
        how (str): kind of computation. It may be: 'l2r', 'r2l', 'both' or
            'smart' (by default).
        use (str): do we use MaxVol method (if 'mv'; default) or k-means
            ('k_means').

    Returns:
        (np.ndarray, float, np.ndarray, float): multi-index (array of length d)
        which relates to minimum TT-tensor element; the related value of the
        tensor item (float); multi-index (array of length d) which relates to
        maximum TT-tensor element; the related value of the tensor item (float).
        I.e., the output looks like i_min, y_min, i_max, y_max.

    """
    if how == 'l2r':
        Y = teneva.orthogonalize(Y, 0)
        I, vecs = _select_top_k_l2r(Y, k=k, use=use)
    elif how == 'r2l':
        Y = teneva.orthogonalize(Y, len(Y) - 1)
        I, vecs = _select_top_k_r2l(Y, k=k, use=use)
    elif how == 'both':
        Y = teneva.orthogonalize(Y, 0)
        I, vecs = _select_top_k_l2r(Y, k=k, use=use)
        i_min = np.argmin(vecs)
        i_max = np.argmax(vecs)
        
        I_min1, min1, I_max1, max1 = I[i_min], vecs[i_min].item(), I[i_max], vecs[i_max].item()
        
        I, vecs = _select_top_k_r2l(Y, k=k, use=use)
        i_min = np.argmin(vecs)
        i_max = np.argmax(vecs)
        
        I_min2, min2, I_max2, max2 = I[i_min], vecs[i_min].item(), I[i_max], vecs[i_max].item()

        I_min = I_min1 if  min1 < min2 else I_min2
        I_max = I_max1 if  max1 > max2 else I_max2
        return I_min, min(min1, min2), I_max, max(max1, max2)
            
        
    elif how == 'smart':
        Y = teneva.orthogonalize(Y, len(Y) - 1)
        I1, vecs1 = _select_top_k_l2r(Y, k=k, save_all=True, use=use)
        return _select_top_k_r2l(Y, k=k, other=(I1, vecs1), use=use)
        
    else:
        raise

    i_min = np.argmin(vecs)
    i_max = np.argmax(vecs)
        
    return I[i_min], vecs[i_min].item(), I[i_max], vecs[i_max].item()
 
 
def _k_means_spere(mat, k, select='min'):
    def unit_vector(x):
        return x / np.linalg.norm(x)

    def fd(x, y):
        x = unit_vector(x)
        y = unit_vector(y)
        return 1.1 + np.dot(x, y)

    
    mat_norm = np.linalg.norm(mat, axis=1)
    clustering = SpectralClustering(n_clusters=k,
         assign_labels='discretize',
         #affinity=fd,
         random_state=0).fit(mat/mat_norm[:, None])
    
    
    res = []
    f_min = np.argmin if select=='min' else np.argmax
    
    for i in range(k):
        idx = np.where(clustering.labels_ == i)[0]
        #print(f"len: {len(idx)}")
        i_min = f_min( mat_norm[idx] )
        
        res.append(idx[i_min])
        

    res = np.array(res)
    return res


def _select_maxvol(vecs, core, k=10, transpose=False, use='mv'):
    if transpose:
        mat = np.einsum("ijk,nk->nji", core, vecs)
    else:
        mat = np.einsum("ni,ijk->njk", vecs, core)

    mat = mat.reshape(-1, mat.shape[-1], order='F')
    
    dr = min(mat.shape[-1] + k, mat.shape[0]) - mat.shape[-1]
    
    if dr == 0:
        idx = np.arange(mat.shape[0])
    else:
        if use=='mv':
            idx = teneva.maxvol_rect(mat, dr_min=dr, dr_max=dr)[0]
        elif  use=='k_means':
            idx = _k_means_spere(mat, k, select='max')
        else:
            assert False, f"Unknown method: {use}"
    return idx, mat[idx]


def _select_top_k_l2r(Y, k=10, save_all=False, use='mv'):
    if save_all:
        I_all = []
        v_all = [np.array([[1]])]
    
    vecs = np.array([[1.]])
    I = np.array([[-100]])
    
    for G in Y:
        i_mv, vecs = _select_maxvol(vecs, G, k=k, use=use)
        n = G.shape[1]
        I = np.hstack([
            np.kron(np.ones(n, dtype=int)[:, None], I)[i_mv],
            np.kron(np.arange(n)[:, None], np.ones(I.shape[0], dtype=int)[:, None])[i_mv]
            ])
        
        
        if save_all:
            I_all.append(I[:, 1:])
            v_all.append(vecs)

        
    if  save_all:
        return I_all, v_all
    else:
        return I[:, 1:], vecs

    
def _select_top_k_r2l(Y, k=10, save_all=False, other=None, use='mv'):
    d = len(Y)
    
    if save_all:
        I_all = []
        v_all = [np.array([[1]])]

    if other is not None:
        best_min = np.inf
        best_max = -np.inf
        best_idx = [None, None]
        other = [[]] + other[0], other[1]
        
    vecs = np.array([[1.]])
    I = np.array([[-100]])
    
    for i, G in enumerate(Y[::-1]):
        i_mv, vecs = _select_maxvol(vecs, G, k=k, transpose=True, use=use)
        n = G.shape[1]
        I = np.hstack([
            np.kron(np.arange(n)[:, None], np.ones(I.shape[0], dtype=int)[:, None])[i_mv],
            np.kron(np.ones(n, dtype=int)[:, None], I)[i_mv]
            ])


        if save_all:
            I_all.append(I[:, :-1])
            v_all = [vecs] + v_all
            
            
        if other is not None:
            o_idx, o_vects = other[0][d - i - 1], other[1][d - i - 1]
            vals = np.einsum("ni,li->nl", vecs, o_vects)
            for idx_i,  row_vals in zip(I, vals):
                for o_idx_i, cur_val in zip(o_idx, row_vals):
                
                    if cur_val > best_max:
                        best_max = cur_val
                        best_idx[0] = list(o_idx_i) + list(idx_i[:-1])
                        
                    if cur_val < best_min:
                        best_min = cur_val
                        best_idx[1] = list(o_idx_i) + list(idx_i[:-1])
            
    if other is not None:
        return best_idx[1], best_min,  best_idx[0], best_max
        
        
    if save_all:
        return I_all, v_all
    else:
        return I[:, :-1], vecs
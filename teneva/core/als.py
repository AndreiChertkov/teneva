"""Package teneva, module core.als: construct TT-tensor, using TT-ALS.

This module contains the function "als" which computes the TT-approximation
for the tensor by TT-ALS algorithm, using given random samples.

"""
import numpy as np
import scipy as sp


from .tensor import copy
from .tensor import get
from .transformation import orthogonalize
from .transformation import orthogonalize_left
from .transformation import orthogonalize_right


def als(I_trn, Y_trn, Y0, nswp=10):
    """Build TT-tensor by TT-ALS from the given random tensor samples.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for multi-indices I in the form
            of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int): number of ALS iterations (sweeps).

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    I_trn = np.asanyarray(I_trn, dtype=int)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    m = I_trn.shape[0]
    d = I_trn.shape[1]
    Y = copy(Y0)

    for k in range(d):
        if np.unique(I_trn[:, k]).size != Y[k].shape[1]:
            raise ValueError('One groundtruth sample is needed for every slice')

    Yl = [np.ones((1, m, Y[k].shape[0])) for k in range(d)]
    Yr = [None for _ in range(d-1)] + [np.ones((1, m, 1))]

    orthogonalize(Y, 0)

    for k in range(d-1, 0, -1):
        i_trn = I_trn[:, k]
        Yr[k-1] = np.einsum('ijk,kjl->ijl', Y[k][:, i_trn, :], Yr[k])

    for _ in range(nswp):
        for k in range(d-1):
            i_trn = I_trn[:, k]
            optimize_core(Y[k], i_trn, Y_trn, Yl[k], Yr[k])
            orthogonalize_left(Y, k)
            Yl[k+1] = np.einsum('ijk,kjl->ijl', Yl[k], Y[k][:, i_trn, :])

        for k in range(d-1, 0, -1):
            i_trn = I_trn[:, k]
            optimize_core(Y[k], i_trn, Y_trn, Yl[k], Yr[k])
            orthogonalize_right(Y, k)
            Yr[k-1] = np.einsum('ijk,kjl->ijl', Y[k][:, i_trn, :], Yr[k])

    return Y


def als2(I_trn, Y_trn, Y0, nswp=10, eps=None):
    """Build TT-tensor by TT-ALS from the given random tensor samples.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for multi-indices I in the form
            of array of the shape [samples].
        Y0 (list): TT-tensor, which is the initial approximation for algorithm.
        nswp (int):  number of ALS iterations (sweeps).
        eps (float): desired accuracy of approximation.

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    Note:
        This is the alternative realization of the algorithm. The version from
        "als" function in many cases works better and much faster.

        Note that the code of this function is not optimized.

    """
    I_trn = np.asanyarray(I_trn, dtype=int)
    Y_trn = np.asanyarray(Y_trn, dtype=float)

    P, d = I_trn.shape

    norm = np.linalg.norm(Y_trn)
    Y_trn = Y_trn.copy()
    Y_trn /= norm

    Y = [G.copy() for G in Y0]

    elist = []

    for swp in range(nswp):

        for k in range(d):
            r1, n, r2 = Y[k].shape

            core = np.zeros([r1, n, r2])

            leftU = Y[:k] if k > 0 else []
            rightV = Y[k+1:] if k < d-1 else []

            for i in range(n):
                thetaI = np.where(I_trn[:, k] == i)[0]
                if len(thetaI) == 0:
                    continue

                A = np.zeros([len(thetaI), r1*r2])
                for j in range(len(thetaI)):
                    A[j:j+1, :] += getRow(leftU, rightV, I_trn[thetaI[j], :])

                vec_slice = np.linalg.lstsq(A, Y_trn[thetaI], rcond=-1)[:1]
                core[:, i, :] += _reshape(vec_slice, [r1, r2])

            Y[k] = core.copy()

        if eps is not None:
            get = lambda x: get(Y, x)
            e = 0.5 * sum((get(I_trn[p, :]) - Y_trn[p])**2 for p in range(P))
            elist.append(e)
            if e < eps or swp > 0 and abs(e - elist[-2]) < eps:
                break

    Y[0] *= norm

    return Y


def getRow(leftU, rightV, jVec):
    jLeft = jVec[:len(leftU)] if len(leftU) > 0 else None
    jRight = jVec[-len(rightV):] if len(rightV) > 0 else None

    multU = np.ones([1, 1])
    for k in range(len(leftU)):
        multU = multU @ leftU[k][:, jLeft[k], :]

    multV= np.ones([1, 1])
    for k in range(len(rightV)-1, -1, -1):
        multV = rightV[k][:, jRight[k], :] @ multV

    return np.kron(multV.T, multU)


def optimize_core(G, i_trn, Y_trn, left, right):
    for k in range(G.shape[1]):
        idx = np.where(i_trn == k)[0]

        leftside = left[0, idx, :]
        rightside = right[:, idx, 0]

        lhs = np.transpose(rightside, [1, 0])[:, :, np.newaxis]
        rhs = leftside[:, np.newaxis, :]
        A = _reshape(lhs * rhs, (len(idx), -1))

        b = Y_trn[idx]

        sol, residuals = sp.linalg.lstsq(A, b)[0:2]
        if residuals.size == 0:
            residuals = np.linalg.norm(A.dot(sol) - b) ** 2

        G[:, k, :] = _reshape(sol, G[:, k, :].shape, 'C')


def _reshape(A, n, order='F'):
    return np.reshape(A, n, order=order)

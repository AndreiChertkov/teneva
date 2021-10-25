import itertools
import numpy as np
import scipy as sp


from .tensor import get


def als(X_trn, Y_trn, Y0, nswp=10):
    P = X_trn.shape[0]
    N = X_trn.shape[1]
    Y = [G.copy() for G in Y0]

    for dim in range(N):
        if np.unique(X_trn[:, dim]).size != Y[dim].shape[1]:
            raise ValueError('One groundtruth sample is needed for every slice')

    def left_orthogonalize(cores, mu):
        coreL = np.reshape(cores[mu], [-1, cores[mu].shape[2]], order="F")
        Q, R = np.linalg.qr(coreL, mode='reduced')
        cores[mu] = np.reshape(Q, cores[mu].shape[:-1] + (Q.shape[1],), order="F")
        rightcoreR = np.reshape(cores[mu+1], [cores[mu+1].shape[0], -1], order="F")
        cores[mu+1] = np.reshape(np.dot(R, rightcoreR), (R.shape[0], ) + cores[mu+1].shape[1:], order="F")
        return R

    def right_orthogonalize(cores, mu):
        coreR = np.reshape(cores[mu], [cores[mu].shape[0], -1], order="F")
        L, Q = sp.linalg.rq(coreR, mode='economic', check_finite=False)
        cores[mu] = np.reshape(Q, (Q.shape[0], ) + cores[mu].shape[1:], order="F")
        leftcoreL = np.reshape(cores[mu-1], [-1, cores[mu-1].shape[2]], order="F")
        cores[mu-1] = np.reshape(np.dot(leftcoreL, L), cores[mu-1].shape[:-1] + (L.shape[1], ), order="F")
        return L

    def orthogonalize(cores, mu):
        L = np.array([[1]])
        R = np.array([[1]])
        for i in range(0, mu):
            R = left_orthogonalize(cores, i)
        for i in range(len(cores)-1, mu, -1):
            L = right_orthogonalize(cores, i)
        return R, L

    def optimize_core(Y, X_trn, Y_trn, mu, direction, lefts, rights):
        sse = 0
        for index in range(Y[mu].shape[1]):
            idx = np.where(X_trn[:, mu] == index)[0]
            leftside = lefts[mu][0, idx, :]
            rightside = rights[mu][:, idx, 0]
            lhs = np.transpose(rightside, [1, 0])[:, :, np.newaxis]
            rhs = leftside[:, np.newaxis, :]
            A = np.reshape(lhs*rhs, [len(idx), -1], order='F')
            b = Y_trn[idx]
            sol, residuals = sp.linalg.lstsq(A, b)[0:2]
            if residuals.size == 0:
                residuals = np.linalg.norm(A.dot(sol) - b) ** 2
            Y[mu][:, index, :] = np.reshape(sol, Y[mu][:, index, :].shape, order='C')
            sse += residuals
        if direction == 'right':
            left_orthogonalize(Y, mu)
            lefts[mu+1] = np.einsum('ijk,kjl->ijl', lefts[mu], Y[mu][:, X_trn[:, mu], :])
        else:
            right_orthogonalize(Y, mu)
            rights[mu-1] = np.einsum('ijk,kjl->ijl', Y[mu][:, X_trn[:, mu], :], rights[mu])
        return lefts, rights

    orthogonalize(Y, 0)

    lefts = [np.ones([1, P, Y[i].shape[0]]) for i in range(N)]
    rights = [None] * N
    rights[-1] = np.ones([1, P, 1])
    for dim in range(N-2, -1, -1):
        rights[dim] = np.einsum('ijk,kjl->ijl', Y[dim+1][:, X_trn[:, dim+1], :], rights[dim+1])

    for swp in range(nswp):
        for mu in range(N-1):
            lefts, rights = optimize_core(
                Y, X_trn, Y_trn, mu, "right", lefts, rights)
        for mu in range(N-1, 0, -1):
            lefts, rights = optimize_core(
                Y, X_trn, Y_trn, mu, "left", lefts, rights)

    return Y


def als2(X_trn, Y_trn, Y0, nswp=10, eps=None):
    P, d = X_trn.shape

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
                thetaI = np.where(X_trn[:, k] == i)[0]
                if len(thetaI) == 0:
                    continue

                A = np.zeros([len(thetaI), r1*r2])
                for j in range(len(thetaI)):
                    A[j:j+1, :] += getRow(leftU, rightV, X_trn[thetaI[j], :])

                vec_slice = np.linalg.lstsq(A, Y_trn[thetaI], rcond=-1)[:1]
                core[:, i, :] += np.reshape(vec_slice, [r1, r2], order='F')

            Y[k] = core.copy()

        if eps is not None:
            get = lambda x: get(Y, x)
            e = 0.5 * sum((get(X_trn[p, :]) - Y_trn[p])**2 for p in range(P))
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

import itertools
import numpy as np


def core_one(n, r):
    return np.kron(np.ones([1, n, 1]), np.eye(r)[:, None, :])


def kron(a, b):
    return np.kron(a, b)


def orthogonalize(Z, k):
    # Z = [G.copy() for G in Y]
    L = np.array([[1.]])
    R = np.array([[1.]])
    for i in range(0, k):
        G = reshape(Z[i], [-1, Z[i].shape[2]])
        Q, R = np.linalg.qr(G, mode='reduced')
        Z[i] = reshape(Q, Z[i].shape[:-1] + (Q.shape[1], ))
        G = reshape(Z[i+1], [Z[i+1].shape[0], -1])
        Z[i+1] = reshape(np.dot(R, G), (R.shape[0], ) + Z[i+1].shape[1:])
    for i in range(len(Z)-1, k, -1):
        G = reshape(Z[i], [Z[i].shape[0], -1])
        L, Q = scipy.linalg.rq(G, mode='economic', check_finite=False)
        Z[i] = reshape(Q, (Q.shape[0], ) + Z[i].shape[1:])
        G = reshape(Z[i-1], [-1, Z[i-1].shape[2]])
        Z[i-1] = reshape(np.dot(G, L), Z[i-1].shape[:-1] + (L.shape[1], ))
    return Z


def reshape(a, sz):
    return np.reshape(a, sz, order='F')


def sum_many(tensors, e=1.E-10, rmax=None, freq_trunc=15):
    cores = tensors[0]
    for i, t in enumerate(tensors[1:]):
        cores = teneva.add(cores, t)
        if (i+1) % freq_trunc == 0:
            cores = teneva.truncate(cores, e=e)
    return teneva.truncate(cores, e=e, rmax=rmax)

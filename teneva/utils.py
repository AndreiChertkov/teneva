import itertools
import numpy as np


def core_one(n, r):
    return np.kron(np.ones([1, n, 1]), np.eye(r)[:, None, :])

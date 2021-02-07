import numpy as np
from scipy.interpolate import interp1d


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
    def __init__(self, d, cores, fun=None):
        self.d = d
        self.fun = fun
        self.fun_eval = 0
        self.edges = [Edge() for i in range(d + 1)]
        self.nodes = [Node(
            cores[i].copy(), self.edges[i], self.edges[i+1]) for i in range(d)]
        for i in range(d - 1):
            self.edges[i].nodes[1] = self.nodes[i]
            self.edges[i + 1].nodes[0] = self.nodes[i]


def confidence(F, alpha=.05):
    r"""
    Constructs a Dvoretzky-Kiefer-Wolfowitz confidence band for the eCDF.

    Parameters
    ----------
    F : array_like
        The empirical distributions
    alpha : float
        Set alpha for a (1 - alpha) % confidence band.

    Notes
    -----
    Based on the DKW inequality.

    .. math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| > \epsilon \right) \leq 2e^{-2n\epsilon^2}

    References
    ----------
    Wasserman, L. 2006. `All of Nonparametric Statistics`. Springer.

    """
    nobs = len(F)
    epsilon = np.sqrt(np.log(2./alpha) / (2 * nobs))
    lower = np.clip(F - epsilon, 0, 1)
    upper = np.clip(F + epsilon, 0, 1)

    return lower, upper


def get_cdf(x):
    _x = np.array(x, copy=True)
    _x.sort()
    _y = np.linspace(1./len(_x), 1, len(_x))

    _x = np.r_[-np.inf, _x]
    _y = np.r_[0, _y]

    def cdf(z):
        tind = np.searchsorted(_x, z, 'right') - 1
        return _y[tind]

    return cdf


def kron(a, b):
    return np.kron(a, b)


def reshape(a, sz):
    return np.reshape(a, sz, order = 'F')

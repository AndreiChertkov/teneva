import numpy as np


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


def kron(a, b):
    return np.kron(a, b)


def reshape(a, sz):
    return np.reshape(a, sz, order = 'F')

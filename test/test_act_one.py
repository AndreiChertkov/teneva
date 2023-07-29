import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestActOneGet(unittest.TestCase):
    def setUp(self):
        self.Y = teneva.rand([10] * 5, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.k = [1, 2, 3, 4, 5]
        self.eps = 1.E-16

    def test_base(self):
        k = np.array(self.k)
        y = teneva.get(self.Y, k)
        z = self.Z[tuple(k)]
        e = np.abs(y-z)
        self.assertLess(e, self.eps)

    def test_list(self):
        k = self.k
        y = teneva.get(self.Y, k)
        z = self.Z[tuple(k)]
        e = np.abs(y-z)
        self.assertLess(e, self.eps)

    def test_many(self):
        K = [self.k, self.k, self.k]
        y = teneva.get(self.Y, K)
        z = np.array([self.Z[tuple(k)] for k in K])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)


class TestActOneGetMany(unittest.TestCase):
    def setUp(self):
        self.Y = teneva.rand([10] * 5, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.K = [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 2, 1],
        ]
        self.eps = 1.E-16

    def test_base(self):
        K = np.array(self.K)
        y = teneva.get_many(self.Y, K)
        z = np.array([self.Z[tuple(k)] for k in K])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)

    def test_list(self):
        K = self.K
        y = teneva.get_many(self.Y, K)
        z = np.array([self.Z[tuple(k)] for k in K])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)


if __name__ == '__main__':
    unittest.main()

import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestActOneCopy(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-16

    def test_base(self):
        Y = teneva.rand([10] * 5, r=3, seed=42)
        Z = teneva.copy(Y)

        y1 = Y[2][1, 2, 0]
        Z[2][1, 2, 0] = 42.
        y2 = Y[2][1, 2, 0]

        e = np.abs(y1-y2)
        self.assertLess(e, self.eps)

    def test_base_rev(self):
        Y = teneva.rand([10] * 5, r=3, seed=42)
        Z = teneva.copy(Y)

        z1 = Z[2][1, 2, 0]
        Y[2][1, 2, 0] = 42.
        z2 = Z[2][1, 2, 0]

        e = np.abs(z1-z2)
        self.assertLess(e, self.eps)

    def test_array(self):
        Y = np.random.randn(3, 4, 5)
        Z = teneva.copy(Y)

        y1 = Y[1, 2, 0]
        Z[1, 2, 0] = 42.
        y2 = Y[1, 2, 0]

        e = np.abs(y1-y2)
        self.assertLess(e, self.eps)

    def test_number(self):
        y = 42.
        z = teneva.copy(y)

        e = np.abs(z-y)
        self.assertLess(e, self.eps)

    def test_none(self):
        y = None
        z = teneva.copy(y)

        self.assertIsNone(z)


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


class TestActOneMean(unittest.TestCase):
    def setUp(self):
        self.n = [5] * 10
        self.Y = teneva.rand(self.n, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.eps = 1.E-16

    def test_base(self):
        m_calc = teneva.mean(self.Y)
        m_real = np.mean(self.Z)

        e = np.abs(m_calc - m_real)
        self.assertLess(e, self.eps)

    def test_prob_zero(self):
        P = [np.zeros(k) for k in self.n]
        m = teneva.mean(self.Y, P)
        self.assertLess(np.abs(m), self.eps)


class TestActOneSum(unittest.TestCase):
    def setUp(self):
        self.n = [5] * 10
        self.Y = teneva.rand(self.n, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.eps = 1.E-13

    def test_base(self):
        s_calc = teneva.sum(self.Y)
        s_real = np.sum(self.Z)

        e = np.abs(s_calc - s_real)
        self.assertLess(e, self.eps)


if __name__ == '__main__':
    unittest.main()

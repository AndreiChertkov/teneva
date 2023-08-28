import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestTensorsConst(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [10, 11, 12, 13, 14]
        self.v = 42.
        self.I_zero = [
            [0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5],
            [9, 9, 9, 9, 9],
        ]
        self.i_non_zero = [5, 5, 5, 5, 5]
        self.eps = 1.E-5

    def test_base(self):
        Y = teneva.const(self.n, self.v)

        self.assertEqual(len(Y), self.d)

        for G in Y:
            self.assertEqual(G.shape[0], 1)
            self.assertEqual(G.shape[2], 1)

        Y_full = teneva.full(Y)
        e_min = abs(np.min(Y_full) - self.v)
        e_max = abs(np.max(Y_full) - self.v)
        self.assertLess(e_min, self.eps)
        self.assertLess(e_max, self.eps)

    def test_with_i_zero(self):
        Y = teneva.const(self.n, self.v, self.I_zero)

        for i in self.I_zero:
            y = teneva.get(Y, i)
            e = abs(y)
            self.assertLess(e, self.eps)

        Y_full = teneva.full(Y)
        mean = np.sum(Y_full) / np.sum(Y_full > 1.E-20)
        e = abs(mean - self.v)
        self.assertLess(e, self.eps)

    def test_with_i_non_zero(self):
        Y = teneva.const(self.n, self.v, self.I_zero, self.i_non_zero)

        for i in self.I_zero:
            y = teneva.get(Y, i)
            e = abs(y)
            self.assertLess(e, self.eps)

        y = teneva.get(Y, self.i_non_zero)
        e = abs(y - self.v)
        self.assertLess(e, self.eps)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

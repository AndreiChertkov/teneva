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

    def test_with_i_non_zero(self):
        Y = teneva.const(self.n, self.v, self.I_zero, self.i_non_zero)

        for i in self.I_zero:
            y = teneva.get(Y, i)
            e = abs(y)
            self.assertLess(e, self.eps)

        y = teneva.get(Y, self.i_non_zero)
        e = abs(y - self.v)
        self.assertLess(e, self.eps)

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


class TestTensorsDelta(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [10, 11, 12, 13, 14]
        self.v = 42.
        self.i = [5, 6, 7, 8, 9]
        self.eps = 1.E-5

    def test_base(self):
        Y = teneva.delta(self.n, self.i, self.v)

        self.assertEqual(len(Y), self.d)

        for G in Y:
            self.assertEqual(G.shape[0], 1)
            self.assertEqual(G.shape[2], 1)

        y = teneva.get(Y, self.i)
        e = abs(y - self.v)
        self.assertLess(e, self.eps)

        s = teneva.norm(Y)
        e = abs(s - self.v)
        self.assertLess(e, self.eps)

        Y_full = teneva.full(Y)

        e = abs(Y_full[tuple(self.i)] - self.v)
        self.assertLess(e, self.eps)

        s = len([y for y in Y_full.flatten() if abs(y) > 1.E-10])
        self.assertEqual(s, 1)


class TestTensorsPoly(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [10, 11, 12, 13, 14]
        self.i = np.array([2, 3, 3, 4, 5])
        self.shift = np.array([2, 3, 5, 3, 8])
        self.shift_scalar = 12.
        self.scale = 5.
        self.power = 3
        self.eps = 1.E-10

    def test_base(self):
        Y = teneva.poly(self.n, self.shift, self.power, self.scale)

        self.assertEqual(len(Y), self.d)

        for G in Y[1:]:
            self.assertEqual(G.shape[0], 2)
        for G in Y[:-1]:
            self.assertEqual(G.shape[2], 2)
        for k, G in zip(self.n, Y):
            self.assertEqual(G.shape[1], k)
        self.assertEqual(Y[0].shape[0], 1)
        self.assertEqual(Y[-1].shape[2], 1)

        y_appr = teneva.get(Y, self.i)
        y_real = self.scale * np.sum((self.i + self.shift)**self.power)

        y = teneva.get(Y, self.i)
        e = abs(y_appr - y_real)
        self.assertLess(e, self.eps)

    def test_shift_scalar(self):
        Y = teneva.poly(self.n, self.shift_scalar, self.power, self.scale)

        self.assertEqual(len(Y), self.d)

        for G in Y[1:]:
            self.assertEqual(G.shape[0], 2)
        for G in Y[:-1]:
            self.assertEqual(G.shape[2], 2)
        for k, G in zip(self.n, Y):
            self.assertEqual(G.shape[1], k)
        self.assertEqual(Y[0].shape[0], 1)
        self.assertEqual(Y[-1].shape[2], 1)

        y_appr = teneva.get(Y, self.i)
        y_real = self.scale * np.sum((self.i + self.shift_scalar)**self.power)

        y = teneva.get(Y, self.i)
        e = abs(y_appr - y_real)
        self.assertLess(e, self.eps)


class TestTensorsRand(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [12, 13, 14, 15, 16]
        self.r = [1, 2, 3, 4, 5, 1]
        self.r_const = 4
        self.eps = 1.E-10

    def test_base(self):
        Y = teneva.rand(self.n, self.r)

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (self.r[k], self.n[k], self.r[k+1]))

    def test_custom_limit(self):
        a = 0.994
        b = 0.995
        Y = teneva.rand(self.n, self.r, a=a, b=b)

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            for g in Y[k].flatten():
                self.assertLess(a, g)
                self.assertLess(g, b)

    def test_rank_const(self):
        Y = teneva.rand(self.n, self.r_const)
        r = [1] + [self.r_const] * (self.d-1) + [1]

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (r[k], self.n[k], r[k+1]))


class TestTensorsRandCustom(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [12, 13, 14, 15, 16]
        self.r = [1, 2, 3, 4, 5, 1]
        self.r_const = 4
        self.eps = 1.E-10

    def test_base(self):
        v = 42.
        f = lambda sz: [v]*sz
        Y = teneva.rand_custom(self.n, self.r, f)

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (self.r[k], self.n[k], self.r[k+1]))

        for k in range(self.d):
            for g in Y[k].flatten():
                self.assertLess(abs(v - g), 1.E-12)

    def test_rank_const(self):
        v = 42.
        f = lambda sz: [v]*sz
        Y = teneva.rand_custom(self.n, self.r_const, f)
        r = [1] + [self.r_const] * (self.d-1) + [1]

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (r[k], self.n[k], r[k+1]))

        for k in range(self.d):
            for g in Y[k].flatten():
                self.assertLess(abs(v - g), 1.E-12)


class TestTensorsRandNorm(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [12, 13, 14, 15, 16]
        self.r = [1, 2, 3, 4, 5, 1]
        self.r_const = 4
        self.eps = 1.E-10

    def test_base(self):
        Y = teneva.rand_norm(self.n, self.r)

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (self.r[k], self.n[k], self.r[k+1]))

    def test_custom_limit(self):
        m = 2.
        s = 0.001
        Y = teneva.rand_norm(self.n, self.r, m=m, s=s)

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            for g in Y[k].flatten():
                self.assertLess(m-5.*s, g)
                self.assertLess(g, m+5.*s)

    def test_rank_const(self):
        Y = teneva.rand_norm(self.n, self.r_const)
        r = [1] + [self.r_const] * (self.d-1) + [1]

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (r[k], self.n[k], r[k+1]))


class TestTensorsRandStab(unittest.TestCase):
    def setUp(self):
        self.d = 10000
        self.n = [10] * self.d
        self.r = 5
        self.i = [0] * self.d
        self.eps = 1.E-6

    def test_base(self):
        Y = teneva.rand_stab(self.n, self.r)
        r = [1] + [self.r] * (self.d-1) + [1]

        self.assertEqual(len(Y), self.d)
        for k in range(self.d):
            self.assertEqual(Y[k].shape, (r[k], self.n[k], r[k+1]))

        y = teneva.get(Y, self.i)
        self.assertLess(abs(y - 1.), self.eps)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

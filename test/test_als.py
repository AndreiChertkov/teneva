import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestAlsAls(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [5] * self.d
        self.r = 3
        self.m = 1.E+4

        def func(I):
            return np.sum(I, axis=1)

        self.I_trn = teneva.sample_lhs(self.n, self.m, seed=42)
        self.y_trn = func(self.I_trn)

        self.I_tst = teneva.sample_rand(self.n, self.m, seed=42)
        self.y_tst = func(self.I_tst)

    def test_base(self):
        Y = teneva.rand(self.n, self.r, seed=42)
        Y = teneva.als(self.I_trn, self.y_trn, Y)

        e_trn = teneva.accuracy_on_data(Y, self.I_trn, self.y_trn)
        e_tst = teneva.accuracy_on_data(Y, self.I_tst, self.y_tst)

        self.assertLess(e_trn, 1.E-5)
        self.assertLess(e_tst, 1.E-5)

    def test_rank(self):
        Y = teneva.rand(self.n, 1, seed=42)
        Y = teneva.als(self.I_trn, self.y_trn, Y, r=10)

        e_trn = teneva.accuracy_on_data(Y, self.I_trn, self.y_trn)
        e_tst = teneva.accuracy_on_data(Y, self.I_tst, self.y_tst)

        self.assertLess(e_trn, 4.E-4)
        self.assertLess(e_tst, 4.E-4)

    def test_weight(self):
        Y = teneva.rand(self.n, self.r, seed=42)
        w = np.arange(self.m) + 1.
        Y = teneva.als(self.I_trn, self.y_trn, Y, w=w)

        e_trn = teneva.accuracy_on_data(Y, self.I_trn, self.y_trn)
        e_tst = teneva.accuracy_on_data(Y, self.I_tst, self.y_tst)

        self.assertLess(e_trn, 4.E-4)
        self.assertLess(e_tst, 4.E-4)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

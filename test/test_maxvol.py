import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


np.random.seed(42)


class TestMaxvol(unittest.TestCase):
    def setUp(self):
        self.n = 5000   # Number of rows
        self.r = 50     # Number of columns
        self.e = 1.01   # Accuracy parameter
        self.k = 500    # Maximum number of iterations

        # Random tall matrix:
        rand = np.random.default_rng(42)
        self.A = rand.normal(size=(self.n, self.r))

        self.eps = 1.E-12

    def test_base(self):
        I, B = teneva.maxvol(self.A, self.e, self.k)
        C = self.A[I, :]

        e = np.max(np.abs(B))
        self.assertTrue(1. <= e <= 1.01)

        e = np.max(np.abs(self.A - B @ C))
        self.assertLess(e, self.eps)


class TestMaxvolRect(unittest.TestCase):
    def setUp(self):
        self.n = 5000   # Number of rows
        self.r = 50     # Number of columns
        self.e = 1.01   # Accuracy parameter
        self.k = 500    # Maximum number of iterations

        # Random tall matrix:
        rand = np.random.default_rng(42)
        self.A = rand.normal(size=(self.n, self.r))

        self.eps = 1.E-12

    def test_base(self):
        dr_min = 2  # Minimum number of added rows
        dr_max = 8  # Maximum number of added rows

        I, B = teneva.maxvol_rect(self.A, self.e, dr_min, dr_max)
        C = self.A[I, :]

        e = np.max(np.abs(B))
        self.assertTrue(1. <= e <= 1.01)

        e = np.max(np.abs(self.A - B @ C))
        self.assertLess(e, self.eps)


if __name__ == '__main__':
    unittest.main()

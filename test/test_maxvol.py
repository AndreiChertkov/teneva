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
        self.A = np.random.randn(self.n, self.r)

    def test_maxvol(self):
        I, B = teneva.maxvol(self.A, self.e, self.k)
        C = self.A[I, :]

        e = np.max(np.abs(B))
        self.assertTrue(1. <= e <= 1.01)

        e = np.max(np.abs(self.A - B @ C))
        self.assertLess(e, 1.E-14)

    def test_maxvol_rect(self):
        dr_min = 2  # Minimum number of added rows
        dr_max = 8  # Maximum number of added rows

        I, B = teneva.maxvol_rect(self.A, self.e, dr_min, dr_max)
        C = self.A[I, :]

        e = np.max(np.abs(B))
        self.assertTrue(1. <= e <= 1.01)

        e = np.max(np.abs(self.A - B @ C))
        self.assertLess(e, 1.E-14)


if __name__ == '__main__':
    unittest.main()

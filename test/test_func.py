import numpy as np
from scipy.optimize import rosen
import teneva
from time import perf_counter as tpc
import unittest


class TestFuncFuncGet(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-10

    def test_base(self):
        f = lambda X: rosen(X.T)        # Target function
        f_grid = lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb'))
        a = [-2., -4., -3., -2.]        # Grid lower bounds
        b = [+2., +3., +4., +2.]        # Grid upper bounds
        n = [5, 6, 7, 8]                # Grid size
        Y0 = teneva.rand(n, r=2)        # Initial approximation for TT-cross
        e = 1.E-3                       # Accuracy for TT-CROSS
        eps = 1.E-6                     # Accuracy for truncation
        Y = teneva.cross(f_grid, Y0, e=e)
        Y = teneva.truncate(Y, eps)

        A = teneva.func_int(Y)

        X = np.array([
            [0., 0., 0., 0.],
            [0., 2., 3., 2.],
            [1., 1., 1., 1.],
        ])

        Z = teneva.func_get(X, A, a, b)

        err = np.linalg.norm(Z - f(X))
        self.assertLess(err, self.eps)

        x = X[1]
        z = teneva.func_get(x, A, a, b)
        err = np.linalg.norm(z - f(x))
        self.assertLess(err, self.eps)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

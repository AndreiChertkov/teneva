import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestGridPoiScale(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-16

    def test_base_cheb(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]

        X = np.array([       # We prepare 4 spatial points:
            [-5., -3., -1.], # Point near the lower bound
            [ 0.,  0.,  0.], # Zero point
            [+5., +3., +1.], # Point near the upper bound
        ])

        Xsc = teneva.poi_scale(X, a, b, 'cheb')

        self.assertLess(abs(Xsc[0, 0] + 1.), self.eps)
        self.assertLess(abs(Xsc[0, 1] + 1.), self.eps)
        self.assertLess(abs(Xsc[0, 2] + 1.), self.eps)

        self.assertLess(abs(Xsc[1, 0] - 0.), self.eps)
        self.assertLess(abs(Xsc[1, 1] - 0.), self.eps)
        self.assertLess(abs(Xsc[1, 2] - 0.), self.eps)

        self.assertLess(abs(Xsc[2, 0] - 1.), self.eps)
        self.assertLess(abs(Xsc[2, 1] - 1.), self.eps)
        self.assertLess(abs(Xsc[2, 2] - 1.), self.eps)

    def test_base_uni(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]

        X = np.array([       # We prepare 4 spatial points:
            [-5., -3., -1.], # Point near the lower bound
            [ 0.,  0.,  0.], # Zero point
            [+5., +3., +1.], # Point near the upper bound
        ])

        Xsc = teneva.poi_scale(X, a, b)

        self.assertLess(abs(Xsc[0, 0] - 0.), self.eps)
        self.assertLess(abs(Xsc[0, 1] - 0.), self.eps)
        self.assertLess(abs(Xsc[0, 2] - 0.), self.eps)

        self.assertLess(abs(Xsc[1, 0] - 0.5), self.eps)
        self.assertLess(abs(Xsc[1, 1] - 0.5), self.eps)
        self.assertLess(abs(Xsc[1, 2] - 0.5), self.eps)

        self.assertLess(abs(Xsc[2, 0] - 1.), self.eps)
        self.assertLess(abs(Xsc[2, 1] - 1.), self.eps)
        self.assertLess(abs(Xsc[2, 2] - 1.), self.eps)

    def test_same_cheb(self):
        X = np.array([
            [0., 0.5, 0.7, 1.0],
            [-1., -0.9, 0.4, 0.9]])

        Xsc = teneva.poi_scale(X, -1., +1., 'cheb')

        err = np.linalg.norm(X - Xsc)
        self.assertLess(err, self.eps)

    def test_same_uni(self):
        X = np.array([
            [0., 0.5, 0.7, 1.0],
            [0.1, 0.2, 0.4, 0.8]])

        Xsc = teneva.poi_scale(X, 0., 1.)

        err = np.linalg.norm(X - Xsc)
        self.assertLess(err, self.eps)


class TestGridPoiToInd(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-14

    def test_base_cheb(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]
        n = [9, 8, 7]

        X = np.array([       # We prepare 4 spatial points:
            [-5., -3., -1.], # Point near the lower bound
            [ 0.,  0.,  0.], # Zero point
            [-1., +2.,  0.], # Random point
            [+5., +3., +1.], # Point near the upper bound
        ])

        I = teneva.poi_to_ind(X, a, b, n, 'cheb')

        self.assertLess(abs(I[0, 0] - 8), self.eps)
        self.assertLess(abs(I[0, 1] - 7), self.eps)
        self.assertLess(abs(I[0, 2] - 6), self.eps)

        self.assertLess(abs(I[1, 0] - 4), self.eps)
        self.assertLess(abs(I[1, 1] - 4), self.eps)
        self.assertLess(abs(I[1, 2] - 3), self.eps)

        self.assertLess(abs(I[2, 0] - 5), self.eps)
        self.assertLess(abs(I[2, 1] - 2), self.eps)
        self.assertLess(abs(I[2, 2] - 3), self.eps)

        self.assertLess(abs(I[3, 0] - 0), self.eps)
        self.assertLess(abs(I[3, 1] - 0), self.eps)
        self.assertLess(abs(I[3, 2] - 0), self.eps)

    def test_base_uni(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]
        n = [9, 8, 7]

        X = np.array([       # We prepare 4 spatial points:
            [-5., -3., -1.], # Point near the lower bound
            [ 0.,  0.,  0.], # Zero point
            [-1., +2.,  0.], # Random point
            [+5., +3., +1.], # Point near the upper bound
        ])

        I = teneva.poi_to_ind(X, a, b, n)

        self.assertLess(abs(I[0, 0] - 0), self.eps)
        self.assertLess(abs(I[0, 1] - 0), self.eps)
        self.assertLess(abs(I[0, 2] - 0), self.eps)

        self.assertLess(abs(I[1, 0] - 4), self.eps)
        self.assertLess(abs(I[1, 1] - 4), self.eps)
        self.assertLess(abs(I[1, 2] - 3), self.eps)

        self.assertLess(abs(I[2, 0] - 3), self.eps)
        self.assertLess(abs(I[2, 1] - 6), self.eps)
        self.assertLess(abs(I[2, 2] - 3), self.eps)

        self.assertLess(abs(I[3, 0] - 8), self.eps)
        self.assertLess(abs(I[3, 1] - 7), self.eps)
        self.assertLess(abs(I[3, 2] - 6), self.eps)

    def test_one_poi(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]
        n = [9, 8, 7]

        X = [-5., -3., -1.]

        I = teneva.poi_to_ind(X, a, b, n)

        self.assertEqual(len(I.shape), 1)
        self.assertEqual(I.shape[0], len(n))
        self.assertLess(abs(I[0] - 0), self.eps)
        self.assertLess(abs(I[0] - 0), self.eps)
        self.assertLess(abs(I[0] - 0), self.eps)

    def test_same_cheb(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]
        n = [7, 5, 3]

        X = np.array([
            [-5., -3., -1.],  # Point near the lower bound
            [ 0.,  0.,  0.],  # Zero point
            [+5., +3., +1.],  # Point near the upper bound
        ])

        I = teneva.poi_to_ind(X, a, b, n, 'cheb')
        Z = teneva.ind_to_poi(I, a, b, n, 'cheb')

        err = np.linalg.norm(X - Z)
        self.assertLess(err, self.eps)

    def test_same_uni(self):
        a = [-5., -3., -1.]
        b = [+5., +3., +1.]
        n = [7, 5, 3]

        X = np.array([
            [-5., -3., -1.],  # Point near the lower bound
            [ 0.,  0.,  0.],  # Zero point
            [+5., +3., +1.],  # Point near the upper bound
        ])

        I = teneva.poi_to_ind(X, a, b, n)
        Z = teneva.ind_to_poi(I, a, b, n)

        err = np.linalg.norm(X - Z)
        self.assertLess(err, self.eps)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

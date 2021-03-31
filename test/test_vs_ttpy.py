import numpy as np
import unittest


import teneva


import tt
from tt.cross.rectcross import cross as ttpy_cross


np.random.seed(42)


def _err(A, B):
    if isinstance(A, list):
        A = tt.tensor.from_list(A)
    if isinstance(B, list):
        B = tt.tensor.from_list(B)
    if isinstance(A, (int, float)):
        return abs((A - B) / A)
    if isinstance(A, np.ndarray):
        return np.linalg.norm(A - B) / np.linalg.norm(A)
    return (A - B).norm() / A.norm()


class TestAdd(unittest.TestCase):

    def setUp(self):
        self.Z1 = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Z2 = tt.rand(np.array([5, 6, 7, 8, 9]), r=12)
        self.Y1 = tt.tensor.to_list(self.Z1)
        self.Y2 = tt.tensor.to_list(self.Z2)
        self.e = 1.E-14

    def test_base(self):
        C1 = self.Z1 + self.Z2
        C2 = teneva.add(self.Y1, self.Y2)

        self.assertTrue(_err(C1, C2) < self.e)


class TestCross(unittest.TestCase):

    def test_base(self):
        N = [9, 8, 8, 9, 8]  # Shape of the tensor
        d = len(N)           # Dimension of the problem
        M = 10000            # Number of test cases
        nswp     = 6         # Sweep number
        eps      = 1.E-6     # Desired accuracy
        kickrank = 2         # Cross parameter
        rf       = 2         # Cross parameter

        def f(x):
            y = np.sum([x[i]**(i+1) for i in range(d)])
            y = np.sin(y)**2 / (1 + y**2)
            return  y

        def f_vect(X):
            """Naively vectorized model function."""
            return np.array([f(x) for x in X])

        X_tst = np.vstack([np.random.choice(N[i], M) for i in range(d)]).T
        Y_tst = np.array([f(x) for x in X_tst])

        Y0 = teneva.rand(N, 2)

        Y = teneva.cross(f_vect, Y0, nswp, kickrank, rf)
        Y = teneva.truncate(Y, eps)

        get = teneva.getter(Y)
        Z = np.array([get(x) for x in X_tst])
        e1 = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
        r1 = teneva.erank(Y)

        def solver_cro(f, Y0, nswp=10, eps=None, kickrank=2, rf=2):
            Y0 = tt.tensor.from_list(Y0)
            Y = ttpy_cross(myfun=f, x0=Y0, nswp=nswp, eps=1.E-16,
                      eps_abs=0., kickrank=kickrank, rf=rf,
                      verbose=False, stop_fun=None, approx_fun=None)
            if eps:
                Y = Y.round(eps)
            return tt.tensor.to_list(Y)

        Yr_cro = solver_cro(f_vect, Y0, nswp, eps, kickrank, rf)

        get = teneva.getter(Yr_cro)
        Z = np.array([get(x) for x in X_tst])
        e2 = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
        r2 = teneva.erank(Y)

        self.assertTrue(_err(e1, e2) < 1.E-5)
        self.assertTrue(_err(r1, r2) < 5.E-14)


class TestErank(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        m1 = self.Z.erank
        m2 = teneva.erank(self.Y)

        self.assertTrue(_err(m1, m2) < self.e)


class TestFull(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8]), r=5)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        X1 = self.Z.full()
        X2 = teneva.full(self.Y)

        self.assertTrue(_err(X1, X2) < self.e)


class TestGet(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.x = np.array([0, 1, 2, 1, 0])
        self.e = 1.E-14

    def test_base(self):
        y1 = self.Z[self.x]
        y2 = teneva.get(self.Y, self.x)

        self.assertTrue(_err(y1, y2) < self.e)

    def test_getter(self):
        get = teneva.getter(self.Y)

        y1 = self.Z[self.x]
        y2 = get(self.x)

        self.assertTrue(_err(y1, y2) < self.e)


class TestGetCdf(unittest.TestCase):

    def test_base(self):

        def cdf0(x, X):
            return np.array([np.sum(X <= p) for p in x]) / len(X)

        X = [3, 3, 1, 4]
        x = np.array([3, 55, 0.5, 1.5, 2])

        ecdf = teneva.get_cdf(X)

        v1 = ecdf(x)
        v2 = cdf0(x, X)

        self.assertTrue(_err(v1, v2) < 1.E-16)


class TestMean(unittest.TestCase):
    """
    Todo:
        Add tests for the given non uniform P.
    """

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        m1 = tt.sum(self.Z) / np.prod([G.shape[1] for G in self.Y])
        m2 = teneva.mean(self.Y)

        self.assertTrue(_err(m1, m2) < self.e)


class TestMul(unittest.TestCase):

    def setUp(self):
        self.Z1 = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Z2 = tt.rand(np.array([5, 6, 7, 8, 9]), r=12)
        self.Y1 = tt.tensor.to_list(self.Z1)
        self.Y2 = tt.tensor.to_list(self.Z2)
        self.e = 1.E-14

    def test_base(self):
        C1 = self.Z1 * self.Z2
        C2 = teneva.mul(self.Y1, self.Y2)

        self.assertTrue(_err(C1, C2) < self.e)


class TestNorm(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        n1 = self.Z.norm()
        n2 = teneva.norm(self.Y)

        self.assertTrue(_err(n1, n2) < self.e)


class TestRand(unittest.TestCase):

    def setUp(self):
        self.e = 1.E-15

    def test_base(self):
        def sf(k):
            return np.array([0.1] * k)

        n = np.array([5, 6, 7, 8, 9])
        r = 4
        Z = tt.rand(n, r=r, samplefunc=sf)
        Y = teneva.rand(n, r, sf)

        self.assertTrue(_err(Z, Y) < self.e)


class TestSum(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        m1 = tt.sum(self.Z)
        m2 = teneva.sum(self.Y)

        self.assertTrue(_err(m1, m2) < self.e)


class TestTruncate(unittest.TestCase):

    def setUp(self):
        self.Z = tt.rand(np.array([5, 6, 7, 8, 9]), r=10)
        self.Y = tt.tensor.to_list(self.Z)
        self.e = 1.E-14

    def test_base(self):
        e = 1.E-2
        Z = self.Z.round(e)
        Y = teneva.truncate(self.Y, e)

        self.assertTrue(_err(Z, Y) < self.e)


if __name__ == '__main__':
    unittest.main()

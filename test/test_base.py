import numpy as np
import unittest


import teneva


np.random.seed(42)


class TestCross(unittest.TestCase):

    def test_base(self):
        N = [9, 8, 8, 9, 8]  # Shape of the tensor
        d = len(N)           # Dimension of the problem
        M = 10000            # Number of test cases
        nswp     = 10        # Sweep number
        eps      = 1.E-6     # Desired accuracy
        kickrank = 1         # Cross parameter
        rf       = 1         # Cross parameter

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
        e = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
        r = teneva.erank(Y)

        self.assertTrue(e < 5.E-7)
        self.assertTrue(r < 13.81)


if __name__ == '__main__':
    unittest.main()

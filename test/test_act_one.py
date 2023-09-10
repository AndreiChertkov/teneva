import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestActOneCopy(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-16

    def test_array(self):
        Y = np.random.randn(3, 4, 5)
        Z = teneva.copy(Y)

        y1 = Y[1, 2, 0]
        Z[1, 2, 0] = 42.
        y2 = Y[1, 2, 0]

        e = np.abs(y1-y2)
        self.assertLess(e, self.eps)

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

    def test_none(self):
        y = None
        z = teneva.copy(y)

        self.assertIsNone(z)

    def test_number(self):
        y = 42.
        z = teneva.copy(y)

        e = np.abs(z-y)
        self.assertLess(e, self.eps)


class TestActOneGet(unittest.TestCase):
    def setUp(self):
        self.Y = teneva.rand([10] * 5, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.i = [1, 2, 3, 4, 5]
        self.eps = 1.E-16

    def test_base(self):
        i = np.array(self.i)
        y = teneva.get(self.Y, i)
        z = self.Z[tuple(i)]
        e = np.abs(y-z)
        self.assertLess(e, self.eps)

    def test_list(self):
        i = self.i
        y = teneva.get(self.Y, i)
        z = self.Z[tuple(i)]
        e = np.abs(y-z)
        self.assertLess(e, self.eps)

    def test_many(self):
        i = np.array(self.i)
        I = [i, i-1, i+1]
        y = teneva.get(self.Y, I)
        z = np.array([self.Z[tuple(i)] for i in I])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)


class TestActOneGetAndGrad(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [10] * self.d
        self.Y = teneva.rand(self.n, r=3, seed=42)
        self.i = [1, 2, 3, 4, 5]
        self.lr = 1.E-4
        self.eps = 1.E-16

    def test_base(self):
        y, dY = teneva.get_and_grad(self.Y, self.i)

        Z = teneva.copy(self.Y)
        for k in range(self.d):
            Z[k] -= self.lr * dY[k]

        z = teneva.get(Z, self.i)
        e = teneva.accuracy(self.Y, Z)

        self.assertLess(z, y)
        self.assertLess(e, self.lr)

    def test_get(self):
        y, dY = teneva.get_and_grad(self.Y, self.i)
        z = teneva.get(self.Y, self.i)
        e = np.abs(y-z)
        self.assertLess(e, self.eps)


class TestActOneGetMany(unittest.TestCase):
    def setUp(self):
        self.Y = teneva.rand([10] * 5, r=3, seed=42)
        self.Z = teneva.full(self.Y)
        self.I = [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 2, 1],
        ]
        self.eps = 1.E-16

    def test_base(self):
        I = np.array(self.I)
        y = teneva.get_many(self.Y, I)
        z = np.array([self.Z[tuple(i)] for i in I])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)

    def test_list(self):
        I = self.I
        y = teneva.get_many(self.Y, I)
        z = np.array([self.Z[tuple(i)] for i in I])
        e = np.abs(np.max(y-z))
        self.assertLess(e, self.eps)


class TestActOneInterface(unittest.TestCase):
    def setUp(self):
        self.n = [6, 5, 4, 3, 2]
        self.d = len(self.n)
        self.Y = teneva.rand(self.n, r=3, seed=42)
        self.i = [5, 4, 3, 2, 1]
        self.P = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3],
            [0.1, 0.2]]

        self.d2 = 8
        self.n2 = [5] * self.d2
        self.Y2 = teneva.rand(self.n2, r=4, seed=42)
        self.p2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        self.eps = 1.E-15

    def test_base(self):
        phi_r = teneva.interface(self.Y)
        phi_l = teneva.interface(self.Y, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_i(self):
        phi_r = teneva.interface(self.Y, i=self.i)
        phi_l = teneva.interface(self.Y, i=self.i, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_i_p(self):
        phi_r = teneva.interface(self.Y, self.P, self.i)
        phi_l = teneva.interface(self.Y, self.P, self.i, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_norm_natural(self):
        phi_r = teneva.interface(self.Y, norm='n')
        phi_l = teneva.interface(self.Y, norm='n', ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_norm_none(self):
        phi_r = teneva.interface(self.Y, norm=None)
        phi_l = teneva.interface(self.Y, norm=None, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_p(self):
        phi_r = teneva.interface(self.Y, self.P)
        phi_l = teneva.interface(self.Y, self.P, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
        self.assertLess(e, self.eps)

    def test_p_equal(self):
        phi_r = teneva.interface(self.Y2, self.p2)
        phi_l = teneva.interface(self.Y2, self.p2, ltr=True)

        self.assertEqual(len(phi_r[0]), 1)
        self.assertEqual(len(phi_r[-1]), 1)
        self.assertEqual(len(phi_l[0]), 1)
        self.assertEqual(len(phi_l[-1]), 1)

        v_r = phi_r[0].item()
        v_l = phi_l[-1].item()
        e = np.abs(v_r-v_l)
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
    np.random.seed(42)
    unittest.main()

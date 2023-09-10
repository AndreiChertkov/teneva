import numpy as np
import teneva
from time import perf_counter as tpc
import unittest


class TestCrossCross(unittest.TestCase):
    def setUp(self):
        self.d = 5
        self.n = [ 20,  18,  16,  14,  12]
        self.func = lambda I: np.sum(I, axis=1)

        self.I_tst = teneva.sample_rand(self.n, 1.E+4, seed=42)
        self.y_tst = self.func(self.I_tst)

    def test_base(self):
        r = 1
        m = None
        e = 1.E-8
        nswp = None
        dr_min = 1
        info = {}
        Y = teneva.rand(self.n, r)
        Y = teneva.cross(self.func, Y, m, e, nswp, dr_min=dr_min, dr_max=3,
            info=info)

        r1 = teneva.erank(Y)
        r2 = r + dr_min * info['nswp']
        self.assertTrue(r1 >= r2)

        Y = teneva.truncate(Y, e)

        err = teneva.accuracy_on_data(Y, self.I_tst, self.y_tst)
        self.assertLess(err, e*10)

    def test_cb(self):
        nswp = 5
        info = {}

        def cb(Y, info, opts):
            if info['nswp'] == nswp:
                # Stop the algorithm's work (just for demo!)
                return True

        Y = teneva.rand(self.n, 1)
        Y = teneva.cross(self.func, Y, nswp=100, cb=cb, info=info)

        self.assertEqual(info['nswp'], nswp)

    def test_e(self):
        r = 1
        m = None
        e = 1.E-4
        nswp = None
        dr_min = 1
        info = {}
        Y = teneva.rand(self.n, r)
        Y = teneva.cross(self.func, Y, m, e, nswp, dr_min=dr_min, dr_max=3,
            info=info)

        self.assertLess(info['e'], e)

    def test_m(self):
        r = 1
        m = 1.E+4
        e = None
        nswp = None
        dr_min = 1
        info = {}
        Y = teneva.rand(self.n, r)
        Y = teneva.cross(self.func, Y, m, e, nswp, dr_min=dr_min, dr_max=3,
            info=info)

        self.assertLess(info['m'], m+1)

    def test_m_cache_scale(self):
        nswp = 5
        cache = {}
        info = {}

        Y = teneva.rand(self.n, 1)
        Y = teneva.cross(self.func, Y, nswp=100,
            m_cache_scale=2, cache=cache, info=info)

        self.assertEqual(info['stop'], 'conv')

    def test_nswp(self):
        r = 1
        m = None
        e = None
        nswp = 4
        dr_min = 1
        info = {}
        Y = teneva.rand(self.n, r)
        Y = teneva.cross(self.func, Y, m, e, nswp, dr_min=dr_min, dr_max=3,
            info=info)

        self.assertEqual(info['nswp'], nswp)

    def test_vld(self):
        r = 1
        m = 1.E+4
        e = None
        e_vld=1.E-6
        nswp = None
        dr_min = 1
        info = {}
        I_vld = teneva.sample_lhs(self.n, 1.E+4, seed=42)
        y_vld = self.func(I_vld)
        Y = teneva.rand(self.n, r)
        Y = teneva.cross(self.func, Y, m, e, nswp, dr_min=dr_min, dr_max=3,
            info=info, I_vld=I_vld, y_vld=y_vld, e_vld=e_vld)

        err = teneva.accuracy_on_data(Y, I_vld, y_vld)
        self.assertLess(err, e_vld)

        err = teneva.accuracy_on_data(Y, self.I_tst, self.y_tst)
        self.assertLess(err, e_vld*10)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

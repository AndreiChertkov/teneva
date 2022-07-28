"""Package teneva, module func.demo.func_demo_rosenbrock: function.

This module contains class that implements analytical Rosenbrock function
for demo and tests.

"""
import numpy as np


from ..func import Func
from teneva import grid_flat
from teneva import ind_to_poi


class FuncDemoRosenbrock(Func):
    def __init__(self, d, dy=0.):
        """Rosenbrock function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/rosen.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194.

        """
        super().__init__(d, name='Rosenbrock')

        self.dy = dy

        self.set_lim(-2.048, +2.048)
        self.set_min([1.]*self.d, 0. + dy)

        self.with_cores = True

    def _calc(self, x):
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return np.sum(y1 + y2) + self.dy

    def _comp(self, X):
        y1 = 100. * (X[:, 1:] - X[:, :-1]**2)**2
        y2 = (X[:, :-1] - 1.)**2
        return np.sum(y1 + y2, axis=1) + self.dy

    def _cores(self, X):
        d = len(X)
        res = []
        for i, x in enumerate(X):
            x2 = x*x
            if i == 0:
                core = np.zeros([1, len(x), 3])
                core[0, :, 0] = 1
                core[0, :, 1] = x2
                core[0, :, 2] = 100*(x2**2) + (1-x)**2
            elif i == d-1:
                core = np.zeros([3, len(x), 1])
                core[2, :, 0] = 1
                core[1, :, 0] = -200*x
                core[0, :, 0] = 100*x2
            else:
                core = np.zeros([3, len(x), 3])
                core[0, :, 0] = core[2, :, 2] = 1.
                core[0, :, 1] = x2
                core[0, :, 2] = 100*x2 + 100*(x2**2) + (1-x)**2
                core[1, :, 2] = -200*x

            res.append(core)
        return res

    def build_cores(self):
        # TODO! It works only for square grids.
        self.method = 'CORES'

        def rosen_core_m(x, core):
            core[:] = 0.
            core[0, 0] = core[2, 2] = 1.
            core[0, 1] = x*x
            core[0, 2] = 100*x*x+100*(x**4) + (1-x)**2
            core[1, 2] = -200*x

        def rosen_core_f(x, core):
            core[0, :] = [1, x*x, 100*(x**4) + (1-x)**2]

        def rosen_core_l(x, core):
            core[:, 0] = [100*(x**2) , -200*x, 1]

        I = grid_flat([self.n[0]])[:, 0]
        X = ind_to_poi(I, self.a[0], self.b[0], self.n[0], self.kind)
        n = len(X)

        core_f = np.zeros([1, n, 3])
        core_m = np.zeros([3, n, 3])
        core_l = np.zeros([3, n, 1])

        for i, x in enumerate(X):
            rosen_core_f(x, core_f[:, i, :])
            rosen_core_m(x, core_m[:, i, :])
            rosen_core_l(x, core_l[:, i, :])

        self.prep([core_f] + [core_m]*(self.d-2) + [core_l])

"""Package teneva, module func.demo.func_demo_rosenbrock: function.

This module contains class that implements analytical Rosenbrock function
for demo and tests.

"""
import numpy as np


from ..func import Func


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

    def _calc(self, x):
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return np.sum(y1 + y2) + self.dy

    def _comp(self, X):
        y1 = 100. * (X[:, 1:] - X[:, :-1]**2)**2
        y2 = (X[:, :-1] - 1.)**2
        return np.sum(y1 + y2, axis=1) + self.dy

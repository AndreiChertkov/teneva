"""Package teneva, module func.demo.func_demo_rosenbrock: function.

This module contains class that implements analytical Rosenbrock function
for demo and tests.

"""
import numpy as np
from scipy.optimize import rosen


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
        return rosen(x) + self.dy

    def _comp(self, X):
        return rosen(X.T) + self.dy

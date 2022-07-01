"""Package teneva, module func.demo.func_demo_grienwank: function.

This module contains class that implements analytical Grienwank function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoGrienwank(Func):
    def __init__(self, d, dy=0.):
        """Grienwank function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/griewank.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194.

        """
        super().__init__(d, name='Grienwank')

        self.dy = dy

        self.set_lim(-600., +600.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        y1 = np.sum(x**2) / 4000

        y2 = np.cos(x / np.sqrt(np.arange(self.d) + 1.))
        y2 = - np.prod(y2)

        y3 = 1.

        return y1 + y2 + y3 + self.dy

    def _comp(self, X):
        y1 = np.sum(X**2, axis=1) / 4000

        y2 = np.cos(X / np.sqrt(np.arange(self.d) + 1))
        y2 = - np.prod(y2, axis=1)

        y3 = 1.

        return y1 + y2 + y3 + self.dy

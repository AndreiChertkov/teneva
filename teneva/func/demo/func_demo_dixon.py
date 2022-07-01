"""Package teneva, module func.demo.func_demo_dixon: function.

This module contains class that implements analytical Dixon function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoDixon(Func):
    def __init__(self, d, dy=0.):
        """Dixon function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/dixonpr.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194.

        """
        super().__init__(d, name='Dixon')

        self.dy = dy

        self.set_lim(-10., +10.)

        i = np.arange(1, self.d+1)
        x = 2**(-(2**i - 2) / 2**i)
        self.set_min(x, 0. + dy)


    def _calc(self, x):
        y1 = (x[0] - 1)**2
        y2 = np.arange(2, self.d+1) * (x[1:]**2 - x[:-1])**2
        y2 = np.sum(y2)
        return y1 + y2 + self.dy

    def _comp(self, X):
        y1 = (X[:, 0] - 1)**2
        y2 = np.arange(2, self.d+1) * (X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)
        return y1 + y2 + self.dy

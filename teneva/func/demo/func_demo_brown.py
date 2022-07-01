"""Package teneva, module func.demo.func_demo_brown: function.

This module contains class that implements analytical Brown function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoBrown(Func):
    def __init__(self, d, dy=0.):
        """Brown function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            for details.

        """
        super().__init__(d, name='Brown')

        self.dy = dy

        self.set_lim(-1., +4.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        y = (x[:-1]**2)**(x[1:]**2+1) + (x[1:]**2)**(x[:-1]**2+1)
        return np.sum(y) + self.dy

    def _comp(self, X):
        Y = (X[:, :-1]**2)**(X[:, 1:]**2+1) + (X[:, 1:]**2)**(X[:, :-1]**2+1)
        return np.sum(Y, axis=1) + self.dy

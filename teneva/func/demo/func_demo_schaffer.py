"""Package teneva, module func.demo.func_demo_schaffer: function.

This module contains class that implements analytical Schaffer function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoSchaffer(Func):
    def __init__(self, d, dy=0.):
        """Schaffer function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            for details.

        """
        super().__init__(d, name='Schaffer')

        self.dy = dy

        self.set_lim(-100., +100.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (np.sin(np.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return np.sum(y) + self.dy

    def _comp(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1) + self.dy

"""Package teneva, module func.demo.func_demo_exponential: function.

This module contains class that implements analytical Exponential function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoExponential(Func):
    def __init__(self, d):
        """Exponential function for demo and tests.

        Args:
            d (int): number of dimensions.

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194.

        """
        super().__init__(d, name='Exponential')

        self.set_lim(-1., +1.)
        self.set_min([0.]*self.d, -1.)

    def _calc(self, x):
        return -np.exp(-0.5 * np.sum(x**2))

    def _comp(self, X):
        return -np.exp(-0.5 * np.sum(X**2, axis=1))

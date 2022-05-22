"""Package teneva, module func.demo.func_demo_qing: function.

This module contains class that implements analytical Qing function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoQing(Func):
    def __init__(self, d):
        """Qing function for demo and tests.

        Args:
            d (int): number of dimensions.

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194.

        """
        super().__init__(d, name='Qing')

        self.set_lim(0., +500.)
        self.set_min(np.sqrt(np.arange(1, self.d+1)), 0.)

    def _calc(self, x):
        return np.sum((x**2 - np.arange(1, self.d+1))**2)

    def _comp(self, X):
        return np.sum((X**2 - np.arange(1, self.d+1))**2, axis=1)

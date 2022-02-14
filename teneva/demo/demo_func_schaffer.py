"""Package teneva, module demo.demo_func_schaffer: function.

This module contains class that implements analytical Schaffer function
for demo and tests.

"""

import numpy as np


from .demo_func import DemoFunc


class DemoFuncSchaffer(DemoFunc):
    def __init__(self, d):
        """Schaffer function for demo and tests.

        Args:
            d (int): number of dimensions.

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            for details.

        """
        super().__init__(d, 'Schaffer')

        self.set_lim(-100., +100.)
        self.set_min([0.]*self.d, 0.)

    def _calc(self, x):
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (np.sin(np.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return np.sum(y)

    def _comp(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

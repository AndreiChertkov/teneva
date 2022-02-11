"""Package teneva, module demo.demo_func_rastrigin: function.

This module contains class that implements analytical Rastrigin function
for demo and tests.

"""
import numpy as np


from .demo_func import DemoFunc


class DemoFuncRastrigin(DemoFunc):
    def __init__(self, d, A=10.):
        """Rastrigin function for demo and tests.

        See https://www.sfu.ca/~ssurjano/rastr.html for details.

        Args:
            d (int): number of dimensions.
            A (float): parameter of the function.

        """
        super().__init__(d, 'Rastrigin')

        self.par_A = A

        self.set_lim(-5.12, +5.12)
        self.set_min([0.]*self.d, 0.)

    def _calc(self, x):
        y1 = self.par_A * self.d
        y2 = np.sum(x**2 - self.par_A * np.cos(2. * np.pi * x))
        return y1 + y2

    def _comp(self, X):
        y1 = self.par_A * self.d
        y2 = np.sum(X**2 - self.par_A * np.cos(2. * np.pi * X), axis=1)
        return y1 + y2

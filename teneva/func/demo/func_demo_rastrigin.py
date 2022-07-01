"""Package teneva, module func.demo.func_demo_rastrigin: function.

This module contains class that implements analytical Rastrigin function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoRastrigin(Func):
    def __init__(self, d, dy=0., A=10.):
        """Rastrigin function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).
            A (float): parameter of the function.

        Note:
            See https://www.sfu.ca/~ssurjano/rastr.html for details.

            See also Johannes M Dieterich, Bernd Hartke. "Empirical review of
            standard benchmark functions using evolutionary global
            optimization". Applied Mathematics 2012; 3:1552-1564.

        """
        super().__init__(d, name='Rastrigin')

        self.dy = dy

        self.par_A = A

        self.set_lim(-5.12, +5.12)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        y1 = self.par_A * self.d
        y2 = np.sum(x**2 - self.par_A * np.cos(2. * np.pi * x))
        return y1 + y2 + self.dy

    def _comp(self, X):
        y1 = self.par_A * self.d
        y2 = np.sum(X**2 - self.par_A * np.cos(2. * np.pi * X), axis=1)
        return y1 + y2 + self.dy

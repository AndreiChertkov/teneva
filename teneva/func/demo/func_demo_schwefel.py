"""Package teneva, module func.demo.func_demo_schwefel: function.

This module contains class that implements analytical Schwefel function
for demo and tests.

"""
import numpy as np


from ..func import Func


class FuncDemoSchwefel(Func):
    def __init__(self, d, dy=0.):
        """Schwefel function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/schwef.html for details.

            See also Johannes M Dieterich, Bernd Hartke. "Empirical review of
            standard benchmark functions using evolutionary global
            optimization". Applied Mathematics 2012; 3:1552-1564.

        """
        super().__init__(d, name='Schwefel')

        self.dy = dy

        self.set_lim(-500., +500.)
        self.set_min([420.9687]*self.d, 0. + dy)

    def _calc(self, x):
        y0 = 418.9829 * self.d
        return y0 - np.sum(x * np.sin(np.sqrt(np.abs(x)))) + self.dy

    def _comp(self, X):
        y0 = 418.9829 * self.d
        return y0 - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1) + self.dy

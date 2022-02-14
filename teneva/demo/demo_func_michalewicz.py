"""Package teneva, module demo.demo_func_michalewicz: function.

This module contains class that implements analytical Michalewicz function
for demo and tests.

"""
import numpy as np


from .demo_func import DemoFunc


class DemoFuncMichalewicz(DemoFunc):
    def __init__(self, d, m=10.):
        """Michalewicz function for demo and tests.

        Args:
            d (int): number of dimensions.
            m (float): parameter of the function.

        Note:
            See https://www.sfu.ca/~ssurjano/michal.html for details.

            See also Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand,
            Jean-Marc Alliot. "Certified global minima for a benchmark of
            difficult optimization problems". arXiv preprint arXiv:2003.09867
            2020.

        """
        super().__init__(d, 'Michalewicz')

        self.par_m = m

        self.set_lim(0., np.pi)

        if self.d == 2:
            self.set_min([2.20, 1.57], -1.8013)
        if self.d == 5:
            self.set_min(None, -4.687658)
        if self.d == 10:
            self.set_min(None, -9.66015)

    def _calc(self, x):
        y1 = np.sin(((np.arange(self.d) + 1) * x**2 / np.pi))

        y = -np.sum(np.sin(x) * y1**(2 * self.par_m))

        return y

    def _comp(self, X):
        y1 = np.sin(((np.arange(self.d) + 1) * X**2 / np.pi))

        y = -np.sum(np.sin(X) * y1**(2 * self.par_m), axis=1)

        return y

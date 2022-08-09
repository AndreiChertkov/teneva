"""Package teneva, module func.demo.func_demo_rastrigin: function.

This module contains class that implements analytical Rastrigin function
for demo and tests.

"""
import numpy as np
try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


from ..func import Func
from ..utils import _cores_addition


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

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        par_A = torch.tensor(self.par_A)
        dy = torch.tensor(self.dy)
        pi = torch.tensor(np.pi)

        y1 = par_A * d
        y2 = torch.sum(x**2 - par_A * torch.cos(2. * pi * x))

        return y1 + y2 + dy

    def _comp(self, X):
        y1 = self.par_A * self.d
        y2 = np.sum(X**2 - self.par_A * np.cos(2. * np.pi * X), axis=1)
        return y1 + y2 + self.dy

    def _cores(self, X):
        return _cores_addition(
            [x**2 - self.par_A * np.cos(2 * np.pi * x) for x in X.T],
            a0=self.par_A*self.d)

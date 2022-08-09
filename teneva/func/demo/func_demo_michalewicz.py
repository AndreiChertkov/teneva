"""Package teneva, module func.demo.func_demo_michalewicz: function.

This module contains class that implements analytical Michalewicz function
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


class FuncDemoMichalewicz(Func):
    def __init__(self, d, dy=0., m=10.):
        """Michalewicz function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).
            m (float): parameter of the function.

        Note:
            See https://www.sfu.ca/~ssurjano/michal.html for details.

            See also Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand,
            Jean-Marc Alliot. "Certified global minima for a benchmark of
            difficult optimization problems". arXiv preprint arXiv:2003.09867
            2020.

            Note that the value of the global minimum is known only for the
            case of dimensions 2, 5, and 10. In this cases, only the
            corresponding value of the function is known, but not the argument.

        """
        super().__init__(d, name='Michalewicz')

        self.dy = dy

        self.par_m = m

        self.set_lim(0., np.pi)

        if self.d == 2:
            self.set_min([2.20, 1.57], -1.8013 + dy)
        if self.d == 5:
            self.set_min(None, -4.687658 + dy)
        if self.d == 10:
            self.set_min(None, -9.66015 + dy)

    def _calc(self, x):
        y1 = np.sin(((np.arange(self.d) + 1) * x**2 / np.pi))

        y = -np.sum(np.sin(x) * y1**(2 * self.par_m))

        return y + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        par_m = torch.tensor(self.par_m)
        dy = torch.tensor(self.dy)
        pi = torch.tensor(np.pi)

        y1 = torch.sin(((torch.arange(d) + 1) * x**2 / pi))

        y = -torch.sum(torch.sin(x) * y1**(2 * par_m))

        return y + dy

    def _comp(self, X):
        y1 = np.sin(((np.arange(self.d) + 1) * X**2 / np.pi))

        y = -np.sum(np.sin(X) * y1**(2 * self.par_m), axis=1)

        return y + self.dy

    def _cores(self, X):
        Y = _cores_addition(
            [np.sin(x) * np.sin(i*x**2/np.pi)**(2*self.par_m)
                for i, x in enumerate(X.T, 1)])
        Y[-1] *= -1.
        return Y

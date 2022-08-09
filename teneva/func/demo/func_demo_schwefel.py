"""Package teneva, module func.demo.func_demo_schwefel: function.

This module contains class that implements analytical Schwefel function
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
            optimization". Applied Mathematics 2012; 3:1552-1564
            ("128. Schwefel 2.26 Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).

        """
        super().__init__(d, name='Schwefel')

        self.dy = dy

        self.par_a = 418.9829

        self.set_lim(-500., +500.)
        self.set_min([420.9687]*self.d, 0. + dy)

    def _calc(self, x):
        y0 = self.par_a * self.d
        return y0 - np.sum(x * np.sin(np.sqrt(np.abs(x)))) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        par_a = torch.tensor(self.par_a)
        dy = torch.tensor(self.dy)

        y0 = par_a * d

        return y0 - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x)))) + dy

    def _comp(self, X):
        y0 = self.par_a * self.d
        return y0 - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1) + self.dy

    def _cores(self, X):
        return _cores_addition(
            [-x * np.sin(np.sqrt(np.abs(x))) for x in X.T],
            a0=self.par_a*self.d)

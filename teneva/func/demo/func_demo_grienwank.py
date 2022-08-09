"""Package teneva, module func.demo.func_demo_grienwank: function.

This module contains class that implements analytical Grienwank function
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
from ..utils import _cores_mults
from teneva import add


class FuncDemoGrienwank(Func):
    def __init__(self, d, dy=0.):
        """Grienwank function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/griewank.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("40. Griewank Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).

        """
        super().__init__(d, name='Grienwank')

        self.dy = dy

        self.set_lim(-600., +600.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        y1 = np.sum(x**2) / 4000

        y2 = np.cos(x / np.sqrt(np.arange(self.d) + 1.))
        y2 = - np.prod(y2)

        y3 = 1.

        return y1 + y2 + y3 + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        dy = torch.tensor(self.dy)

        y1 = torch.sum(x**2) / 4000

        y2 = torch.cos(x / torch.sqrt(torch.arange(d) + 1.))
        y2 = - torch.prod(y2)

        y3 = 1.

        return y1 + y2 + y3 + dy

    def _comp(self, X):
        y1 = np.sum(X**2, axis=1) / 4000

        y2 = np.cos(X / np.sqrt(np.arange(self.d) + 1))
        y2 = - np.prod(y2, axis=1)

        y3 = 1.

        return y1 + y2 + y3 + self.dy

    def _cores(self, X):
        Y = _cores_mults([np.cos(x / np.sqrt(i)) for i, x in enumerate(X.T, 1)])
        Y[-1] *= -1
        return add(Y, _cores_addition([x**2 / 4000. for x in X.T], a0=1))

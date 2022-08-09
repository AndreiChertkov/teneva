"""Package teneva, module func.demo.func_demo_qing: function.

This module contains class that implements analytical Qing function
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


class FuncDemoQing(Func):
    def __init__(self, d, dy=0.):
        """Qing function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("98. Qing Function"; Continuous, Differentiable, Separable
            Scalable, Multimodal).

            Note that we limit this function to the [0, 500] domain to make
            sure it has a single global minimum.

        """
        super().__init__(d, name='Qing')

        self.dy = dy

        self.set_lim(0., +500.)
        self.set_min(np.sqrt(np.arange(1, self.d+1)), 0. + dy)

    def _calc(self, x):
        return np.sum((x**2 - np.arange(1, self.d+1))**2) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        dy = torch.tensor(self.dy)

        return torch.sum((x**2 - torch.arange(1, d+1))**2) + dy

    def _comp(self, X):
        return np.sum((X**2 - np.arange(1, self.d+1))**2, axis=1) + self.dy

    def _cores(self, X):
        return _cores_addition([(x**2 - i)**2 for i, x in enumerate(X.T, 1)])

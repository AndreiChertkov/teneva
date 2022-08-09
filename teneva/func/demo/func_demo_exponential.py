"""Package teneva, module func.demo.func_demo_exponential: function.

This module contains class that implements analytical Exponential function
for demo and tests.

"""
import numpy as np
try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


from ..func import Func
from ..utils import _cores_mults


class FuncDemoExponential(Func):
    def __init__(self, d, dy=0.):
        """Exponential function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("54. Exponential Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).

        """
        super().__init__(d, name='Exponential')

        self.dy = dy

        self.set_lim(-1., +1.)
        self.set_min([0.]*self.d, -1. + dy)

    def _calc(self, x):
        return -np.exp(-0.5 * np.sum(x**2)) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        dy = torch.tensor(self.dy)

        return -torch.exp(-0.5 * torch.sum(x**2)) + dy

    def _comp(self, X):
        return -np.exp(-0.5 * np.sum(X**2, axis=1)) + self.dy

    def _cores(self, X):
        Y = _cores_mults([np.exp(-0.5 *x**2) for x in X.T])
        Y[-1] *= -1.
        return Y

"""Package teneva, module func.demo.func_demo_alpine: function.

This module contains class that implements analytical Alpine function
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


class FuncDemoAlpine(Func):
    def __init__(self, d, dy=0.):
        """Alpine function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("6. Alpine 1 Function"; Continuous, Non-Differentiable, Separable,
            Non-Scalable, Multimodal).

        """
        super().__init__(d, name='Alpine')

        self.dy = dy

        self.set_lim(-10., +10.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x)) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        dy = torch.tensor(self.dy)

        return torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x)) + dy

    def _comp(self, X):
        return np.sum(np.abs(X * np.sin(X) + 0.1 * X), axis=1) + self.dy

    def _cores(self, X):
        return _cores_addition([np.abs(x * (np.sin(x) + 0.1)) for x in X.T])

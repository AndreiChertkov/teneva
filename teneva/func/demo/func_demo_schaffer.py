"""Package teneva, module func.demo.func_demo_schaffer: function.

This module contains class that implements analytical Schaffer function
for demo and tests.

"""
import numpy as np
try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


from ..func import Func


class FuncDemoSchaffer(Func):
    def __init__(self, d, dy=0.):
        """Schaffer function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See Momin Jamil, Xin-She Yang. "A literature survey of benchmark
            functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("136. Schaffer F6 Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Multimodal).

        """
        super().__init__(d, name='Schaffer')

        self.dy = dy

        self.set_lim(-100., +100.)
        self.set_min([0.]*self.d, 0. + dy)

    def _calc(self, x):
        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (np.sin(np.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2
        return np.sum(y) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        dy = torch.tensor(self.dy)

        z = x[:-1]**2 + x[1:]**2
        y = 0.5 + (torch.sin(torch.sqrt(z))**2 - 0.5) / (1. + 0.001 * z)**2

        return torch.sum(y) + dy

    def _comp(self, X):
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1) + self.dy

"""Package teneva, module func.demo.func_demo_rosenbrock: function.

This module contains class that implements analytical Rosenbrock function
for demo and tests.

"""
import numpy as np
try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


from ..func import Func


class FuncDemoRosenbrock(Func):
    def __init__(self, d, dy=0.):
        """Rosenbrock function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/rosen.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("105. Rosenbrock Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Unimodal).

        """
        super().__init__(d, name='Rosenbrock')

        self.dy = dy

        self.set_lim(-2.048, +2.048)
        self.set_min([1.]*self.d, 0. + dy)

    def _calc(self, x):
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return np.sum(y1 + y2) + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        dy = torch.tensor(self.dy)

        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2

        return torch.sum(y1 + y2) + dy

    def _comp(self, X):
        y1 = 100. * (X[:, 1:] - X[:, :-1]**2)**2
        y2 = (X[:, :-1] - 1.)**2
        return np.sum(y1 + y2, axis=1) + self.dy

    def _cores(self, X):
        Y = []
        for i, x in enumerate(X.T):
            x2 = x*x
            if i == 0:
                G = np.zeros([1, len(x), 3])
                G[0, :, 0] = 1
                G[0, :, 1] = x2
                G[0, :, 2] = 100*(x2**2) + (1-x)**2
            elif i == self.d-1:
                G = np.zeros([3, len(x), 1])
                G[2, :, 0] = 1
                G[1, :, 0] = -200*x
                G[0, :, 0] = 100*x2
            else:
                G = np.zeros([3, len(x), 3])
                G[0, :, 0] = 1.
                G[2, :, 2] = 1.
                G[0, :, 1] = x2
                G[0, :, 2] = 100*x2 + 100*(x2**2) + (1-x)**2
                G[1, :, 2] = -200*x
            Y.append(G)
        return Y

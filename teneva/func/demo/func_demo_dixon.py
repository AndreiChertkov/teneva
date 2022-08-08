"""Package teneva, module func.demo.func_demo_dixon: function.

This module contains class that implements analytical Dixon function
for demo and tests.

"""
import numpy as np
try:
    import torch
    with_torch = True
except Exception as e:
    with_torch = False


from ..func import Func


class FuncDemoDixon(Func):
    def __init__(self, d, dy=0.):
        """Dixon function for demo and tests.

        Args:
            d (int): number of dimensions.
            dy (float): optional function shift (y -> y + dy).

        Note:
            See https://www.sfu.ca/~ssurjano/dixonpr.html for details.

            See also Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("48. Dixon & Price Function"; Continuous, Differentiable,
            Non-Separable, Scalable, Unimodal).

            Note that this function achieves a global minimum at more than one
            point.

        """
        super().__init__(d, name='Dixon')

        self.dy = dy

        self.set_lim(-10., +10.)

        # TODO: check this formula one more time:
        x = [1.]
        for _ in range(d-1):
            x.append(np.sqrt(x[-1]/2.))
        self.set_min(np.array(x), 0. + dy)

    def _calc(self, x):
        y1 = (x[0] - 1)**2
        y2 = np.arange(2, self.d+1) * (2. * x[1:]**2 - x[:-1])**2
        y2 = np.sum(y2)
        return y1 + y2 + self.dy

    def _calc_pt(self, x):
        if not with_torch:
            raise ValueError('Torch is not available')

        d = torch.tensor(self.d)
        dy = torch.tensor(self.dy)

        y1 = (x[0] - 1)**2
        y2 = torch.arange(2, d+1) * (2. * x[1:]**2 - x[:-1])**2
        y2 = torch.sum(y2)
        return y1 + y2 + dy

    def _comp(self, X):
        y1 = (X[:, 0] - 1)**2
        y2 = np.arange(2, self.d+1) * (2. * X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)
        return y1 + y2 + self.dy

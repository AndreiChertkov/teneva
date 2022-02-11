"""Package teneva, module demo.demo_func_schwefel: function.

This module contains class that implements analytical Schwefel function
for demo and tests.

"""

import numpy as np


from .demo_func import DemoFunc


class DemoFuncSchwefel(DemoFunc):
    def __init__(self, d):
        """Schwefel function for demo and tests.

        See https://www.sfu.ca/~ssurjano/schwef.html for details.

        Args:
            d (int): number of dimensions.

        """
        super().__init__(d, 'Schwefel')

        self.set_lim(-500., +500.)
        self.set_min([420.9687]*self.d, 0.)

    def _calc(self, x):
        y1 = 418.9829 * self.d
        y2 = - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return y1 + y2

    def _comp(self, X):
        y1 = 418.9829 * self.d
        y2 = - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)
        return y1 + y2

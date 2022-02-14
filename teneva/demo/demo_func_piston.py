"""Package teneva, module demo.demo_func_piston: function.

This module contains class that implements analytical Piston function
for demo and tests.

"""
import numpy as np


from .demo_func import DemoFunc


class DemoFuncPiston(DemoFunc):
    def __init__(self, d=7):
        """Piston 7-dimensional function for demo and tests.

        Args:
            d (int): number of dimensions. It should be 7.

        Note:
            See Vitaly Zankin, Gleb Ryzhakov, Ivan Oseledets. "Gradient descent
            based D-optimal design for the least-squares polynomial
            approximation". arXiv preprint arXiv:1806.06631 2018 for details.

        """
        super().__init__(d, 'Piston')

        if self.d != 7:
            raise ValueError('DemoFuncPiston is available only for 7-d case')

        self.set_lim(
            [30., 0.005, 0.002, 1000,  90000, 290, 340],
            [60., 0.020, 0.010, 5000, 110000, 296, 360])

    def _calc(self, x):
        return self._comp(x.reshape((1, -1)))[0]

    def _comp(self, X):
        _M  = X[:, 0]
        _S  = X[:, 1]
        _V0 = X[:, 2]
        _k  = X[:, 3]
        _P0 = X[:, 4]
        _Ta = X[:, 5]
        _T0 = X[:, 6]

        _A = _P0 * _S + 19.62 * _M - _k * _V0 / _S
        _Q = _P0 * _V0 / _T0
        _V = _S / 2 / _k * (np.sqrt(_A**2 + 4 * _k * _Q * _Ta) - _A)
        _C = 2 * np.pi * np.sqrt(_M / (_k + _S**2 * _Q * _Ta / _V**2))

        return _C

import numpy as np
from scipy.optimize import rosen


class DemoFunc:
    def __init__(self, d, name='Demo'):
        """Demo function.

        Args:
            d (int): number of dimensions.
            name (str): display name of the function.

        """
        self.d = d
        self.name = name

        self.set_lim(-1., +1.)
        self.set_min(None, None)

    def comp(self, X):
        """Compute the function in the given points.

        Args:
            X (np.ndarray): points (function inputs) in the form of array of the
                shape [samples, d], where "samples" is the number of samples and
                "d" is the dimension of the input.

        """
        raise NotImplementedError()

    def set_lim(self, a, b):
        """Set grid bounds.

        Args:
            a (list): grid lower bounds for each dimension (list or np.ndarray
                of length "d"). It may be also float, then the lower bounds for each dimension will be the same.
            b (list): grid upper bounds for each dimension (list or np.ndarray
                of length "d"). It may be also float, then the upper bounds for
                each dimension will be the same.

        """
        if isinstance(a, (int, float)): a = [a] * self.d
        if isinstance(b, (int, float)): b = [b] * self.d

        if isinstance(a, list): a = np.array(a, dtype=float)
        if isinstance(b, list): b = np.array(b, dtype=float)

        self.a = a
        self.b = b

    def set_min(self, x_min, y_min):
        """Set the exact global minimum of the function.

        Args:
            x_min (np.ndarray): argument of the function that provides its
                minimum value (list or np.ndarray of length "d").
            y_min (float): minimum value of the function.

        """
        if isinstance(x_min, list): x_min = np.array(x_min, dtype=float)

        self.x_min = x_min
        self.y_min = y_min


class DemoFuncAckley(DemoFunc):
    def __init__(self, d, a=20., b=0.2, c=2.*np.pi):
        """Ackley function for demo and tests.

        See https://www.sfu.ca/~ssurjano/ackley.html for details.

        Args:
            d (int): number of dimensions.
            a (float): parameter of the function.
            b (float): parameter of the function.
            c (float): parameter of the function.

        """
        super().__init__(d, 'Ackley')

        self.par_a = a
        self.par_b = b
        self.par_c = c

        self.set_lim(-32.768, +32.768)
        self.set_min([0.]*self.d, 0.)

    def comp(self, X):
        y1 = np.sqrt(np.sum(X.T**2, axis=0) / self.d)
        y1 = - self.par_a * np.exp(-self.par_b * y1)

        y2 = np.sum(np.cos(self.par_c * X.T), axis=0) / self.d
        y2 = - np.exp(y2)

        y3 = self.par_a + np.exp(1.0)

        return y1 + y2 + y3


class DemoFuncGrienwank(DemoFunc):
    def __init__(self, d):
        """Grienwank function for demo and tests.

        See https://www.sfu.ca/~ssurjano/griewank.html for details.

        Args:
            d (int): number of dimensions.

        """
        super().__init__(d, 'Grienwank')

        self.set_lim(-600., +600.)
        self.set_min([0.]*self.d, 0.)

    def comp(self, X):
        y1 = np.sum(X.T**2, axis=0)/4000

        y2 = np.cos(X / np.sqrt(np.arange(self.d) + 1))
        y2 = - np.prod(y2.T)

        y3 = 1.

        return y1 + y2 + y3


class DemoFuncMichalewicz(DemoFunc):
    def __init__(self, d, m=10.):
        """Michalewicz function for demo and tests.

        See https://www.sfu.ca/~ssurjano/michal.html for details.

        Args:
            d (int): number of dimensions.
            m (float): parameter of the function.

        """
        super().__init__(d, 'Michalewicz')

        self.par_m = m

        self.set_lim(0., np.pi)

        if self.d == 2:
            self.set_min([2.20, 1.57], -1.8013)
        if self.d == 5:
            self.set_min(None, -4.687658)
        if self.d == 10:
            self.set_min(None, -9.66015)

    def comp(self, X):
        y1 = np.arange(self.d) + 1
        y1 = np.sin((y1 * X**2 / np.pi))

        y = -np.sum(np.sin(X.T) * y1.T**(2 * self.par_m), axis=0)

        return y / self.d


class DemoFuncPiston(DemoFunc):
    def __init__(self):
        """Piston 7-dimensional function for demo and tests.

        See https://arxiv.org/pdf/1806.06631.pdf for details.

        """
        super().__init__(7, 'Piston')

        self.set_lim(
            [30., 0.005, 0.002, 1000,  90000, 290, 340],
            [60., 0.020, 0.010, 5000, 110000, 296, 360])

    def comp(self, X):
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


class DemoFuncRastrigin(DemoFunc):
    def __init__(self, d, A=10.):
        """Rastrigin function for demo and tests.

        See https://www.sfu.ca/~ssurjano/rastr.html for details.

        Args:
            d (int): number of dimensions.
            A (float): parameter of the function.

        """
        super().__init__(d, 'Rastrigin')

        self.par_A = A

        self.set_lim(-5.12, +5.12)
        self.set_min([0.]*self.d, 0.)

    def comp(self, X):
        y1 = self.par_A * self.d
        y2 = np.sum(X.T**2 - self.par_A * np.cos(2. * np.pi * X.T), axis=0)
        return y1 + y2


class DemoFuncRosenbrock(DemoFunc):
    def __init__(self, d):
        """Rosenbrock function for demo and tests.

        See https://www.sfu.ca/~ssurjano/rosen.html for details.

        Args:
            d (int): number of dimensions.

        """
        super().__init__(d, 'Rosenbrock')

        self.set_lim(-2.048, +2.048)
        self.set_min([1.]*self.d, 0.)

    def comp(self, X):
        return rosen(X.T)


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

    def comp(self, X):
        y1 = 418.9829 * self.d
        y2 = - np.sum(X.T * np.sin(np.sqrt(np.abs(X.T))), axis=0)
        return y1 + y2

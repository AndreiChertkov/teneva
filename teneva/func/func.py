"""Package teneva, module func.func: class that represents the function."""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from time import perf_counter as tpc


from .utils import _data_check
from .utils import _data_load
from .utils import _data_prep
from .utils import _data_save
from .utils import _noise
from teneva import als
from teneva import anova
from teneva import cache_to_data
from teneva import cheb_int
from teneva import cheb_get
from teneva import cheb_gets
from teneva import cross
from teneva import erank
from teneva import getter
from teneva import grid_flat
from teneva import ind_to_poi
from teneva import sample_lhs
from teneva import size
from teneva import tensor_rand
from teneva import truncate


class Func:
    def __init__(self, d, f_calc=None, f_comp=None, name='Demo'):
        """Multivariable function with approximation methods and helpers.

        Class represents the multivariable function (black box) with various
        methods (including TT-ALS, TT-ANOVA and TT-CROSS) for constructing its
        low-rank tensor approximation in the TT-format and utilities for
        building datasets (train, validation, test), check the accuracy, etc.

        Args:
            d (int): number of dimensions of the target function's input.
            f_calc (function): python function that returns the value (float)
                of the target function for the given multidimensional point
                (1D np.ndarray of the length "d"). If it is not set, then the
                corresponding function code is expected to be written in the
                class method "_calc". This function is used to build the
                train, validation and test datasets. If "f_calc" argument and
                "_calc" method are not set, then the "f_comp" / "_comp" method
                will be automatically used for the function evaluation for one
                point.
            f_comp (function): python function that returns the values (1D
                np.ndarray of the length "samples") of the target function for
                the given multidimensional points (2D np.ndarray of the shape
                [samples, d]). If it is not set, then the corresponding
                function code is expected to be written in the class method
                "_comp". This function is used to build the train, validation
                and test datasets. If "f_comp" argument and "_comp" method are
                not set, then the "f_calc" / "_calc" method will be
                automatically used for the function evaluation for the batch of
                points.
            name (str): optional display name for the function.

        Note:
            If "f_calc" / "_calc" / "f_comp" / "_comp" methods are not set,
            then train, validation and test datasets should be set manually by
            "set_trn_ind" / "set_trn_poi", "set_vld_ind" / "set_vld_poi" and
            "set_tst_ind" / "set_tst_poi" methods.

            Class instance method "get_f_poi" should be used for the target
            function evaluation (for one point or for the batch of points). To
            evaluate the target function for the grid multi-index (one or
            batch), use method "get_f_ind" (this function is very usefull for
            tensor approximation methods).

            To evaluate the TT-approximation of the target function for one
            point or for the batch of points, use the "get_poi" method of just
            call the class instance. To evaluate the TT-approximation of the
            target function on the grid multi-index, use "get_ind" method or
            "getitem" notation.

        """
        self.d = d
        self.f_calc = f_calc
        self.f_comp = f_comp
        self.name = name

        self.I_trn_ind = None
        self.I_trn_poi = None
        self.I_tst_ind = None
        self.I_tst_poi = None
        self.I_vld_ind = None
        self.I_vld_poi = None
        self.X_trn_ind = None
        self.X_trn_poi = None
        self.X_tst_ind = None
        self.X_tst_poi = None
        self.X_vld_ind = None
        self.X_vld_poi = None
        self.Y_trn_ind = None
        self.Y_trn_poi = None
        self.Y_tst_ind = None
        self.Y_tst_poi = None
        self.Y_vld_ind = None
        self.Y_vld_poi = None

        self.m_trn_ind = 0
        self.m_trn_poi = 0
        self.m_tst_ind = 0
        self.m_tst_poi = 0
        self.m_vld_ind = 0
        self.m_vld_poi = 0

        self.t_trn_ind_build = 0.
        self.t_trn_poi_build = 0.
        self.t_tst_ind_build = 0.
        self.t_tst_poi_build = 0.
        self.t_vld_ind_build = 0.
        self.t_vld_poi_build = 0.

        self.set_grid()
        self.set_lim()
        self.set_min()
        self.set_noise()
        self.clear()

        # TODO!
        self.with_int = None

    def __call__(self, X):
        return self.get_poi(X)

    def __getitem__(self, I):
        return self.get_ind(I)

    @property
    def with_cores(self):
        return hasattr(self, '_cores')

    @property
    def with_min(self):
        return self.y_min is not None

    @property
    def with_min_x(self):
        return self.x_min is not None

    def als(self, nswp=50, e=1.E-16, info={}, e_vld=None, r=None, e_adap=1.E-3, log=False):
        """Build approximation, using TT-ALS.

        See "teneva.core.als" for more details. Initial approximation should
        be prepared before the call of this function.

        """
        self.method = self.method + '-ALS' if self.method else 'ALS'
        if r is not None:
            self.method += '(A)'

        if not self.m_trn_ind:
            raise ValueError('Train data is not ready')
        if self.Y is None:
            raise ValueError('Initial approximation is not ready')

        t = tpc()
        Y = als(self.I_trn_ind, self.Y_trn_ind, self.Y, nswp, e, info,
            self.I_vld_ind, self.Y_vld_ind, e_vld, r, e_adap, log)
        self.t += tpc() - t

        if not self.m:
            self.m = self.m_trn_ind
        self.nswp = info['nswp']

        self.prep(Y)

    def anova(self, r=2, order=1, noise=1.E-10):
        """Build approximation, using TT-ANOVA.

        See "teneva.core.anova" for more details.

        """
        self.method = self.method + '-ANO' if self.method else 'ANO'

        if not self.m_trn_ind:
            raise ValueError('Train data is not ready')

        t = tpc()
        Y = anova(self.I_trn_ind, self.Y_trn_ind, r, order, noise)
        self.t += tpc() - t

        if not self.m:
            self.m = self.m_trn_ind

        self.prep(Y)

    def build_trn_ind(self, m):
        """Generate train dataset from random indices.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables I_trn_ind (grid indices), X_trn_ind
            (spatial points related to grid indices) and Y_trn_ind (function
            values) will be built. Indices I_trn_ind will be generated from the
            LHS distribution. If the noise parameters are not None, then the
            corresponding random noise will be added for the data.

        """
        t = tpc()
        a, b, n, f = self.a, self.b, self.n, sample_lhs
        self.m_trn_ind = int(m)
        self.I_trn_ind = f(n, self.m_trn_ind)
        self.X_trn_ind = ind_to_poi(self.I_trn_ind, a, b, n, self.kind)
        self.Y_trn_ind = self.get_f_poi_spec(self.X_trn_ind)
        self.t_trn_ind_build = tpc() - t

    def build_trn_poi(self, m):
        """Generate train dataset from spatial points.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables X_trn_poi (spatial points) and Y_trn_poi
            (function values) will be built. If the noise parameters are not
            None, then the corresponding random noise will be added for the
            data.

        """
        t = tpc()
        a, b, f = self.a, self.b, np.random.uniform
        self.m_trn_poi = int(m)
        self.I_trn_poi = None
        self.X_trn_poi = np.vstack([f(a, b) for _ in range(self.m_trn_poi)])
        self.Y_trn_poi = self.get_f_poi_spec(self.X_trn_poi)
        self.t_trn_poi_build = tpc() - t

    def build_tst_ind(self, m):
        """Generate test dataset from random tensor indices.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables I_tst_ind (grid indices), X_tst_ind
            (spatial points related to grid indices) and Y_tst_ind (function
            values) will be built. Indices I_trn_ind will be generated from
            uniform random choice of indices.

        """
        t = tpc()
        a, b, n, f = self.a, self.b, self.n, np.random.choice
        self.m_tst_ind = int(m)
        self.I_tst_ind = np.vstack([f(k, self.m_tst_ind) for k in n]).T
        self.X_tst_ind = ind_to_poi(self.I_tst_ind, a, b, n, self.kind)
        self.Y_tst_ind = self.get_f_poi(self.X_tst_ind)
        self.t_tst_ind_build = tpc() - t

    def build_tst_poi(self, m):
        """Generate test dataset from spatial points.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables X_tst_poi (spatial points) and Y_tst_poi
            (function values) will be built.

        """
        t = tpc()
        a, b, f = self.a, self.b, np.random.uniform
        self.m_tst_poi = int(m)
        self.I_tst_poi = None
        self.X_tst_poi = np.vstack([f(a, b) for _ in range(self.m_tst_poi)])
        self.Y_tst_poi = self.get_f_poi(self.X_tst_poi)
        self.t_tst_poi_build = tpc() - t

    def build_vld_ind(self, m):
        """Generate validation dataset from random tensor indices.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables I_vld_ind (grid indices), X_vld_ind
            (spatial points related to grid indices) and Y_vld_ind (function
            values) will be built. Indices I_vld_ind will be generated from
            uniform random choice of indices. If the noise parameters are not
            None, then the corresponding random noise will be added for the
            data.

        """
        t = tpc()
        a, b, n, f = self.a, self.b, self.n, np.random.choice
        self.m_vld_ind = int(m)
        self.I_vld_ind = np.vstack([f(k, self.m_vld_ind) for k in n]).T
        self.X_vld_ind = ind_to_poi(self.I_vld_ind, a, b, n, self.kind)
        self.Y_vld_ind = self.get_f_poi_spec(self.X_vld_ind)
        self.t_vld_ind_build = tpc() - t

    def build_vld_poi(self, m):
        """Generate validation dataset from spatial points.

        Args:
            m (int or float): number of points to generate.

        Note:
            Class instance variables X_vld_poi (spatial points) and Y_vld_poi
            (function values) will be built. If the noise parameters are not
            None, then the corresponding random noise will be added for the
            data.

        """
        t = tpc()
        a, b, f = self.a, self.b, np.random.uniform
        self.m_vld_poi = int(m)
        self.I_vld_poi = None
        self.X_vld_poi = np.vstack([f(a, b) for _ in range(self.m_vld_poi)])
        self.Y_vld_poi = self.get_f_poi_spec(self.X_vld_poi)
        self.t_vld_poi_build = tpc() - t

    def check(self):
        """Compute the error of the TT-approximation on all datasets."""
        if self.m_trn_ind:
            self.check_trn_ind()
        if self.m_trn_poi:
            self.check_trn_poi()
        if self.m_tst_ind:
            self.check_tst_ind()
        if self.m_tst_poi:
            self.check_tst_poi()
        if self.m_vld_ind:
            self.check_vld_ind()
        if self.m_vld_poi:
            self.check_vld_poi()

    def check_trn_ind(self):
        """Calculate the TT-approximation error for train indices."""
        self.e_trn_ind, self.t_trn_ind_check = _data_check(
            self.I_trn_ind, self.Y_trn_ind, self.get_ind)
        return self.e_trn_ind

    def check_trn_poi(self):
        """Calculate the TT-approximation error for train points."""
        if self.kind != 'cheb':
            raise ValueError('Can check only "cheb" spatial points')

        self.e_trn_poi, self.t_trn_poi_check = _data_check(
            self.X_trn_poi, self.Y_trn_poi, self.get_poi)
        return self.e_trn_poi

    def check_tst_ind(self):
        """Calculate the TT-approximation error for test indices."""
        self.e_tst_ind, self.t_tst_ind_check = _data_check(
            self.I_tst_ind, self.Y_tst_ind, self.get_ind)
        return self.e_tst_ind

    def check_tst_poi(self):
        """Calculate the TT-approximation error for test points."""
        if self.kind != 'cheb':
            raise ValueError('Can check only "cheb" spatial points')

        self.e_tst_poi, self.t_tst_poi_check = _data_check(
            self.X_tst_poi, self.Y_tst_poi, self.get_poi)
        return self.e_tst_poi

    def check_vld_ind(self):
        """Calculate the TT-approximation error for validation indices."""
        self.e_vld_ind, self.t_vld_ind_check = _data_check(
            self.I_vld_ind, self.Y_vld_ind, self.get_ind)
        return self.e_vld_ind

    def check_vld_poi(self):
        """Calculate the TT-approximation error for validation points."""
        if self.kind != 'cheb':
            raise ValueError('Can check only "cheb" spatial points')

        self.e_vld_poi, self.t_vld_poi_check = _data_check(
            self.X_vld_poi, self.Y_vld_poi, self.get_poi)
        return self.e_vld_poi

    def clear(self):
        """Remove all results of the previous approximations.

        Note:
            Note that this function does not remove the datasets. So, for
            example, TT-CROSS cache (train data) may be used in the following
            approximation.

        """
        self.Y = None             # TT-tensor of function values on the grid
        self.r = 0                # Effective TT-rank for Y

        self._get = None          # Getter for the TT-tensor
        self.A = None             # TT-tensor of interpolation coefficients

        self.t = 0.               # Total time of approximation
        self.e = -1.              # Expected error of approximation
        self.m = 0                # Total number of requests to target function
        self.m_cache = 0          # Total number of requests to cache
        self.method = ''          # Name of the approximation method
        self.nswp = 0             # Number of (CROSS or ALS) sweeps
        self.stop = 'm'           # Stop condition (CROSS) for approximation

        self.e_trn_ind = -1.      # Relative errors on the datasets
        self.e_trn_poi = -1.
        self.e_tst_ind = -1.
        self.e_tst_poi = -1.
        self.e_vld_ind = -1.
        self.e_vld_poi = -1.

        self.t_trn_ind_check = 0. # Time of error check for datasets
        self.t_trn_poi_check = 0.
        self.t_tst_ind_check = 0.
        self.t_tst_poi_check = 0.
        self.t_vld_ind_check = 0.
        self.t_vld_poi_check = 0.

    def cores(self):
        self.method = 'CORES'

        if not self.with_cores:
            msg = 'Can not build the TT-cores explicitly for this function'
            raise NotImplementedError(msg)

        if hasattr(self, 'dy') and abs(self.dy) > 1.E-100:
            msg = 'Option "dy" is not supported for this mode'
            raise NotImplementedError(msg)

        t = tpc()

        I = np.array([grid_flat(k) for k in self.n], dtype=int).T
        X = ind_to_poi(I, self.a, self.b, self.n, self.kind)
        Y = self._cores(X)

        self.t += tpc() - t

        self.prep(Y)

    def cross(self, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=2, tau0=1.05, k0=100, info={}, cache=True, eps=1.E-8, e_vld=None, r_max=None, log=False, func=None):
        """Build approximation, using TT-CROSS algorithm.

        See "teneva.core.cross" for more details. Initial approximation should
        be prepared before the call of this function.

        Note:
            Parameter "eps" is the accuracy of truncation of the TT-CROSS
            result. Other parameters are described in the base "cross" function.

        """
        self.method = self.method + '-CRO' if self.method else 'CRO'

        if self.Y is None:
            raise ValueError('Initial approximation is not ready')

        t = tpc()
        cache = {} if cache else None
        f = self.get_f_ind_spec
        Y = cross(f, self.Y,
            m, e, nswp, tau, dr_min, dr_max, tau0, k0, info, cache,
            I_vld=self.I_vld_ind, Y_vld=self.Y_vld_ind, e_vld=e_vld, log=log,
            func=func)
        if eps is not None:
            Y = truncate(Y, eps)
        if log:
            print()
        self.t += tpc() - t

        self.m += info['m']
        self.m_cache += info['m_cache']
        self.e = info['e']
        self.nswp = info['nswp']
        self.stop = info['stop']

        if cache is not None and len(cache) > 0:
            I_, Y_ = cache_to_data(cache)
            X_ = ind_to_poi(I_, self.a, self.b, self.n, self.kind)
            self.set_trn_ind(I_, X_, Y_)

        self.prep(Y)

    def get_f_ind(self, I):
        """Calculate the target function in the given grid multi-index.

        Args:
            I (list, np.ndarray): grid index in the form of array of the shape
                [d], where "d" is the dimension of the input or batch of grid
                indices in the form of array of the shape [samples, d], where
                "samples" is the number of samples.

        Returns:
            float or np.ndarray: the function value in the point related to
            the given grid index or the values of the function in the points
            related to given grid indices in the form of array of the shape
            [samples].

        Note:
            Grid parameters may be set using "set_grid" function.

        """
        X = ind_to_poi(I, self.a, self.b, self.n, self.kind)
        return self.get_f_poi(X)

    def get_f_ind_spec(self, I):
        """Compute the target function values for indices with special options.

        Args:
            I (list, np.ndarray): grid multi-indices in the form of array of
                the shape [samples, d], where "samples" is the number of
                samples and "d" is the dimension of the input.

        Returns:
            float or np.ndarray: the values of the function in the points
            related to given grid indices in the form of array of the shape
            [samples].

        Note:
            If noise parameters are set (see "set_noise" function), then noise
            will be added to the result of this function.

        """
        y = self.get_f_ind(I)
        y = _noise(y, self.noise_add, self.noise_mul)
        return y

    def get_f_poi(self, X):
        """Calculate the target function in the given point or points.

        Args:
            X (list, np.ndarray): point (function input) in the form of array
                of the shape [d], where "d" is the dimension of the input or
                batch of points (function inputs) in the form of array of the
                shape [samples, d], where "samples" is the number of samples.

        Returns:
            float or np.ndarray: the value of the target function in given
            point or the values in given points in the form of array of the
            shape [samples].

        """
        X = np.asanyarray(X, dtype=float)

        if len(X.shape) == 1:
            return self._calc(X)
        else:
            return self._comp(X)

    def get_f_poi_spec(self, X):
        """Compute the target function values for points with special options.

        Args:
            X (list, np.ndarray): spatial points in the form of array of
                the shape [samples, d], where "samples" is the number of
                samples and "d" is the dimension of the input.

        Returns:
            float or np.ndarray: the values of the function in the given points
            in the form of array of the shape [samples].

        Note:
            If noise parameters are set (see "set_noise" function), then noise
            will be added to the result of this function.

        """
        y = self.get_f_poi(X)
        y = _noise(y, self.noise_add, self.noise_mul)
        return y

    def get_ind(self, I):
        """Calculate the approximation in the given grid multi-index.

        Args:
            I (list, np.ndarray): grid index in the form of array of the shape
                [d], where "d" is the dimension of the input or batch of grid
                indices in the form of array of the shape [samples, d], where
                "samples" is the number of samples.

        Returns:
            float or np.ndarray: the value of the approximation for the target
            function in the point related to given grid index or the values in
            the points related to given grid indices in the form of array of
            the shape [samples].

        Note:
            Grid parameters may be set using "set_grid" function.

        """
        if self._get is None:
            raise ValueError('Approximation is not ready')

        I = np.asanyarray(I, dtype=int)

        if len(I.shape) == 1:
            return self._get(I)
        else:
            return np.array([self._get(i) for i in I])

    def get_poi(self, X):
        """Calculate the approximation of function in the given point or points.

        Args:
            X (list, np.ndarray): point (function input) in the form of array
                of the shape [d], where "d" is the dimension of the input or
                batch of points (function inputs) in the form of array of the
                shape [samples, d], where "samples" is the number of samples.

        Returns:
            float or np.ndarray: the value of the approximation for the target
            function in given point or the values in given points in the form
            of array of the shape [samples].

        """
        if self.Y is None:
            raise ValueError('Approximation is not ready')

        if self.A is None:
            raise ValueError('Interpolation coefficients are not ready')

        X = np.asanyarray(X, dtype=float)

        if len(X.shape) == 1:
            return cheb_get(X.reshape(1, -1), self.A, self.a, self.b)[0]
        else:
            return cheb_get(X, self.A, self.a, self.b)

    def info(self, text_post='', full=False):
        """Present (print) the info about approximation result.

        Note:
            Errors (if available) are presented in the following order: trn
            ind, trn poi, vld ind, vld poi, tst ind, tst poi.

        """
        if full:
            return self.info_full()

        text = ''

        name = self.name or ''
        meth = self.method or ''

        text += name + ' ' * max(0, 15-len(name)) + ' ['
        text += meth + ' ' * max(0, 12-len(meth)) + ' ] > '

        text += f'error: '
        if self.e_trn_ind >= 0:
            text += f'{self.e_trn_ind:-7.1e} / '
        if self.e_trn_poi >= 0:
            text += f'{self.e_trn_poi:-7.1e} / '
        if self.e_vld_ind >= 0:
            text += f'{self.e_vld_ind:-7.1e} / '
        if self.e_vld_poi >= 0:
            text += f'{self.e_vld_poi:-7.1e} / '
        if self.e_tst_ind >= 0:
            text += f'{self.e_tst_ind:-7.1e} / '
        if self.e_tst_poi >= 0:
            text += f'{self.e_tst_poi:-7.1e} / '
        text = text[:-2] + '| '

        text += f'rank: {self.r:-4.1f} | '

        text += f'time: {self.t:-7.3f}'

        if text_post:
            text += ' | ' + text_post

        print(text)

    def info_full(self):
        """Present (print) the full info about approximation result."""
        text = ''
        text += f'=' * 50 + '\n'

        text += f'------------------- | {self.name} function\n'
        if self.method:
            text += f'Method              : {self.method:s}\n'

        text += '\n'

        if self.m > 0:
            text += f'Evals function      : {self.m:-7.1e}\n'
        if self.m_cache > 0:
            text += f'Evals cache         : {self.m_cache:-7.1e}\n'
        if self.r:
            text += f'TT-rank             : {self.r:-7.1f}\n'
        if self.Y is not None:
            text += f'Number of params    : {size(self.Y):-7.1e}\n'

        text += '\n'

        if self.m_trn_ind > 0:
            text += f'Samples trn ind     : {self.m_trn_ind:-7.1e}\n'
        if self.m_trn_poi > 0:
            text += f'Samples trn poi     : {self.m_trn_poi:-7.1e}\n'
        if self.m_vld_ind > 0:
            text += f'Samples vld ind     : {self.m_vld_ind:-7.1e}\n'
        if self.m_vld_poi > 0:
            text += f'Samples vld poi     : {self.m_vld_poi:-7.1e}\n'
        if self.m_tst_ind > 0:
            text += f'Samples tst ind     : {self.m_tst_ind:-7.1e}\n'
        if self.m_tst_poi > 0:
            text += f'Samples tst poi     : {self.m_tst_poi:-7.1e}\n'

        text += '\n'

        if self.e_trn_ind >= 0:
            text += f'Error trn ind       : {self.e_trn_ind:-7.1e}\n'
        if self.e_trn_poi >= 0:
            text += f'Error trn poi       : {self.e_trn_poi:-7.1e}\n'
        if self.e_vld_ind >= 0:
            text += f'Error vld ind       : {self.e_vld_ind:-7.1e}\n'
        if self.e_vld_poi >= 0:
            text += f'Error vld poi       : {self.e_vld_poi:-7.1e}\n'
        if self.e_tst_ind >= 0:
            text += f'Error tst ind       : {self.e_tst_ind:-7.1e}\n'
        if self.e_tst_poi >= 0:
            text += f'Error tst poi       : {self.e_tst_poi:-7.1e}\n'

        text += '\n'

        if self.t > 0:
            text += f'Time approximation  : {self.t:-7.3f}\n'

        if self.t_trn_ind_build > 0:
            text += f'Time trn build ind  : {self.t_trn_ind_build:-7.3f}\n'
        if self.t_trn_ind_check > 0:
            text += f'Time trn check ind  : {self.t_trn_ind_check:-7.3f}\n'
        if self.t_trn_poi_build > 0:
            text += f'Time trn build poi  : {self.t_trn_poi_build:-7.3f}\n'
        if self.t_trn_poi_check > 0:
            text += f'Time trn check poi  : {self.t_trn_poi_check:-7.3f}\n'

        if self.t_vld_ind_build > 0:
            text += f'Time vld build ind  : {self.t_vld_ind_build:-7.3f}\n'
        if self.t_vld_ind_check > 0:
            text += f'Time vld check ind  : {self.t_vld_ind_check:-7.3f}\n'
        if self.t_vld_poi_build > 0:
            text += f'Time vld build poi  : {self.t_vld_poi_build:-7.3f}\n'
        if self.t_vld_poi_check > 0:
            text += f'Time vld check poi  : {self.t_vld_poi_check:-7.3f}\n'

        if self.t_tst_ind_build > 0:
            text += f'Time tst build ind  : {self.t_tst_ind_build:-7.3f}\n'
        if self.t_tst_ind_check > 0:
            text += f'Time tst check ind  : {self.t_tst_ind_check:-7.3f}\n'
        if self.t_tst_poi_build > 0:
            text += f'Time tst build poi  : {self.t_tst_poi_build:-7.3f}\n'
        if self.t_tst_poi_check > 0:
            text += f'Time tst check poi  : {self.t_tst_poi_check:-7.3f}\n'

        text += '\n'

        if self.nswp > 0:
            text += f'Sweeps              : {self.nswp:-7.0f}\n'

        text += f'=' * 50

        print(text)

    def load_trn_ind(self, fpath, m=None):
        """Load the train dataset indices of size m from the npz-file.

        Note:
            If the noise parameters are not None, then the corresponding random
            noise will be added for the loaded data.

        """
        self.set_trn_ind(*_data_load(fpath, m, self.n, self.kind))
        self.Y_trn_ind = _noise(self.Y_trn_ind, self.noise_add, self.noise_mul)

    def load_trn_poi(self, fpath, m=None):
        """Load the train dataset points of size m from the npz-file.

        Note:
            If the noise parameters are not None, then the corresponding random
            noise will be added for the loaded data.

        """
        self.set_trn_poi(*_data_load(fpath, m, self.n, self.kind))
        self.Y_trn_poi = _noise(self.Y_trn_poi, self.noise_add, self.noise_mul)

    def load_tst_ind(self, fpath, m=None):
        """Load the test dataset indices of size m from the npz-file."""
        self.set_tst_ind(*_data_load(fpath, m, self.n, self.kind))

    def load_tst_poi(self, fpath, m=None):
        """Load the test dataset points of size m from the npz-file."""
        self.set_tst_poi(*_data_load(fpath, m, self.n, self.kind))

    def load_vld_ind(self, fpath, m=None):
        """Load the validation dataset indices of size m from the npz-file.

        Note:
            If the noise parameters are not None, then the corresponding random
            noise will be added for the loaded data.

        """
        self.set_vld_ind(*_data_load(fpath, m, self.n, self.kind))
        self.Y_vld_ind = _noise(self.Y_vld_ind, self.noise_add, self.noise_mul)

    def load_vld_poi(self, fpath, m=None):
        """Load the validation dataset points of size m from the npz-file.

        Note:
            If the noise parameters are not None, then the corresponding random
            noise will be added for the loaded data.

        """
        self.set_vld_poi(*_data_load(fpath, m, self.n, self.kind))
        self.Y_vld_poi = _noise(self.Y_vld_poi, self.noise_add, self.noise_mul)

    def plot(self, k=1000):
        """Plot the target function for the 2D case.

        Args:
            k (int): number of points for each dimension.

        """
        if self.d != 2:
            raise ValueError('Plot is supported only for 2D case')

        X1 = np.linspace(self.a[0], self.b[0], k)
        X2 = np.linspace(self.a[1], self.b[1], k)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])

        Y = self.get_f_poi(X)
        Y = Y.reshape(X1.shape)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.set_title(self.name + ' function')
        surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.3, aspect=10)
        plt.show()

    def prep(self, Y, with_int=True):
        """Prepare getter and interpolation coefficients for approximation.

        Args:
            Y (list): TT-tensor with function values on the grid.
            with_int (bool): if flag is set, then the getter for tensor and
                Chebyshev interpolation coefficients (in the case if Chebyshev
                grid is used) will be prepared.

        """
        if Y is None:
            raise ValueError('Approximation is not ready')

        self.Y = Y
        self.r = erank(self.Y)

        self._get = getter(self.Y)

        if with_int and (self.with_int is None or self.with_int == True):
            self.A = cheb_int(self.Y) if self.kind == 'cheb' else None
        else:
            self.A = None

    def rand(self, r=2):
        """Build random approximation in the TT-format.

        Note:
            This function should be called before "als" and "cross" methods
            to prepare the initial approximation for TT-ALS and TT-CROSS
            methods, if other approaches (for example, TT-ANOVA from function
            "anova") are not used to construct the initial approximation.

            Note that interpolation parameters ("_get", "A") will not be
            prepared while call of this function in contrast to the case of
            using other approximation methods ("als", "anova", "cross", etc.).

        """
        self.method = ''

        t = tpc()
        Y = tensor_rand(self.n, r)
        self.t += tpc() - t

        self.prep(Y, with_int=False)

    def save_trn_ind(self, fpath):
        """Save train dataset indices to the npz-file."""
        _data_save(fpath, self.I_trn_ind, self.X_trn_ind, self.Y_trn_ind,
            self.t_trn_ind_build, self.n, self.kind)

    def save_trn_poi(self, fpath):
        """Save train dataset points to the npz-file."""
        _data_save(fpath, self.I_trn_poi, self.X_trn_poi, self.Y_trn_poi,
            self.t_trn_poi_build, self.n, self.kind)

    def save_tst_ind(self, fpath):
        """Save test dataset indices to the npz-file."""
        _data_save(fpath, self.I_tst_ind, self.X_tst_ind, self.Y_tst_ind,
            self.t_tst_ind_build, self.n, self.kind)

    def save_tst_poi(self, fpath):
        """Save test dataset points to the npz-file."""
        _data_save(fpath, self.I_tst_poi, self.X_tst_poi, self.Y_tst_poi,
            self.t_tst_poi_build, self.n, self.kind)

    def save_vld_ind(self, fpath):
        """Save validation dataset indices to the npz-file."""
        _data_save(fpath, self.I_vld_ind, self.X_vld_ind, self.Y_vld_ind,
            self.t_vld_ind_build, self.n, self.kind)

    def save_vld_poi(self, fpath):
        """Save validation dataset points to the npz-file."""
        _data_save(fpath, self.I_vld_poi, self.X_vld_poi, self.Y_vld_poi,
            self.t_vld_poi_build, self.n, self.kind)

    def set_grid(self, n=10, kind='cheb'):
        """Set grid options for function discretization.

        Args:
            n (int, float, list, np.ndarray): grid size for each dimension
                (list or np.ndarray of length "d"). It may be also int/float,
                then the size for each dimension will be the same.
            kind (str): the grid kind, it may be "uni" (uniform grid) and "cheb"
                (Chebyshev grid).

        Note:
            If a grid type other than "cheb" is specified, then it will be
            impossible to calculate the value of the approximation result at an
            arbitrary point (method "get_poi") but the calculation in an
            arbitrary tensor multi-index (method "get_ind") will be still
            available.

        """
        if isinstance(n, (int, float)):
            n = np.ones(self.d, dtype=int) * int(n)
        self.n = np.asanyarray(n, dtype=int)
        if len(self.n) != self.d:
            raise ValueError('Invalid length of n')

        if not kind in ['uni', 'cheb']:
            raise ValueError(f'Unknown grid type "{kind}"')
        self.kind = kind

    def set_lim(self, a=-1., b=+1.):
        """Set bounds for the spatial rectangular region.

        Args:
            a (float, list, np.ndarray): lower bounds for each dimension (list
                or np.ndarray of length "d"). It may be also float, then the
                lower bounds for each dimension will be the same.
            b (float, list, np.ndarray): upper bounds for each dimension (list
                or np.ndarray of length "d"). It may be also float, then the
                upper bounds for each dimension will be the same.

        """
        if isinstance(a, (int, float)):
            a = [a] * self.d
        self.a = np.asanyarray(a, dtype=float)

        if isinstance(b, (int, float)):
            b = [b] * self.d
        self.b = np.asanyarray(b, dtype=float)

    def set_min(self, x_min=None, y_min=None):
        """Set the exact global minimum of the function.

        Args:
            x_min (list, np.ndarray): argument of the function that provides its
                minimum value (list or np.ndarray of the length "d").
            y_min (float): minimum value of the function (y_min = f(x_min)).

        """
        if x_min is not None:
            x_min = np.asanyarray(x_min, dtype=float)
        self.x_min = x_min

        self.y_min = y_min

    def set_noise(self, noise_add=None, noise_mul=None):
        """Set the noise parameters.

        If the noise parameters are not None, then the corresponding random
        noise will be added for train and validation data (see functions
        "get_f_ind_spec" and "get_f_poi_spec"; and "build_trn_ind",
        "build_trn_poi", "build_vld_ind", "build_vld_poi"). The TT-CROSS
        method will also use noise data (see function "cross").

        Args:
            noise_add (float): additive noise parameter.
            noise_mul (float): multiplicative noise parameter.

        """
        self.noise_add = noise_add
        self.noise_mul = noise_mul

    def set_trn_ind(self, I=None, X=None, Y=None, t=0.):
        """Set train data indices (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_trn_ind, self.X_trn_ind, self.Y_trn_ind, self.m_trn_ind = res
        self.t_trn_ind_build = t

    def set_trn_poi(self, I=None, X=None, Y=None, t=0.):
        """Set train data points (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_trn_poi, self.X_trn_poi, self.Y_trn_poi, self.m_trn_poi = res
        self.t_trn_poi_build = t

    def set_tst_ind(self, I=None, X=None, Y=None, t=0.):
        """Set test data indices (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_tst_ind, self.X_tst_ind, self.Y_tst_ind, self.m_tst_ind = res
        self.t_tst_ind_build = t

    def set_tst_poi(self, I=None, X=None, Y=None, t=0.):
        """Set test data points (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_tst_poi, self.X_tst_poi, self.Y_tst_poi, self.m_tst_poi = res
        self.t_tst_poi_build = t

    def set_vld_ind(self, I=None, X=None, Y=None, t=0.):
        """Set validation data indices (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_vld_ind, self.X_vld_ind, self.Y_vld_ind, self.m_vld_ind = res
        self.t_vld_ind_build = t

    def set_vld_poi(self, I=None, X=None, Y=None, t=0.):
        """Set validation data points (indices, points, values and time)."""
        res = _data_prep(I, X, Y)
        self.I_vld_poi, self.X_vld_poi, self.Y_vld_poi, self.m_vld_poi = res
        self.t_vld_poi_build = t

    def _calc(self, x):
        if self.f_calc is None:
            return self._comp(x.reshape(1, -1))[0]
        else:
            return self.f_calc(x)

    def _comp(self, X):
        if self.f_comp is None:
            return np.array([self._calc(x) for x in X])
        else:
            return self.f_comp(X)

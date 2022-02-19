"""Package teneva, module func.func: class that represents the function."""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


import teneva


class Func:
    def __init__(self, d, f_calc=None, f_comp=None, name='Func'):
        """Multivariable function with approximation methods.

        Multivariable function with methods for constructing a low-rank tensor
        approximation and auxiliary utilities for constructing data sets,
        interpolation, etc.

        Args:
            d (int): number of dimensions.
            f_calc (function): python function that returns the value (float)
                of the target function for the given multidimensional point
                (1D np.ndarray of the length "d"). If is not set, then the
                corresponding function code is expected to be written in the
                class method "_calc".
            f_comp (function): python function that returns the values (1D
                np.ndarray of the length "samples") of the target function for
                the given multidimensional points (2D np.ndarray of the shape "samples" x "d"). If is not set, then the corresponding
                function code is expected to be written in the class method
                "_comp".
            name (str): display name for the function.

        """
        self.d = d
        self.f_calc = f_calc
        self.f_comp = f_comp
        self.name = name

        self.m_trn = 0
        self.m_tst = 0

        self.set_grid()
        self.set_lim(-1., +1.)
        self.set_min(None, None)
        self.set_perm()

    def build_trn(self, m, is_ind=False):
        """Generate train dataset from random indices or spatial points.

        Args:
            m (int or float): number of points to generate.
            is_ind (bool): if True, then grid indices will be generated
                (I_trn), otherwise the spatial points (X_trn) will be generated.

        Note:
            In case is_ind = False, class instance variables X_trn (spatial
                points) and Y_trn (function values) will be saved.

            In case is_ind = True, class instance variables I_trn (grid
                indices), X_trn (spatial points related to grid indices) and
                Y_trn (function values) will be saved. Indices I_trn will be
                generated from LHS distribution.

        """
        self.m_trn = int(m)
        a, b, n = self.a, self.b, self.n

        if is_ind:
            func = teneva.sample_lhs
            self.I_trn = func(self.n, self.m_trn)
            I = self.I_trn
            if self.q is not None:
                I = self.qtt_parse_many(I)
            self.X_trn = teneva.ind2poi(I, a, b, n, self.kind)
            self.Y_trn = self.comp(self.X_trn)
        else:
            func = np.random.uniform
            self.I_trn = None
            self.X_trn = np.vstack([func(a, b) for _ in range(self.m_trn)])
            self.Y_trn = self.comp(self.X_trn)

    def build_tst(self, m, is_ind=False):
        """Generate test dataset from random indices or spatial points.

        Args:
            m (int or float): number of points to generate.
            is_ind (bool): if True, then grid indices will be generated
                (I_trn), otherwise the spatial points (X_trn) will be generated.

        Note:
            In case is_ind = False, class instance variables X_tst (spatial
                points) and Y_tst (function values) will be saved.

            In case is_ind = True, class instance variables I_tst (grid
                indices), X_tst (spatial points related to grid indices) and
                Y_tst (function values) will be saved. Indices I_trn will be
                generated from uniform random choice of indices.

        """
        self.m_tst = int(m)
        a, b, n = self.a, self.b, self.n

        if is_ind:
            func = np.random.choice
            self.I_tst = np.vstack([func(k, self.m_tst) for k in self.n]).T
            I = self.I_tst
            if self.q is not None:
                I = self.qtt_parse_many(I)
            self.X_tst = teneva.ind2poi(I, a, b, n, self.kind)
            self.Y_tst = self.comp(self.X_tst)
        else:
            func = np.random.uniform
            self.I_tst = None
            self.X_tst = np.vstack([func(a, b) for _ in range(self.m_tst)])
            self.Y_tst = self.comp(self.X_tst)

    def calc(self, x):
        """Calculate the function in the given point.

        Args:
            x (np.ndarray): point (function input) in the form of array of the
                shape [d], where "d" is the dimension of the input.

        Returns:
            float: the value of the function in given point.

        """
        z = x if self.perm is None else x[self.perm]
        return self._calc(z)

    def calc_grid(self, i):
        """Calculate the function in the given grid index.

        Args:
            i (np.ndarray): grid index in the form of array of the shape
                [d], where "d" is the dimension of the input.

        Returns:
            float: the function value in the point related to given grid index.

        Note:
            Grid params may be set using "set_grid" function.

        """
        x = teneva.ind2poi(i, self.a, self.b, self.n, self.kind)
        return self.calc(x)

    def check_trn(self, Y):
        """Compute the error of TT-tensor on the dataset.

        Args:
            Y (list): TT-tensor.

        Returns:
            float: the relative difference between two lists of values, i.e.,
                norm(Y - Y_trn) / norm(Y_trn).

        """
        if not self.m_trn:
            # raise ValueError('Train points are not prepared')
            return -1

        if self.I_trn is None:
            # Train points are spatial:
            if self.kind != 'cheb':
                raise ValueError('Can check only "cheb" spatial train points')
            A = teneva.cheb_int(Y)
            Z = teneva.cheb_get(self.X_trn, A, self.a, self.b)
        else:
            # Train points are indices:
            get = teneva.getter(Y)
            Z = np.array([get(i) for i in self.I_trn])

        return np.linalg.norm(Z - self.Y_trn) / np.linalg.norm(self.Y_trn)

    def check_tst(self, Y):
        """Compute the error of TT-tensor on the test dataset.

        Args:
            Y (list): TT-tensor.

        Returns:
            float: the relative difference between two lists of values, i.e.,
                norm(Y - Y_tst) / norm(Y_tst).

        """
        if not self.m_tst:
            # raise ValueError('Test points are not prepared')
            return -1

        if self.I_tst is None:
            # Test points are spatial:
            if self.kind != 'cheb':
                raise ValueError('Can check only "cheb" spatial test points')
            A = teneva.cheb_int(Y)
            Z = teneva.cheb_get(self.X_tst, A, self.a, self.b)
        else:
            # Test points are indices:
            get = teneva.getter(Y)
            Z = np.array([get(i) for i in self.I_tst])

        return np.linalg.norm(Z - self.Y_tst) / np.linalg.norm(self.Y_tst)

    def comp(self, X):
        """Compute the function in the given points.

        Args:
            X (np.ndarray): points (function inputs) in the form of array of the
                shape [samples, d], where "samples" is the number of samples and
                "d" is the dimension of the input.

        Returns:
            np.ndarray: the values of the function in given points in the form
                of array of the shape [samples].

        """
        Z = X if self.perm is None else X[:, self.perm]
        return self._comp(X)

    def comp_grid(self, I):
        """Compute the function in the given grid indices.

        Args:
            I (np.ndarray): grid indices in the form of array of the shape
                [samples, d], where "samples" is the number of samples and "d"
                is the dimension of the input.

        Returns:
            np.ndarray: the values of the function in the points related to
                given grid indices in the form of array of the shape [samples].

        Note:
            Grid params may be set using "set_grid" function.

        """
        X = teneva.ind2poi(I, self.a, self.b, self.n, self.kind)
        return self.comp(X)

    def plot(self, k=1000):
        """Plot the function for the 2D case.

        Args:
            k (int): number of points for each dimension.

        Todo:
            Add also 2D suplot.

        """
        if self.d != 2:
            raise ValueError('Plot is supported only for 2D case')

        X1 = np.linspace(self.a[0], self.b[0], k)
        X2 = np.linspace(self.a[1], self.b[1], k)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])

        Y = self.comp(X)
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

    def qtt_parse_many(self, I_qtt):
        """Transforms tensor indices from QTT (long) to base (short) format."""
        samples = I_qtt.shape[0]
        n_qtt = [self.n[0]]*self.q
        I = np.zeros((samples, self.d))
        for i in range(self.d):
            J_curr = I_qtt[:, self.q*i:self.q*(i+1)].T
            I[:, i] = np.ravel_multi_index(J_curr, n_qtt, order='F')
        return I

    def set_grid(self, n=10, p=None, q=None, kind='uni'):
        """Set grid options for function discretization.

        Args:
            n (list): grid size for each dimension (list or np.ndarray of length
                "d"). It may be also float, then the size for each dimension
                will be the same.
            p (int): the grid size factor (if is given, then there will be n=p^q
                points for each dimension). This parameter can be specified
                instead of "n". If this parameter is specified, then the
                parameter "q" must also be specified, and in this case the
                QTT-based approach will be used.
            q (int): the grid size factor (if is given, then there will be n=p^q
                points for each dimension). This parameter can be specified
                instead of "n". If this parameter is specified, then the
                parameter "p" must also be specified, and in this case the
                QTT-based approach will be used.
            kind (str): the grid kind, it may be "uni" (uniform grid) and "cheb"
                (Chebyshev grid).

        """
        if n is None:
            if p is None or q is None:
                raise ValueError('If n is not set, then p and q should be set')
            self.p = int(p)
            self.q = int(q)
            self.n = np.ones(self.d * self.q, dtype=int) * self.p
            self.n_func = np.ones(self.d, dtype=int) * self.p**self.q
        else:
            if p is not None or q is not None:
                raise ValueError('If n is set, then p and q should be None')
            self.p = None
            self.q = None
            if isinstance(n, (int, float)):
                self.n = np.ones(self.d, dtype=int) * int(n)
            self.n = np.asanyarray(self.n, dtype=int)
            self.n_func = self.n.copy()
        if self.n_func.size != self.d:
            raise ValueError('Invalid length of n')

        if not kind in ['uni', 'cheb']:
            raise ValueError(f'Unknown grid type "{kind}"')
        self.kind = kind

    def set_lim(self, a, b):
        """Set spatial grid bounds.

        Args:
            a (list): grid lower bounds for each dimension (list or np.ndarray
                of length "d"). It may be also float, then the lower bounds for each dimension will be the same.
            b (list): grid upper bounds for each dimension (list or np.ndarray
                of length "d"). It may be also float, then the upper bounds for
                each dimension will be the same.

        """
        if isinstance(a, (int, float)):
            a = [a] * self.d
        self.a = np.asanyarray(a, dtype=float)

        if isinstance(b, (int, float)):
            b = [b] * self.d
        self.b = np.asanyarray(b, dtype=float)

    def set_min(self, x_min, y_min):
        """Set the exact global minimum of the function.

        Args:
            x_min (np.ndarray): argument of the function that provides its
                minimum value (list or np.ndarray of length "d").
            y_min (float): minimum value of the function.

        """
        self.x_min = x_min
        self.y_min = y_min

        if self.x_min is not None:
            self.x_min = np.asanyarray(self.x_min, dtype=float)

    def set_perm(self, perm=None):
        """Set the permutation of the function arguments.

        Args:
            perm (list, np.ndarray): new ordering of indices.

        Todo:
            Add permutation support to QTT representation.

        """
        if perm is None:
            self.perm = None
            return
        else:
            self.perm = np.asanyarray(perm, dtype=int)

        self.a = self.a[self.perm]
        self.b = self.b[self.perm]
        self.n = self.n[self.perm]

        if self.m_trn > 0 and self.I_trn is not None:
            self.I_trn = self.I_trn[:, self.perm]
        if self.m_trn > 0 and self.X_trn is not None:
            self.X_trn = self.X_trn[:, self.perm]

        if self.m_tst > 0 and self.I_tst is not None:
            self.I_tst = self.I_tst[:, self.perm]
        if self.m_tst > 0 and self.X_tst is not None:
            self.X_tst = self.X_tst[:, self.perm]

    def _calc(self, x):
        raise NotImplementedError()

    def _comp(self, X):
        raise NotImplementedError()

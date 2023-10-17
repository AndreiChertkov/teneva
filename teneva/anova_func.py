"""Package teneva, module anova_func: Functional ANOVA in the TT-format.

This module contains the function "anova_func" which computes the
TT-approximation of Chebyshev interpolation coefficients' tensor, using given
train dataset with random points.

"""
import numpy as np
import scipy as sp
import teneva


class ANOVA_func:
    def __init__(self, X_trn, y_trn, n, a=-1., b=+1., lamb=1.E-7):
        self.X_trn = teneva.poi_scale(X_trn, a, b, kind='cheb')
        self.y_trn = np.asarray(y_trn, dtype=float)
        self.lamb = lamb
        self._cfs = None
        self.d = self.X_trn.shape[1]
        self.n = n

    @property
    def coeffs(self):
        if self._cfs is not None:
            return self._cfs

        self._cfs = cfs = []

        y0 = np.mean(self.y_trn)
        cfs.append(y0)

        y = self.y_trn - y0
        for xd in self.X_trn.T:
            A = teneva.func_basis(xd, m=self.n, kind='cheb').T
            AtA = A.T @ A
            Aty = A.T @ y

            cur_cf = sp.linalg.lstsq(AtA + self.lamb * np.identity(A.shape[1]),
                Aty, overwrite_a=True, overwrite_b=True,
                lapack_driver='gelsy')[0]
            cfs.append(cur_cf[1:])

            cfs[0] += cur_cf[0]

        return self._cfs

    def cores(self, e=1.E-8):
        cfs = self.coeffs
        idx = np.zeros(self.d, dtype=int)
        A = teneva.delta([self.n]*self.d, idx, cfs[0])

        for i, cf in enumerate(cfs[1:]):
            idx[:] = 0
            for pi, p in enumerate(cf):
                idx[i] = pi + 1
                A = teneva.add(A, teneva.delta([self.n]*self.d, idx, p))

        return A if e is None else teneva.truncate(A, e)


def anova_func(X_trn, y_trn, n, a=-1., b=+1., lamb=1.E-7, e=1.E-8):
    """Build functional TT-tensor by TT-ANOVA from the given random points.

    Args:
        X_trn (np.ndarray): train points in the form of array of the shape
            [samples, d].
        y_trn (np.ndarray): values of the tensor for train points X_trn in the
            form of array of the shape [samples].
        n (int): mode size of the tensor (i.e, the maximum power of the
            interpolation coefficient plus 1).
        a (float, list, np.ndarray): grid lower bounds for each dimension (list
            or np.ndarray of length d or float).
        b (float, list, np.ndarray): grid upper bounds for each dimension (list
            or np.ndarray of length d or float).
        lamb (float): regularization parameter.
        e (float): optional truncation accuracy for resulting TT-tensor.

    Returns:
        list: TT-tensor, which represents the interpolation coefficients in the
        TT-format.

    Note:
        A class "ANOVA_func" that represents a wider set of methods for working
        with this decomposition is also available; see "anova_func.py".

    """
    return ANOVA_func(X_trn, y_trn, n, a, b, lamb).cores(e)

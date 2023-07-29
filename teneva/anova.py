"""Package teneva, module anova: ANOVA decomposition in the TT-format.

This module contains the function "anova" which computes the TT-approximation
for the tensor, using given random samples.

"""
import numpy as np
import pickle
import teneva


class ANOVA:
    # TODO: add docstring and add into demo
    def __init__(self, I_trn=None, y_trn=None, order=1, seed=None, fpath=None):
        self.rand = teneva._rand(seed)

        if not order in [1, 2]:
            raise ValueError('Invalid value for ANOVA order (should be 1 or 2')
        self.order = order

        if fpath is None:
            if I_trn is None or y_trn is None:
                raise ValueError('"(I_trn, y_trn)" or "fpath" should be set')
            self.build(I_trn, y_trn)
        else:
            if I_trn is not None or y_trn is not None:
                raise ValueError('Can`t use both "(I_trn, y_trn)" and "fpath"')
            self.load(fpath)

    def __call__(self, I):
        I = np.asanyarray(I, dtype=self.dtype)

        if len(I.shape) == 1:
            return self.calc(I)
        elif len(I.shape) == 2:
            return np.array([self.calc(i) for i in I])
        else:
            raise ValueError('Invalid input multi-indices for ANOVA')

    def __getitem__(self, I):
        return self(I)

    @property
    def f1_arr(self):
        try:
            return self._f1_arr
        except AttributeError:
            pass

        f1_arr = self._f1_arr = [np.array([f1_curr[x] for x in dm])
            for dm, f1_curr in zip(self.domain, self.f1)]
        return f1_arr

    @property
    def f2_arr(self):
        try:
            return self._f2_arr
        except AttributeError:
            pass

        f2_arr = self._f2_arr = [
            np.array([f2_curr[x1, x2] for x1 in dm1 for x2 in dm2 ])
                for (dm1, dm2), f2_curr in  zip([(dm1, dm2)
                    for k1, dm1 in enumerate(self.domain[:-1])
                        for dm2 in self.domain[k1+1:]], self.f2)]
        return f2_arr

    def build(self, I_trn, y_trn):
        I_trn = np.asanyarray(I_trn)
        self.dtype = I_trn.dtype

        y_trn = np.asanyarray(y_trn, dtype=float)

        self.y_max = np.max(y_trn)
        self.y_min = np.min(y_trn)

        self.d = I_trn.shape[1]

        self.domain = []
        self.shapes = np.zeros(self.d, dtype=int)
        for k in range(self.d):
            points = np.unique(I_trn[:, k])
            self.domain.append(points)
            self.shapes[k] = len(points)

        self.build_0(I_trn, y_trn)

        if self.order >= 1:
            self.build_1(I_trn, y_trn)
        else:
            self.f1 = []

        if self.order >= 2:
            self.build_2(I_trn, y_trn)
        else:
            self.f2 = []

    def build_0(self, I_trn, y_trn):
        self.f0 = np.mean(y_trn)

    def build_1(self, I_trn, y_trn):
        self.f1 = []
        for k, dm in enumerate(self.domain):
            f1_curr = {}
            for x in dm:
                idx = I_trn[:, k] == x
                value = np.mean(y_trn[idx]) - self.f0
                f1_curr[x] = value
            self.f1.append(f1_curr)

    def build_2(self, I_trn, y_trn):
        cache = dict()
        self.f2 = []
        for k1, dm1 in enumerate(self.domain[:-1]):
            for k2, dm2 in enumerate(self.domain[k1+1:], start=k1+1):
                f2_curr = {}
                for x1 in dm1:
                    for x2 in dm2:
                        try:
                            idx1 = cache[k1, x1]
                        except KeyError:
                            idx1 = I_trn[:, k1] == x1
                            cache[k1, x1] = idx1

                        try:
                            idx2 = cache[k2, x2]
                        except KeyError:
                            idx2 = I_trn[:, k2] == x2
                            cache[k2, x2] = idx2

                        idx = idx1 & idx2

                        if idx.sum() == 0:
                            value = 0.
                        else:
                            value = np.mean(y_trn[idx]) - self.f0
                            value = value - self.f1[k1][x1] - self.f1[k2][x2]
                        f2_curr[x1, x2] = value
                self.f2.append(f2_curr)

    def calc(self, i):
        res = self.calc_0()
        if self.order >= 1:
            res += self.calc_1(i)
        if self.order >= 2:
            res += self.calc_2(i)
        return res

    def calc_0(self):
        return self.f0

    def calc_1(self, x):
        res = 0.
        for num, x1 in enumerate(x):
            res += self.f1[num][x1]
        return res

    def calc_2(self, x):
        # We allow x to be of smaller length
        res = 0.
        for i1, x1 in enumerate(x[:-1]):
            for i2, x2 in enumerate(x[i1+1:], start=i1+1):
                num = self.pair_num_to_num(i1, i2)
                res += self.f2[num][x1, x2]
        return res

    def cores(self, r=2, noise=1.E-10, only_near=False, rel_noise=None):
        if self.order < 1:
            raise ValueError('TT-cores may be constructed only if order >= 1')

        if rel_noise is not None:
            noise = rel_noise * max(abs(self.y_max), abs(self.y_min))

        cores = self.cores_1(r, noise)

        if self.order >= 2:
            cores2_many = self.cores_2(r, only_near)
            cores = teneva.add_many([cores] + cores2_many, r=r)

        return cores

    def cores_1(self, r=2, noise=1.E-10):
        cores = []

        core = noise * self.rand.normal(size=(1, self.shapes[0], r))
        core[0, :, 0] = 1.
        core[0, :, 1] = self.f1_arr[0]
        cores.append(core)

        for i in range(1, self.d-1):
            core = noise * self.rand.normal(size=(r, self.shapes[i], r))
            core[0, :, 0] = 1.
            core[1, :, 1] = 1.
            core[0, :, 1] = self.f1_arr[i]
            cores.append(core)

        core = noise * self.rand.normal(size=(r, self.shapes[self.d-1], 1))
        core[0, :, 0] = self.f1_arr[self.d-1] + self.f0
        core[1, :, 0] = 1.
        cores.append(core)

        return cores

    def cores_2(self, r=2, only_near=False):
        mats = []
        num = 0
        for i1 in range(self.d-1):
            for i2 in ([i1+1] if only_near else range(i1+1, self.d)):
                shape = (self.shapes[i1], self.shapes[i2])
                mat = self.f2_arr[num].reshape(shape, order='C')
                mats.append(mat)
                num += 1

        cores = []
        num = 0
        for i1 in range(self.d-1):
            for i2 in ([i1+1] if only_near else range(i1+1, self.d)):
                cores.append(_second_order_2_tt(mats[num], i1, i2, self.shapes))
                num += 1

        return cores

    def load(self, fpath):
        if not '.' in fpath:
            fpath += '.pickle'

        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        self.order = data['order']
        self.dtype = data['dtype']
        self.y_max = data['y_max']
        self.y_min = data['y_min']
        self.domain = data['domain']
        self.shapes = data['shapes']
        self.d = data['d']
        self.f0 = data['f0']
        self.f1 = data['f1']
        self.f2 = data['f2']

    def max(self, minmax=max):
        """Get min or max value of tensor from 1th order ANOVA.

        Args:
            minmax (function): what to find (min or max).

        Returns:
            (value, np.ndarray): value of the found optimum and multi-index.

        """
        max_np = {min: np.argmin, max: np.argmax}[minmax]
        val = self.f0
        ind = [None] * self.d
        for i, fi in enumerate(self.f1):
            xx = list(fi)
            xx_max = xx[max_np([fi[x] for x in xx])]
            ind[i] = xx_max
            val += fi[xx_max]
        return val, ind

    def pair_num_to_num(self, x1, x2):
        assert x1 != x2
        if x1 > x2:
            x1, x2 = x2, x1
        return x2 - 1 + ((-3 + 2*self.d - x1)*x1) // 2

    def sample(self, xi=None, eps=1.E-10, with_square=False):
        if xi is None:
            xi = self.d - 1

        if xi > 0:
            prev_vals = self.sample(xi-1, eps, with_square)
        else:
            prev_vals = []

        dm = self.domain[xi]
        p = np.full(len(dm), self(prev_vals))

        if self.order >= 1:
            f1 = self.f1[xi]
            for i, x in enumerate(dm):
                p[i] += f1[x]

        if self.order >= 2:
            for x_prev_num, x_prev_val in enumerate(prev_vals):
                f2 = self.f2[self.pair_num_to_num(x_prev_num, xi)]
                for i, x in enumerate(dm):
                    p[i] += f2[x_prev_val, x]

        p = p**2 if with_square else np.maximum(p, 0)
        if p.sum() < eps * len(p):
            print('Warning (ANOVA): probabilities are zeros') # TODO
            p += eps

        p /= p.sum()
        p_sample = self.rand.choice(len(p), p=p)
        cur_sample = dm[p_sample]

        prev_vals.append(cur_sample)

        return prev_vals

    def save(self, fpath):
        if not '.' in fpath:
            fpath += '.pickle'

        with open(fpath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'dtype': self.dtype,
                'y_max': self.y_max,
                'y_min': self.y_min,
                'domain': self.domain,
                'shapes': self.shapes,
                'd': self.d,
                'f0': self.f0,
                'f1': self.f1,
                'f2': self.f2
            }, f, protocol=pickle.HIGHEST_PROTOCOL)


def anova(I_trn, y_trn, r=2, order=1, noise=1.E-10, seed=None, fpath=None):
    """Build TT-tensor by TT-ANOVA from the given random tensor samples.

    Args:
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d].
        y_trn (np.ndarray): values of the tensor for multi-indices I in the
            form of array of the shape [samples].
        r (int): rank of the constructed TT-tensor.
        order (int): order of the ANOVA decomposition (may be only 1 or 2).
        noise (float): noise added to formally zero elements of TT-cores.
        seed (int): random seed. It should be an integer number or a numpy
            Generator class instance.
        fpath (str): optional path for train data (I_trn, y_trn).

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    Note:
        A class "ANOVA" that represents a wider set of methods for working with
        this decomposition is also available. See "anova.py" for more details
        (detailed documentation for this class will be prepared later). This
        function is just a wrapper for "ANOVA" class. Maybe later this class
        will be replaced by the function.

    """
    return ANOVA(I_trn, y_trn, order, seed, fpath).cores(r, noise)


def _core_one(n, r):
    return np.kron(np.ones([1, n, 1]), np.eye(r)[:, None, :])


def _second_order_2_tt(A, i, j, shapes):
    if i > j:
        j, i = i, j
        A = A.T

    U, V = teneva.matrix_skeleton(A)
    r = U.shape[1]

    core1 = U.reshape(1, U.shape[0], r)
    core2 = V.reshape(r, V.shape[1], 1)

    cores = []
    for num, n in enumerate(shapes):
        if num < i or num > j:
            cores.append(np.ones([1, n, 1]))
        if num == i:
            cores.append(core1)
        if num == j:
            cores.append(core2)
        if i < num < j:
            cores.append(_core_one(n, r))

    return cores

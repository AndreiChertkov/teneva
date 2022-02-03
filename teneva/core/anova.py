"""Package teneva, module core.anova: ANOVA decomposition in the TT-format.

This module contains the function "anova" which computes the TT-approximation
for the tensor, using given random samples.

"""
import numpy as np


from .svd import matrix_skeleton
from .tensor import add_many


class ANOVA:
    def __init__(self, I_trn, Y_trn, order=1):
        if not order in [1, 2]:
            raise ValueError('Invalid value for ANOVA order (should be 1 or 2')
        self.order = order
        self.build(I_trn, Y_trn)

    def __call__(self, I):
        I = np.asanyarray(I)
        if len(I.shape) == 1:
            return self.calc(I)
        if len(I.shape) == 2:
            return np.array([self.calc(i) for i in I])
        return None

    def __getitem__(self, I):
        return self(I)

    def build(self, I_trn, Y_trn):
        self.d = I_trn.shape[1]
        self.domain = []
        self.shapes = np.zeros(self.d, dtype=int)
        for i in range(self.d):
            points = np.unique(I_trn[:, i])
            self.domain.append(points)
            self.shapes[i] = len(points)

        self.build_0(I_trn, Y_trn)
        if self.order >= 1:
            self.build_1(I_trn, Y_trn)
        if self.order >= 2:
            self.build_2(I_trn, Y_trn)

    def build_0(self, I_trn, Y_trn):
        self.f0 = np.mean(Y_trn)

    def build_1(self, I_trn, Y_trn):
        self.f1 = []
        self.f1_arr = []
        for i, dm in enumerate(self.domain):
            f1_curr = dict()
            f1_curr_arr = []
            for x in dm:
                value = np.mean(Y_trn[I_trn[:, i] == x]) - self.f0
                f1_curr[x] = value
                f1_curr_arr.append(value)
            self.f1.append(f1_curr)
            self.f1_arr.append(np.array(f1_curr_arr))

    def build_2(self, I_trn, Y_trn):
        self.f2 = []
        self.f2_arr = []
        for i1, dm1 in enumerate(self.domain[:-1]):
            for i2, dm2 in enumerate(self.domain[i1 + 1:], start=i1+1):
                f2_curr = dict()
                f2_curr_arr = []
                for x1 in dm1:
                    for x2 in dm2:
                        idx = (I_trn[:, i1] == x1) & (I_trn[:, i2] == x2)
                        if idx.sum() == 0:
                            value = 0.
                        else:
                            df = np.mean(Y_trn[idx]) - self.f0
                            value = df - self.f1[i1][x1] - self.f1[i2][x2]
                        f2_curr[(x1, x2)] = value
                        f2_curr_arr.append(value)
                self.f2.append(f2_curr)
                self.f2_arr.append(np.array(f2_curr_arr))

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
        for i, x1 in enumerate(x):
            try:
                res += self.f1[i][x1]
            except:
                print(f'Error: non val for f1[{i}][{x1}]')
        return res

    def calc_2(self, x):
        res = 0.
        num = 0
        for i1, x1 in enumerate(x[:-1]):
            for x2 in x[i1 + 1:]:
                res += self.f2[num][(x1, x2)]
                num += 1
        return res

    def cores(self, r=2, noise=1.E-10, only_near=False):
        if self.order < 1:
            raise ValueError('TT-cores may be constructed only if order >= 1')

        cores = self.cores_1(r, noise)

        if self.order >= 2:
            cores2_many = self.cores_2(r, only_near)
            cores = add_many([cores] + cores2_many, r=r)

        return cores

    def cores_1(self, r=2, noise=1.E-10):
        cores = []

        core = noise * np.random.randn(1, self.shapes[0], r)
        core[0, :, 0] = 1.
        core[0, :, 1] = self.f1_arr[0]
        cores.append(core)

        for i in range(1, self.d-1):
            core = noise * np.random.randn(r, self.shapes[i], r)
            core[0, :, 0] = 1.
            core[1, :, 1] = 1.
            core[0, :, 1] = self.f1_arr[i]
            cores.append(core)

        core = noise * np.random.randn(r, self.shapes[self.d-1], 1)
        core[0, :, 0] = self.f1_arr[self.d-1] + self.f0
        core[1, :, 0] = 1.
        cores.append(core)

        return cores

    def cores_2(self, r=2, only_near=False):
        mats = []
        num = 0
        for i1 in range(self.d-1):
            for i2 in range(i1+1, self.d):
                shape = (self.shapes[i1], self.shapes[i2])
                mat = self.f2_arr[num].reshape(shape, order='C')
                mats.append(mat)
                num += 1

        cores = []
        num = 0
        for i1 in range(self.d-1):
            for i2 in ([i1+1] if only_near else range(i1+1, self.d)):
                cores.append(second_order_2_TT(mats[num], i1, i2, self.shapes))
                num += 1

        return cores


def anova(I_trn, Y_trn, r=2, order=1):
    """Build TT-tensor by TT-ANOVA from the given random tensor samples.

    Args:
        I_trn (np.ndarray): multiindices for the tensor in the form of array
            of the shape [samples, d].
        Y_trn (np.ndarray): values of the tensor for multiindices I in the form
            of array of the shape [samples].
        r (int): maximum rank of the constructed TT-tensor (should be > 0).
        order (int): order of the ANOVA decomposition (may be only 1 or 2).

    Returns:
        list: TT-tensor, which represents the TT-approximation for the tensor.

    """
    return ANOVA(I_trn, Y_trn, order).cores(r)


def core_one(n, r):
    return np.kron(np.ones([1, n, 1]), np.eye(r)[:, None, :])


def second_order_2_TT(A, i, j, shapes):
    if i > j:
        j, i = i, j
        A = A.T

    U, V = matrix_skeleton(A)
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
            cores.append(core_one(n, r))

    return cores
import numpy as np


from .tensor import add
from .tensor import truncate


class ANOVA:
    def __init__(self, order, X=None, y=None):
        if not order in [1, 2]:
            raise ValueError('Invalid value for ANOVA order (should be 1 or 2')
        self.order = order

        if X is not None and y is not None:
            self.build(X, y)

    def build(self, X, y):
        self.dim = X.shape[1]
        self.domain = []
        self.shapes = np.zeros(self.dim, dtype=int)
        for i in range(self.dim):
            points = np.unique(X[:, i])
            self.domain.append(points)
            self.shapes[i] = len(points)

        self.build_0(X, y)
        if self.order >= 1:
            self.build_1(X, y)
        if self.order >= 2:
            self.build_2(X, y)

    def build_0(self, X, y):
        self.f0 = np.mean(y)

    def build_1(self, X, y):
        self.f1 = []
        self.f1_arr = []
        for i, dm in enumerate(self.domain):
            f1_curr = dict()
            f1_curr_arr = []
            for x in dm:
                value = np.mean(y[X[:, i] == x]) - self.f0
                f1_curr[x] = value
                f1_curr_arr.append(value)
            self.f1.append(f1_curr)
            self.f1_arr.append(np.array(f1_curr_arr))

    def build_2(self, X, y):
        self.f2 = []
        self.f2_arr = []
        for i1, dm1 in enumerate(self.domain[:-1]):
            for i2, dm2 in enumerate(self.domain[i1 + 1:], start=i1+1):
                f2_curr = dict()
                f2_curr_arr = []
                for x1 in dm1:
                    for x2 in dm2:
                        idx = (X[:, i1] == x1) & (X[:, i2] == x2)
                        if idx.sum() == 0:
                            #print(f"Warn: for (i1, i2)=({i1}, {i2}) there is no values")
                            value = 0.
                        else:
                            df = np.mean(y[idx]) - self.f0
                            value = df - self.f1[i1][x1] - self.f1[i2][x2]
                        f2_curr[(x1, x2)] = value
                        f2_curr_arr.append(value)
                self.f2.append(f2_curr)
                self.f2_arr.append(np.array(f2_curr_arr))

    def calc(self, X):
        res = self.calc_0()
        if self.order >= 1:
            res += self.calc_1(X)
        if self.order >= 2:
            res += self.calc_2(X)
        return res

    def calc_0(self):
        return self.f0

    def calc_1(self, X):
        res = 0.
        for i, x in enumerate(X):
            try:
                res += self.f1[i][x]
            except:
                print(f'Error: non val for f1[{i}][{x}]')
        return res

    def calc_2(self, X):
        res = 0.
        num = 0
        for i1, x1 in enumerate(X[:-1]):
            for x2 in X[i1 + 1:]:
                res += self.f2[num][(x1, x2)]
                num += 1
        return res

    def cores(self, rank=2, noise=0., only_near=False):
        if self.order < 1:
            raise ValueError('TT-cores may be constructed only if order >= 1')

        cores = self.cores_1(rank)

        if self.order >= 2:
            cores2_many = self.cores_2(rank, only_near)
            cores = sum_many([cores] + cores2_many, rmax=rank)

        return cores

    def cores_1(self, rank=2, noise=1.E-10):
        cores = []

        core = noise * np.random.randn(1, self.shapes[0], rank)
        core[0, :, 0] = 1.
        core[0, :, 1] = self.f1_arr[0]
        cores.append(core)

        for i in range(1, self.dim-1):
            core = noise * np.random.randn(rank, self.shapes[i], rank)
            core[0, :, 0] = 1.
            core[1, :, 1] = 1.
            core[0, :, 1] = self.f1_arr[i]
            cores.append(core)

        core = noise * np.random.randn(rank, self.shapes[self.dim-1], 1)
        core[0, :, 0] = self.f1_arr[self.dim-1] + self.f0
        core[1, :, 0] = 1.
        cores.append(core)

        return cores

    def cores_2(self, rank=2, only_near=False):
        mats = []
        num = 0
        for i1 in range(self.dim-1):
            for i2 in range(i1+1, self.dim):
                shape = (self.shapes[i1], self.shapes[i2])
                mat = self.f2_arr[num].reshape(shape, order='C')
                mats.append(mat)
                num += 1

        cores = []
        num = 0
        for i1 in range(self.dim-1):
            for i2 in [i1+1] if only_near else range(i1+1, self.dim):
                cores.append(second_order_2_TT(mats[num], i1, i2, self.shapes))
                num += 1

        return cores

    def __call__(self, X):
        X = np.asanyarray(X)
        if len(X.shape) == 1:
            return self.calc(X)
        if len(X.shape) == 2:
            return np.array([self.calc(x) for x in X])
        return None

    def __getitem__(self, X):
        return self(X)


def skeleton(a, eps=1.E-10, r=int(1e10), hermitian=False):
    u, s, v = np.linalg.svd(a, full_matrices=False,
        compute_uv=True, hermitian=hermitian)
    r = min(r, sum(s/s[0] > eps))
    un = u[:, :r]
    sn = np.diag(np.sqrt(s[:r]))
    vn = v[:r]
    return un @ sn, sn @ vn


def core_one(n, r):
    core = np.zeros([r, n, r])
    for x in range(n):
        core[range(r), x, range(r)] = 1.
    return core


def second_order_2_TT(A, i, j, shapes):
    if i > j: # Так не должно быть
        j, i = i, j
        A = A.T

    U, V = skeleton(A)
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


def sum_many(tensors, e=1.E-10, rmax=None, freq_trunc=15):
    cores = tensors[0]
    for i, t in enumerate(tensors[1:]):
        cores = add(cores, t)
        if (i+1) % freq_trunc == 0:
            cores = truncate(cores, e=e)
    return truncate(cores, e=e, rmax=rmax)

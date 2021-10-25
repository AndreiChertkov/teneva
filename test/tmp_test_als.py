import numpy as np
import teneva
from time import perf_counter as tpc


np.random.seed(42)


from teneva import als
from teneva import als2
from teneva import ANOVA
from teneva import getter
from teneva import lhs


def solve(data, order=1, rank=2, nswp=10, name='test', is_als_2=False):
    I_trn, Y_trn, I_tst, Y_tst = data
    M_trn = Y_trn.size
    M_tst = Y_tst.size

    t = tpc()
    anova = ANOVA(order, I_trn, Y_trn)
    Y0 = anova.cores(rank)
    if is_als_2:
        Y = als2(I_trn, Y_trn, Y0, nswp)
    else:
        Y = als(I_trn, Y_trn, Y0, nswp)
    t = tpc() - t

    get = getter(Y)
    Z_trn = np.array([get(i) for i in I_trn])
    Z_tst = np.array([get(i) for i in I_tst])

    e_trn = np.linalg.norm(anova(I_trn) - Y_trn) / np.linalg.norm(Y_trn)
    e_tst = np.linalg.norm(anova(I_tst) - Y_tst) / np.linalg.norm(Y_tst)
    e_trn_tt = np.linalg.norm(Z_trn - Y_trn) / np.linalg.norm(Y_trn)
    e_tst_tt = np.linalg.norm(Z_tst - Y_tst) / np.linalg.norm(Y_tst)

    text = name + ' ' * (15 - len(name)) + '| '
    text += f't: {t:-6.2f} | '
    text += f'r: {rank:-3d} | '
    text += f'data: {M_trn} / {M_tst} | '
    text += f'err: {e_trn:-8.2e} / {e_tst:-8.2e} | '
    text += f'err (TT): {e_trn_tt:-8.2e} / {e_tst_tt:-8.2e} | '
    print(text)


def run():
    d = 4
    shape = [100] * d

    def f(I):
        y = I[:, 0] + I[:, 1] + I[:, 2] + I[:, 3]
        return y * 1.0

    M_trn = 10000
    I_trn = lhs(shape, M_trn)
    Y_trn = f(I_trn)
    I_trn.shape


    M_tst = 10000
    I_tst = np.vstack([np.random.choice(shape[i], M_tst) for i in range(d)]).T
    Y_tst = f(I_tst)

    data = [I_trn, Y_trn, I_tst, Y_tst]

    solve(data, order=1, rank=3, name='ANOVA-1-ALS')
    solve(data, order=2, rank=3, name='ANOVA-2-ALS')

    solve(data, order=1, rank=3, name='ANOVA-1-ALS-2', is_als_2=True)
    solve(data, order=2, rank=3, name='ANOVA-2-ALS-2', is_als_2=True)

if __name__ == '__main__':
    run()

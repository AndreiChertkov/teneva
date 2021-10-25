import numpy as np
import teneva
from time import perf_counter as tpc


np.random.seed(42)


from teneva import ANOVA
from teneva import getter
from teneva import lhs


def solve(I_trn, Y_trn, I_tst, Y_tst, order=1, rank=2, name='test'):
    M_trn = Y_trn.size
    M_tst = Y_tst.size

    t = tpc()
    anova = ANOVA(order, I_trn, Y_trn)
    Y = anova.cores(rank)
    t = tpc() - t

    get = getter(Y)
    Z_trn = np.array([get(i) for i in I_trn])
    Z_tst = np.array([get(i) for i in I_tst])

    e_trn = np.linalg.norm(anova(I_trn) - Y_trn) / np.linalg.norm(Y_trn)
    e_tst = np.linalg.norm(anova(I_tst) - Y_tst) / np.linalg.norm(Y_tst)
    e_trn_tt = np.linalg.norm(Z_trn - Y_trn) / np.linalg.norm(Y_trn)
    e_tst_tt = np.linalg.norm(Z_tst - Y_tst) / np.linalg.norm(Y_tst)

    text = name + ' ' * (10 - len(name)) + '| '
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
        return y

    M_trn = 10000
    I_trn = lhs(shape, M_trn)
    Y_trn = f(I_trn)
    I_trn.shape


    M_tst = 10000
    I_tst = np.vstack([np.random.choice(shape[i], M_tst) for i in range(d)]).T
    Y_tst = f(I_tst)

    solve(I_trn, Y_trn, I_tst, Y_tst, order=1, rank=2, name='ANOVA-1')
    solve(I_trn, Y_trn, I_tst, Y_tst, order=2, rank=9, name='ANOVA-2')


if __name__ == '__main__':
    run()

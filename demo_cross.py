"""The demo of using TT-CROSS. Example for the Rosenbrock function.

As a result of the script work we expect the output in console like this:
"
Rosenbrock     | e=1.4e-15 | r=3.0 | t= 0.8 | evals=5.9e+04 / 3.9e+03 | sweep=  2 | eps=4.0e-08 | stop=e
"

"""

import teneva
import numpy as np
from time import perf_counter as tpc


def run(d, n, evals, e=1.E-6, r=2, m_tst=1.E+5, kind='cheb', is_grid=False):
    func = teneva.DemoFuncRosenbrock(d)
    func.set_grid(n, kind)
    func.build_tst(m_tst, is_grid)

    info = {}
    cache = {}
    nswp = 100
    dr_min = 1
    dr_max = 2

    t_trn = tpc()
    Y = teneva.rand([n] * func.d, r)
    Y = teneva.cross(
        func.comp_grid, Y, e, evals, nswp, dr_min, dr_max, cache, info)
    t_trn = tpc() - t_trn

    t_tst = tpc()
    err = func.check_tst(Y)
    t_tst = tpc() - t_tst

    text = func.name + ' ' * max(0, 15 - len(func.name)) + '| '
    text += f'e={err:-7.1e} | r={teneva.erank(Y):-3.1f} | t={t_trn:-4.1f} | '
    text += f'evals={info["k_evals"]:-7.1e} / {info["k_cache"]:-7.1e} | '
    text += f'sweep={info["nswp"]:-3d} | eps={info["e"]:-7.1e} | '
    text += f'stop={info["stop"]}'
    print(text)


if __name__ == '__main__':
    run(d=10, n=100, evals=1.E+5, kind='cheb', is_grid=False)

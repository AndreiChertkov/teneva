import numpy as np
import re
import teneva


class Log:
    def __init__(self, fpath='log.txt'):
        self.fpath = fpath
        self.is_new = True
        self.len_pref = 19

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False

    def name(self, name):
        text = name + ' ' * max(0, self.len_pref-len(name)) + ' > '
        self(text)

    def res(self, res):
        name = res['method']
        text = '  - ' + name + ' ' * max(0, self.len_pref-len(name)-4) + ' | '

        text += 'error (trn/tst) : '
        text += f'{res["e_trn"]:-7.1e} / '
        text += f'{res["e_tst"]:-7.1e}'

        if 't' in res:
            text += ' | time : '
            text += f'{res["t"]:-7.3f}'

        self(text)


def calc_als(func, log, r=5, nswp=50, with_log=True):
    func.clear()
    func.rand(r=r)
    func.als(nswp=nswp, log=False)

    t = func.t
    e_trn = func.check_trn_ind()
    e_tst = func.check_tst_ind()
    if with_log:
        log.res({'method': func.method, 't': t, 'e_trn': e_trn, 'e_tst': e_tst})

    return t, e_trn, e_tst


def calc_als_many(func, log, reps=10):
    t = []
    e_trn = []
    e_tst = []
    for rep in range(reps):
        t_cur, e_trn_cur, e_tst_cur = calc_als(func, log, with_log=False)
        t.append(t_cur)
        e_trn.append(e_trn_cur)
        e_tst.append(e_tst_cur)

    t = np.mean(t)
    e_trn = np.mean(e_trn)
    e_tst = np.mean(e_tst)

    method = f'ALS (* mean {reps})'
    log.res({'method': method, 't': t, 'e_trn': e_trn, 'e_tst': e_tst})

    return t, e_trn, e_tst


def calc_anova(func, log, r=5, order=1, noise_ano=1.E-10):
    func.clear()
    func.anova(r=r, order=order, noise=noise_ano)

    t = func.t
    e_trn = func.check_trn_ind()
    e_tst = func.check_tst_ind()
    log.res({'method': func.method, 't': t, 'e_trn': e_trn, 'e_tst': e_tst})

    return t, e_trn, e_tst


def calc_anova_als(func, log, r=5, nswp=50, order=1, noise_ano=1.E-10):
    func.clear()
    func.anova(r=r, order=order, noise=noise_ano)
    func.als(nswp=nswp, log=False)

    t = func.t
    e_trn = func.check_trn_ind()
    e_tst = func.check_tst_ind()
    log.res({'method': func.method, 't': t,
        'e_trn': e_trn, 'e_tst': e_tst})

    return t, e_trn, e_tst


def calc_cross(func, log, m, r=1):
    func.clear()
    func.set_trn_ind()
    func.rand(r=r)
    func.cross(m, log=False)

    t = func.t
    e_trn = func.check_trn_ind()
    e_tst = func.check_tst_ind()
    log.res({'method': func.method, 't': t, 'e_trn': e_trn, 'e_tst': e_tst})

    return t, e_trn, e_tst


def check(d=7, n=10, m_trn=1.E+4, m_tst=1.E+4):
    log = Log(f'check/result_{get_version()}.txt')
    for func in teneva.func_demo_all(d, with_piston=True):
        log.name(func.name)

        func.set_grid(n, kind='uni')
        func.build_trn_ind(m_trn)
        func.build_tst_ind(m_tst)

        calc_als_many(func, log)
        calc_anova(func, log)
        calc_anova_als(func, log)
        calc_cross(func, log, m_trn)


def get_version():
    with open('teneva/__init__.py', encoding='utf-8') as f:
        text = f.read()
        version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
        version = version.group(1)
        version = version.replace('.', '-')
    return version


if __name__ == '__main__':
    np.random.seed(42)
    check()

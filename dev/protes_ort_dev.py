import jax
import jax.numpy as jnp
from teneva import core_jax as teneva
from time import perf_counter as tpc


def protes_ort_dev(f, d, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, seed=42, is_max=False, log=False, log_ind=False, info={}, P=None, with_info_i_opt_list=False):
    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'k_top': k_top,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})

    rng = jax.random.PRNGKey(seed)

    sample = jax.jit(jax.vmap(teneva.sample, (None, None, 0)))
    interface = jax.jit(teneva.interface_rtl)
    likelihood = jax.jit(jax.vmap(teneva.get_log, (None, 0)))
    orthogonalize = jax.jit(teneva.orthogonalize_rtl_stab)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(d, n, r, key)
        P, p = orthogonalize(P)

    elif len(P[1].shape) != 4:
        raise ValueError('Initial P tensor should have special format')

    @jax.jit
    def loss(P_cur, I_cur):
        l = likelihood(P_cur, I_cur)
        return jnp.mean(-l)

    loss_grad = jax.grad(loss)

    @jax.jit
    def update_orth(Y, G, lr):
        def body(q, data):
            Q, G = data
            r1, n1, r2 = Q.shape
            Q = Q.reshape(r1, n1*r2).T
            G = G.reshape(r1, n1*r2).T
            A = G @ Q.T - Q @ G.T
            I = jnp.eye(A.shape[0])
            Q = jnp.linalg.inv(I + lr/2*A) @ (I - lr/2*A) @ Q
            return None, Q.T.reshape(r1, n1, r2)

        Yl, Ym, Yr = Y
        Gl, Gm, Gr = G

        _, Yl = body(None, (Yl, Gl))
        _, Ym = jax.lax.scan(body, None, (Ym, Gm))
        _, Yr = body(None, (Yr, Gr))

        return Yl, Ym, Yr

    @jax.jit
    def optimize(P_cur, I_cur):
        return update_orth(P_cur, loss_grad(P_cur, I_cur), lr)

    while True:
        rng, key = jax.random.split(rng)
        zl, zm = interface(P)
        I = sample(P, zm, jax.random.split(key, k))

        y = f(I)
        y = jnp.array(y)
        info['m'] += y.shape[0]

        is_new = _check(I, y, info, with_info_i_opt_list)

        if info['m'] >= m:
            info['t'] = tpc() - time
            break

        ind = jnp.argsort(y, kind='stable')
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            P = optimize(P, I[ind, :])

        info['t'] = tpc() - time

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info, with_info_i_opt_list):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr

        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(info['y_opt'])

        if with_info_i_opt_list:
            info['i_opt_list'].append(info['i_opt'].copy())

    return is_new


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]


def _log(info, log=False, log_ind=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if log_ind:
        text += f' | i {" ".join([str(i) for i in info["i_opt"]])}'

    if is_end:
        text += ' <<< DONE'

    print(text)


##############################################################################
###### TMP (code to check the optimizer)


def func_build(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*jnp.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (b - a) + a

        y1 = jnp.sqrt(jnp.sum(X**2, axis=1) / d)
        y1 = - par_a * jnp.exp(-par_b * y1)

        y2 = jnp.sum(jnp.cos(par_c * X), axis=1)
        y2 = - jnp.exp(y2 / d)

        y3 = par_a + jnp.exp(1.)

        return y1 + y2 + y3

    return func


def demo():
    """A simple demonstration for discretized multivariate analytic function.

    Note that base protes gives result:

    protes > m 5.0e+01 | t 3.338e+00 | y  1.5434e+01
    protes > m 1.2e+03 | t 5.441e+00 | y  1.5239e+01
    protes > m 2.2e+03 | t 7.157e+00 | y  1.4116e+01
    protes > m 3.2e+03 | t 8.809e+00 | y  1.3057e+01
    protes > m 4.2e+03 | t 1.043e+01 | y  8.4726e+00
    protes > m 5.8e+03 | t 1.322e+01 | y  0.0000e+00
    protes > m 1.0e+04 | t 2.049e+01 | y  0.0000e+00 <<< DONE

    RESULT | y opt = 0.00000e+00 | time = 20.501465051

    """
    d = 7                # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes_ort_dev(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t}')


if __name__ == '__main__':
    demo()

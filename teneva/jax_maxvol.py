# This is draft!


import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu as jlu
from jax.scipy.linalg import solve_triangular as jsolve_triangular


@jax.jit
def maxvol(A, e=1.05, K=100):
    N, r = A.shape
    P, L, U = jlu(A)
    I = P.argmax(axis=0)
    Q = jsolve_triangular(U, A.T, trans=1, lower=False)
    B = jsolve_triangular(L[:r, :], Q, trans=1, unit_diagonal=True, lower=True)

    @jax.jit
    def step(args):
        I, B, i, j = args
        x = B[i, :]
        y = B[:, j]
        y = jax.ops.index_update(y, i, y[i] - 1)
        I = jax.ops.index_update(I, i, j)
        B-= jnp.outer(y / B[i, j], x)
        return I, B, i, j

    @jax.jit
    def cond(args):
        I, B, b, k = args
        return jnp.logical_and(k < K, jnp.abs(b) > e)

    @jax.jit
    def body(args):
        I, B, b, k = args
        i, j = jnp.divmod(jnp.abs(B).argmax(), N)
        b = B[i, j]
        I, B, i, j = jax.lax.cond(jnp.abs(b) > e, step,
            lambda args: args, operand=(I, B, i, j))
        return I, B, b, k+1

    I, B, b, k = jax.lax.while_loop(cond, body, (I, B, 2 * e, 0))
    return I[:r], B.T


def rect_maxvol(A, e, N_min, N_max, e0=1.05, K0=10): # DRAFT !!!
    N, r = A.shape
    I_tmp, B = maxvol(A, e0, K0)
    I = jnp.hstack([I_tmp, jnp.zeros(N_max-r, dtype=I_tmp.dtype)])
    S = jnp.ones(N, dtype=I_tmp.dtype)
    for s in I_tmp:
        S = jax.ops.index_update(S, s, 0)
    F = S * jnp.linalg.norm(B, axis=1)**2

    # @jax.jit
    def cond(args):
        I, B, F, S, f, k = args
        return jnp.logical_and(k < N_max, f > e*e)

    # @jax.jit
    def body(args):
        I, B, F, S, f, k = args
        i = jnp.argmax(F)
        # if k >= N_min and F[i] <= e*e: break
        I = jax.ops.index_update(I, k, i)
        S = jax.ops.index_update(S, i, 0)
        v = B.dot(B[i])
        l = 1. / (1 + v[i])
        B = jnp.hstack([B - l * jnp.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
        return I, B, F, S, F[i], k+1

    I, B, F, S, f, k = jax.lax.while_loop(cond, body, (I, B, F, S, 2 * e*e, r))
    I = I[:B.shape[1]]
    #B[I] = jnp.eye(B.shape[1], dtype=B.dtype)
    return I, B

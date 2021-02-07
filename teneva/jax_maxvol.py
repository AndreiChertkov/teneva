import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu as jlu
from jax.scipy.linalg import solve_triangular as jsolve_triangular


@jax.jit
def jax_maxvol(A, e=1.05, K=100):
    N, r = A.shape
    P, L, U = jlu(A)
    I = P.argmax(axis=0)
    Q = jsolve_triangular(U, A.T, trans=1, lower=False)
    B = jsolve_triangular(L[:r, :], Q, trans=1, unit_diagonal=True, lower=True)

    @jax.jit
    def cond(args):
        I, B, b, k = args
        return jnp.logical_and(k < K, jnp.abs(b) > e)

    @jax.jit
    def body(args):
        I, B, b, k = args
        i, j = jnp.divmod(jnp.abs(B).argmax(), N)
        b = B[i, j]

        @jax.jit
        def run(args):
            I, B = args
            x = B[i, :]
            y = B[:, j]
            y = jax.ops.index_update(y, i, y[i] - 1)
            I = jax.ops.index_update(I, i, j)
            B-= jnp.outer(y / b, x)
            return I, B

        I, B = jax.lax.cond(jnp.abs(b) > e, run, lambda a: a, operand=(I, B))
        return I, B, b, k+1

    I, B, b, k = jax.lax.while_loop(cond, body, (I, B, 2 * e, 0))
    return I[:r], B.T

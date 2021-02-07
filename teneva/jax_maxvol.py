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

def rect_maxvol(A, e, maxK, min_add_K=0, start_maxvol_iters=10): # DRAFT!
    N, r = A.shape
    minK = min(maxK, r + min_add_K)
    I_tmp, B = maxvol(A, 1.05, start_maxvol_iters)
    I = jnp.hstack([I_tmp, jnp.zeros(N-r, dtype=np.int32)])
    C = jnp.ones(N, dtype=np.int32)
    for k in I_tmp:
        C = jax.ops.index_update(C, k, 0)
    F = C * jnp.linalg.norm(B, axis=1)**2

    @jax.jit
    def cond(args):
        I, B, F, C, f, k = args
        return jnp.logical_and(k < maxK, jnp.abs(f) > e*e)

    @jax.jit
    def body(args):
        I, B, F, C, f, k = args
        i = jnp.argmax(F)
        # if k >= minK and F[i] <= e*e: break
        I = jax.ops.index_update(I, k, i)
        C = jax.ops.index_update(C, i, 0)
        c = B[i]
        v = B.dot(c)
        l = 1. / (1 + v[i])
        B = jnp.hstack([B - l * jnp.outer(v, c), l * v.reshape(-1, 1)])
        F = C * (F - l * v[:N] * v[:N])

        return I, B, F, C, F[i], k+1

    I, B, F, C, f, k = jax.lax.while_loop(cond, body, (I, B, F, C, 2 * e*e, r))
    I = I[:B.shape[1]]
    #B[I] = jnp.eye(B.shape[1], dtype=B.dtype)
    return I, B

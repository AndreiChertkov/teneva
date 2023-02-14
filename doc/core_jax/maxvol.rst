Module maxvol: compute the maximal-volume submatrix
---------------------------------------------------


.. automodule:: teneva.core_jax.maxvol


-----




|
|

.. autofunction:: teneva.core_jax.maxvol.maxvol

  **Examples**:

  .. code-block:: python

    n = 5000                           # Number of rows
    r = 50                             # Number of columns
    rng, key = jax.random.split(rng)
    A = jax.random.normal(key, (n, r)) # Random tall matrix

  .. code-block:: python

    e = 1.01  # Accuracy parameter
    k = 500   # Maximum number of iterations

  .. code-block:: python

    # Compute row numbers and coefficient matrix:
    I, B = teneva.maxvol(A, e, k)
    
    # Maximal-volume square submatrix:
    C = A[I, :]

  .. code-block:: python

    print(f'|Det C|        : {np.abs(np.linalg.det(C)):-10.2e}')
    print(f'Max |B|        : {np.max(np.abs(B)):-10.2e}')
    print(f'Max |A - B C|  : {np.max(np.abs(A - B @ C)):-10.2e}')
    print(f'Selected rows  : {I.size:-10d} > ', np.sort(I))

    # >>> ----------------------------------------
    # >>> Output:

    # |Det C|        :        inf
    # Max |B|        :   1.00e+00
    # Max |A - B C|  :   5.78e-06
    # Selected rows  :         50 >  [  57  110  310  531  590  623  699 1074 1294 1369 1429 1485 1504 1723
    #  1781 1835 1917 1952 2120 2147 2342 2687 2762 2785 3027 3088 3147 3236
    #  3393 3491 3619 3714 3820 3870 3879 4030 4037 4039 4049 4087 4118 4285
    #  4328 4446 4582 4587 4664 4694 4805 4931]
    # 




|
|

.. autofunction:: teneva.core_jax.maxvol.maxvol_rect

  **Examples**:

  .. code-block:: python

    n = 5000                           # Number of rows
    r = 50                             # Number of columns
    rng, key = jax.random.split(rng)
    A = jax.random.normal(key, (n, r)) # Random tall matrix

  .. code-block:: python

    e = 1.01    # Accuracy parameter
    dr_min = 2  # Minimum number of added rows
    dr_max = 8  # Maximum number of added rows
    e0 = 1.05   # Accuracy parameter for the original maxvol algorithm
    k0 = 50     # Maximum number of iterations for the original maxvol algorithm

  THIS IS DRAFT !!!

  .. code-block:: python

    # Row numbers and coefficient matrix:
    I, B = teneva.maxvol_rect(A, e,
        dr_min, dr_max, e0, k0)
    
    # Maximal-volume rectangular submatrix:
    C = A[I, :]

    # >>> ----------------------------------------
    # >>> Output:

    # /Users/andrei/opt/anaconda3/envs/teneva/lib/python3.8/site-packages/jax-0.4.3-py3.8.egg/jax/_src/ops/scatter.py:87: FutureWarning: scatter inputs have incompatible types: cannot safely cast value from dtype=float32 to dtype=int32. In future JAX releases this will result in an error.
    #   warnings.warn("scatter inputs have incompatible types: cannot safely cast "
    # 




|
|


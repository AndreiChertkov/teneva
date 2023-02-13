Module svd: SVD-based algorithms for matrices and tensors
---------------------------------------------------------


.. automodule:: teneva.core_jax.svd


-----




|
|

.. autofunction:: teneva.core_jax.svd.matrix_skeleton

  **Examples**:

  .. code-block:: python

    # Shape of the matrix:
    m, n = 100, 30
    
    # Build random matrix, which has rank 3,
    # as a sum of rank-1 matrices:
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 6)
    u = [jax.random.normal(keys[i], (m, )) for i in range(3)]
    v = [jax.random.normal(keys[i], (m, )) for i in range(3, 6)]
    A = np.outer(u[0], v[0]) + np.outer(u[1], v[1]) + np.outer(u[2], v[2])

  .. code-block:: python

    # Compute skeleton decomp.:
    U, V = teneva.matrix_skeleton(A, r=3)
    
    # Approximation error
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :', V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 100)
    # Error      : 2.82e-07
    # 

  .. code-block:: python

    # Compute skeleton decomp with small rank:
    U, V = teneva.matrix_skeleton(A, r=2)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :', V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 100)
    # Error      : 5.11e-01
    # 




|
|


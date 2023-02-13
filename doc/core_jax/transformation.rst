Module transformation: orthogonalization, truncation and other transformations of the TT-tensors
------------------------------------------------------------------------------------------------


.. automodule:: teneva.core_jax.transformation


-----




|
|

.. autofunction:: teneva.core_jax.transformation.full

  **Examples**:

  .. code-block:: python

    d = 5     # Dimension of the tensor
    n = 6     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    teneva.show(Y)
    
    Z = teneva.full(Y)
    
    # Compare original tensor and reconstructed tensor
    k = np.array([0, 1, 2, 3, 4])
    y = teneva.get(Y, k)
    z = Z[tuple(k)]
    e = np.abs(z-y)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D (shape =     6; rank =     4)
    # Error : 9.7e-08
    # 




|
|


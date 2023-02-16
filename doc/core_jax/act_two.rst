Module act_two: operations with a pair of TT-tensors
----------------------------------------------------


.. automodule:: teneva.core_jax.act_two


-----




|
|

.. autofunction:: teneva.core_jax.act_two.accuracy

  **Examples**:

  .. code-block:: python

    d = 20  # Dimension of the tensor
    n = 10  # Mode size of the tensor
    r = 2   # TT-rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r, key)

  Let construct the TT-tensor Y2 = Y1 + eps * Y1 (eps = 1.E-4):

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Z2 = teneva.rand(d, n, r, key)
    Z2[0] = Z2[0] * 1.E-4
    
    Y2 = teneva.add(Y1, Z2) 

  .. code-block:: python

    eps = teneva.accuracy(Y1, Y2)
    
    print(f'Accuracy     : {eps.item():-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 8.99e-05
    # 

  Note that this function works correctly even for very large dimension values due to the use of balancing (stabilization) in the scalar product:

  .. code-block:: python

    for d in [10, 50, 100, 250, 1000, 10000]:
        rng, key = jax.random.split(rng)
        Y1 = teneva.rand(d, n, r, key)
        Y2 = teneva.add(Y1, Y1)
        eps = teneva.accuracy(Y1, Y2).item()
    
        print(f'd = {d:-5d} | eps = {eps:-8.1e} | expected value 0.5')

    # >>> ----------------------------------------
    # >>> Output:

    # d =    10 | eps =  5.0e-01 | expected value 0.5
    # d =    50 | eps =  5.0e-01 | expected value 0.5
    # d =   100 | eps =  5.0e-01 | expected value 0.5
    # d =   250 | eps =  5.0e-01 | expected value 0.5
    # d =  1000 | eps =  5.0e-01 | expected value 0.5
    # d = 10000 | eps =  5.0e-01 | expected value 0.5
    # 




|
|

.. autofunction:: teneva.core_jax.act_two.add

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r1 = 2  # TT-rank of the 1th tensor
    r2 = 3  # TT-rank of the 2th tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r1, key)
    
    rng, key = jax.random.split(rng)
    Y2 = teneva.rand(d, n, r2, key)

  .. code-block:: python

    Y = teneva.add(Y1, Y2)
    teneva.show(Y)  # Note that the result has TT-rank 2 + 3 = 5

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     6 | r =     5 |
    # 

  Let check the result:

  .. code-block:: python

    Y1_full = teneva.full(Y1)
    Y2_full = teneva.full(Y2)
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.60e-08
    # 




|
|

.. autofunction:: teneva.core_jax.act_two.mul

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r1 = 2  # TT-rank of the 1th tensor
    r2 = 3  # TT-rank of the 2th tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r1, key)
    
    rng, key = jax.random.split(rng)
    Y2 = teneva.rand(d, n, r2, key)

  .. code-block:: python

    Y = teneva.mul(Y1, Y2)
    teneva.show(Y)  # Note that the result has TT-rank 2 * 3 = 6

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     6 | r =     6 |
    # 

  Let check the result:

  .. code-block:: python

    Y1_full = teneva.full(Y1)
    Y2_full = teneva.full(Y2)
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full * Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 1.51e-07
    # 




|
|

.. autofunction:: teneva.core_jax.act_two.mul_scalar

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r1 = 2  # TT-rank of the 1th tensor
    r2 = 3  # TT-rank of the 2th tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r1, key)
    
    rng, key = jax.random.split(rng)
    Y2 = teneva.rand(d, n, r2, key)

  .. code-block:: python

    v = teneva.mul_scalar(Y1, Y2)
    
    print(v) # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [2.6991057]
    # 

  Let check the result:

  .. code-block:: python

    Y1_full = teneva.full(Y1)
    Y2_full = teneva.full(Y2)
    
    v_full = np.sum(Y1_full * Y2_full)
    
    print(v_full) # Print the resulting value from full tensor
    
    # Compute error for TT-tensor vs full tensor :
    e = np.abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # 2.6991096
    # Error     : 1.41e-06
    # 




|
|

.. autofunction:: teneva.core_jax.act_two.mul_scalar_stab

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r1 = 2  # TT-rank of the 1th tensor
    r2 = 3  # TT-rank of the 2th tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r1, key)
    
    rng, key = jax.random.split(rng)
    Y2 = teneva.rand(d, n, r2, key)

  .. code-block:: python

    v, p = teneva.mul_scalar_stab(Y1, Y2)
    print(v) # Print the scaled value
    print(p) # Print the scale factors
    
    v = v * 2**np.sum(p) # Resulting value
    print(v)   # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [-1.0362672]
    # [-1.  1.  0.  2. -3.]
    # [-0.5181336]
    # 

  Let check the result:

  .. code-block:: python

    Y1_full = teneva.full(Y1)
    Y2_full = teneva.full(Y2)
    
    v_full = np.sum(Y1_full * Y2_full)
    
    print(v_full) # Print the resulting value from full tensor
    
    # Compute error for TT-tensor vs full tensor :
    e = abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # -0.51813036
    # Error     : 6.21e-06
    # 




|
|

.. autofunction:: teneva.core_jax.act_two.sub

  **Examples**:

  .. code-block:: python

    d = 5   # Dimensions of the tensors
    n = 6   # Mode sizes of the tensors
    r1 = 2  # TT-rank of the 1th tensor
    r2 = 3  # TT-rank of the 2th tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y1 = teneva.rand(d, n, r1, key)
    
    rng, key = jax.random.split(rng)
    Y2 = teneva.rand(d, n, r2, key)

  .. code-block:: python

    Y = teneva.sub(Y1, Y2)
    teneva.show(Y)  # Note that the result has TT-rank 2 + 3 = 5

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     6 | r =     5 |
    # 

  Let check the result:

  .. code-block:: python

    Y1_full = teneva.full(Y1)
    Y2_full = teneva.full(Y2)
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full - Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.56e-08
    # 




|
|


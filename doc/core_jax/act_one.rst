Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva.core_jax.act_one


-----




|
|

.. autofunction:: teneva.core_jax.act_one.copy

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with mode size 4 and TT-rank 12:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(10, 4, 12, key)
    
    Z = teneva.copy(Y) # The copy of Y  
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # 0.798532
    # 0.798532
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.get

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = np.arange(2**d) # Tensor will be 2^d
    Y0 = np.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    k = np.array([0, 1, 0, 1, 0])
    y1 = teneva.get(Y1, k)
    
    # Compute the same element of the original tensor:
    y0 = Y0[tuple(k)]
    
    # Compare values:
    e = np.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # Error : 4.8e-07
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.get_many

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = np.arange(2**d) # Tensor will be 2^d
    Y0 = np.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    K = np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    y1 = teneva.get_many(Y1, K)
    
    # Compute the same elements of the original tensor:
    y0 = np.array([Y0[tuple(k)] for k in K])
    
    # Compare values:
    e = np.max(np.abs(y1-y0))
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # Error : 6.6e-07
    # 

  We can compare the calculation time using the base function ("get") with "jax.vmap" and the function "get_many":

  .. code-block:: python

    d = 1000   # Dimension of the tensor
    n = 100    # Mode size of the tensor
    r = 10     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    get1 = jax.jit(jax.vmap(teneva.get, (None, 0)))
    get2 = jax.jit(teneva.get_many)
    
    for m in [1, 1.E+1, 1.E+2, 1.E+3, 1.E+4]:
        # TODO: remove teneva_base here
        I = np.array(teneva_base.sample_lhs([n]*d, int(m)))
    
        t1 = tpc()
        y1 = get1(Y, I)
        t1 = tpc() - t1
    
        t2 = tpc()
        y2 = get2(Y, I)
        t2 = tpc() - t2
    
        print(f'm: {m:-7.1e} | T1 : {t1:-8.4f} | T2 : {t2:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    # m: 1.0e+00 | T1 :   0.0650 | T2 :   0.0623
    # m: 1.0e+01 | T1 :   0.0894 | T2 :   0.0965
    # m: 1.0e+02 | T1 :   0.0938 | T2 :   0.0960
    # m: 1.0e+03 | T1 :   0.1464 | T2 :   0.1485
    # m: 1.0e+04 | T1 :   0.4759 | T2 :   0.4678
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.get_stab

  **Examples**:

  .. code-block:: python

    d = 5  # Dimension of the tensor
    n = 4  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct d-dim full array:
    t = np.arange(2**d) # Tensor will be 2^d
    Y0 = np.cos(t).reshape([2] * d, order='F')
    
    # Compute TT-tensor from Y0 by TT-SVD:  
    Y1 = teneva.svd(Y0, r)
    
    # Print the TT-tensor:
    teneva.show(Y1)
    
    # Select some tensor element and compute the value:
    k = np.array([0, 1, 0, 1, 0])
    y1, p1 = teneva.get_stab(Y1, k)
    print(y1)
    print(p1)
    
    # Reconstruct the value:
    y1 = y1 * 2.**np.sum(p1)
    print(y1)
    
    # Compute the same element of the original tensor:
    y0 = Y0[tuple(k)]
    
    # Compare values:
    e = np.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     5 | n =     2 | r =     2 |
    # -1.6781421
    # [ 0.  0.  0. -1.  0.]
    # -0.83907104
    # Error : 4.8e-07
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    k = np.zeros(1000, dtype=np.int32)
    y, p = teneva.get_stab(Y, k)
    print(y, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.1663944 808.0
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.mean

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m = teneva.mean(Y)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = np.mean(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.59e-09
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    teneva.mean(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # Array(0., dtype=float32)
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.mean_stab

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m, p = teneva.mean_stab(Y)
    print(m)
    print(p)
    
    # Reconstruct the value:
    m = m * 2.**np.sum(p)
    print(m)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = np.mean(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # -1.5127699
    # [-2. -2. -1.  0. -2. -3.]
    # -0.0014773144
    # Error     : 1.28e-09
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    m, p = teneva.mean_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.5990865 -2530.0
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.sum

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m = teneva.sum(Y)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = np.sum(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 7.63e-05
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    teneva.sum(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # Array(0., dtype=float32)
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.sum_stab

  **Examples**:

  .. code-block:: python

    d = 6     # Dimension of the tensor
    n = 5     # Mode size of the tensor
    r = 4     # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    
    m, p = teneva.sum_stab(Y)
    print(m)
    print(p)
    
    # Reconstruct the value:
    m = m * 2.**np.sum(p)
    print(m)
    
    # Compute tensor in the full format to check the result:
    Y_full = teneva.full(Y)
    m_full = np.sum(Y_full)
    e = abs(m - m_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # 1.3956219
    # [1. 1. 1. 1. 1. 1.]
    # 89.3198
    # Error     : 2.67e-04
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    m, p = teneva.sum_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # 1.654193 -2525.0
    # 




|
|


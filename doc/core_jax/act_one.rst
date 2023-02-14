Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva.core_jax.act_one


-----




|
|

.. autofunction:: teneva.core_jax.act_one.convert

  **Examples**:

  .. code-block:: python

    import numpy as onp

  Let build jax TT-tensor and convert it to numpy (base) version:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(6, 5, 4, key)
    teneva.show(Y)
    
    print('Is jax   : ', isinstance(Y[0], np.ndarray))
    print('Is numpy : ', isinstance(Y[0], onp.ndarray))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Is jax   :  True
    # Is numpy :  False
    # 

  .. code-block:: python

    Y_base = teneva.convert(Y)
    teneva_base.show(Y_base)
    
    print('Is jax   : ', isinstance(Y_base[0], np.ndarray))
    print('Is numpy : ', isinstance(Y_base[0], onp.ndarray))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     6D : |5| |5| |5| |5| |5| |5|
    # <rank>  =    4.0 :   \4/ \4/ \4/ \4/ \4/
    # Is jax   :  False
    # Is numpy :  True
    # 

  And now let convert the numpy (base) TT-tensor back into jax format:

  .. code-block:: python

    Z = teneva.convert(Y_base)
    teneva.show(Z)
    
    # Check that it is the same:
    e = np.max(np.abs(teneva.full(Y) - teneva.full(Z)))
    
    print('Is jax   : ', isinstance(Z[0], np.ndarray))
    print('Is numpy : ', isinstance(Z[0], onp.ndarray))
    print('Error    : ', e)   

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Is jax   :  True
    # Is numpy :  False
    # Error    :  0.0
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.copy

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with mode size 4 and TT-rank 12:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(10, 9, 7, key)
    
    Z = teneva.copy(Y) # The copy of Y  
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # 0.19049883
    # 0.19049883
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

    # m: 1.0e+00 | T1 :   0.0624 | T2 :   0.0596
    # m: 1.0e+01 | T1 :   0.0949 | T2 :   0.0972
    # m: 1.0e+02 | T1 :   0.1019 | T2 :   0.0962
    # m: 1.0e+03 | T1 :   0.1455 | T2 :   0.1498
    # m: 1.0e+04 | T1 :   0.4869 | T2 :   0.4776
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

    # -1.1911157 792.0
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.interface_ltr

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zm, zr = teneva.interface_ltr(Y)
    
    for z in zm:
        print(z)
    print(zr)

    # >>> ----------------------------------------
    # >>> Output:

    # [ 0.6159539   0.05393152 -0.59480584 -0.51371026]
    # [-0.09011538  0.28736737  0.9508705   0.07172485]
    # [-0.10756405 -0.65141314 -0.5013524   0.55922854]
    # [ 0.41511503 -0.6911681  -0.5902755   0.03925423]
    # [-0.66395193 -0.09679152 -0.04487634 -0.74012524]
    # [ 0.17707978 -0.15525422 -0.07720309  0.96880263]
    # [ 0.17707978 -0.15525422 -0.07720309  0.96880263]
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.interface_rtl

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zl, zm = teneva.interface_rtl(Y)
    
    print(zl)
    for z in zm:
        print(z)

    # >>> ----------------------------------------
    # >>> Output:

    # [-0.95776975  0.17451975  0.18542242 -0.13356045]
    # [-0.20214106 -0.9513638   0.19919026 -0.11987165]
    # [ 0.34267044  0.69565064 -0.52662045  0.34830743]
    # [ 0.2049764   0.15211944  0.96648353 -0.02745909]
    # [ 0.03999171 -0.23093204 -0.9616975  -0.14215827]
    # [ 0.8828743  -0.3793065   0.2701475   0.06066062]
    # [-0.4844946   0.3101102   0.69839716 -0.42583805]
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

    # Error     : 2.61e-08
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

    # 1.7866555
    # [-2. -1. -3. -1.  0. -3.]
    # 0.0017447808
    # Error     : 5.82e-09
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    m, p = teneva.mean_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # 1.968759 -2512.0
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

    # Error     : 2.29e-04
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

    # -1.7985754
    # [1. 1. 1. 0. 1. 1.]
    # -57.554413
    # Error     : 8.77e-05
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    m, p = teneva.sum_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # 1.4154698 -2540.0
    # 




|
|


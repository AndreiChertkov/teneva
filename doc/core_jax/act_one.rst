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

  Let compare this function with numpy realization:

  .. code-block:: python

    Y1_base = teneva.convert(Y1) # Convert tensor to numpy version
    y1_base = teneva_base.get(Y1_base, k)
    
    print(y1)
    print(y1_base)

    # >>> ----------------------------------------
    # >>> Output:

    # -0.83907104
    # -0.8390711
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.get_log

  **Examples**:

  .. code-block:: python

    d = 6  # Dimension of the tensor
    n = 5  # Mode size of the tensor
    r = 2  # Rank of the tensor
    
    # Construct random d-dim non-negative TT-tensor:
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    Y = teneva.mul(Y, Y)
    
    # Print the TT-tensor:
    teneva.show(Y)
    
    # Compute the full tensor from the TT-tensor:  
    Y0 = teneva.full(Y)
    
    # Select some tensor element and compute the value:
    k = np.array([3, 1, 2, 1, 0, 4])
    y1 = teneva.get_log(Y, k)
    
    # Compute the same element of the original tensor:
    y0 = np.log(Y0[tuple(k)])
    
    # Compare values:
    e = np.abs(y1-y0)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # Error : 4.8e-06
    # 

  We may also use vmap and jit for this function:

  .. code-block:: python

    d = 10   # Dimension of the tensor
    n = 10   # Mode size of the tensor
    r = 3    # Rank of the tensor
    m = 1000 # Batch size
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)
    Y = teneva.mul(Y, Y)
    
    rng, key = jax.random.split(rng)
    K = teneva.sample_lhs(d, n, m, key)
    
    get_log = jax.vmap(jax.jit(teneva.get_log), (None, 0))
    y = get_log(Y, K)
    print(y[:2])

    # >>> ----------------------------------------
    # >>> Output:

    # [-5.508591  0.88134 ]
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

    # m: 1.0e+00 | T1 :   0.0613 | T2 :   0.0596
    # m: 1.0e+01 | T1 :   0.0853 | T2 :   0.0862
    # m: 1.0e+02 | T1 :   0.0964 | T2 :   0.0981
    # m: 1.0e+03 | T1 :   0.1682 | T2 :   0.1493
    # m: 1.0e+04 | T1 :   0.4604 | T2 :   0.4519
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

.. autofunction:: teneva.core_jax.act_one.grad

  **Examples**:

  .. code-block:: python

    l = 1.E-4   # Learning rate
    d = 5       # Dimension of the tensor
    n = 4       # Mode size of the tensor
    r = 2       # Rank of the tensor
    
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key=key)
    
    # Targer multi-index for gradient:
    i = np.array([0, 1, 2, 3, 0])
    
    y = teneva.get(Y, i)
    dY = teneva.grad(Y, i)

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    y_base, dY_base = teneva_base.get_and_grad(Y_base, i)
    dY_base = [G[:, k, :] for G, k in zip(dY_base, i)]
    dY_base = [dY_base[0], np.array(dY_base[1:-1]), dY_base[-1]]
    print('Error : ', np.max(np.array([np.max(np.abs(g-g_base)) for g, g_base in zip(dY, dY_base)])))

    # >>> ----------------------------------------
    # >>> Output:

    # Error :  2.9802322e-08
    # 

  Let apply the gradient:

  .. code-block:: python

    Z = teneva.copy(Y) # TODO
    Z[0] = Z[0].at[:, i[0], :].add(-l * dY[0])
    for k in range(1, d-1):
        Z[1] = Z[1].at[k-1, :, i[k], :].add(-l * dY[1][k-1])
    Z[2] = Z[2].at[:, i[d-1], :].add(-l * dY[2])
    
    z = teneva.get(Z, i)
    e = np.max(np.abs(teneva.full(Y) - teneva.full(Z)))
    
    print(f'Old value at multi-index : {y:-12.5e}')
    print(f'New value at multi-index : {z:-12.5e}')
    print(f'Difference for tensors   : {e:-12.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Old value at multi-index :  3.88889e-01
    # New value at multi-index :  3.88468e-01
    # Difference for tensors   :      4.2e-04
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
    # [-0.4625181   0.40195844 -0.76607376 -0.19400343]
    # [-0.38513428 -0.07560533 -0.8746828   0.28440356]
    # [-0.81773984  0.2169487   0.16309422  0.5075776 ]
    # [ 0.44751287 -0.7911021  -0.34725943  0.23086922]
    # [ 0.5872935   0.5877588  -0.24633797 -0.49894252]
    # [-0.6880904   0.66989774  0.22775562  0.16092257]
    # 

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    phi_l = teneva_base.interface(Y_base, ltr=True)
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # [1.]
    # [ 0.61595389  0.05393152 -0.59480584 -0.51371024]
    # [-0.46251811  0.4019585  -0.76607378 -0.19400342]
    # [-0.38513432 -0.07560532 -0.87468279  0.28440359]
    # [-0.81773988  0.21694874  0.16309422  0.50757759]
    # [ 0.44751292 -0.79110206 -0.34725933  0.23086941]
    # [ 0.58729329  0.58775886 -0.24633791 -0.49894264]
    # [-0.68809042  0.6698976   0.2277557   0.16092271]
    # [-1.]
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

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    phi_r = teneva_base.interface(Y_base, ltr=False)
    for phi in phi_r:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # [1.]
    # [-0.95776981  0.1745197   0.18542233 -0.13356056]
    # [-0.20214109 -0.95136373  0.19919037 -0.11987174]
    # [ 0.34267048  0.69565067 -0.52662039  0.34830741]
    # [ 0.20497638  0.15211939  0.96648351 -0.02745911]
    # [ 0.0399917 -0.230932  -0.9616975 -0.1421583]
    # [ 0.88287439 -0.3793065   0.2701475   0.06066064]
    # [-0.4844946   0.31011021  0.69839716 -0.42583805]
    # [1.]
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

.. autofunction:: teneva.core_jax.act_one.norm

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r = 3   # TT-rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)

  .. code-block:: python

    v = teneva.norm(Y)  # Compute the Frobenius norm
    print(v)            # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [58.251373]
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)
    
    e = abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}') 

    # >>> ----------------------------------------
    # >>> Output:

    # 58.25131
    # Error     : 1.11e-06
    # 




|
|

.. autofunction:: teneva.core_jax.act_one.norm_stab

  **Examples**:

  .. code-block:: python

    d = 5   # Dimension of the tensor
    n = 6   # Mode size of the tensor
    r = 3   # Rank of the tensor

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)

  .. code-block:: python

    v, p = teneva.norm_stab(Y) # Compute the Frobenius norm
    print(v) # Print the scaled value
    print(p) # Print the scale factors
    
    v = v * 2**np.sum(p) # Resulting value
    print(v)   # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # [1.0796704]
    # [0.5 1.  1.  1.  1.5]
    # [34.549454]
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)
    
    e = abs((v - v_full)/v_full).item()
    print(f'Error     : {e:-8.2e}') 

    # >>> ----------------------------------------
    # >>> Output:

    # 34.549458
    # Error     : 1.10e-07
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

    # Error     : 8.77e-05
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

    # -1.4871672
    # [ 0.  2.  2.  1.  1. -2.]
    # -23.794676
    # Error     : 6.10e-05
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    m, p = teneva.sum_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.108925 -2526.0
    # 




|
|


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

    # m: 1.0e+00 | T1 :   0.0661 | T2 :   0.0627
    # m: 1.0e+01 | T1 :   0.0935 | T2 :   0.0884
    # m: 1.0e+02 | T1 :   0.0928 | T2 :   0.1028
    # m: 1.0e+03 | T1 :   0.1571 | T2 :   0.1447
    # m: 1.0e+04 | T1 :   0.4959 | T2 :   0.5186
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

    # Error :  5.9604645e-08
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

    # Old value at multi-index :  2.11916e-01
    # New value at multi-index :  2.11819e-01
    # Difference for tensors   :      1.7e-04
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

    # [-0.46863997  0.5238826  -0.04652507 -0.7097598 ]
    # [-0.3369044  -0.22432232  0.8403437  -0.3605514 ]
    # [ 0.20289943  0.80434436  0.54934067 -0.10043337]
    # [-0.7899511  -0.04141854  0.5076321  -0.34142572]
    # [ 0.0697167   0.09638917 -0.20285943 -0.9719551 ]
    # [ 0.37379828 -0.00510346 -0.5085104  -0.7756713 ]
    # [ 0.47192457 -0.48247743  0.6201052  -0.3999652 ]
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
    # [-0.46863996  0.52388265 -0.04652507 -0.7097598 ]
    # [-0.33690442 -0.22432231  0.84034375 -0.36055138]
    # [ 0.20289945  0.80434431  0.54934069 -0.10043337]
    # [-0.7899511  -0.04141856  0.50763206 -0.34142562]
    # [ 0.06971664  0.09638914 -0.2028594  -0.97195514]
    # [ 0.37379827 -0.00510348 -0.50851045 -0.77567128]
    # [ 0.47192461 -0.48247745  0.62010522 -0.39996525]
    # [1.]
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

    # [ 0.26914886  0.32372084  0.86451733 -0.27454227]
    # [-0.44757462  0.32706285  0.83166337 -0.03229539]
    # [ 0.07637926 -0.8496607   0.2919713  -0.4324299 ]
    # [ 0.57197976 -0.48798612 -0.15620424  0.6405536 ]
    # [-0.31179526 -0.57576454 -0.01529724 -0.7556752 ]
    # [ 0.2509977   0.8163165   0.04465267 -0.51829886]
    # [ 0.49726936  0.15992026 -0.22335434  0.822959  ]
    # 

  Let compare this function with numpy (base) realization:

  .. code-block:: python

    Y_base = teneva.convert(Y) # Convert it to numpy version
    phi_r = teneva_base.interface(Y_base, ltr=False)
    for phi in phi_r:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # [-1.]
    # [ 0.26914881  0.32372084  0.86451738 -0.27454224]
    # [-0.44757464  0.32706281  0.83166331 -0.03229539]
    # [ 0.07637926 -0.84966072  0.29197133 -0.43242989]
    # [ 0.57197979 -0.48798613 -0.15620422  0.64055359]
    # [-0.31179527 -0.57576449 -0.01529725 -0.75567517]
    # [ 0.2509977   0.81631648  0.04465267 -0.51829886]
    # [ 0.49726936  0.15992026 -0.22335435  0.82295901]
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

    # Error     : 1.86e-09
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

    # 1.4634157
    # [-2. -1. -1. -1. -2. -1.]
    # 0.0057164677
    # Error     : 1.68e-08
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)
    m, p = teneva.mean_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # 1.6542045 -2525.0
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

    # [34.549458]
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
    # Error     : 0.00e+00
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

    # [1.3547028]
    # [0.  1.5 1.  1.5 1. ]
    # [43.35049]
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

    # 43.350483
    # Error     : 1.76e-07
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

    # Error     : 1.22e-04
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

    # -1.0640556
    # [ 1.  1.  0. -1.  2.  2.]
    # -34.049778
    # Error     : 2.56e-04
    # 

  We can check it also for big random tensor:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key, a=-0.01, b=+0.01)
    m, p = teneva.sum_stab(Y)
    print(m, np.sum(p))

    # >>> ----------------------------------------
    # >>> Output:

    # -1.3892597 -2537.0
    # 




|
|


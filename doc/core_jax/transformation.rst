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

    # TT-tensor-jax | d =     5 | n =     6 | r =     4 |
    # Error : 9.7e-08
    # 




|
|

.. autofunction:: teneva.core_jax.transformation.orthogonalize_rtl

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d=7, n=4, r=3, key=key)
    Z = teneva.orthogonalize_rtl(Y)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     7 | n =     4 | r =     3 |
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    Y_full = teneva.full(Y)
    Z_full = teneva.full(Z)
    e = np.max(np.abs(Y_full - Z_full))
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 1.22e-04
    # 

  And we can make sure that all TT-cores, except the first one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    Zl, Zm, Zr = Z
    
    v = [Zl[:, j, :] @ Zl[:, j, :].T for j in range(Zl.shape[1])]
    print(np.sum(np.array(v), axis=0))
    
    for G in Zm:
        v = [G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]
        print(np.sum(np.array(v), axis=0))
        
    v = [Zr[:, j, :] @ Zr[:, j, :].T for j in range(Zr.shape[1])]
    print(np.sum(np.array(v), axis=0))

    # >>> ----------------------------------------
    # >>> Output:

    # [[4379525.]]
    # [[ 9.9999988e-01 -1.4901161e-08  3.9115548e-08]
    #  [-1.4901161e-08  1.0000000e+00 -3.7252903e-09]
    #  [ 3.9115548e-08 -3.7252903e-09  1.0000001e+00]]
    # [[ 9.9999994e-01 -1.1455268e-07  1.4901161e-08]
    #  [-1.1455268e-07  9.9999994e-01 -2.9802322e-08]
    #  [ 1.4901161e-08 -2.9802322e-08  1.0000002e+00]]
    # [[ 9.9999994e-01  7.4505806e-09 -3.4226105e-08]
    #  [ 7.4505806e-09  1.0000000e+00 -3.5390258e-08]
    #  [-3.4226105e-08 -3.5390258e-08  9.9999994e-01]]
    # [[9.9999988e-01 5.9604645e-08 2.9802322e-08]
    #  [5.9604645e-08 1.0000002e+00 3.7252903e-08]
    #  [2.9802322e-08 3.7252903e-08 9.9999976e-01]]
    # [[ 9.9999994e-01  3.7252903e-08 -1.4901161e-08]
    #  [ 3.7252903e-08  1.0000001e+00 -1.1175871e-08]
    #  [-1.4901161e-08 -1.1175871e-08  9.9999994e-01]]
    # [[ 1.0000001e+00 -5.2154064e-08  2.9802322e-08]
    #  [-5.2154064e-08  1.0000001e+00 -2.9802322e-08]
    #  [ 2.9802322e-08 -2.9802322e-08  1.0000001e+00]]
    # 




|
|

.. autofunction:: teneva.core_jax.transformation.orthogonalize_rtl_stab

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d=7, n=4, r=3, key=key)
    Z_stab, p_stab = teneva.orthogonalize_rtl_stab(Y)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     7D (shape =     4; rank =     3)
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    Z = teneva.copy(Z_stab)
    Z[0] *= 2**np.sum(zp)
    
    Y_full = teneva.full(Y)
    Z_full = teneva.full(Z)
    e = np.max(np.abs(Y_full - Z_full))
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 1.53e-04
    # 

  .. code-block:: python

    Zl, Zm, Zr = Z_stab
    
    v = [Zl[:, j, :] @ Zl[:, j, :].T for j in range(Zl.shape[1])]
    print(np.sum(np.array(v), axis=0))
    
    for G in Zm:
        v = [G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]
        print(np.sum(np.array(v), axis=0))
        
    v = [Zr[:, j, :] @ Zr[:, j, :].T for j in range(Zr.shape[1])]
    print(np.sum(np.array(v), axis=0))

    # >>> ----------------------------------------
    # >>> Output:

    # [[5.2260284]]
    # [[ 9.9999994e-01 -1.1874363e-08 -4.4703484e-08]
    #  [-1.1874363e-08  9.9999988e-01  0.0000000e+00]
    #  [-4.4703484e-08  0.0000000e+00  9.9999994e-01]]
    # [[ 9.9999988e-01  0.0000000e+00  2.2351742e-08]
    #  [ 0.0000000e+00  9.9999976e-01 -2.9802322e-08]
    #  [ 2.2351742e-08 -2.9802322e-08  9.9999976e-01]]
    # [[ 1.0000000e+00  5.4715201e-09 -2.9802322e-08]
    #  [ 5.4715201e-09  1.0000001e+00  7.8231096e-08]
    #  [-2.9802322e-08  7.8231096e-08  1.0000000e+00]]
    # [[ 1.0000001e+00  1.4901161e-08 -5.9604645e-08]
    #  [ 1.4901161e-08  1.0000000e+00  0.0000000e+00]
    #  [-5.9604645e-08  0.0000000e+00  9.9999976e-01]]
    # [[ 9.9999982e-01  2.9802322e-08  7.4505806e-09]
    #  [ 2.9802322e-08  1.0000000e+00 -1.4901161e-08]
    #  [ 7.4505806e-09 -1.4901161e-08  9.9999976e-01]]
    # [[ 1.0000000e+00 -1.0617077e-07 -1.4901161e-08]
    #  [-1.0617077e-07  9.9999976e-01 -2.9802322e-08]
    #  [-1.4901161e-08 -2.9802322e-08  1.0000000e+00]]
    # 




|
|


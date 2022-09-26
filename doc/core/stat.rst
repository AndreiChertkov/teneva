Module stat: helper functions for processing statistics
-------------------------------------------------------


.. automodule:: teneva.core.stat


-----


.. autofunction:: teneva.cdf_confidence

  **Examples**:

  .. code-block:: python

    # Statistical points:
    points = np.random.randn(15)
    
    # Compute the confidence:
    cdf_min, cdf_max = teneva.cdf_confidence(points)
    for p, c_min, c_max in zip(points, cdf_min, cdf_max):
        print(f'{p:-8.4f} | {c_min:-8.4f} | {c_max:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    #   0.4967 |   0.1461 |   0.8474
    #  -0.1383 |   0.0000 |   0.2124
    #   0.6477 |   0.2970 |   0.9983
    #   1.5230 |   1.0000 |   1.0000
    #  -0.2342 |   0.0000 |   0.1165
    #  -0.2341 |   0.0000 |   0.1165
    #   1.5792 |   1.0000 |   1.0000
    #   0.7674 |   0.4168 |   1.0000
    #  -0.4695 |   0.0000 |   0.0000
    #   0.5426 |   0.1919 |   0.8932
    #  -0.4634 |   0.0000 |   0.0000
    #  -0.4657 |   0.0000 |   0.0000
    #   0.2420 |   0.0000 |   0.5926
    #  -1.9133 |   0.0000 |   0.0000
    #  -1.7249 |   0.0000 |   0.0000
    # 


.. autofunction:: teneva.cdf_getter

  **Examples**:

  .. code-block:: python

    # Statistical points:
    x = np.random.randn(1000)
    
    # Build the CDF getter:
    cdf = teneva.cdf_getter(x)

  .. code-block:: python

    z = -9999  # Point for CDF computations
    
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # 0.0
    # 

  .. code-block:: python

    z = +9999  # Point for CDF computations
    
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0
    # 

  .. code-block:: python

    # Several points for CDF computations:
    z = [-10000, -10, -1, 0, 100]
    
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # array([0.   , 0.   , 0.145, 0.485, 1.   ])
    # 


.. autofunction:: teneva.sample_ind_rand

  **Examples**:

  .. code-block:: python

    Y = np.array([       # We generate 2D tensor for demo
        [0.1, 0.2, 0.3],
        [0. , 0. , 0. ],
        [0.2, 0.2, 0. ],
        [0. , 0. , 0. ],
    ])
    Y = teneva.svd(Y)    # We compute its TT-representation
    print(teneva.sum(Y)) # We print the sum of tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000000000002
    # 

  .. code-block:: python

    m = 3 # Number of requested samples
    I = teneva.sample_ind_rand(Y, m)
    
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 0] 0.10000000000000003
    # [0 1] 0.19999999999999993
    # [0 2] 0.3000000000000001
    # 

  We may also generate multi-indices with repeats:

  .. code-block:: python

    m = 10
    I = teneva.sample_ind_rand(Y, m, unique=False)
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [2 0] 0.20000000000000004
    # [2 0] 0.20000000000000004
    # [2 0] 0.20000000000000004
    # [0 1] 0.19999999999999993
    # [2 0] 0.20000000000000004
    # [0 2] 0.3000000000000001
    # [0 2] 0.3000000000000001
    # [0 0] 0.10000000000000003
    # [2 1] 0.19999999999999998
    # [2 1] 0.19999999999999998
    # 

  And now let check this function for big random TT-tensor:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 5:
    Y = teneva.tensor_rand([4]*5, 5)
    
    # Compute the square of Y:
    Y = teneva.mul(Y, Y)
    
    # Normalize the tensor:
    p = teneva.sum(Y)
    Y = teneva.mul(Y, 1./p)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)
    
    I = teneva.sample_ind_rand(Y, m=10)
    
    print('\n--- Result:')
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4|  |4|  |4|  |4|  |4|
    # <rank>  =   25.0 :   \25/ \25/ \25/ \25/
    # 
    # --- Result:
    # [2 0 3 1 0] 0.0017819542111315733
    # [2 0 3 1 3] 0.036792827938648416
    # [2 0 3 2 0] 0.0035760224198429177
    # [2 0 3 2 2] 0.04238010233786789
    # [2 0 3 3 2] 0.016255410950474433
    # [2 0 3 3 3] 0.022511352864702686
    # [3 1 0 1 3] 0.018721237013247793
    # [3 1 1 3 3] 0.011129383148418694
    # [3 2 3 1 3] 0.019752890888528732
    # [3 2 3 2 2] 0.025114732463777094
    # 



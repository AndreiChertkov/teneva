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

    # [2 0] 0.20000000000000004
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

    # [0 2] 0.3000000000000001
    # [0 2] 0.3000000000000001
    # [0 2] 0.3000000000000001
    # [0 2] 0.3000000000000001
    # [0 1] 0.19999999999999993
    # [0 2] 0.3000000000000001
    # [2 0] 0.20000000000000004
    # [2 1] 0.19999999999999998
    # [2 1] 0.19999999999999998
    # [0 2] 0.3000000000000001
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
    # [1 2 1 2 0] 0.034088829237965905
    # [0 2 0 1 1] 0.006165874517144947
    # [3 2 0 1 0] 0.0032721436587474792
    # [0 2 1 2 0] 0.010021111432564143
    # [3 2 3 2 1] 0.008131246814427893
    # [1 2 1 0 3] 0.006175679712646391
    # [1 2 1 2 1] 0.02220029134708088
    # [0 2 3 2 1] 0.007693077224856762
    # [3 2 1 1 1] 0.009168314901172975
    # [3 2 2 0 1] 0.019823516988107307
    # 



Module sample: random sampling for/from the TT-tensor
-----------------------------------------------------


.. automodule:: teneva.sample


-----




|
|

.. autofunction:: teneva.sample.sample

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
    I = teneva.sample(Y, m)
    
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 2] 0.30000000000000004
    # [2 0] 0.20000000000000012
    # [2 0] 0.20000000000000012
    # 

  And now let check this function for big random TT-tensor:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 5:
    Y = teneva.rand([4]*5, 5)
    
    # Compute the square of Y:
    Y = teneva.mul(Y, Y)
    
    # Normalize the tensor:
    p = teneva.sum(Y)
    Y = teneva.mul(Y, 1./p)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)
    
    I = teneva.sample(Y, m=10)
    
    print('\n--- Result:')
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4|  |4|  |4|  |4|  |4|
    # <rank>  =   25.0 :   \25/ \25/ \25/ \25/
    # 
    # --- Result:
    # [1 0 3 3 1] 0.0043818698268434046
    # [0 2 3 0 1] 0.0013502182372207051
    # [1 3 0 3 1] 0.005326614808275069
    # [1 1 0 0 2] 0.01908349282834068
    # [1 1 0 0 2] 0.01908349282834068
    # [3 2 0 2 1] 0.001117893228468234
    # [1 1 0 2 2] 0.01064118457402128
    # [0 2 2 3 1] 0.021937081751779743
    # [0 2 0 3 1] 0.011157886814384966
    # [1 1 1 3 2] 0.00383177829736998
    # 




|
|

.. autofunction:: teneva.sample.sample_square

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
    I = teneva.sample_square(Y, m)
    
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 1] 0.19999999999999993
    # [0 0] 0.1
    # [2 0] 0.20000000000000012
    # 

  We may also generate multi-indices with repeats:

  .. code-block:: python

    m = 10
    I = teneva.sample_square(Y, m, unique=False)
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 2] 0.30000000000000004
    # [0 1] 0.19999999999999993
    # [0 1] 0.19999999999999993
    # [2 1] 0.19999999999999998
    # [0 2] 0.30000000000000004
    # [0 2] 0.30000000000000004
    # [0 2] 0.30000000000000004
    # [2 1] 0.19999999999999998
    # [0 2] 0.30000000000000004
    # [2 1] 0.19999999999999998
    # 

  And now let check this function for big random TT-tensor:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 5:
    Y = teneva.rand([4]*5, 5)
    
    # Compute the square of Y:
    Y = teneva.mul(Y, Y)
    
    # Normalize the tensor:
    p = teneva.sum(Y)
    Y = teneva.mul(Y, 1./p)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)
    
    I = teneva.sample_square(Y, m=10)
    
    print('\n--- Result:')
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4|  |4|  |4|  |4|  |4|
    # <rank>  =   25.0 :   \25/ \25/ \25/ \25/
    # 
    # --- Result:
    # [1 0 1 2 2] 0.006794672157333136
    # [1 1 0 1 1] 0.004450109674681529
    # [2 0 2 0 1] 0.007509342017359113
    # [1 0 2 0 1] 0.003448569920689505
    # [2 1 1 0 0] 0.0022855464248411166
    # [3 3 1 0 0] 0.002569138931693821
    # [0 1 1 2 0] 0.004202312004487534
    # [1 2 2 1 3] 0.005067429232471865
    # [0 2 3 0 2] 0.006373841841063713
    # [0 1 3 1 3] 0.00399751823255496
    # 




|
|

.. autofunction:: teneva.sample.sample_lhs

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    n = [5] * d     # Shape of the tensor/grid
    m = 8           # Number of samples
    
    I = teneva.sample_lhs(n, m)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 0 3]
    #  [3 4 2]
    #  [0 4 1]
    #  [1 2 4]
    #  [4 3 4]
    #  [3 3 0]
    #  [4 2 3]
    #  [2 1 1]]
    # 

  Note that we can set the random seed value also:

  .. code-block:: python

    I = teneva.sample_lhs([3, 4], 3, seed=42)
    print(I)
    I = teneva.sample_lhs([3, 4], 3, seed=0)
    print(I)
    I = teneva.sample_lhs([3, 4], 3, 42)
    print(I)
    I = teneva.sample_lhs([3, 4], 3, seed=np.random.default_rng(42))
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[2 2]
    #  [1 1]
    #  [0 3]]
    # [[2 3]
    #  [0 0]
    #  [1 2]]
    # [[2 2]
    #  [1 1]
    #  [0 3]]
    # [[2 2]
    #  [1 1]
    #  [0 3]]
    # 




|
|

.. autofunction:: teneva.sample.sample_rand

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    n = [5] * d     # Shape of the tensor/grid
    m = 8           # Number of samples
    
    I = teneva.sample_rand(n, m)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 1 2]
    #  [3 0 0]
    #  [3 2 4]
    #  [2 4 2]
    #  [2 3 2]
    #  [4 3 1]
    #  [0 3 0]
    #  [3 3 4]]
    # 

  Note that we can set the random seed value also:

  .. code-block:: python

    I = teneva.sample_rand([3, 4], 3, seed=42)
    print(I)
    I = teneva.sample_rand([3, 4], 3, seed=0)
    print(I)
    I = teneva.sample_rand([3, 4], 3, 42)
    print(I)
    I = teneva.sample_rand([3, 4], 3, seed=np.random.default_rng(42))
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 1]
    #  [2 1]
    #  [1 3]]
    # [[2 1]
    #  [1 1]
    #  [1 0]]
    # [[0 1]
    #  [2 1]
    #  [1 3]]
    # [[0 1]
    #  [2 1]
    #  [1 3]]
    # 




|
|

.. autofunction:: teneva.sample.sample_tt

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    n = [5] * d     # Shape of the tensor/grid
    m = 2           # The expected TT-rank
    
    I, idx, idx_many = teneva.sample_tt(n, m)
    
    print(I.shape)
    print(idx.shape)
    print(idx_many.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (40, 3)
    # (4,)
    # (3,)
    # 




|
|


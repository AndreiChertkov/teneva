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
    # [0 0] 0.1
    # [0 1] 0.19999999999999993
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
    # [1 3 0 0 3] 0.0005467688008461928
    # [2 2 3 3 0] 0.006627906923442645
    # [1 1 1 1 0] 0.0017246632716313237
    # [3 2 1 2 0] 0.0008373295031647166
    # [1 3 1 3 3] 0.00037274612504040714
    # [0 0 2 1 3] 0.0039011799572751522
    # [1 2 0 0 2] 0.003242585201425543
    # [1 2 0 0 0] 0.00931349693960367
    # [3 1 2 2 0] 0.0011575363014772284
    # [1 2 3 0 3] 0.03366467522253799
    # 

  Note that we can also set a random seed value:

  .. code-block:: python

    Y = teneva.rand([4]*5, 5)
    Y = teneva.mul(Y, Y)
    Y = teneva.mul(Y, 1./teneva.sum(Y))
    I = teneva.sample(Y, m=10, seed=42)
    
    print('\n--- Result:')
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # --- Result:
    # [3 0 2 2 2] 0.014619862447557261
    # [2 3 1 3 2] 0.0011271606287902576
    # [3 1 3 1 1] 0.00031715064859554894
    # [3 2 3 1 1] 0.007394218666406085
    # [0 1 2 1 3] 0.0047701130856797995
    # [3 0 0 0 2] 0.00788182131582216
    # [3 1 1 0 1] 0.0016185689029685083
    # [3 0 0 2 0] 0.0021319553328506173
    # [0 3 0 0 3] 0.002736837128643397
    # [2 2 3 2 0] 0.0006140683439889205
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
    # [2 0] 0.20000000000000012
    # [0 0] 0.1
    # 

  .. code-block:: python

    m = 10
    I = teneva.sample_square(Y, m, unique=False)
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 2] 0.30000000000000004
    # [2 1] 0.19999999999999998
    # [2 0] 0.20000000000000012
    # [2 1] 0.19999999999999998
    # [2 1] 0.19999999999999998
    # [0 1] 0.19999999999999993
    # [0 1] 0.19999999999999993
    # [0 2] 0.30000000000000004
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
    # [3 1 2 3 0] 0.0014598207440115493
    # [1 2 1 1 1] 0.0038227446850823184
    # [3 0 1 2 0] 0.002431550307598073
    # [3 2 3 0 0] 0.014927727284710889
    # [1 2 3 0 1] 0.010419554168995405
    # [3 3 2 0 2] 0.005610355321104711
    # [1 3 0 2 0] 0.00813428871624301
    # [1 0 3 3 1] 0.0059010351201360806
    # [1 0 0 0 2] 0.002413513966839127
    # [1 2 3 0 2] 0.006713745601688902
    # 

  Note that we can also set a random seed value:

  .. code-block:: python

    Y = teneva.rand([4]*5, 5)
    Y = teneva.mul(Y, Y)
    Y = teneva.mul(Y, 1./teneva.sum(Y))
    I = teneva.sample_square(Y, m=10, seed=42)
    
    print('\n--- Result:')
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # --- Result:
    # [0 2 3 0 0] 0.018207262948340554
    # [0 2 2 0 1] 0.012200195167159356
    # [0 1 0 1 2] 0.0010835371784504821
    # [2 2 2 3 3] 0.013183247314231279
    # [0 2 1 3 1] 0.0029869590066619453
    # [2 1 3 2 0] 0.0019840125638689735
    # [2 1 1 3 2] 0.0009026844205923414
    # [2 0 2 1 3] 0.0008211322913717347
    # [2 2 2 0 0] 0.03449546103487785
    # [3 2 3 2 1] 0.001954351929649005
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

    # [[0 4 3]
    #  [2 1 1]
    #  [3 3 0]
    #  [4 2 1]
    #  [0 3 2]
    #  [4 2 4]
    #  [1 4 0]
    #  [2 0 3]]
    # 

  Note that we can also set a random seed value:

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

    # [[0 1 4]
    #  [0 4 3]
    #  [4 2 0]
    #  [3 2 4]
    #  [4 2 2]
    #  [0 1 3]
    #  [4 2 4]
    #  [4 1 4]]
    # 

  Note that we can also set a random seed value:

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

  Note that we can also set a random seed value:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    n = [5] * d     # Shape of the tensor/grid
    m = 2           # The expected TT-rank
    
    I, idx, idx_many = teneva.sample_tt(n, m, seed=42)
    
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


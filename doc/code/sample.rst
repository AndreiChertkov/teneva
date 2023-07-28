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

    # [2 1] 0.19999999999999998
    # [0 0] 0.1
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
    # [3 2 3 3 3] 0.010639344526473096
    # [2 3 1 3 3] 0.01609006806646322
    # [3 2 3 1 3] 0.004617820281068656
    # [2 3 2 0 2] 0.009130960046487956
    # [0 1 2 1 3] 0.0002672009173486632
    # [3 0 0 0 3] 0.0017268323022795688
    # [3 2 2 1 0] 0.002871965728257122
    # [3 0 0 2 1] 0.001582995845371316
    # [1 3 0 0 2] 0.003428035968489353
    # [2 3 2 2 0] 0.003872785554759354
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
    # [3 0 2 3 2] 0.004826459330111369
    # [2 3 1 3 2] 0.005509947742579449
    # [3 2 3 0 3] 0.001571199616084186
    # [2 3 3 0 1] 0.006887102818501732
    # [0 2 2 1 2] 0.0014199710711856265
    # [3 0 0 0 3] 0.005972717330588275
    # [3 1 1 0 1] 0.0025027137121134617
    # [3 0 0 0 1] 0.010819263132409041
    # [0 3 0 0 2] 0.005087334274275927
    # [2 2 2 1 1] 0.000984958168111335
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
    # [0 2] 0.30000000000000004
    # 

  We may also generate multi-indices with repeats:

  .. code-block:: python

    m = 10
    I = teneva.sample_square(Y, m, unique=False)
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [0 0] 0.1
    # [0 2] 0.30000000000000004
    # [0 2] 0.30000000000000004
    # [2 1] 0.19999999999999998
    # [2 0] 0.20000000000000012
    # [2 1] 0.19999999999999998
    # [0 1] 0.19999999999999993
    # [2 0] 0.20000000000000012
    # [2 0] 0.20000000000000012
    # [0 2] 0.30000000000000004
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
    # [3 3 0 1 1] 0.004134279846409279
    # [2 0 1 0 1] 0.008187765618947818
    # [3 3 0 1 0] 0.009760899526764417
    # [3 0 1 0 0] 0.008878223130488012
    # [0 2 1 1 0] 0.011130758226864677
    # [1 1 0 1 0] 0.002712013561471569
    # [3 2 1 1 0] 0.009778037273774277
    # [1 3 1 1 0] 0.020951495468461447
    # [1 3 1 0 0] 0.013765490389836495
    # [2 0 1 0 0] 0.01529832123173697
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

    # [[0 1 2]
    #  [3 0 0]
    #  [3 2 4]
    #  [2 4 2]
    #  [2 3 2]
    #  [4 3 1]
    #  [0 3 0]
    #  [3 3 4]]
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


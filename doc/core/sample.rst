Module sample: random sampling for/from the TT-tensor
-----------------------------------------------------


.. automodule:: teneva.core.sample


-----




|
|

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




|
|

.. autofunction:: teneva.sample_lhs

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    n = [5] * d     # Shape of the tensor/grid
    m = 8           # Number of samples
    
    I = teneva.sample_lhs(n, m)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 3 0]
    #  [0 2 4]
    #  [3 3 2]
    #  [1 0 3]
    #  [1 2 0]
    #  [3 0 1]
    #  [4 4 1]
    #  [2 1 4]]
    # 




|
|

.. autofunction:: teneva.sample_tt

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


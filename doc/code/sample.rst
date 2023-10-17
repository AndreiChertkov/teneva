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
    # [1 1 0 0 0] 0.0007588471820855217
    # [2 1 1 3 0] 0.002706677398128092
    # [1 0 2 3 0] 0.0013416595874856504
    # [1 3 0 3 2] 0.005268780663755161
    # [1 1 3 2 3] 0.020850249604089342
    # [2 3 3 1 3] 0.005235806763642116
    # [2 2 2 3 0] 0.007237919123950909
    # [2 2 2 3 0] 0.007237919123950909
    # [2 3 1 2 2] 0.00386137356877238
    # [2 3 2 2 3] 0.0101601246676211
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
    # [2 2 3 2 1] 0.006701399784256296
    # [2 3 0 2 3] 0.0009199062997398363
    # [3 2 3 1 1] 0.007034058195361783
    # [2 2 3 0 1] 0.020689174925208352
    # [0 2 3 1 1] 0.007385551333875279
    # [3 1 0 1 3] 0.00044101891370375306
    # [2 2 2 0 1] 0.01161300890685018
    # [3 1 0 3 2] 0.008949552792504081
    # [0 2 1 0 2] 0.0007910849983416415
    # [2 2 3 2 0] 0.007089705144309747
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

    # [0 2] 0.30000000000000004
    # [2 0] 0.20000000000000012
    # [2 1] 0.19999999999999998
    # 

  .. code-block:: python

    m = 10
    I = teneva.sample_square(Y, m, unique=False)
    for i in I:
        print(i, teneva.get(Y, i))

    # >>> ----------------------------------------
    # >>> Output:

    # [2 1] 0.19999999999999998
    # [0 2] 0.30000000000000004
    # [0 1] 0.19999999999999993
    # [2 0] 0.20000000000000012
    # [2 0] 0.20000000000000012
    # [2 0] 0.20000000000000012
    # [0 2] 0.30000000000000004
    # [0 2] 0.30000000000000004
    # [2 0] 0.20000000000000012
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
    # [0 3 1 0 0] 0.017523367961620212
    # [1 1 3 0 3] 0.008094385528619455
    # [0 0 0 3 1] 0.012882844366531889
    # [2 0 2 2 1] 0.005396751620488374
    # [0 3 2 3 1] 0.005793914853250297
    # [0 3 1 0 3] 0.031223275738127162
    # [0 3 3 1 1] 0.00849159306905711
    # [0 0 2 1 1] 0.009221683168084471
    # [3 3 1 0 3] 0.010801785940475796
    # [0 1 3 0 1] 0.002037091228806436
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
    # [3 1 0 1 1] 0.017963333094387288
    # [0 2 2 1 1] 0.012152286440489418
    # [1 3 0 1 1] 0.0008635241295761303
    # [0 2 1 3 2] 0.003238965367713351
    # [2 3 0 1 1] 0.014700631383368384
    # [2 1 1 3 1] 0.0024888432721324153
    # [1 1 1 1 1] 0.0051630663903024536
    # [2 3 3 2 3] 0.0032659562682485228
    # [2 1 2 3 1] 0.004955517050227165
    # [1 0 0 2 0] 0.0038466183942745245
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

    # [[1 4 4]
    #  [0 3 3]
    #  [1 2 0]
    #  [0 0 3]
    #  [4 4 2]
    #  [2 0 0]
    #  [3 1 1]
    #  [4 2 4]]
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

    # [[3 3 3]
    #  [3 3 0]
    #  [2 2 3]
    #  [2 1 4]
    #  [3 0 2]
    #  [3 4 0]
    #  [1 2 3]
    #  [2 2 2]]
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

.. autofunction:: teneva.sample.sample_rand_poi

  **Examples**:

  .. code-block:: python

    d = 4               # Dimension
    a = [-2, -4, 0, 3]  # Lower grid bounds
    b = [+2, -2, 3, 6]  # Lower grid bounds
    m = 3               # Number of samples
    
    X = teneva.sample_rand_poi(a, b, m)
    
    print(X)

    # >>> ----------------------------------------
    # >>> Output:

    # [[-0.96523525 -2.34653896  1.38316722  3.21546859]
    #  [ 0.38706846 -3.6737769   2.41898011  3.02479311]
    #  [-0.6563158  -2.81430498  0.93096971  3.11847307]]
    # 

  Let generate many samples and check that limits are valid:

  .. code-block:: python

    d = 10
    a = -3.
    b = +4.
    m = int(1.E+5)
    X = teneva.sample_rand_poi([a]*d, [b]*d, m)
    print(np.min(X))
    print(np.max(X))

    # >>> ----------------------------------------
    # >>> Output:

    # -2.999982269034018
    # 3.9999884743761758
    # 

  Note that we can also set a random seed value:

  .. code-block:: python

    d = 3
    X = teneva.sample_rand_poi([-2]*d, [+2]*d, 2, seed=42)
    print(X)
    X = teneva.sample_rand_poi([-2]*d, [+2]*d, 2, seed=0)
    print(X)
    X = teneva.sample_rand_poi([-2]*d, [+2]*d, 2, seed=42)
    print(X)
    X = teneva.sample_rand_poi([-2]*d, [+2]*d, 2,
        seed=np.random.default_rng(42))
    print(X)

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.09582419  1.43439168 -1.62329061]
    #  [-0.24448624  0.78947212  1.90248941]]
    # [[ 0.54784675 -1.8361059   1.25308096]
    #  [-0.92085314 -1.93388946  1.65102231]]
    # [[ 1.09582419  1.43439168 -1.62329061]
    #  [-0.24448624  0.78947212  1.90248941]]
    # [[ 1.09582419  1.43439168 -1.62329061]
    #  [-0.24448624  0.78947212  1.90248941]]
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


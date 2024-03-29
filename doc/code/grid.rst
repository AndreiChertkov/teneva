Module grid: create and transform multidimensional grids
--------------------------------------------------------


.. automodule:: teneva.grid


-----




|
|

.. autofunction:: teneva.grid.grid_flat

  **Examples**:

  .. code-block:: python

    n = [2, 3, 4]           # This is the 3D grid 2 x 3 x 4
    I = teneva.grid_flat(n) # This is the full list of indices (flatten grid)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 0 0]
    #  [1 0 0]
    #  [0 1 0]
    #  [1 1 0]
    #  [0 2 0]
    #  [1 2 0]
    #  [0 0 1]
    #  [1 0 1]
    #  [0 1 1]
    #  [1 1 1]
    #  [0 2 1]
    #  [1 2 1]
    #  [0 0 2]
    #  [1 0 2]
    #  [0 1 2]
    #  [1 1 2]
    #  [0 2 2]
    #  [1 2 2]
    #  [0 0 3]
    #  [1 0 3]
    #  [0 1 3]
    #  [1 1 3]
    #  [0 2 3]
    #  [1 2 3]]
    # 




|
|

.. autofunction:: teneva.grid.grid_prep_opt

  **Examples**:

  .. code-block:: python

    teneva.grid_prep_opt(-5., d=3)

    # >>> ----------------------------------------
    # >>> Output:

    # array([-5., -5., -5.])
    # 

  .. code-block:: python

    teneva.grid_prep_opt([-5., +4])

    # >>> ----------------------------------------
    # >>> Output:

    # array([-5.,  4.])
    # 

  .. code-block:: python

    teneva.grid_prep_opt([5., +4.21], kind=int)

    # >>> ----------------------------------------
    # >>> Output:

    # array([5, 4])
    # 

  .. code-block:: python

    teneva.grid_prep_opt([-5., +4], reps=3)

    # >>> ----------------------------------------
    # >>> Output:

    # array([[-5.,  4.],
    #        [-5.,  4.],
    #        [-5.,  4.]])
    # 




|
|

.. autofunction:: teneva.grid.grid_prep_opts

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    a = -5.         # Lower bounds for grid
    b = +5.         # Upper bounds for grid
    n = 7           # Shape of the tensor/grid
    
    teneva.grid_prep_opts(a, b, n, d)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([-5., -5., -5.]), array([5., 5., 5.]), array([7, 7, 7]))
    # 

  .. code-block:: python

    d = None        # Dimension of the tensor/grid
    a = -5.         # Lower bounds for grid
    b = +5.         # Upper bounds for grid
    n = [7, 4, 7]   # Shape of the tensor/grid
    
    teneva.grid_prep_opts(a, b, n, d)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([-5., -5., -5.]), array([5., 5., 5.]), array([7, 4, 7]))
    # 

  .. code-block:: python

    d = None        # Dimension of the tensor/grid
    a = [-5., -4.]  # Lower bounds for grid
    b = +5.         # Upper bounds for grid
    n = 6           # Shape of the tensor/grid
    
    teneva.grid_prep_opts(a, b, n, d)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([-5., -4.]), array([5., 5.]), array([6, 6]))
    # 

  .. code-block:: python

    a = [-5., -4.]  # Lower bounds for grid
    b = [+5., +4.]  # Upper bounds for grid
    n = [100, 200]  # Shape of the tensor/grid
    
    teneva.grid_prep_opts(a, b, n)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([-5., -4.]), array([5., 4.]), array([100, 200]))
    # 

  .. code-block:: python

    a = [-5., -4., +3.]  # Lower bounds for grid
    b = [+5., +4., +3.]  # Upper bounds for grid
    n = [100, 200, 300]  # Shape of the tensor/grid
    
    teneva.grid_prep_opts(a, b, n, reps=2)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([[-5., -4.,  3.],
    #         [-5., -4.,  3.]]),
    #  array([[5., 4., 3.],
    #         [5., 4., 3.]]),
    #  array([[100, 200, 300],
    #         [100, 200, 300]]))
    # 




|
|

.. autofunction:: teneva.grid.ind_qtt_to_tt

  **Examples**:

  .. code-block:: python

    d = 4             # Dimension of the TT-tensor
    q = 4             # Quantization value
                      # (note that TT mode size will be n=2^q)
    i_qtt = [         # Multi-index in the QTT-format
        1, 0, 0, 0,   # -> 1 in TT
        0, 1, 0, 0,   # -> 2 in TT
        0, 0, 0, 1,   # -> 8 in TT
        1, 1, 1, 1]   # -> 2^q-1 in TT
    
    i = teneva.ind_qtt_to_tt(i_qtt, q)
    
    print(i)          # Multi-index in the TT-format

    # >>> ----------------------------------------
    # >>> Output:

    # [ 1  2  8 15]
    # 

  We can also calculate several multi-indices at once:

  .. code-block:: python

    d = 3
    q = 3
    
    I_qtt = [         # Multi-indices in the QTT-format
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ] 
    
    I = teneva.ind_qtt_to_tt(I_qtt, q)
    
    print(I)          # Multi-indices in the TT-format

    # >>> ----------------------------------------
    # >>> Output:

    # [[1 2 4]
    #  [3 6 5]
    #  [7 7 7]]
    # 




|
|

.. autofunction:: teneva.grid.ind_to_poi

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    a = [-5.] * d   # Lower bounds for grid
    b = [+5.] * d   # Upper bounds for grid
    n = [7] * d     # Shape of the tensor/grid

  .. code-block:: python

    # Random multi-indices (samples x dimension):
    I = np.vstack([np.random.choice(k, 50) for k in n]).T
    
    print(I.shape)
    print(I[0, :]) # The 1th sample

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [6 4 5]
    # 

  .. code-block:: python

    X = teneva.ind_to_poi(I, a, b, n)
    
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [5.         1.66666667 3.33333333]
    # 

  Grid bounds and tensor shape may be also numbers:

  .. code-block:: python

    X = teneva.ind_to_poi(I, -5, 5, 7)
    
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [5.         1.66666667 3.33333333]
    # 

  We may also compute only one point while function call:

  .. code-block:: python

    X = teneva.ind_to_poi(I[0, :], -5, 5, 7)
    
    print(X)

    # >>> ----------------------------------------
    # >>> Output:

    # [5.         1.66666667 3.33333333]
    # 

  By default the uniform (kind="uni") grid is used. We may also use the Chebyshev grid:

  .. code-block:: python

    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [-5.         -2.5        -4.33012702]
    # 




|
|

.. autofunction:: teneva.grid.ind_tt_to_qtt

  **Examples**:

  .. code-block:: python

    d = 4             # Dimension of the TT-tensor
    n = 8             # Mode size of the TT-tensor
    i = [ 1, 3, 5, 7] # Multi-index in the TT-format
    
    i_qtt = teneva.ind_tt_to_qtt(i, n)
    
    print(i_qtt)      # Multi-index in the QTT-format

    # >>> ----------------------------------------
    # >>> Output:

    # [1 0 0 1 1 0 1 0 1 1 1 1]
    # 

  We can also calculate several multi-indices at once:

  .. code-block:: python

    d = 4
    n = 8
    
    I = [             # Multi-indices in the TT-format
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [2, 3, 4, 5],
        [7, 7, 7, 7],
    ] 
    
    I_qtt = teneva.ind_tt_to_qtt(I, n)
    
    print(I_qtt)      # Multi-indices in the QTT-format

    # >>> ----------------------------------------
    # >>> Output:

    # [[1 0 0 0 0 0 0 0 0 0 0 0]
    #  [1 0 0 1 0 0 0 0 0 0 0 0]
    #  [1 0 0 1 0 0 1 0 0 1 0 0]
    #  [0 1 0 1 1 0 0 0 1 1 0 1]
    #  [1 1 1 1 1 1 1 1 1 1 1 1]]
    # 




|
|

.. autofunction:: teneva.grid.poi_scale

  **Examples**:

  .. code-block:: python

    d = 3                 # Dimension of the tensor/grid
    a = [-5., -3., -1.]   # Lower bounds for grid
    b = [+5., +3., +1.]   # Upper bounds for grid
    
    X = np.array([       # We prepare 4 spatial points:
        [-5., -3., -1.], # Point near the lower bound
        [ 0.,  0.,  0.], # Zero point
        [-1., +2.,  0.], # Random point
        [+5., +3., +1.], # Point near the upper bound
    ])

  .. code-block:: python

    Xsc = teneva.poi_scale(X, a, b)
    
    print(Xsc)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0.         0.         0.        ]
    #  [0.5        0.5        0.5       ]
    #  [0.4        0.83333333 0.5       ]
    #  [1.         1.         1.        ]]
    # 

  We can also consider the Chebyshev grid:

  .. code-block:: python

    Xsc = teneva.poi_scale(X, a, b, 'cheb')
    
    print(Xsc)

    # >>> ----------------------------------------
    # >>> Output:

    # [[-1.         -1.         -1.        ]
    #  [ 0.          0.          0.        ]
    #  [-0.2         0.66666667  0.        ]
    #  [ 1.          1.          1.        ]]
    # 

  For a uniform grid, scaling from an interval of [0, 1] will be the identical operation:

  .. code-block:: python

    teneva.poi_scale(([
        [0., 0.5, 0.7, 1.0],
        [-2., 0.2, 0.4, 2.0],
    ]), 0., 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # array([[0. , 0.5, 0.7, 1. ],
    #        [0. , 0.2, 0.4, 1. ]])
    # 

  For a Chebyshev grid, scaling from an interval of [-1, 1] will be the identical operation:

  .. code-block:: python

    teneva.poi_scale(([
        [-1., -0.2, 0., 0.7, 1.0],
        [-1., -0.4, 0., 0.4, 1.0],
    ]), -1., 1., 'cheb')

    # >>> ----------------------------------------
    # >>> Output:

    # array([[-1. , -0.2,  0. ,  0.7,  1. ],
    #        [-1. , -0.4,  0. ,  0.4,  1. ]])
    # 




|
|

.. autofunction:: teneva.grid.poi_to_ind

  **Examples**:

  .. code-block:: python

    d = 3                 # Dimension of the tensor/grid
    a = [-5., -3., -1.]   # Lower bounds for grid
    b = [+5., +3., +1.]   # Upper bounds for grid
    n = [9, 8, 7]         # Shape of the tensor/grid
    
    X = np.array([       # We prepare 4 spatial points:
        [-5., -3., -1.], # Point near the lower bound
        [ 0.,  0.,  0.], # Zero point
        [-1., +2.,  0.], # Random point
        [+5., +3., +1.], # Point near the upper bound
    ])

  We can build multi-indices for the uniform grid:

  .. code-block:: python

    I = teneva.poi_to_ind(X, a, b, n)
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 0 0]
    #  [4 4 3]
    #  [3 6 3]
    #  [8 7 6]]
    # 

  We can also build multi-indices for the Chebyshev grid:

  .. code-block:: python

    I = teneva.poi_to_ind(X, a, b, n, 'cheb')
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[8 7 6]
    #  [4 4 3]
    #  [5 2 3]
    #  [0 0 0]]
    # 

  Grid bounds and tensor shape may be also numbers:

  .. code-block:: python

    I = teneva.poi_to_ind(X, -1., +1., 10, 'cheb')
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[9 9 9]
    #  [4 4 4]
    #  [9 0 4]
    #  [0 0 0]]
    # 

  We may also compute only one point while function call:

  .. code-block:: python

    x = [-5., -3., -1.]
    I = teneva.poi_to_ind(x, -1., +1., 10, 'cheb')
    
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [9 9 9]
    # 

  We can apply "ind_to_poi" function to the generated multi-indices and check the result:

  .. code-block:: python

    d = 3                 # Dimension of the tensor/grid
    a = [-5., -3., -1.]   # Lower bounds for grid
    b = [+5., +3., +1.]   # Upper bounds for grid
    n = [7, 5, 3]         # Shape of the tensor/grid
    
    X = np.array([
        [-5., -3., -1.],  # Point near the lower bound
        [ 0.,  0.,  0.],  # Zero point
        [+5., +3., +1.],  # Point near the upper bound
    ])

  .. code-block:: python

    I = teneva.poi_to_ind(X, a, b, n)
    Z = teneva.ind_to_poi(I, a, b, n)
    
    print(X) # Used spacial points
    print(Z) # Generated spacial points
    print(I) # Multi-indices

    # >>> ----------------------------------------
    # >>> Output:

    # [[-5. -3. -1.]
    #  [ 0.  0.  0.]
    #  [ 5.  3.  1.]]
    # [[-5. -3. -1.]
    #  [ 0.  0.  0.]
    #  [ 5.  3.  1.]]
    # [[0 0 0]
    #  [3 2 1]
    #  [6 4 2]]
    # 

  .. code-block:: python

    I = teneva.poi_to_ind(X, a, b, n, 'cheb')
    Z = teneva.ind_to_poi(I, a, b, n, 'cheb')
    
    print(X) # Used spacial points
    print(Z) # Generated spacial points
    print(I) # Multi-indices

    # >>> ----------------------------------------
    # >>> Output:

    # [[-5. -3. -1.]
    #  [ 0.  0.  0.]
    #  [ 5.  3.  1.]]
    # [[-5.0000000e+00 -3.0000000e+00 -1.0000000e+00]
    #  [ 3.0616170e-16  1.8369702e-16  6.1232340e-17]
    #  [ 5.0000000e+00  3.0000000e+00  1.0000000e+00]]
    # [[6 4 2]
    #  [3 2 1]
    #  [0 0 0]]
    # 




|
|


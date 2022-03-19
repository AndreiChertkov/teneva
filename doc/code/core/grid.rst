grid: create and transform multidimensional grids
-------------------------------------------------


.. automodule:: teneva.core.grid


-----


.. autofunction:: teneva.cache_to_data

  **Examples**:

  Let apply TT-CROSS for benchmark function:

  .. code-block:: python

    a         = [-5., -4., -3., -2., -1.] # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.] # Upper bounds for spatial grid
    n         = [ 20,  18,  16,  14,  12] # Shape of the tensor
    m         = 8.E+3                     # Number of calls to function
    r         = 3                         # TT-rank of the initial tensor
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)
    
    cache = {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, cache=cache)

  Now cache contains the requested function values and related tensor multi-indices:

  .. code-block:: python

    I, Y = teneva.cache_to_data(cache)
    
    print(I.shape)
    print(Y.shape)
    
    i = I[0, :]                       # The 1th multi-index
    y = Y[0]                          # Saved value in cache
    
    print(i)
    print(y)
    print(func(i))

    # >>> ----------------------------------------
    # >>> Output:

    # (6830, 5)
    # (6830,)
    # [ 0 17  2 10  2]
    # 63079.06485373773
    # 63079.06485373773
    # 


-----


.. autofunction:: teneva.grid_flat

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


-----


.. autofunction:: teneva.grid_prep_opt

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


-----


.. autofunction:: teneva.grid_prep_opts

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

    a = [-5., -4.]  # Lower bounds for grid
    b = [+5., +4.]  # Upper bounds for grid
    n = [100, 200]  # Shape of the tensor/grid
    teneva.grid_prep_opts(a, b, n, reps=2)

    # >>> ----------------------------------------
    # >>> Output:

    # (array([[-5., -4.],
    #         [-5., -4.]]),
    #  array([[5., 4.],
    #         [5., 4.]]),
    #  array([[100, 200],
    #         [100, 200]]))
    # 


-----


.. autofunction:: teneva.ind_to_poi

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    a = [-5.] * d   # Lower bounds for grid
    b = [+5.] * d   # Upper bounds for grid
    n = [7] * d     # Shape of the tensor/grid

  .. code-block:: python

    # Random multi-indices (samples x dimension):
    I = np.vstack([np.random.choice(n[i], 50) for i in range(d)]).T
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


-----


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

    # [[3 4 2]
    #  [0 1 4]
    #  [1 3 1]
    #  [2 1 4]
    #  [4 0 3]
    #  [1 2 0]
    #  [0 2 0]
    #  [2 3 2]]
    # 


-----


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



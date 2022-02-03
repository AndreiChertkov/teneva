grid: create and transform multidimensional grids
-------------------------------------------------


.. automodule:: teneva.core.grid



-----

.. autofunction:: teneva.grid_flat

  **Examples**:

  .. code-block:: python

    n = [2, 3, 4]
    I = teneva.grid_flat(n)
    print(I)

    # >>> ----------------------------------------
    # >>> Output:

    # [[0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]
    #  [0 0 1 1 2 2 0 0 1 1 2 2 0 0 1 1 2 2 0 0 1 1 2 2]
    #  [0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3]]
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

  .. code-block:: python

    d = None        # Dimension of the tensor/grid
    a = -5.         # Lower bounds for grid
    b = +5.         # Upper bounds for grid
    n = [7, 7, 7]   # Shape of the tensor/grid
    teneva.grid_prep_opts(a, b, n, d)

  .. code-block:: python

    d = None        # Dimension of the tensor/grid
    a = [-5., -4.]  # Lower bounds for grid
    b = +5.         # Upper bounds for grid
    n = 6           # Shape of the tensor/grid
    teneva.grid_prep_opts(a, b, n, d)

  .. code-block:: python

    a = [-5., -4.]  # Lower bounds for grid
    b = [+5., +4.]  # Upper bounds for grid
    n = [100, 200]  # Shape of the tensor/grid
    teneva.grid_prep_opts(a, b, n)

  .. code-block:: python

    a = [-5., -4.]  # Lower bounds for grid
    b = [+5., +4.]  # Upper bounds for grid
    n = [100, 200]  # Shape of the tensor/grid
    teneva.grid_prep_opts(a, b, n, reps=2)



-----

.. autofunction:: teneva.ind2poi

  **Examples**:

  .. code-block:: python

    d = 3           # Dimension of the tensor/grid
    a = [-5.] * d   # Lower bounds for grid
    b = [+5.] * d   # Upper bounds for grid
    n = [7] * d     # Shape of the tensor/grid

  .. code-block:: python

    # Random inidices (samples x dimension):
    I = np.vstack([np.random.choice(n[i], 50) for i in range(d)]).T
    print(I.shape)
    print(I[0, :]) # The 1th sample

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [6 4 5]
    # 

  .. code-block:: python

    X = teneva.ind2poi(I, a, b, n)
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [5.         1.66666667 3.33333333]
    # 

  Grid bounds and tensor shape may be also numbers:

  .. code-block:: python

    X = teneva.ind2poi(I, -5, 5, 7)
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [5.         1.66666667 3.33333333]
    # 

  We may also compute only one point while function call:

  .. code-block:: python

    X = teneva.ind2poi(I[0, :], -5, 5, 7)
    print(X)

    # >>> ----------------------------------------
    # >>> Output:

    # [5.         1.66666667 3.33333333]
    # 

  By default the uniform (kind="uni") grid is used. We may also use the Chebyshev grid:

  .. code-block:: python

    X = teneva.ind2poi(I, a, b, n, 'cheb')
    print(X.shape)
    print(X[0, :]) # The 1th point

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 3)
    # [-5.         -2.5        -4.33012702]
    # 



-----

.. autofunction:: teneva.ind2str

  **Examples**:

  .. code-block:: python

    i = [1, 2, 3, 4, 5]
    s = teneva.ind2str(i)
    print(s)

    # >>> ----------------------------------------
    # >>> Output:

    # 1-2-3-4-5
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



-----

.. autofunction:: teneva.str2ind

  **Examples**:

  .. code-block:: python

    s = '1-2-3-4-5'
    i = teneva.str2ind(s)
    print(i)

    # >>> ----------------------------------------
    # >>> Output:

    # [1 2 3 4 5]
    # 


als: construct TT-tensor by TT-ALS
----------------------------------


.. automodule:: teneva.core.als

---


.. autofunction:: teneva.core.als.als

  **Examples**:

  .. code-block:: python

    d         = 10          # Dimension of the function
    A         = [-2.] * d   # Lower bound for spatial grid
    B         = [+2.] * d   # Upper bound for spatial grid
    N         = [10] * d    # Shape of the tensor (it may be non-uniform)
    M_tst     = 10000       # Number of test points

  .. code-block:: python

    evals     = 10000       # Number of calls to target function
    nswp      = 50          # Sweep number for ALS iterations
    r         = 3           # TT-rank of the initial random tensor

  .. code-block:: python

    # Target function
    # (the function takes as input a set of tensor indices I of the shape [samples, dim], which
    # are transformed into points X of a uniform spatial grid using the function "ind2poi"):
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind2poi(I, A, B, N)
        return rosen(X.T)

  .. code-block:: python

    # Train data:
    
    I_trn = teneva.sample_lhs(N, evals) 
    Y_trn = func(I_trn)

  .. code-block:: python

    # Test data
    # (we generate M_tst random tensor elements for accuracy check):
    
    I_tst = np.vstack([np.random.choice(N[i], M_tst) for i in range(d)]).T
    Y_tst = func(I_tst)

  .. code-block:: python

    # Build tensor
    # (we generate random initial r-rank approximation in the TT-format using
    # the function "rand") and then compute the resulting TT-tensor by TT-ALS):
    
    t = tpc()
    Y = teneva.rand(N, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       3.17
    # 

  .. code-block:: python

    # Check result:
    
    get = teneva.getter(Y)
    
    Z = np.array([get(i) for i in I_trn])
    e_trn = np.linalg.norm(Z - Y_trn) / np.linalg.norm(Y_trn)
    
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   8.44e-01
    # Error on test  :   3.91e+00
    # 

---


.. autofunction:: teneva.core.als.als2

  **Examples**:

  .. code-block:: python

    d         = 5           # Dimension of the function
    A         = [-2.] * d   # Lower bound for spatial grid
    B         = [+2.] * d   # Upper bound for spatial grid
    N         = [10] * d    # Shape of the tensor (it may be non-uniform)
    M_tst     = 10000       # Number of test points

  .. code-block:: python

    evals     = 10000       # Number of calls to target function
    nswp      = 50          # Sweep number for ALS iterations
    r         = 3           # TT-rank of the initial random tensor

  .. code-block:: python

    # Target function
    # (the function takes as input a set of tensor indices I of the shape [samples, dim], which
    # are transformed into points X of a uniform spatial grid using the function "ind2poi"):
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind2poi(I, A, B, N)
        return rosen(X.T)

  .. code-block:: python

    # Train data:
    
    I_trn = teneva.sample_lhs(N, evals) 
    Y_trn = func(I_trn)

  .. code-block:: python

    # Test data
    # (we generate M_tst random tensor elements for accuracy check):
    
    I_tst = np.vstack([np.random.choice(N[i], M_tst) for i in range(d)]).T
    Y_tst = func(I_tst)

  .. code-block:: python

    # Build tensor
    # (we generate random initial r-rank approximation in the TT-format using
    # the function "rand") and then compute the resulting TT-tensor by TT-ALS):
    
    t = tpc()
    Y = teneva.rand(N, r)
    Y = teneva.als2(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :     106.83
    # 

  .. code-block:: python

    # Check result:
    
    get = teneva.getter(Y)
    
    Z = np.array([get(i) for i in I_trn])
    e_trn = np.linalg.norm(Z - Y_trn) / np.linalg.norm(Y_trn)
    
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   1.54e-15
    # Error on test  :   1.60e-15
    # 

---

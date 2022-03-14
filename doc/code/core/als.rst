als: construct TT-tensor by TT-ALS
----------------------------------


.. automodule:: teneva.core.als


-----


.. autofunction:: teneva.als

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 20,  18,  16,  14,  12]   # Shape of the tensor

  .. code-block:: python

    m         = 1.E+4  # Number of calls to target function
    nswp      = 50     # Sweep number for ALS iterations
    r         = 3      # TT-rank of the initial random tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)

  We prepare train data from the LHS random distribution:

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m) 
    Y_trn = func(I_trn)

  We prepare test data from as a random tensor multi-indices:

  .. code-block:: python

    # Test data:
    
    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    
    # Function values for the test points:
    Y_tst = func(I_tst)

  We build the TT-tensor, which approximates the target function (we generate random initial r-rank approximation in the TT-format using the function "rand" and then compute the resulting TT-tensor by TT-ALS):

  .. code-block:: python

    t = tpc()
    Y = teneva.rand(n, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       1.58
    # 

  And now we can check the result:

  .. code-block:: python

    # Fast getter for TT-tensor values:
    get = teneva.getter(Y)                     
    
    # Compute approximation in train points:
    Z = np.array([get(i) for i in I_trn])
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(Z - Y_trn)          
    e_trn /= np.linalg.norm(Y_trn)
    
    # Compute approximation in test points:
    Z = np.array([get(i) for i in I_tst])
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(Z - Y_tst)          
    e_tst /= np.linalg.norm(Y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   3.23e-15
    # Error on test  :   3.51e-15
    # 


-----


.. autofunction:: teneva.als2

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 20,  18,  16,  14,  12]   # Shape of the tensor

  .. code-block:: python

    m         = 1.E+4  # Number of calls to target function
    nswp      = 50     # Sweep number for ALS iterations
    r         = 3      # TT-rank of the initial random tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)

  We prepare train data from the LHS random distribution:

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m) 
    Y_trn = func(I_trn)

  We prepare test data from as a random tensor multi-indices:

  .. code-block:: python

    # Test data:
    
    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    
    # Function values for the test points:
    Y_tst = func(I_tst)

  We build the TT-tensor, which approximates the target function (we generate random initial r-rank approximation in the TT-format using the function "rand" and then compute the resulting TT-tensor by TT-ALS):

  .. code-block:: python

    t = tpc()
    Y = teneva.rand(n, r)
    Y = teneva.als2(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :      89.15
    # 

  And now we can check the result:

  .. code-block:: python

    # Fast getter for TT-tensor values:
    get = teneva.getter(Y)                     
    
    # Compute approximation in train points:
    Z = np.array([get(i) for i in I_trn])
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(Z - Y_trn)          
    e_trn /= np.linalg.norm(Y_trn)
    
    # Compute approximation in test points:
    Z = np.array([get(i) for i in I_tst])
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(Z - Y_tst)          
    e_tst /= np.linalg.norm(Y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   1.22e-15
    # Error on test  :   1.33e-15
    # 



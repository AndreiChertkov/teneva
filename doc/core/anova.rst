Module anova: construct TT-tensor by TT-ANOVA
---------------------------------------------


.. automodule:: teneva.core.anova


-----




|
|

.. autofunction:: teneva.anova

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 20,  18,  16,  14,  12]   # Shape of the tensor

  .. code-block:: python

    m         = 1.E+4  # Number of calls to target function
    order     = 1      # Order of ANOVA decomposition (1 or 2)
    r         = 2      # TT-rank of the resulting tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)

  We prepare train data from the LHS random distribution:

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m) 
    y_trn = func(I_trn)

  We prepare test data from random tensor multi-indices:

  .. code-block:: python

    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
    
    # Function values for the test points:
    y_tst = func(I_tst)

  We build the TT-tensor, which approximates the target function:

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r, order)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.00
    # 

  And now we can check the result:

  .. code-block:: python

    # Fast getter for TT-tensor values:
    get = teneva.getter(Y)                     
    
    # Compute approximation in train points:
    y_our = teneva.get_many(Y, I_trn)
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    # Compute approximation in test points:
    y_our = teneva.get_many(Y, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   1.11e-01
    # Error on test  :   1.14e-01
    # 

  We can also build approximation using 2-th order ANOVA decomposition:

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r, order=2)
    t = tpc() - t
    
    get = teneva.getter(Y)                     
    
    y_our = teneva.get_many(Y, I_trn)
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    y_our = teneva.get_many(Y, I_tst)
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.13
    # Error on train :   8.67e-02
    # Error on test  :   8.18e-02
    # 

  Let's look at the quality of approximation for a linear function:

  .. code-block:: python

    d = 4
    a = -2.
    b = +3.
    n = [10] * d
    r = 3
    m_trn = int(1.E+5)
    m_tst = int(1.E+4)

  .. code-block:: python

    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return 5. + 0.1 * X[:, 0] + 0.2 * X[:, 1] + 0.3 * X[:, 2] + 0.4 * X[:, 3]

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m_trn) 
    y_trn = func(I_trn)
    
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    y_tst = func(I_tst)

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r, order=1)
    t = tpc() - t
    
    get = teneva.getter(Y)                     
    
    y_our = teneva.get_many(Y, I_trn)
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    y_our = teneva.get_many(Y, I_tst)
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.02
    # Error on train :   2.50e-03
    # Error on test  :   2.49e-03
    # 

  Let's look at the quality of approximation for a quadratic function

  .. code-block:: python

    d = 4
    a = -2.
    b = +3.
    n = [10] * d
    r = 3
    m_trn = int(1.E+5)
    m_tst = int(1.E+4)

  .. code-block:: python

    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return 5. + 0.1 * X[:, 0]**2 + 0.2 * X[:, 1]**2 + 0.3 * X[:, 2]**2 + 0.4 * X[:, 3]**2

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m_trn) 
    y_trn = func(I_trn)
    
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    y_tst = func(I_tst)

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r, order=1)
    t = tpc() - t
    
    get = teneva.getter(Y)                     
    
    y_our = teneva.get_many(Y, I_trn)
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    y_our = teneva.get_many(Y, I_tst)
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.02
    # Error on train :   2.60e-03
    # Error on test  :   2.59e-03
    # 




|
|


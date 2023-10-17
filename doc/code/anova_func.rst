Module anova_func: construct TT-tensor of interpolation coefs by TT-ANOVA
-------------------------------------------------------------------------


.. automodule:: teneva.anova_func


-----




|
|

.. autofunction:: teneva.anova_func.anova_func

  **Examples**:

  .. code-block:: python

    d = 5                           # Dimension of the function
    a = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n = 3                           # Shape of interpolation tensor

  .. code-block:: python

    m     = 1.E+4  # Number of calls to target function
    e     = 1.E-8  # Truncation accuracy

  We set the target function (the function takes as input a set of multidimensional points X of the shape [samples, dimension]):

  .. code-block:: python

    def func(X):
        return np.sum(X, axis=1)

  We prepare train data from the random distribution:

  .. code-block:: python

    X_trn = teneva.sample_rand_poi(a, b, m) 
    y_trn = func(X_trn)

  We prepare test data from random points:

  .. code-block:: python

    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random points:
    X_tst = teneva.sample_rand_poi(a, b, m_tst) 
    
    # Function values for the test points:
    y_tst = func(X_tst)

  We build the TT-tensor of interpolation coefficients:

  .. code-block:: python

    t = tpc()
    A = teneva.anova_func(X_trn, y_trn, n, a, b, 1.E-5, e=e)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.01
    # 

  And now we can check the result:

  .. code-block:: python

    # Compute approximation in train points:
    y_our = teneva.func_get(X_trn, A, a, b)
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    # Compute approximation in test points:
    y_our = teneva.func_get(X_tst, A, a, b)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   2.87e-02
    # Error on test  :   2.78e-02
    # 




|
|


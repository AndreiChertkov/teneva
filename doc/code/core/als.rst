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

    m         = 1.E+4                       # Number of calls to target function
    nswp      = 50                          # Sweep number for ALS iterations
    r         = 3                           # TT-rank of the initial random tensor

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
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
    
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

    # Build time     :       1.01
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

    # Error on train :   6.78e-10
    # Error on test  :   8.44e-10
    # 

  We can also set a validation data set and specify as a stop criterion the accuracy of the TT-approximation on this data (and we can also present the logs):

  .. code-block:: python

    # Validation data:
    
    # Number of validation points:
    m_vld = int(1.E+3)
    
    # Random multi-indices for the validation points:
    I_vld = np.vstack([np.random.choice(k, m_vld) for k in n]).T
    
    # Function values for the validation points:
    Y_vld = func(I_vld)

  .. code-block:: python

    t = tpc()
    Y = teneva.rand(n, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp, I_vld=I_vld, Y_vld=Y_vld, e_vld=1.E-5, log=True)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.594 | rank:   3.0 | err: 1.0e+00 | 
    # #   1 | time:      1.108 | rank:   3.0 | err: 1.1e+00 | eps: 3.3e+03 | 
    # #   2 | time:      1.679 | rank:   3.0 | err: 1.2e+00 | eps: 8.2e-01 | 
    # #   3 | time:      2.298 | rank:   3.0 | err: 1.1e+00 | eps: 4.7e-01 | 
    # #   4 | time:      2.807 | rank:   3.0 | err: 6.7e-01 | eps: 1.1e+00 | 
    # #   5 | time:      3.384 | rank:   3.0 | err: 1.6e-01 | eps: 6.8e-01 | 
    # #   6 | time:      3.973 | rank:   3.0 | err: 5.8e-02 | eps: 1.3e-01 | 
    # #   7 | time:      4.548 | rank:   3.0 | err: 1.7e-02 | eps: 5.2e-02 | 
    # #   8 | time:      5.037 | rank:   3.0 | err: 3.4e-03 | eps: 1.6e-02 | 
    # #   9 | time:      5.580 | rank:   3.0 | err: 7.0e-04 | eps: 2.7e-03 | 
    # #  10 | time:      6.145 | rank:   3.0 | err: 1.8e-04 | eps: 5.4e-04 | 
    # #  11 | time:      6.661 | rank:   3.0 | err: 4.8e-05 | eps: 1.3e-04 | 
    # #  12 | time:      7.237 | rank:   3.0 | err: 1.4e-05 | eps: 3.4e-05 | 
    # #  13 | time:      7.815 | rank:   3.0 | err: 4.0e-06 | eps: 9.7e-06 | stop: e_vld | 
    # Build time     :       7.82
    # 

  We can use helper functions to present the resulting accuracy:

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   3.03e-06
    # Error on valid.:   4.01e-06
    # Error on test  :   3.74e-06
    # 

  We may also set the value of relative rate of solution change to stop the iterations:

  .. code-block:: python

    t = tpc()
    Y = teneva.rand(n, r)
    Y = teneva.als(I_trn, Y_trn, Y, e=1.E-6, I_vld=I_vld, Y_vld=Y_vld, log=True)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.485 | rank:   3.0 | err: 1.0e+00 | 
    # #   1 | time:      1.033 | rank:   3.0 | err: 1.1e+00 | eps: 3.5e+03 | 
    # #   2 | time:      1.621 | rank:   3.0 | err: 1.3e+00 | eps: 8.3e-01 | 
    # #   3 | time:      2.199 | rank:   3.0 | err: 1.3e+00 | eps: 4.8e-01 | 
    # #   4 | time:      2.698 | rank:   3.0 | err: 1.3e+00 | eps: 3.7e-01 | 
    # #   5 | time:      3.240 | rank:   3.0 | err: 1.2e+00 | eps: 4.8e-01 | 
    # #   6 | time:      3.803 | rank:   3.0 | err: 5.9e-01 | eps: 9.7e-01 | 
    # #   7 | time:      4.354 | rank:   3.0 | err: 1.5e-01 | eps: 5.6e-01 | 
    # #   8 | time:      4.899 | rank:   3.0 | err: 9.8e-02 | eps: 1.3e-01 | 
    # #   9 | time:      5.491 | rank:   3.0 | err: 4.3e-02 | eps: 5.3e-02 | 
    # #  10 | time:      6.056 | rank:   3.0 | err: 1.0e-02 | eps: 3.6e-02 | 
    # #  11 | time:      6.541 | rank:   3.0 | err: 3.5e-03 | eps: 7.6e-03 | 
    # #  12 | time:      7.078 | rank:   3.0 | err: 9.5e-04 | eps: 2.3e-03 | 
    # #  13 | time:      7.647 | rank:   3.0 | err: 1.9e-04 | eps: 6.4e-04 | 
    # #  14 | time:      8.145 | rank:   3.0 | err: 4.5e-05 | eps: 1.2e-04 | 
    # #  15 | time:      8.694 | rank:   3.0 | err: 1.2e-05 | eps: 3.0e-05 | 
    # #  16 | time:      9.269 | rank:   3.0 | err: 3.2e-06 | eps: 7.8e-06 | 
    # #  17 | time:      9.781 | rank:   3.0 | err: 8.8e-07 | eps: 2.1e-06 | 
    # #  18 | time:     10.359 | rank:   3.0 | err: 2.5e-07 | eps: 5.9e-07 | stop: e | 
    # Build time     :      10.36
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   1.76e-07
    # Error on valid.:   2.47e-07
    # Error on test  :   2.22e-07
    # 

  Note that we can use TT-ANOVA as an initial approximation (it will lead to faster convergence):

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp, I_vld=I_vld, Y_vld=Y_vld, e_vld=1.E-5, log=True)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.561 | rank:   3.0 | err: 1.1e-01 | 
    # #   1 | time:      1.101 | rank:   3.0 | err: 2.8e-02 | eps: 1.1e-01 | 
    # #   2 | time:      1.680 | rank:   3.0 | err: 4.2e-03 | eps: 2.7e-02 | 
    # #   3 | time:      2.298 | rank:   3.0 | err: 2.7e-03 | eps: 3.1e-03 | 
    # #   4 | time:      2.897 | rank:   3.0 | err: 3.5e-04 | eps: 3.1e-03 | 
    # #   5 | time:      3.469 | rank:   3.0 | err: 5.2e-05 | eps: 3.2e-04 | 
    # #   6 | time:      4.024 | rank:   3.0 | err: 1.1e-05 | eps: 4.1e-05 | 
    # #   7 | time:      4.615 | rank:   3.0 | err: 2.8e-06 | eps: 8.5e-06 | stop: e_vld | 
    # Build time     :       4.62
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   2.13e-06
    # Error on valid.:   2.75e-06
    # Error on test  :   2.61e-06
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

    m         = 1.E+4                       # Number of calls to target function
    nswp      = 50                          # Sweep number for ALS iterations
    r         = 3                           # TT-rank of the initial random tensor

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
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
    
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

    # Build time     :      87.32
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

    # Error on train :   7.53e-01
    # Error on test  :   2.61e+00
    # 



Module als: construct TT-tensor by TT-ALS
-----------------------------------------


.. automodule:: teneva.core.als


-----


.. autofunction:: teneva.als

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 10,  12,  14,  16,  18]   # Shape of the tensor

  .. code-block:: python

    m         = 1.E+4                       # Number of calls to target function
    nswp      = 50                          # Sweep number for ALS iterations
    r         = 5                           # TT-rank of the initial random tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    def func(I):
        """Schaffer function."""
        X = teneva.ind_to_poi(I, a, b, n)
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

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

  And now we will build the TT-tensor, which approximates the target function by the TT-ALS method. Note that we use TT-ANOVA as an initial approximation (we can instead generate random initial r-rank approximation in the TT-format using the function "tensor_rand", but convergence will be worse, and there will also be instability of the solution).

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       2.92
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

    # Error on train :   2.19e-03
    # Error on test  :   2.56e-03
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
    Y = teneva.anova(I_trn, Y_trn, r)
    Y = teneva.als(I_trn, Y_trn, Y, nswp, I_vld=I_vld, Y_vld=Y_vld, e_vld=1.E-2, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.642 | rank:   5.0 | err: 2.1e-01 | 
    # #   1 | time:      1.296 | rank:   5.0 | err: 7.5e-02 | eps: 2.0e-01 | 
    # #   2 | time:      1.862 | rank:   5.0 | err: 2.8e-02 | eps: 6.3e-02 | 
    # #   3 | time:      2.519 | rank:   5.0 | err: 2.0e-02 | eps: 1.8e-02 | 
    # #   4 | time:      3.183 | rank:   5.0 | err: 1.8e-02 | eps: 4.1e-03 | 
    # #   5 | time:      3.892 | rank:   5.0 | err: 1.8e-02 | eps: 2.5e-03 | 
    # #   6 | time:      4.447 | rank:   5.0 | err: 1.7e-02 | eps: 1.6e-03 | 
    # #   7 | time:      5.149 | rank:   5.0 | err: 1.6e-02 | eps: 1.3e-03 | 
    # #   8 | time:      5.794 | rank:   5.0 | err: 1.5e-02 | eps: 1.3e-03 | 
    # #   9 | time:      6.478 | rank:   5.0 | err: 1.5e-02 | eps: 1.2e-03 | 
    # #  10 | time:      7.155 | rank:   5.0 | err: 1.4e-02 | eps: 1.0e-03 | 
    # #  11 | time:      7.720 | rank:   5.0 | err: 1.4e-02 | eps: 9.4e-04 | 
    # #  12 | time:      8.432 | rank:   5.0 | err: 1.3e-02 | eps: 8.8e-04 | 
    # #  13 | time:      9.121 | rank:   5.0 | err: 1.3e-02 | eps: 8.7e-04 | 
    # #  14 | time:      9.841 | rank:   5.0 | err: 1.2e-02 | eps: 8.8e-04 | 
    # #  15 | time:     10.551 | rank:   5.0 | err: 1.1e-02 | eps: 9.0e-04 | 
    # #  16 | time:     11.142 | rank:   5.0 | err: 1.1e-02 | eps: 9.0e-04 | 
    # #  17 | time:     11.813 | rank:   5.0 | err: 1.0e-02 | eps: 8.9e-04 | 
    # #  18 | time:     12.529 | rank:   5.0 | err: 9.5e-03 | eps: 8.6e-04 | stop: e_vld | 
    # 
    # Build time     :      12.53
    # 

  We can use helper functions to present the resulting accuracy:

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   8.60e-03
    # Error on valid.:   9.52e-03
    # Error on test  :   1.02e-02
    # 

  We may also set the value of relative rate of solution change to stop the iterations:

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r)
    Y = teneva.als(I_trn, Y_trn, Y, e=1.E-3, I_vld=I_vld, Y_vld=Y_vld, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.679 | rank:   5.0 | err: 2.1e-01 | 
    # #   1 | time:      1.385 | rank:   5.0 | err: 8.1e-02 | eps: 2.0e-01 | 
    # #   2 | time:      2.060 | rank:   5.0 | err: 8.0e-02 | eps: 4.2e-02 | 
    # #   3 | time:      2.766 | rank:   5.0 | err: 7.5e-02 | eps: 3.2e-02 | 
    # #   4 | time:      3.424 | rank:   5.0 | err: 6.4e-02 | eps: 2.9e-02 | 
    # #   5 | time:      4.074 | rank:   5.0 | err: 4.9e-02 | eps: 2.7e-02 | 
    # #   6 | time:      4.750 | rank:   5.0 | err: 4.0e-02 | eps: 1.8e-02 | 
    # #   7 | time:      5.424 | rank:   5.0 | err: 3.7e-02 | eps: 1.0e-02 | 
    # #   8 | time:      6.002 | rank:   5.0 | err: 3.3e-02 | eps: 1.1e-02 | 
    # #   9 | time:      6.708 | rank:   5.0 | err: 2.7e-02 | eps: 1.6e-02 | 
    # #  10 | time:      7.409 | rank:   5.0 | err: 2.1e-02 | eps: 1.5e-02 | 
    # #  11 | time:      8.096 | rank:   5.0 | err: 1.9e-02 | eps: 6.2e-03 | 
    # #  12 | time:      8.747 | rank:   5.0 | err: 1.8e-02 | eps: 3.7e-03 | 
    # #  13 | time:      9.298 | rank:   5.0 | err: 1.7e-02 | eps: 2.3e-03 | 
    # #  14 | time:      9.939 | rank:   5.0 | err: 1.6e-02 | eps: 2.2e-03 | 
    # #  15 | time:     10.626 | rank:   5.0 | err: 1.5e-02 | eps: 2.2e-03 | 
    # #  16 | time:     11.311 | rank:   5.0 | err: 1.4e-02 | eps: 1.9e-03 | 
    # #  17 | time:     12.016 | rank:   5.0 | err: 1.3e-02 | eps: 1.7e-03 | 
    # #  18 | time:     12.612 | rank:   5.0 | err: 1.3e-02 | eps: 1.5e-03 | 
    # #  19 | time:     13.310 | rank:   5.0 | err: 1.2e-02 | eps: 1.3e-03 | 
    # #  20 | time:     13.993 | rank:   5.0 | err: 1.2e-02 | eps: 1.2e-03 | 
    # #  21 | time:     14.663 | rank:   5.0 | err: 1.1e-02 | eps: 1.0e-03 | 
    # #  22 | time:     15.374 | rank:   5.0 | err: 1.1e-02 | eps: 9.1e-04 | stop: e | 
    # 
    # Build time     :      15.38
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   9.58e-03
    # Error on valid.:   1.06e-02
    # Error on test  :   1.11e-02
    # 

  We can also use rank-adaptive version of the TT-ALS method:

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r=2)
    Y = teneva.als(I_trn, Y_trn, Y, e=1.E-3, I_vld=I_vld, Y_vld=Y_vld, log=True, e_adap=1.E-3, r=5)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.478 | rank:   2.0 | err: 2.1e-01 | 
    # #   1 | time:      1.320 | rank:   5.0 | err: 2.3e-01 | eps: 2.0e-01 | 
    # #   2 | time:      2.155 | rank:   5.0 | err: 1.8e-01 | eps: 1.8e-01 | 
    # #   3 | time:      3.091 | rank:   5.0 | err: 3.4e-01 | eps: 2.7e-01 | 
    # #   4 | time:      3.833 | rank:   5.0 | err: 2.3e-01 | eps: 2.8e-01 | 
    # #   5 | time:      4.685 | rank:   5.0 | err: 1.9e-01 | eps: 1.9e-01 | 
    # #   6 | time:      5.546 | rank:   5.0 | err: 2.5e-01 | eps: 1.9e-01 | 
    # #   7 | time:      6.387 | rank:   5.0 | err: 2.0e-01 | eps: 2.0e-01 | 
    # #   8 | time:      7.140 | rank:   5.0 | err: 1.7e-01 | eps: 1.7e-01 | 
    # #   9 | time:      8.002 | rank:   5.0 | err: 1.7e-01 | eps: 1.6e-01 | 
    # #  10 | time:      8.860 | rank:   5.0 | err: 2.0e-01 | eps: 1.8e-01 | 
    # #  11 | time:      9.699 | rank:   5.0 | err: 2.4e-01 | eps: 2.2e-01 | 
    # #  12 | time:     10.552 | rank:   5.0 | err: 3.0e-01 | eps: 3.3e-01 | 
    # #  13 | time:     11.307 | rank:   5.0 | err: 2.7e-01 | eps: 3.3e-01 | 
    # #  14 | time:     12.134 | rank:   5.0 | err: 2.3e-01 | eps: 2.8e-01 | 
    # #  15 | time:     12.990 | rank:   5.0 | err: 1.6e+00 | eps: 2.1e+00 | 
    # #  16 | time:     13.829 | rank:   5.0 | err: 3.3e-01 | eps: 1.1e+00 | 
    # #  17 | time:     14.661 | rank:   5.0 | err: 2.9e-01 | eps: 3.9e-01 | 
    # #  18 | time:     15.490 | rank:   5.0 | err: 2.8e-01 | eps: 3.0e-01 | 
    # #  19 | time:     16.285 | rank:   5.0 | err: 2.7e-01 | eps: 3.1e-01 | 
    # #  20 | time:     17.161 | rank:   5.0 | err: 1.3e+00 | eps: 1.7e+00 | 
    # #  21 | time:     17.988 | rank:   5.0 | err: 5.9e-01 | eps: 1.3e+00 | 
    # #  22 | time:     18.847 | rank:   5.0 | err: 1.9e-01 | eps: 6.1e-01 | 
    # #  23 | time:     19.606 | rank:   5.0 | err: 1.8e-01 | eps: 2.0e-01 | 
    # #  24 | time:     20.465 | rank:   5.0 | err: 2.1e-01 | eps: 2.0e-01 | 
    # #  25 | time:     21.315 | rank:   5.0 | err: 2.6e-01 | eps: 2.9e-01 | 
    # #  26 | time:     22.178 | rank:   5.0 | err: 1.9e-01 | eps: 2.7e-01 | 
    # #  27 | time:     23.043 | rank:   5.0 | err: 2.1e-01 | eps: 2.0e-01 | 
    # #  28 | time:     23.903 | rank:   5.0 | err: 2.3e-01 | eps: 2.4e-01 | 
    # #  29 | time:     24.711 | rank:   5.0 | err: 1.8e-01 | eps: 2.1e-01 | 
    # #  30 | time:     25.537 | rank:   5.0 | err: 2.2e-01 | eps: 2.0e-01 | 
    # #  31 | time:     26.356 | rank:   5.0 | err: 1.9e-01 | eps: 2.0e-01 | 
    # #  32 | time:     27.164 | rank:   5.0 | err: 1.8e-01 | eps: 1.7e-01 | 
    # #  33 | time:     27.991 | rank:   5.0 | err: 1.8e-01 | eps: 1.5e-01 | 
    # #  34 | time:     28.700 | rank:   5.0 | err: 2.0e-01 | eps: 1.6e-01 | 
    # #  35 | time:     29.514 | rank:   5.0 | err: 2.0e-01 | eps: 1.8e-01 | 
    # #  36 | time:     30.315 | rank:   5.0 | err: 1.9e-01 | eps: 1.9e-01 | 
    # #  37 | time:     31.147 | rank:   5.0 | err: 1.0e+00 | eps: 1.1e+00 | 
    # #  38 | time:     32.009 | rank:   5.0 | err: 1.9e-01 | eps: 8.1e-01 | 
    # #  39 | time:     32.746 | rank:   5.0 | err: 1.8e-01 | eps: 1.6e-01 | 
    # #  40 | time:     33.591 | rank:   5.0 | err: 2.9e-01 | eps: 2.6e-01 | 
    # #  41 | time:     34.394 | rank:   5.0 | err: 2.2e-01 | eps: 2.8e-01 | 
    # #  42 | time:     35.254 | rank:   5.0 | err: 3.0e-01 | eps: 2.7e-01 | 
    # #  43 | time:     36.078 | rank:   5.0 | err: 2.9e-01 | eps: 3.1e-01 | 
    # #  44 | time:     36.801 | rank:   5.0 | err: 2.1e-01 | eps: 2.6e-01 | 
    # #  45 | time:     37.659 | rank:   5.0 | err: 3.5e-01 | eps: 3.4e-01 | 
    # #  46 | time:     38.486 | rank:   5.0 | err: 2.1e-01 | eps: 3.4e-01 | 
    # #  47 | time:     39.312 | rank:   5.0 | err: 2.4e-01 | eps: 1.9e-01 | 
    # #  48 | time:     40.140 | rank:   5.0 | err: 2.4e-01 | eps: 2.4e-01 | 
    # #  49 | time:     40.863 | rank:   5.0 | err: 2.0e-01 | eps: 2.7e-01 | 
    # #  50 | time:     41.695 | rank:   5.0 | err: 3.1e-01 | eps: 3.9e-01 | stop: nswp | 
    # 
    # Build time     :      41.70
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, Y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, Y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, Y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   2.91e-01
    # Error on valid.:   3.09e-01
    # Error on test  :   3.46e-01
    # 


.. autofunction:: teneva.als2

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 10,  12,  14,  16,  18]   # Shape of the tensor

  .. code-block:: python

    m         = 1.E+4                       # Number of calls to target function
    nswp      = 50                          # Sweep number for ALS iterations
    r         = 4                           # TT-rank of the initial random tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    def func(I):
        """Schaffer function."""
        X = teneva.ind_to_poi(I, a, b, n)
        Z = X[:, :-1]**2 + X[:, 1:]**2
        Y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(Y, axis=1)

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

  We build the TT-tensor, which approximates the target function (we generate random initial r-rank approximation in the TT-format using the function "tensor_rand" and then compute the resulting TT-tensor by TT-ALS):

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r)
    Y = teneva.als2(I_trn, Y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :      85.18
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

    # Error on train :   1.32e-02
    # Error on test  :   1.43e-02
    # 



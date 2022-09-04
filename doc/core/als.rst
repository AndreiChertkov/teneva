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

    # Build time     :       3.27
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

    # # pre | time:      0.640 | rank:   5.0 | err: 2.1e-01 | 
    # #   1 | time:      1.361 | rank:   5.0 | err: 7.5e-02 | eps: 2.0e-01 | 
    # #   2 | time:      2.011 | rank:   5.0 | err: 2.8e-02 | eps: 6.3e-02 | 
    # #   3 | time:      2.691 | rank:   5.0 | err: 2.0e-02 | eps: 1.8e-02 | 
    # #   4 | time:      3.350 | rank:   5.0 | err: 1.8e-02 | eps: 4.1e-03 | 
    # #   5 | time:      4.042 | rank:   5.0 | err: 1.8e-02 | eps: 2.5e-03 | 
    # #   6 | time:      4.645 | rank:   5.0 | err: 1.7e-02 | eps: 1.6e-03 | 
    # #   7 | time:      5.331 | rank:   5.0 | err: 1.6e-02 | eps: 1.3e-03 | 
    # #   8 | time:      6.049 | rank:   5.0 | err: 1.5e-02 | eps: 1.3e-03 | 
    # #   9 | time:      6.759 | rank:   5.0 | err: 1.5e-02 | eps: 1.2e-03 | 
    # #  10 | time:      7.441 | rank:   5.0 | err: 1.4e-02 | eps: 1.0e-03 | 
    # #  11 | time:      7.997 | rank:   5.0 | err: 1.4e-02 | eps: 9.4e-04 | 
    # #  12 | time:      8.710 | rank:   5.0 | err: 1.3e-02 | eps: 8.8e-04 | 
    # #  13 | time:      9.440 | rank:   5.0 | err: 1.3e-02 | eps: 8.7e-04 | 
    # #  14 | time:     10.123 | rank:   5.0 | err: 1.2e-02 | eps: 8.8e-04 | 
    # #  15 | time:     10.794 | rank:   5.0 | err: 1.1e-02 | eps: 9.0e-04 | 
    # #  16 | time:     11.458 | rank:   5.0 | err: 1.1e-02 | eps: 9.0e-04 | 
    # #  17 | time:     12.195 | rank:   5.0 | err: 1.0e-02 | eps: 8.9e-04 | 
    # #  18 | time:     12.843 | rank:   5.0 | err: 9.5e-03 | eps: 8.6e-04 | stop: e_vld | 
    # 
    # Build time     :      12.85
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

    # # pre | time:      0.613 | rank:   5.0 | err: 2.1e-01 | 
    # #   1 | time:      1.302 | rank:   5.0 | err: 8.1e-02 | eps: 2.0e-01 | 
    # #   2 | time:      2.012 | rank:   5.0 | err: 8.0e-02 | eps: 4.2e-02 | 
    # #   3 | time:      2.590 | rank:   5.0 | err: 7.5e-02 | eps: 3.2e-02 | 
    # #   4 | time:      3.259 | rank:   5.0 | err: 6.4e-02 | eps: 2.9e-02 | 
    # #   5 | time:      3.944 | rank:   5.0 | err: 4.9e-02 | eps: 2.7e-02 | 
    # #   6 | time:      4.596 | rank:   5.0 | err: 4.0e-02 | eps: 1.8e-02 | 
    # #   7 | time:      5.296 | rank:   5.0 | err: 3.7e-02 | eps: 1.0e-02 | 
    # #   8 | time:      5.950 | rank:   5.0 | err: 3.3e-02 | eps: 1.1e-02 | 
    # #   9 | time:      6.613 | rank:   5.0 | err: 2.7e-02 | eps: 1.6e-02 | 
    # #  10 | time:      7.316 | rank:   5.0 | err: 2.1e-02 | eps: 1.5e-02 | 
    # #  11 | time:      7.938 | rank:   5.0 | err: 1.9e-02 | eps: 6.2e-03 | 
    # #  12 | time:      8.592 | rank:   5.0 | err: 1.8e-02 | eps: 3.7e-03 | 
    # #  13 | time:      9.196 | rank:   5.0 | err: 1.7e-02 | eps: 2.3e-03 | 
    # #  14 | time:      9.890 | rank:   5.0 | err: 1.6e-02 | eps: 2.2e-03 | 
    # #  15 | time:     10.577 | rank:   5.0 | err: 1.5e-02 | eps: 2.2e-03 | 
    # #  16 | time:     11.266 | rank:   5.0 | err: 1.4e-02 | eps: 1.9e-03 | 
    # #  17 | time:     11.934 | rank:   5.0 | err: 1.3e-02 | eps: 1.7e-03 | 
    # #  18 | time:     12.522 | rank:   5.0 | err: 1.3e-02 | eps: 1.5e-03 | 
    # #  19 | time:     13.225 | rank:   5.0 | err: 1.2e-02 | eps: 1.3e-03 | 
    # #  20 | time:     13.901 | rank:   5.0 | err: 1.2e-02 | eps: 1.2e-03 | 
    # #  21 | time:     14.620 | rank:   5.0 | err: 1.1e-02 | eps: 1.0e-03 | 
    # #  22 | time:     15.273 | rank:   5.0 | err: 1.1e-02 | eps: 9.1e-04 | stop: e | 
    # 
    # Build time     :      15.28
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

    # Build time     :      84.03
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



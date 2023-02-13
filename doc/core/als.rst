Module als: construct TT-tensor by TT-ALS
-----------------------------------------


.. automodule:: teneva.core.als


-----




|
|

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
        y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(y, axis=1)

  We prepare train data from the LHS random distribution:

  .. code-block:: python

    I_trn = teneva.sample_lhs(n, m) 
    y_trn = func(I_trn)

  We prepare test data from as a random tensor multi-indices:

  .. code-block:: python

    # Test data:
    
    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
    
    # Function values for the test points:
    y_tst = func(I_tst)

  And now we will build the TT-tensor, which approximates the target function by the TT-ALS method. Note that we use TT-ANOVA as an initial approximation (we can instead generate random initial r-rank approximation in the TT-format using the function "tensor_rand", but convergence will be worse, and there will also be instability of the solution).

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       2.77
    # 

  And now we can check the result:

  .. code-block:: python

    # Fast getter for TT-tensor values:
    get = teneva.getter(Y)                     
    
    # Compute approximation in train points:
    y_our = np.array([get(i) for i in I_trn])
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    # Compute approximation in test points:
    y_our = np.array([get(i) for i in I_tst])
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
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
    y_vld = func(I_vld)

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, nswp, I_vld=I_vld, y_vld=y_vld, e_vld=1.E-2, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.001 | rank:   5.0 | 
    # #   1 | time:      0.742 | rank:   5.0 | e_vld: 7.5e-02 | e: 2.0e-01 | 
    # #   2 | time:      1.377 | rank:   5.0 | e_vld: 2.8e-02 | e: 6.3e-02 | 
    # #   3 | time:      2.093 | rank:   5.0 | e_vld: 2.0e-02 | e: 1.8e-02 | 
    # #   4 | time:      2.735 | rank:   5.0 | e_vld: 1.8e-02 | e: 4.1e-03 | 
    # #   5 | time:      3.475 | rank:   5.0 | e_vld: 1.8e-02 | e: 2.5e-03 | 
    # #   6 | time:      4.225 | rank:   5.0 | e_vld: 1.7e-02 | e: 1.6e-03 | 
    # #   7 | time:      4.834 | rank:   5.0 | e_vld: 1.6e-02 | e: 1.3e-03 | 
    # #   8 | time:      5.543 | rank:   5.0 | e_vld: 1.5e-02 | e: 1.3e-03 | 
    # #   9 | time:      6.185 | rank:   5.0 | e_vld: 1.5e-02 | e: 1.2e-03 | 
    # #  10 | time:      6.920 | rank:   5.0 | e_vld: 1.4e-02 | e: 1.0e-03 | 
    # #  11 | time:      7.530 | rank:   5.0 | e_vld: 1.4e-02 | e: 9.4e-04 | 
    # #  12 | time:      8.194 | rank:   5.0 | e_vld: 1.3e-02 | e: 8.8e-04 | 
    # #  13 | time:      8.924 | rank:   5.0 | e_vld: 1.3e-02 | e: 8.7e-04 | 
    # #  14 | time:      9.534 | rank:   5.0 | e_vld: 1.2e-02 | e: 8.8e-04 | 
    # #  15 | time:     10.267 | rank:   5.0 | e_vld: 1.1e-02 | e: 9.0e-04 | 
    # #  16 | time:     10.886 | rank:   5.0 | e_vld: 1.1e-02 | e: 9.0e-04 | 
    # #  17 | time:     11.646 | rank:   5.0 | e_vld: 1.0e-02 | e: 8.9e-04 | 
    # #  18 | time:     12.301 | rank:   5.0 | e_vld: 9.5e-03 | e: 8.6e-04 | stop: e_vld | 
    # 
    # Build time     :      12.31
    # 

  We can use helper functions to present the resulting accuracy:

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   8.60e-03
    # Error on valid.:   9.52e-03
    # Error on test  :   1.02e-02
    # 

  We may also set the value of relative rate of solution change to stop the iterations:

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, e=1.E-3, I_vld=I_vld, y_vld=y_vld, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.001 | rank:   5.0 | 
    # #   1 | time:      0.620 | rank:   5.0 | e_vld: 8.1e-02 | e: 2.0e-01 | 
    # #   2 | time:      1.300 | rank:   5.0 | e_vld: 8.0e-02 | e: 4.2e-02 | 
    # #   3 | time:      2.010 | rank:   5.0 | e_vld: 7.5e-02 | e: 3.2e-02 | 
    # #   4 | time:      2.620 | rank:   5.0 | e_vld: 6.4e-02 | e: 2.9e-02 | 
    # #   5 | time:      3.368 | rank:   5.0 | e_vld: 4.9e-02 | e: 2.7e-02 | 
    # #   6 | time:      3.983 | rank:   5.0 | e_vld: 4.0e-02 | e: 1.8e-02 | 
    # #   7 | time:      4.704 | rank:   5.0 | e_vld: 3.7e-02 | e: 1.0e-02 | 
    # #   8 | time:      5.357 | rank:   5.0 | e_vld: 3.3e-02 | e: 1.1e-02 | 
    # #   9 | time:      6.072 | rank:   5.0 | e_vld: 2.7e-02 | e: 1.6e-02 | 
    # #  10 | time:      6.876 | rank:   5.0 | e_vld: 2.1e-02 | e: 1.5e-02 | 
    # #  11 | time:      7.526 | rank:   5.0 | e_vld: 1.9e-02 | e: 6.2e-03 | 
    # #  12 | time:      8.258 | rank:   5.0 | e_vld: 1.8e-02 | e: 3.7e-03 | 
    # #  13 | time:      8.878 | rank:   5.0 | e_vld: 1.7e-02 | e: 2.3e-03 | 
    # #  14 | time:      9.584 | rank:   5.0 | e_vld: 1.6e-02 | e: 2.2e-03 | 
    # #  15 | time:     10.183 | rank:   5.0 | e_vld: 1.5e-02 | e: 2.2e-03 | 
    # #  16 | time:     10.843 | rank:   5.0 | e_vld: 1.4e-02 | e: 1.9e-03 | 
    # #  17 | time:     11.544 | rank:   5.0 | e_vld: 1.3e-02 | e: 1.7e-03 | 
    # #  18 | time:     12.148 | rank:   5.0 | e_vld: 1.3e-02 | e: 1.5e-03 | 
    # #  19 | time:     12.854 | rank:   5.0 | e_vld: 1.2e-02 | e: 1.3e-03 | 
    # #  20 | time:     13.456 | rank:   5.0 | e_vld: 1.2e-02 | e: 1.2e-03 | 
    # #  21 | time:     14.222 | rank:   5.0 | e_vld: 1.1e-02 | e: 1.0e-03 | 
    # #  22 | time:     14.869 | rank:   5.0 | e_vld: 1.1e-02 | e: 9.1e-04 | stop: e | 
    # 
    # Build time     :      14.87
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   9.58e-03
    # Error on valid.:   1.06e-02
    # Error on test  :   1.11e-02
    # 

  We can also use rank-adaptive version of the TT-ALS method (DRAFT !!!):

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r=2)
    Y = teneva.als(I_trn, y_trn, Y, nswp=10,
        I_vld=I_vld, y_vld=y_vld, r=5, e_adap=1.E-2, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.001 | rank:   2.0 | 
    # #   1 | time:      0.718 | rank:   5.0 | e_vld: 2.3e-01 | e: 2.0e-01 | 
    # #   2 | time:      1.530 | rank:   5.0 | e_vld: 1.8e-01 | e: 1.8e-01 | 
    # #   3 | time:      2.456 | rank:   5.0 | e_vld: 3.4e-01 | e: 2.7e-01 | 
    # #   4 | time:      3.254 | rank:   5.0 | e_vld: 2.3e-01 | e: 2.8e-01 | 
    # #   5 | time:      4.151 | rank:   5.0 | e_vld: 1.9e-01 | e: 1.8e-01 | 
    # #   6 | time:      4.902 | rank:   5.0 | e_vld: 2.6e-01 | e: 2.9e-01 | 
    # #   7 | time:      5.775 | rank:   5.0 | e_vld: 1.9e-01 | e: 2.9e-01 | 
    # #   8 | time:      6.508 | rank:   5.0 | e_vld: 1.9e-01 | e: 1.9e-01 | 
    # #   9 | time:      7.327 | rank:   5.0 | e_vld: 2.1e-01 | e: 2.3e-01 | 
    # #  10 | time:      8.201 | rank:   5.0 | e_vld: 3.1e-01 | e: 3.9e-01 | stop: nswp | 
    # 
    # Build time     :       8.21
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   2.85e-01
    # Error on valid.:   3.05e-01
    # Error on test  :   3.65e-01
    # 




|
|


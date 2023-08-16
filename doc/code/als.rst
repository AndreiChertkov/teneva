Module als: construct TT-tensor by TT-ALS
-----------------------------------------


.. automodule:: teneva.als


-----




|
|

.. autofunction:: teneva.als.als

  **Examples**:

  .. code-block:: python

    d    = 5                          # Dimension of the function
    a    = [-5., -4., -3., -2., -1.]  # Lower bounds for spatial grid
    b    = [+6., +3., +3., +1., +2.]  # Upper bounds for spatial grid
    n    = [ 10,  12,  14,  16,  18]  # Shape of the tensor

  .. code-block:: python

    m    = 1.E+4                      # Number of calls to target function
    nswp = 50                         # Sweep number for ALS iterations
    r    = 5                          # TT-rank of the initial random tensor

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

    I_trn = teneva.sample_lhs(n, m, seed=42) 
    y_trn = func(I_trn)

  We prepare test data from the random tensor multi-indices:

  .. code-block:: python

    I_tst = teneva.sample_rand(n, 1.E+4, seed=42) 
    y_tst = func(I_tst)

  And now we will build the TT-tensor, which approximates the target function by the TT-ALS method. Note that we use TT-ANOVA as an initial approximation (we can instead generate random initial r-rank approximation in the TT-format using the function "rand", but convergence will be worse, and there will also be instability of the solution).

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       1.79
    # 

  And now we can check the result:

  .. code-block:: python

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

    # Error on train :   1.23e-03
    # Error on test  :   1.39e-03
    # 

  We can also set a validation data set and specify as a stop criterion the accuracy of the TT-approximation on this data (and we can also present the logs):

  .. code-block:: python

    I_vld = teneva.sample_rand(n, 1.E+3, seed=99) 
    y_vld = func(I_vld)

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, nswp, I_vld=I_vld, y_vld=y_vld, e_vld=1.E-2, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.002 | rank:   5.0 | e_vld: 2.1e-01 | 
    # #   1 | time:      0.047 | rank:   5.0 | e_vld: 1.1e-01 | e: 1.9e-01 | 
    # #   2 | time:      0.083 | rank:   5.0 | e_vld: 1.5e-02 | e: 1.1e-01 | 
    # #   3 | time:      0.117 | rank:   5.0 | e_vld: 3.8e-03 | e: 1.5e-02 | stop: e_vld | 
    # 
    # Build time     :       0.12
    # 

  We can use helper functions to present the resulting accuracy:

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   3.44e-03
    # Error on valid.:   3.81e-03
    # Error on test  :   3.79e-03
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

    # # pre | time:      0.002 | rank:   5.0 | e_vld: 2.1e-01 | 
    # #   1 | time:      0.049 | rank:   5.0 | e_vld: 1.1e-01 | e: 1.9e-01 | 
    # #   2 | time:      0.087 | rank:   5.0 | e_vld: 2.2e-02 | e: 1.1e-01 | 
    # #   3 | time:      0.124 | rank:   5.0 | e_vld: 1.0e-02 | e: 1.6e-02 | 
    # #   4 | time:      0.158 | rank:   5.0 | e_vld: 6.5e-03 | e: 4.1e-03 | 
    # #   5 | time:      0.197 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.3e-03 | 
    # #   6 | time:      0.239 | rank:   5.0 | e_vld: 3.7e-03 | e: 1.2e-03 | 
    # #   7 | time:      0.278 | rank:   5.0 | e_vld: 3.1e-03 | e: 7.7e-04 | stop: e | 
    # 
    # Build time     :       0.28
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   2.63e-03
    # Error on valid.:   3.11e-03
    # Error on test  :   2.94e-03
    # 

  We may also pass callback function (it will be called after every sweep):

  .. code-block:: python

    def cb(Y, info, opts):
        e = teneva.accuracy(Y, opts['Yold'])
        print(f'Callback : e={e:-7.1e}')
        if info['nswp'] == 5:
            # Stop the algorithm's work
            return True

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, e=1.E-10, cb=cb, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.001 | rank:   5.0 | 
    # Callback : e=1.9e-01
    # #   1 | time:      0.047 | rank:   5.0 | e: 1.9e-01 | 
    # Callback : e=1.1e-01
    # #   2 | time:      0.082 | rank:   5.0 | e: 1.1e-01 | 
    # Callback : e=1.8e-02
    # #   3 | time:      0.122 | rank:   5.0 | e: 1.8e-02 | 
    # Callback : e=2.3e-03
    # #   4 | time:      0.162 | rank:   5.0 | e: 2.3e-03 | 
    # Callback : e=8.8e-04
    # #   5 | time:      0.202 | rank:   5.0 | e: 8.8e-04 | stop: cb | 
    # 
    # Build time     :       0.21
    # 

  We can also use rank-adaptive version of the TT-ALS method (note that result is very sensitive to "r" and "lamb" parameter values):

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r=2)
    Y = teneva.als(I_trn, y_trn, Y, nswp=5,
        I_vld=I_vld, y_vld=y_vld, r=5, e_adap=1.E-2, lamb=0.00001, log=True)
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.002 | rank:   2.0 | e_vld: 2.1e-01 | 
    # #   1 | time:      0.180 | rank:   5.0 | e_vld: 2.1e-02 | e: 2.2e-01 | 
    # #   2 | time:      0.353 | rank:   3.9 | e_vld: 6.9e-03 | e: 2.0e-02 | 
    # #   3 | time:      0.514 | rank:   3.9 | e_vld: 6.5e-03 | e: 2.6e-03 | 
    # #   4 | time:      0.675 | rank:   4.2 | e_vld: 7.6e-03 | e: 6.3e-03 | 
    # #   5 | time:      0.840 | rank:   3.9 | e_vld: 6.6e-03 | e: 6.3e-03 | stop: nswp | 
    # 
    # Build time     :       0.85
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   6.54e-03
    # Error on valid.:   6.58e-03
    # Error on test  :   7.22e-03
    # 

  We can also specify weights for elements of the training dataset. In the following example, we set increased weights for the first 1000 points from the set and expect that the accuracy of the result on them will be higher than on the rest:

  .. code-block:: python

    m = len(I_trn)
    dm = 1000
    I_trn1, y_trn1 = I_trn[:dm], y_trn[:dm]
    I_trn2, y_trn2 = I_trn[dm:], y_trn[dm:]
    
    w = np.ones(m)
    w[:dm] = 100.

  .. code-block:: python

    t = tpc()
    Y = teneva.anova(I_trn, y_trn, r)
    Y = teneva.als(I_trn, y_trn, Y, w=w)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       1.96
    # 

  .. code-block:: python

    print(f'Error full data : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error for part1 : {teneva.accuracy_on_data(Y, I_trn1, y_trn1):-10.2e}')
    print(f'Error for part2 : {teneva.accuracy_on_data(Y, I_trn2, y_trn2):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error full data :   9.89e-03
    # Error for part1 :   2.68e-03
    # Error for part2 :   1.04e-02
    # 




|
|


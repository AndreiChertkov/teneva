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

    # Build time     :       1.74
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

    # Error on train :   1.27e-03
    # Error on test  :   1.43e-03
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
    # #   1 | time:      0.043 | rank:   5.0 | e_vld: 1.1e-01 | e: 1.9e-01 | 
    # #   2 | time:      0.078 | rank:   5.0 | e_vld: 1.7e-02 | e: 1.1e-01 | 
    # #   3 | time:      0.113 | rank:   5.0 | e_vld: 4.8e-03 | e: 1.5e-02 | stop: e_vld | 
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

    # Error on train :   4.03e-03
    # Error on valid.:   4.82e-03
    # Error on test  :   4.86e-03
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
    # #   1 | time:      0.047 | rank:   5.0 | e_vld: 1.1e-01 | e: 1.9e-01 | 
    # #   2 | time:      0.083 | rank:   5.0 | e_vld: 2.1e-02 | e: 1.1e-01 | 
    # #   3 | time:      0.118 | rank:   5.0 | e_vld: 1.2e-02 | e: 1.6e-02 | 
    # #   4 | time:      0.153 | rank:   5.0 | e_vld: 1.0e-02 | e: 4.1e-03 | 
    # #   5 | time:      0.188 | rank:   5.0 | e_vld: 8.1e-03 | e: 2.5e-03 | 
    # #   6 | time:      0.224 | rank:   5.0 | e_vld: 6.6e-03 | e: 1.6e-03 | 
    # #   7 | time:      0.260 | rank:   5.0 | e_vld: 5.5e-03 | e: 1.2e-03 | 
    # #   8 | time:      0.296 | rank:   5.0 | e_vld: 4.6e-03 | e: 9.7e-04 | stop: e | 
    # 
    # Build time     :       0.30
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   3.41e-03
    # Error on valid.:   4.62e-03
    # Error on test  :   4.04e-03
    # 

  .. code-block:: python

    print(f'Error on train : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error on valid.: {teneva.accuracy_on_data(Y, I_vld, y_vld):-10.2e}')
    print(f'Error on test  : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   3.41e-03
    # Error on valid.:   4.62e-03
    # Error on test  :   4.04e-03
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
    # #   1 | time:      0.044 | rank:   5.0 | e: 1.9e-01 | 
    # Callback : e=1.1e-01
    # #   2 | time:      0.080 | rank:   5.0 | e: 1.1e-01 | 
    # Callback : e=2.3e-02
    # #   3 | time:      0.116 | rank:   5.0 | e: 2.3e-02 | 
    # Callback : e=3.6e-03
    # #   4 | time:      0.152 | rank:   5.0 | e: 3.6e-03 | 
    # Callback : e=1.4e-03
    # #   5 | time:      0.188 | rank:   5.0 | e: 1.4e-03 | stop: cb | 
    # 
    # Build time     :       0.19
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
    # #   1 | time:      0.163 | rank:   5.0 | e_vld: 2.1e-02 | e: 2.2e-01 | 
    # #   2 | time:      0.340 | rank:   3.9 | e_vld: 6.9e-03 | e: 2.0e-02 | 
    # #   3 | time:      0.497 | rank:   3.9 | e_vld: 6.5e-03 | e: 2.6e-03 | 
    # #   4 | time:      0.653 | rank:   4.2 | e_vld: 7.6e-03 | e: 6.3e-03 | 
    # #   5 | time:      0.811 | rank:   3.9 | e_vld: 6.6e-03 | e: 6.3e-03 | stop: nswp | 
    # 
    # Build time     :       0.82
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

    # Build time     :       1.92
    # 

  .. code-block:: python

    print(f'Error full data : {teneva.accuracy_on_data(Y, I_trn, y_trn):-10.2e}')
    print(f'Error for part1 : {teneva.accuracy_on_data(Y, I_trn1, y_trn1):-10.2e}')
    print(f'Error for part2 : {teneva.accuracy_on_data(Y, I_trn2, y_trn2):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error full data :   8.23e-03
    # Error for part1 :   2.34e-03
    # Error for part2 :   8.64e-03
    # 

  (NEW OPTION) We can also use batch updates:

  .. code-block:: python

    t = tpc()
    
    Y = teneva.anova(I_trn, y_trn, r)
    
    Nn = I_trn.shape[0]
    bs = 1000 # batch size
    iters = 10
    
    for i_iter in range(iters):
        idx = np.random.permutation(Nn)
        for i in range(0, Nn, bs):
            I_trn_cur = I_trn[idx[i:i+bs]]
            y_trn_cur = y_trn[idx[i:i+bs]]
    
            Y = teneva.als(I_trn_cur, y_trn_cur, Y, e=1.E-3, I_vld=I_vld, y_vld=y_vld, log=True, 
                           update_sol=True, lamb=2**(i_iter/(iters/30)))
            
    t = tpc() - t
    
    print(f'\nBuild time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.001 | rank:   5.0 | e_vld: 2.1e-01 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 1.6e-01 | e: 1.5e-01 | 
    # #   2 | time:      0.035 | rank:   5.0 | e_vld: 1.4e-01 | e: 8.2e-02 | 
    # #   3 | time:      0.052 | rank:   5.0 | e_vld: 9.6e-02 | e: 7.1e-02 | 
    # #   4 | time:      0.068 | rank:   5.0 | e_vld: 5.4e-02 | e: 7.6e-02 | 
    # #   5 | time:      0.084 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.9e-02 | 
    # #   6 | time:      0.100 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.2e-02 | 
    # #   7 | time:      0.116 | rank:   5.0 | e_vld: 5.1e-02 | e: 7.5e-03 | 
    # #   8 | time:      0.132 | rank:   5.0 | e_vld: 5.1e-02 | e: 5.9e-03 | 
    # #   9 | time:      0.148 | rank:   5.0 | e_vld: 5.2e-02 | e: 5.0e-03 | 
    # #  10 | time:      0.164 | rank:   5.0 | e_vld: 5.2e-02 | e: 4.3e-03 | 
    # #  11 | time:      0.181 | rank:   5.0 | e_vld: 5.2e-02 | e: 3.8e-03 | 
    # #  12 | time:      0.198 | rank:   5.0 | e_vld: 5.2e-02 | e: 3.4e-03 | 
    # #  13 | time:      0.216 | rank:   5.0 | e_vld: 5.2e-02 | e: 3.1e-03 | 
    # #  14 | time:      0.232 | rank:   5.0 | e_vld: 5.1e-02 | e: 3.1e-03 | 
    # #  15 | time:      0.248 | rank:   5.0 | e_vld: 5.1e-02 | e: 2.8e-03 | 
    # #  16 | time:      0.263 | rank:   5.0 | e_vld: 5.1e-02 | e: 2.4e-03 | 
    # #  17 | time:      0.279 | rank:   5.0 | e_vld: 5.1e-02 | e: 2.2e-03 | 
    # #  18 | time:      0.294 | rank:   5.0 | e_vld: 5.1e-02 | e: 2.1e-03 | 
    # #  19 | time:      0.310 | rank:   5.0 | e_vld: 5.1e-02 | e: 2.0e-03 | 
    # #  20 | time:      0.326 | rank:   5.0 | e_vld: 5.1e-02 | e: 1.9e-03 | 
    # #  21 | time:      0.342 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.7e-03 | 
    # #  22 | time:      0.358 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.6e-03 | 
    # #  23 | time:      0.374 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.5e-03 | 
    # #  24 | time:      0.391 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.4e-03 | 
    # #  25 | time:      0.408 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.3e-03 | 
    # #  26 | time:      0.425 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.2e-03 | 
    # #  27 | time:      0.442 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.2e-03 | 
    # #  28 | time:      0.461 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.1e-03 | 
    # #  29 | time:      0.479 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.1e-03 | 
    # #  30 | time:      0.495 | rank:   5.0 | e_vld: 5.0e-02 | e: 1.0e-03 | 
    # #  31 | time:      0.512 | rank:   5.0 | e_vld: 5.0e-02 | e: 9.9e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.0e-02 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 3.4e-02 | e: 3.8e-02 | 
    # #   2 | time:      0.034 | rank:   5.0 | e_vld: 3.2e-02 | e: 9.5e-03 | 
    # #   3 | time:      0.051 | rank:   5.0 | e_vld: 3.1e-02 | e: 5.3e-03 | 
    # #   4 | time:      0.068 | rank:   5.0 | e_vld: 3.1e-02 | e: 3.6e-03 | 
    # #   5 | time:      0.088 | rank:   5.0 | e_vld: 3.0e-02 | e: 2.8e-03 | 
    # #   6 | time:      0.108 | rank:   5.0 | e_vld: 3.0e-02 | e: 2.3e-03 | 
    # #   7 | time:      0.127 | rank:   5.0 | e_vld: 3.0e-02 | e: 2.0e-03 | 
    # #   8 | time:      0.144 | rank:   5.0 | e_vld: 2.9e-02 | e: 1.8e-03 | 
    # #   9 | time:      0.159 | rank:   5.0 | e_vld: 2.9e-02 | e: 1.6e-03 | 
    # #  10 | time:      0.174 | rank:   5.0 | e_vld: 2.9e-02 | e: 1.4e-03 | 
    # #  11 | time:      0.189 | rank:   5.0 | e_vld: 2.8e-02 | e: 1.3e-03 | 
    # #  12 | time:      0.205 | rank:   5.0 | e_vld: 2.8e-02 | e: 1.2e-03 | 
    # #  13 | time:      0.220 | rank:   5.0 | e_vld: 2.8e-02 | e: 1.1e-03 | 
    # #  14 | time:      0.235 | rank:   5.0 | e_vld: 2.8e-02 | e: 9.7e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 2.8e-02 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 2.2e-02 | e: 2.1e-02 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 2.1e-02 | e: 5.1e-03 | 
    # #   3 | time:      0.046 | rank:   5.0 | e_vld: 2.1e-02 | e: 2.9e-03 | 
    # #   4 | time:      0.063 | rank:   5.0 | e_vld: 2.0e-02 | e: 2.1e-03 | 
    # #   5 | time:      0.079 | rank:   5.0 | e_vld: 2.0e-02 | e: 1.6e-03 | 
    # #   6 | time:      0.096 | rank:   5.0 | e_vld: 1.9e-02 | e: 1.3e-03 | 
    # #   7 | time:      0.113 | rank:   5.0 | e_vld: 1.9e-02 | e: 1.1e-03 | 
    # #   8 | time:      0.129 | rank:   5.0 | e_vld: 1.9e-02 | e: 9.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 1.9e-02 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 1.5e-02 | e: 1.4e-02 | 
    # #   2 | time:      0.032 | rank:   5.0 | e_vld: 1.4e-02 | e: 3.0e-03 | 
    # #   3 | time:      0.047 | rank:   5.0 | e_vld: 1.3e-02 | e: 1.8e-03 | 
    # #   4 | time:      0.062 | rank:   5.0 | e_vld: 1.3e-02 | e: 1.2e-03 | 
    # #   5 | time:      0.077 | rank:   5.0 | e_vld: 1.3e-02 | e: 9.4e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 1.3e-02 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 1.1e-02 | e: 9.2e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 1.1e-02 | e: 2.4e-03 | 
    # #   3 | time:      0.048 | rank:   5.0 | e_vld: 1.1e-02 | e: 1.4e-03 | 
    # #   4 | time:      0.065 | rank:   5.0 | e_vld: 1.1e-02 | e: 9.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 1.1e-02 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 9.2e-03 | e: 7.5e-03 | 
    # #   2 | time:      0.034 | rank:   5.0 | e_vld: 9.1e-03 | e: 2.0e-03 | 
    # #   3 | time:      0.050 | rank:   5.0 | e_vld: 9.0e-03 | e: 1.1e-03 | 
    # #   4 | time:      0.065 | rank:   5.0 | e_vld: 9.0e-03 | e: 8.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 9.0e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 8.1e-03 | e: 6.1e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 8.1e-03 | e: 1.7e-03 | 
    # #   3 | time:      0.046 | rank:   5.0 | e_vld: 8.2e-03 | e: 9.4e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 8.2e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 7.5e-03 | e: 5.8e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 7.6e-03 | e: 1.5e-03 | 
    # #   3 | time:      0.046 | rank:   5.0 | e_vld: 7.6e-03 | e: 8.6e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 7.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 7.1e-03 | e: 5.1e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 7.2e-03 | e: 1.4e-03 | 
    # #   3 | time:      0.046 | rank:   5.0 | e_vld: 7.2e-03 | e: 7.7e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 7.2e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 6.3e-03 | e: 4.6e-03 | 
    # #   2 | time:      0.032 | rank:   5.0 | e_vld: 6.2e-03 | e: 1.2e-03 | 
    # #   3 | time:      0.049 | rank:   5.0 | e_vld: 6.2e-03 | e: 7.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 6.2e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.9e-03 | e: 2.9e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 5.8e-03 | e: 8.2e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.8e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.5e-03 | e: 2.5e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 5.5e-03 | e: 8.3e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.5e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.5e-03 | e: 3.0e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 5.6e-03 | e: 7.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.5e-03 | e: 3.1e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 5.5e-03 | e: 8.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.5e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.4e-03 | e: 2.9e-03 | 
    # #   2 | time:      0.032 | rank:   5.0 | e_vld: 5.5e-03 | e: 7.9e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.5e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 5.5e-03 | e: 2.8e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.6e-03 | e: 7.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 5.4e-03 | e: 3.3e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.5e-03 | e: 7.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.5e-03 | 
    # #   1 | time:      0.021 | rank:   5.0 | e_vld: 5.2e-03 | e: 2.8e-03 | 
    # #   2 | time:      0.038 | rank:   5.0 | e_vld: 5.3e-03 | e: 7.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.3e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 5.0e-03 | e: 3.0e-03 | 
    # #   2 | time:      0.035 | rank:   5.0 | e_vld: 5.1e-03 | e: 7.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.1e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 5.2e-03 | e: 3.0e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.3e-03 | e: 7.6e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.3e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 5.1e-03 | e: 2.3e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.1e-03 | e: 5.6e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.1e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.9e-03 | e: 2.0e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.0e-03 | e: 6.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.0e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.9e-03 | e: 1.8e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 5.0e-03 | e: 5.0e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.0e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.7e-03 | e: 1.8e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 4.8e-03 | e: 5.2e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.8e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.8e-03 | e: 2.0e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 4.9e-03 | e: 5.4e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.9e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 5.0e-03 | e: 2.1e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 5.1e-03 | e: 5.3e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 5.1e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.8e-03 | e: 1.8e-03 | 
    # #   2 | time:      0.031 | rank:   5.0 | e_vld: 4.8e-03 | e: 5.4e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.8e-03 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 4.8e-03 | e: 2.0e-03 | 
    # #   2 | time:      0.040 | rank:   5.0 | e_vld: 4.9e-03 | e: 5.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.9e-03 | 
    # #   1 | time:      0.022 | rank:   5.0 | e_vld: 4.8e-03 | e: 2.2e-03 | 
    # #   2 | time:      0.040 | rank:   5.0 | e_vld: 4.9e-03 | e: 5.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.9e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.8e-03 | e: 1.9e-03 | 
    # #   2 | time:      0.033 | rank:   5.0 | e_vld: 4.9e-03 | e: 5.0e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.9e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.8e-03 | e: 9.7e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.8e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.7e-03 | e: 9.4e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.8e-03 | e: 8.0e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.8e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.7e-03 | e: 8.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.7e-03 | e: 8.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.7e-03 | e: 8.6e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.7e-03 | e: 1.1e-03 | 
    # #   2 | time:      0.035 | rank:   5.0 | e_vld: 4.7e-03 | e: 3.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 9.5e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.7e-03 | e: 7.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.7e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 8.9e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.3e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.3e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.2e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.1e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.9e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.6e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.8e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.7e-04 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.4e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.8e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.6e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.3e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.5e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.3e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.2e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.8e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.4e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.4e-05 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.4e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.7e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.1e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.2e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.9e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.5e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.4e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.0e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.4e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.0e-06 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 5.2e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.6e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.0e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.6e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.1e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.5e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.9e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.8e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.7e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.7e-07 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.6e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.2e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.5e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 5.7e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 4.6e-03 | e: 6.1e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.019 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.2e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 5.8e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.5e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 6.6e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.2e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 4.2e-09 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.5e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 2.0e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 0.0e+00 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.016 | rank:   5.0 | e_vld: 4.6e-03 | e: 0.0e+00 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.1e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 1.7e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.018 | rank:   5.0 | e_vld: 4.6e-03 | e: 3.3e-08 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 0.0e+00 | stop: e | 
    # # pre | time:      0.001 | rank:   5.0 | e_vld: 4.6e-03 | 
    # #   1 | time:      0.017 | rank:   5.0 | e_vld: 4.6e-03 | e: 0.0e+00 | stop: e | 
    # 
    # Build time     :       3.17
    # 




|
|


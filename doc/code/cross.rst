Module cross: construct TT-tensor by TT-cross
---------------------------------------------


.. automodule:: teneva.cross


-----




|
|

.. autofunction:: teneva.cross.cross

  **Examples**:

  .. code-block:: python

    d = 5                           # Dimension of the function
    a = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n = [ 20,  18,  16,  14,  12]   # Shape of the tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape "[samples, dimension]", which are transformed into points "X" of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)

  We prepare test data from random tensor multi-indices:

  .. code-block:: python

    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T
    
    # Function values for the test points:
    y_tst = func(I_tst)

  We set the parameters of the TT-cross algorithm:

  .. code-block:: python

    m         = 8.E+3  # Number of calls to target function
    e         = None   # Desired accuracy
    nswp      = None   # Sweep number
    r         = 1      # TT-rank of the initial tensor
    dr_min    = 1      # Cross parameter (minimum number of added rows)
    dr_max    = 3      # Cross parameter (maximum number of added rows)

  We build the TT-tensor, which approximates the target function (note that "cache" is optional [it may be None] and it is effictive only for complex functions with long computing time for one call):

  .. code-block:: python

    t = tpc()
    info, cache = {}, {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache)
    Y = teneva.truncate(Y, 1.E-4) # We round the result at the end
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.16
    # Evals func           :       6734
    # Cache uses           :       6272
    # Iter accuracy        :   0.00e+00
    # Sweep number         :          3
    # Stop condition       :          m
    # TT-rank of pure res  :       11.0
    # TT-rank of trunc res :        3.0
    # 

  And now we can check the result:

  .. code-block:: python

    # Compute approximation in test points:
    y_our = teneva.get_many(Y, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    print(f'Error on test        : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on test        :   3.36e-15
    # 

  Note that "accuracy_on_data" function may be used instead:

  .. code-block:: python

    e_tst = teneva.accuracy_on_data(Y, I_tst, y_tst)
    print(f'Error on test        : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on test        :   3.36e-15
    # 

  We may not specify a limit on the number of requests ("m") to the objective function. In this case, the algorithm will end when the maximum number of iterations ("nswp") is reached or after convergence ("e") [note the value of the stop condition in the output below]:

  .. code-block:: python

    m         = None   # Number of calls to target function
    e         = 1.E-4  # Desired accuracy
    nswp      = 10     # Sweep number (to ensure that it will not work very long)

  .. code-block:: python

    t = tpc()
    info, cache = {}, {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.07
    # Evals func           :       3614
    # Cache uses           :       3084
    # Iter accuracy        :   1.02e-08
    # Sweep number         :          3
    # Stop condition       :          e
    # TT-rank of pure res  :        8.0
    # TT-rank of trunc res :        3.0
    # Error on test        :   6.43e-16
    # 

  We may disable the cache (note that the number of requests to the objective function in this case will be more, but the running time will be less, since this function is calculated very quickly):

  .. code-block:: python

    t = tpc()
    info, cache = {}, None
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.04
    # Evals func           :       6698
    # Cache uses           :          0
    # Iter accuracy        :   1.02e-08
    # Sweep number         :          3
    # Stop condition       :          e
    # TT-rank of pure res  :        8.0
    # TT-rank of trunc res :        3.0
    # Error on test        :   6.43e-16
    # 

  We may also specify all stop conditions. In this case, the algorithm will terminate when at least one stop criterion is met:

  .. code-block:: python

    m         = 1.E+4  # Number of calls to target function
    e         = 1.E-16 # Desired accuracy
    nswp      = 10     # Sweep number (to ensure that it will not work very long)

  .. code-block:: python

    t = tpc()
    info, cache = {}, None
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.07
    # Evals func           :       9126
    # Cache uses           :          0
    # Iter accuracy        :   2.06e-08
    # Sweep number         :          3
    # Stop condition       :          m
    # TT-rank of pure res  :        9.4
    # TT-rank of trunc res :        3.0
    # Error on test        :   1.46e-14
    # 

  .. code-block:: python

    m         = 1.E+4  # Number of calls to target function
    e         = 1.E-16 # Desired accuracy
    nswp      = 1      # Sweep number (to ensure that it will not work very long)

  .. code-block:: python

    t = tpc()
    info, cache = {}, None
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.01
    # Evals func           :        512
    # Cache uses           :          0
    # Iter accuracy        :   5.73e+05
    # Sweep number         :          1
    # Stop condition       :       nswp
    # TT-rank of pure res  :        3.0
    # TT-rank of trunc res :        3.0
    # Error on test        :   1.46e-01
    # 

  We can also set a validation data set and specify as a stop criterion the accuracy of the TT-approximation on this data (and we can also present the logs):

  .. code-block:: python

    d = 25      # Dimension of the function
    n = 64      # Shape of the tensor
    a = -100.   # Lower bounds for spatial grid
    b = +100.   # Upper bounds for spatial grid

  .. code-block:: python

    def func(I):
        """Schaffer function."""
        X = teneva.ind_to_poi(I, a, b, n)
        Z = X[:, :-1]**2 + X[:, 1:]**2
        y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(y, axis=1)

  .. code-block:: python

    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(n, m_tst) for i in range(d)]).T
    
    # Function values for the test points:
    y_tst = func(I_tst)

  .. code-block:: python

    # Number of validation points:
    m_vld = int(1.E+3)
    
    # Random multi-indices for the validation points:
    I_vld = np.vstack([np.random.choice(n, m_vld) for i in range(d)]).T
    
    # Function values for the validation points:
    y_vld = func(I_vld)

  .. code-block:: python

    e_vld = 1.E-3  # Desired error on validation data

  .. code-block:: python

    t = tpc()
    info = {}
    Y = teneva.rand([n]*d, r=1)
    Y = teneva.cross(func, Y, dr_max=1, I_vld=I_vld, y_vld=y_vld,
        e_vld=e_vld, info=info, log=True)
    Y = teneva.truncate(Y, 1.E-4) # We round the result
    t = tpc() - t
    
    print()
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.016 | evals: 0.00e+00 | rank:   1.0 | e_vld: 1.0e+00 | 
    # #   1 | time:      0.054 | evals: 1.23e+04 | rank:   3.0 | e_vld: 1.7e-01 | e: 6.9e+06 | 
    # #   2 | time:      0.129 | evals: 6.04e+04 | rank:   5.0 | e_vld: 4.3e-02 | e: 2.3e-01 | 
    # #   3 | time:      0.311 | evals: 1.68e+05 | rank:   7.0 | e_vld: 4.8e-02 | e: 7.7e-02 | 
    # #   4 | time:      0.816 | evals: 3.58e+05 | rank:   9.0 | e_vld: 4.7e-02 | e: 2.1e-02 | 
    # #   5 | time:      1.803 | evals: 6.55e+05 | rank:  11.0 | e_vld: 5.5e-03 | e: 4.9e-02 | 
    # #   6 | time:      4.828 | evals: 1.08e+06 | rank:  13.0 | e_vld: 3.9e-03 | e: 5.9e-03 | 
    # #   7 | time:     10.271 | evals: 1.66e+06 | rank:  15.0 | e_vld: 2.3e-03 | e: 4.1e-03 | 
    # #   8 | time:     19.424 | evals: 2.42e+06 | rank:  17.0 | e_vld: 1.6e-03 | e: 2.4e-03 | 
    # #   9 | time:     34.129 | evals: 3.38e+06 | rank:  19.0 | e_vld: 9.9e-04 | e: 1.5e-03 | stop: e_vld | 
    # 
    # Build time           :      34.15
    # Evals func           :    3379200
    # Cache uses           :          0
    # Iter accuracy        :   1.52e-03
    # Sweep number         :          9
    # Stop condition       :      e_vld
    # TT-rank of pure res  :       19.0
    # TT-rank of trunc res :       18.0
    # Error on test        :   9.99e-04
    # 

  We may also, for example, use cache and add restriction on the number of requests:

  .. code-block:: python

    m         = 1.E+6  # Number of calls to target function
    e_vld     = 1.E-3  # Desired error on validation data

  .. code-block:: python

    Y = teneva.rand([n]*d, r=1)
    Y = teneva.cross(func, Y, m=m, dr_max=1, I_vld=I_vld, y_vld=y_vld,
        e_vld=e_vld, info={}, cache={}, log=True)
    Y = teneva.truncate(Y, 1.e-4)
    
    print()
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.017 | evals: 0.00e+00 (+ 0.00e+00) | rank:   1.0 | e_vld: 1.0e+00 | 
    # #   1 | time:      0.139 | evals: 1.20e+04 (+ 3.20e+02) | rank:   3.0 | e_vld: 1.5e-01 | e: 1.4e+07 | 
    # #   2 | time:      0.583 | evals: 5.84e+04 (+ 2.04e+03) | rank:   5.0 | e_vld: 3.2e-02 | e: 2.0e-01 | 
    # #   3 | time:      1.542 | evals: 1.58e+05 (+ 9.95e+03) | rank:   7.0 | e_vld: 4.5e-02 | e: 6.3e-02 | 
    # #   4 | time:      3.442 | evals: 3.24e+05 (+ 3.40e+04) | rank:   9.0 | e_vld: 2.9e-02 | e: 2.6e-02 | 
    # #   5 | time:      6.747 | evals: 5.96e+05 (+ 5.92e+04) | rank:  11.0 | e_vld: 6.3e-03 | e: 2.8e-02 | 
    # #   6 | time:     13.962 | evals: 9.86e+05 (+ 9.59e+04) | rank:  13.0 | e_vld: 3.6e-03 | e: 7.0e-03 | 
    # #   6 | time:     18.060 | evals: 9.94e+05 (+ 1.13e+05) | rank:  13.1 | e_vld: 3.6e-03 | e: 9.1e-04 | stop: m | 
    # 
    # TT-rank of trunc res :       12.3
    # Error on test        :   3.65e-03
    # 

  We may also pass callback function (it will be called after every sweep):

  .. code-block:: python

    def cb(Y, info, opts):
        e = teneva.accuracy(Y, opts['Yold'])
        c = len(opts['cache'])
        print(f'Callback : e={e:-7.1e}; cache size: {c:-8d}')
        if info['nswp'] == 5:
            # Stop the algorithm's work (just for demo!)
            return True

  .. code-block:: python

    Y = teneva.rand([n]*d, r=1)
    Y = teneva.cross(func, Y, m=m, dr_max=1, I_vld=I_vld, y_vld=y_vld,
        e_vld=e_vld, cb=cb, info={}, cache={}, log=True)
    Y = teneva.truncate(Y, 1.e-4)
    
    print()
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')
    print(f'Error on test        : {teneva.accuracy_on_data(Y, I_tst, y_tst):-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.018 | evals: 0.00e+00 (+ 0.00e+00) | rank:   1.0 | e_vld: 1.0e+00 | 
    # Callback : e=8.9e+06; cache size:    11968
    # #   1 | time:      0.179 | evals: 1.20e+04 (+ 3.20e+02) | rank:   3.0 | e_vld: 1.6e-01 | e: 8.9e+06 | 
    # Callback : e=2.2e-01; cache size:    57872
    # #   2 | time:      0.655 | evals: 5.79e+04 (+ 2.54e+03) | rank:   5.0 | e_vld: 4.0e-02 | e: 2.2e-01 | 
    # Callback : e=7.9e-02; cache size:   159684
    # #   3 | time:      1.783 | evals: 1.60e+05 (+ 8.25e+03) | rank:   7.0 | e_vld: 5.3e-02 | e: 7.9e-02 | 
    # Callback : e=2.7e-02; cache size:   333516
    # #   4 | time:      4.218 | evals: 3.34e+05 (+ 2.49e+04) | rank:   9.0 | e_vld: 3.3e-02 | e: 2.7e-02 | 
    # Callback : e=3.4e-02; cache size:   593653
    # #   5 | time:      8.458 | evals: 5.94e+05 (+ 6.17e+04) | rank:  11.0 | e_vld: 6.4e-03 | e: 3.4e-02 | stop: cb | 
    # 
    # TT-rank of trunc res :       10.5
    # Error on test        :   6.38e-03
    # 




|
|


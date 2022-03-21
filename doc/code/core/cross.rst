cross: construct TT-tensor by TT-CROSS
--------------------------------------


.. automodule:: teneva.core.cross


-----


.. autofunction:: teneva.cross

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+6., +3., +3., +1., +2.]   # Upper bounds for spatial grid
    n         = [ 20,  18,  16,  14,  12]   # Shape of the tensor

  We set the target function (the function takes as input a set of tensor multi-indices I of the shape [samples, dimension], which are transformed into points X of a uniform spatial grid using the function "ind_to_poi"):

  .. code-block:: python

    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)

  We prepare test data from as a random tensor multi-indices:

  .. code-block:: python

    # Number of test points:
    m_tst = int(1.E+4)
    
    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    
    # Function values for the test points:
    Y_tst = func(I_tst)

  We set the parameters of the TT-CROSS algorithm:

  .. code-block:: python

    m         = 8.E+3  # Number of calls to target function
    e         = None   # Desired accuracy
    nswp      = None   # Sweep number
    r         = 3      # TT-rank of the initial tensor
    dr_min    = 1      # Cross parameter (minimum number of added rows)
    dr_max    = 3      # Cross parameter (maximum number of added rows)

  We build the TT-tensor, which approximates the target function (note that "cache" is optional [it may be None] and it is effictive only for complex functions with long computing time for one call):

  .. code-block:: python

    t = tpc()
    info, cache = {}, {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["m"]:-10d}')
    print(f'Cache uses     : {info["m_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.11
    # Evals func     :       6881
    # Cache uses     :       5947
    # Iter accuracy  :   1.44e-08
    # Sweep number   :          2
    # Stop condition :          m
    # TT-rank of res :        3.0
    # 

  And now we can check the result:

  .. code-block:: python

    # Fast getter for TT-tensor values:
    get = teneva.getter(Y)                     
    
    # Compute approximation in test points:
    Z = np.array([get(i) for i in I_tst])
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on test  :   2.76e-15
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
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["m"]:-10d}')
    print(f'Cache uses     : {info["m_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')
    
    get = teneva.getter(Y)
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.07
    # Evals func     :       4170
    # Cache uses     :       2282
    # Iter accuracy  :   0.00e+00
    # Sweep number   :          2
    # Stop condition :          e
    # TT-rank of res :        3.0
    # Error on test  :   5.56e-16
    # 

  We may not use the cache (note that the number of requests to the objective function in this case will be more, but the running time will be less, since this function is calculated very quickly):

  .. code-block:: python

    t = tpc()
    info, cache = {}, None
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["m"]:-10d}')
    print(f'Cache uses     : {info["m_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')
    
    get = teneva.getter(Y)
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.04
    # Evals func     :       6564
    # Cache uses     :          0
    # Iter accuracy  :   1.53e-08
    # Sweep number   :          2
    # Stop condition :          e
    # TT-rank of res :        3.0
    # Error on test  :   6.92e-16
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
    Y = teneva.cross(func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max, info=info, cache=cache)
    Y = teneva.truncate(Y, 1.e-4) # We round the result
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["m"]:-10d}')
    print(f'Cache uses     : {info["m_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')
    
    get = teneva.getter(Y)
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.04
    # Evals func     :       5736
    # Cache uses     :          0
    # Iter accuracy  :   0.00e+00
    # Sweep number   :          2
    # Stop condition :          e
    # TT-rank of res :        3.0
    # Error on test  :   6.03e-16
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
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["m"]:-10d}')
    print(f'Cache uses     : {info["m_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')
    
    get = teneva.getter(Y)
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.02
    # Evals func     :       1792
    # Cache uses     :          0
    # Iter accuracy  :   7.84e+03
    # Sweep number   :          1
    # Stop condition :       nswp
    # TT-rank of res :        3.0
    # Error on test  :   6.15e-16
    # 



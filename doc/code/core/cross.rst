cross: construct TT-tensor by TT-CAM
------------------------------------


.. automodule:: teneva.core.cross

---


.. autofunction:: teneva.core.cross.cross

  **Examples**:

  .. code-block:: python

    d         = 5                           # Dimension of the function
    a         = [-5., -4., -3., -2., -1.]   # Lower bounds for spatial grid
    b         = [+5., +4., +3., +2., +1.]   # Upper bounds for spatial grid
    n         = [ 50,  60,  70,  80,  90]   # Shape of the tensor

  .. code-block:: python

    evals     = 10000       # Number of calls to target function
    nswp      = 4           # Sweep number
    dr_min    = 1           # Cross parameter (minimum number of added rows)
    dr_max    = 1           # Cross parameter (maximum number of added rows)
    r         = 3           # TT-rank of the initial tensor
    e         = 1.E-4       # Desired accuracy

  .. code-block:: python

    # Target function
    # (we transform indices into points using "ind2poi" function):
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind2poi(I, a, b, n)
        return rosen(X.T)

  .. code-block:: python

    # Test data:
    
    m_tst = 10000 # Number of test points
    I_tst = np.vstack([np.random.choice(n[i], m_tst) for i in range(d)]).T
    Y_tst = func(I_tst)

  .. code-block:: python

    # Build tensor
    # (note: cache is optional (it may be None) and it is effictive only for
    # complex functions with long computing time for one call):
    
    t = tpc()
    cache, info = {}, {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, e, evals, nswp, dr_min, dr_max, cache, info)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    print(f'Evals func     : {info["k_evals"]:-10d}')
    print(f'Cache uses     : {info["k_cache"]:-10d}')
    print(f'Iter accuracy  : {info["e"]:-10.2e}')
    print(f'Sweep number   : {info["nswp"]:-10d}')
    print(f'Stop condition : {info["stop"]:>10}')
    print(f'TT-rank of res : {teneva.erank(Y):-10.1f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.13
    # Evals func     :       9301
    # Cache uses     :       2689
    # Iter accuracy  :  -1.00e+00
    # Sweep number   :          1
    # Stop condition :      evals
    # TT-rank of res :        3.0
    # 

  .. code-block:: python

    # Check result:
    
    get = teneva.getter(Y)                     # Fast getter for TT-tensor values
    
    Z = np.array([get(i) for i in I_tst])      # Compute approximation in test points
    e_tst = np.linalg.norm(Z - Y_tst)          # Accuracy of the result
    e_tst /= np.linalg.norm(Y_tst)
    
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on test  :   8.02e-15
    # 

---

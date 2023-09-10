Module als_func: construct TT-tensor of coefficients
----------------------------------------------------


.. automodule:: teneva.als_func


-----




|
|

.. autofunction:: teneva.als_func.als_func

  **Examples**:

  We set the target function (the function takes as input a set of function inputs X of the shape [samples, dimension]):

  .. code-block:: python

    def func(X):
        """Schaffer function."""
        Z = X[:, :-1]**2 + X[:, 1:]**2
        y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(y, axis=1)

  .. code-block:: python

    d = 7      # Dimension of the function
    a = -5.    # Lower bounds for spatial grid
    b = +6.    # Upper bounds for spatial grid

  Then we select the parameters:

  .. code-block:: python

    m_trn  = 1.E+5  # Train data size (number of function calls)
    m_vld  = 1.E+3  # Validation data size
    m_tst  = 1.E+5  # Test data size
    nswp   = 6      # Sweep number for ALS iterations
    r      = 5      # TT-rank of the initial random tensor
    n      = 2      # Initial shape of the coefficients' tensor
    n_max  = 20     # Maximum shape of the coefficients' tensor

  We prepare random train, validation and test data:

  .. code-block:: python

    X_trn = np.vstack([np.random.uniform(a, b, int(m_trn)) for k in range(d)]).T
    y_trn = func(X_trn)
    
    X_vld = np.vstack([np.random.uniform(a, b, int(m_vld)) for k in range(d)]).T
    y_vld = func(X_vld)
    
    X_tst = np.vstack([np.random.uniform(a, b, int(m_trn)) for k in range(d)]).T
    y_tst = func(X_tst)

  And now we will build the TT-tensor, which approximates the coefficients' tensor in the TT-format by the functional TT-ALS method:

  .. code-block:: python

    t = tpc()
    A0 = teneva.rand([n]*d, r)
    A = teneva.als_func(X_trn, y_trn, A0, a, b, nswp, e=None,
        X_vld=X_vld, y_vld=y_vld, n_max=n_max, log=True)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.141 | rank:   5.0 | e_vld: 2.6e+00 | 
    # #   1 | time:      0.936 | rank:   5.0 | e_vld: 2.7e-01 | e: 1.0e+00 | 
    # #   2 | time:      2.483 | rank:   5.0 | e_vld: 2.3e-01 | e: 6.2e-01 | 
    # #   3 | time:      4.755 | rank:   5.0 | e_vld: 1.8e-01 | e: 5.2e-01 | 
    # #   4 | time:      7.625 | rank:   5.0 | e_vld: 1.3e-01 | e: 3.4e-01 | 
    # #   5 | time:     11.275 | rank:   5.0 | e_vld: 8.4e-02 | e: 2.5e-01 | 
    # #   6 | time:     15.027 | rank:   5.0 | e_vld: 6.9e-02 | e: 1.3e-01 | stop: nswp | 
    # Build time     :      15.04
    # 

  And now we can check the result. We compute values of our approximation in test points using coefficients' tensor:

  .. code-block:: python

    t = tpc()
    
    y_our = teneva.func_get(X_tst, A, a, b)
    e = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    t = tpc() - t
    print(f'Relative error : {e:-10.1e}')
    print(f'Check time     : {t:-10.2f}')

  Note that that the mode sizes for the coefficients' tensor will be changed, since we passed `n_max` parameter:

  .. code-block:: python

    teneva.show(A)

  Here we have given only one example of the use of method. More related demos can be found in the documentation for the "als" function in "als.py" module.




|
|


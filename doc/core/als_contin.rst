Module als_contin: construct TT-tensor of coefficients
------------------------------------------------------


.. automodule:: teneva.core.als_contin


-----


.. autofunction:: teneva.als_contin

  **Examples**:

  We set the target function (the function takes as input a set of function inputs X of the shape [samples, dimension]):

  .. code-block:: python

    def func(X):
        """Schaffer function."""
        Z = X[:, :-1]**2 + X[:, 1:]**2
        y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(y, axis=1)

  .. code-block:: python

    d         = 7      # Dimension of the function
    a         = -5.    # Lower bounds for spatial grid
    b         = +6.    # Upper bounds for spatial grid

  Then we select the parameters:

  .. code-block:: python

    m_trn     = 1.E+5  # Train data size (number of function calls)
    m_vld     = 1.E+3  # Validation data size
    m_tst     = 1.E+5  # Test data size
    nswp      = 10     # Sweep number for ALS iterations
    r         = 5      # TT-rank of the initial random tensor
    n         = 2      # Initial shape of the coefficients' tensor
    n_max     = 20     # Maximum shape of the coefficients' tensor

  We prepare random train, validation and tes data:

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
    A0 = teneva.tensor_rand([n]*d, r)
    A = teneva.als_contin(X_trn, y_trn, A0, a, b, nswp,
        X_vld=X_vld, y_vld=y_vld, n_max=n_max, log=True)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.000 | rank:   5.0 | 
    # #   1 | time:      6.330 | rank:   5.0 | e_vld: 2.7e-01 | e: 5.3e+05 | 
    # #   2 | time:     21.554 | rank:   5.0 | e_vld: 2.3e-01 | 
    # #   3 | time:     45.039 | rank:   5.0 | e_vld: 1.8e-01 | 
    # #   4 | time:     76.058 | rank:   5.0 | e_vld: 1.3e-01 | e: 8.2e-01 | 
    # #   5 | time:    117.307 | rank:   5.0 | e_vld: 8.8e-02 | 
    # #   6 | time:    168.676 | rank:   5.0 | e_vld: 7.2e-02 | e: 3.2e+00 | 
    # #   7 | time:    233.887 | rank:   5.0 | e_vld: 6.5e-02 | e: 4.5e-01 | 
    # #   8 | time:    313.243 | rank:   5.0 | e_vld: 5.1e-02 | e: 0.0e+00 | stop: e | 
    # Build time     :     313.26
    # 

  And now we can check the result. We compute values of our approximation in test points using coefficients' tensor:

  .. code-block:: python

    t = tpc()
    
    y_our = teneva.cheb_get(X_tst, A, a, b)
    e = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    t = tpc() - t
    print(f'Relative error : {e:-10.1e}')
    print(f'Check time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Relative error :    5.2e-02
    # Check time     :       4.72
    # 

  Note that that the mode sizes for the coefficients' tensor will be changed, since we passed `n_max` parameter:

  .. code-block:: python

    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     7D : |10| |18| |18| |18| |18| |18| |10|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/  \5/  \5/
    # 

  Here we have given only one example of the use of method. More related demos can be found in the documentation for the "als" function in "core.als.py" module.



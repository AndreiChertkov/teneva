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

    # # pre | time:      0.153 | rank:   5.0 | e_vld: 1.7e+00 | 
    # #   1 | time:      1.129 | rank:   5.0 | e_vld: 2.7e-01 | e: 1.0e+00 | 
    # #   2 | time:      2.853 | rank:   5.0 | e_vld: 2.3e-01 | e: 6.6e-01 | 
    # #   3 | time:      5.373 | rank:   5.0 | e_vld: 1.8e-01 | e: 5.4e-01 | 
    # #   4 | time:      8.340 | rank:   5.0 | e_vld: 1.4e-01 | e: 3.3e-01 | 
    # #   5 | time:     11.770 | rank:   5.0 | e_vld: 8.3e-02 | e: 2.5e-01 | 
    # #   6 | time:     15.886 | rank:   5.0 | e_vld: 6.9e-02 | e: 1.3e-01 | stop: nswp | 
    # Build time     :      15.90
    # 

  And now we can check the result. We compute values of our approximation in test points using coefficients' tensor:

  .. code-block:: python

    t = tpc()
    
    y_our = teneva.func_get(X_tst, A, a, b)
    e = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    t = tpc() - t
    print(f'Relative error : {e:-10.1e}')
    print(f'Check time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Relative error :    7.1e-02
    # Check time     :       4.76
    # 

  Note that that the mode sizes for the coefficients' tensor will be changed, since we passed `n_max` parameter:

  .. code-block:: python

    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     7D : |8| |14| |14| |14| |14| |14| |8|
    # <rank>  =    5.0 :   \5/  \5/  \5/  \5/  \5/  \5/
    # 

  (NEW OPTION) We can also use batch updates:

  .. code-block:: python

    t = tpc()
    
    A = teneva.rand([n]*d, r)
    
    Nn = X_trn.shape[0]
    bs = 10000 # batch size
    iters = 5
    
    for i_iter in range(iters):
        idx = np.random.permutation(Nn)
        for i in range(0, Nn, bs):
            X_trn_cur = X_trn[idx[i:i+bs]]
            y_trn_cur = y_trn[idx[i:i+bs]]
    
            A = teneva.als_func(X_trn_cur, y_trn_cur, A, a, b, nswp=1, e=None,
                X_vld=X_vld, y_vld=y_vld, n_max=n_max, log=True, update_sol=True, lamb=2**(i_iter/(iters/30)))
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.064 | rank:   5.0 | e_vld: 2.8e+00 | 
    # #   1 | time:      0.205 | rank:   5.0 | e_vld: 2.7e-01 | e: 1.0e+00 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 2.7e-01 | 
    # #   1 | time:      0.300 | rank:   5.0 | e_vld: 2.4e-01 | e: 8.9e-01 | stop: nswp | 
    # # pre | time:      0.053 | rank:   5.0 | e_vld: 2.4e-01 | 
    # #   1 | time:      0.370 | rank:   5.0 | e_vld: 1.9e-01 | e: 7.3e-01 | stop: nswp | 
    # # pre | time:      0.052 | rank:   5.0 | e_vld: 1.9e-01 | 
    # #   1 | time:      0.425 | rank:   5.0 | e_vld: 1.5e-01 | e: 5.2e-01 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 1.5e-01 | 
    # #   1 | time:      0.497 | rank:   5.0 | e_vld: 9.8e-02 | e: 3.9e-01 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 9.8e-02 | 
    # #   1 | time:      0.576 | rank:   5.0 | e_vld: 7.9e-02 | e: 2.3e-01 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 7.9e-02 | 
    # #   1 | time:      0.691 | rank:   5.0 | e_vld: 7.2e-02 | e: 1.9e-01 | stop: nswp | 
    # # pre | time:      0.052 | rank:   5.0 | e_vld: 7.2e-02 | 
    # #   1 | time:      0.795 | rank:   5.0 | e_vld: 5.8e-02 | e: 1.6e-01 | stop: nswp | 
    # # pre | time:      0.053 | rank:   5.0 | e_vld: 5.8e-02 | 
    # #   1 | time:      0.890 | rank:   5.0 | e_vld: 5.4e-02 | e: 1.5e-01 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 5.4e-02 | 
    # #   1 | time:      0.920 | rank:   5.0 | e_vld: 4.7e-02 | e: 1.4e-01 | stop: nswp | 
    # # pre | time:      0.053 | rank:   5.0 | e_vld: 4.7e-02 | 
    # #   1 | time:      0.885 | rank:   5.0 | e_vld: 4.5e-02 | e: 6.9e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.5e-02 | 
    # #   1 | time:      0.881 | rank:   5.0 | e_vld: 4.3e-02 | e: 6.6e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.3e-02 | 
    # #   1 | time:      0.907 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.1e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.907 | rank:   5.0 | e_vld: 4.3e-02 | e: 7.1e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.3e-02 | 
    # #   1 | time:      0.884 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.3e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.874 | rank:   5.0 | e_vld: 4.2e-02 | e: 5.9e-02 | stop: nswp | 
    # # pre | time:      0.061 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.959 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.2e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.922 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.6e-02 | stop: nswp | 
    # # pre | time:      0.053 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.955 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.2e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.964 | rank:   5.0 | e_vld: 4.2e-02 | e: 6.4e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.933 | rank:   5.0 | e_vld: 4.3e-02 | e: 3.8e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.3e-02 | 
    # #   1 | time:      0.923 | rank:   5.0 | e_vld: 4.2e-02 | e: 3.9e-02 | stop: nswp | 
    # # pre | time:      0.057 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.948 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.6e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.948 | rank:   5.0 | e_vld: 4.2e-02 | e: 3.4e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.2e-02 | 
    # #   1 | time:      0.966 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.4e-02 | stop: nswp | 
    # # pre | time:      0.057 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.941 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.5e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.930 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.7e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.935 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.6e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.968 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.4e-02 | stop: nswp | 
    # # pre | time:      0.053 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.949 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.3e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.968 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.6e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.964 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.5e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.954 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.5e-02 | stop: nswp | 
    # # pre | time:      0.057 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.972 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.6e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.945 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.8e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.936 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.6e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.945 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.5e-02 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.934 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.4e-02 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.961 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.6e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.963 | rank:   5.0 | e_vld: 4.1e-02 | e: 1.6e-02 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.965 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.4e-03 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.969 | rank:   5.0 | e_vld: 4.1e-02 | e: 4.4e-03 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.982 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.4e-03 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.983 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.7e-03 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.0e-02 | 
    # #   1 | time:      0.991 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.6e-03 | stop: nswp | 
    # # pre | time:      0.055 | rank:   5.0 | e_vld: 4.0e-02 | 
    # #   1 | time:      0.971 | rank:   5.0 | e_vld: 4.1e-02 | e: 3.4e-03 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.1e-02 | 
    # #   1 | time:      0.957 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.8e-03 | stop: nswp | 
    # # pre | time:      0.056 | rank:   5.0 | e_vld: 4.0e-02 | 
    # #   1 | time:      0.960 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.3e-03 | stop: nswp | 
    # # pre | time:      0.054 | rank:   5.0 | e_vld: 4.0e-02 | 
    # #   1 | time:      0.952 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.7e-03 | stop: nswp | 
    # # pre | time:      0.057 | rank:   5.0 | e_vld: 4.0e-02 | 
    # #   1 | time:      0.949 | rank:   5.0 | e_vld: 4.0e-02 | e: 3.4e-03 | stop: nswp | 
    # Build time     :      43.72
    # 

  .. code-block:: python

    t = tpc()
    
    y_our = teneva.func_get(X_tst, A, a, b)
    e = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    t = tpc() - t
    print(f'Relative error : {e:-10.1e}')
    print(f'Check time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Relative error :    4.1e-02
    # Check time     :       4.92
    # 

  .. code-block:: python

    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     7D : |20| |20| |20| |20| |20| |20| |20|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/  \5/  \5/
    # 

  Here we have given only one example of the use of method. More related demos can be found in the documentation for the "als" function in "als.py" module.




|
|


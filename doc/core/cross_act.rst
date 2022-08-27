Module cross_act: compute function of TT-tensors
------------------------------------------------


.. automodule:: teneva.core.cross_act


-----


.. autofunction:: teneva.cross_act

  **Examples**:

  Let consider the simple operation in the TT-format Y = X1 * X2 + X3:

  .. code-block:: python

    d = 10     # Dimension of the input tensors
    n = [20]*d # Mode sizes of the input tensors (it may be list)
    
    # Random TT-tensors (inputs):
    X1 = teneva.tensor_rand(n, r=3)
    X2 = teneva.tensor_rand(n, r=4)
    X3 = teneva.tensor_rand(n, r=5)

  We can compute the exact result (output):

  .. code-block:: python

    t = tpc()
    Y_real = teneva.add(teneva.mul(X1, X2), X3)
    Y_real = teneva.truncate(Y_real, e=1.E-16)
    t = tpc() - t
    
    teneva.show(Y_real)
    print(f'Time (sec) : {t:-7.3f}')

    # >>> ----------------------------------------
    # >>> Output:

    #  20 20 20 20 20 20 20 20 20 20 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1 17 17 17 17 17 17 17 17 17  1 
    # 
    # Time (sec) :   0.008
    # 

  We set all parameters (note that only "f", "X_list" and "Y0" are required):

  .. code-block:: python

    def f(X):
        # Function should compute the output elements for the given set
        # of input points X (array "[samples, D]"; in our case, D=3).
        # The function should return 1D np.ndarray of the length "samples"
        # with values of the target function for all provided samples.
        return X[:, 0] * X[:, 1] + X[:, 2]

  .. code-block:: python

    # The input of the function (note that the dimension
    # and mode sizes for all tensors must match):
    X_list = [X1, X2, X3]
    
    # Random initial approximation for the output (note that
    # the shape of this tensor should be same as for X1, X2, X3):
    Y0     = teneva.tensor_rand(n, r=1)
    
    e      = 1.E-6  # Accuracy and convergence criterion (optional)
    nswp   = 10     # Maximum number of iterations (optional)
    r      = 9999   # Maximum rank for SVD operation (optional)
    dr     = 3      # Rank ("kickrank") for AMEN (optional)
    dr2    = 1      # Additional rank for AMEN (optional)
    log    = True   # If true, then logs will be presented (optional)

  And now we can run the function:

  .. code-block:: python

    t = tpc()
    Y = teneva.cross_act(f, X_list, Y0, e, nswp, r, dr, dr2, log)
    Y = teneva.truncate(Y, e=1.E-16)
    t = tpc() - t
    
    print('\nResult:')
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # == cross-act #    1 | e:  1.3e+00 | r:   7.0
    # == cross-act #    2 | e:  1.2e+00 | r:  13.0
    # == cross-act #    3 | e:  2.8e-01 | r:  19.0
    # == cross-act #    4 | e:  3.1e-14 | r:  20.0
    # 
    # Result:
    #  20 20 20 20 20 20 20 20 20 20 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1 18 19 19 19 18 18 18 18 17  1 
    # 
    # 

  Finally, we can check the result:

  .. code-block:: python

    eps = teneva.accuracy(Y, Y_real)
    
    print(f'Time (sec) : {t:-7.3f}')
    print(f'Error      : {eps:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Time (sec) :   1.155
    # Error      : 1.0e-08
    # 

  Note that for this example, we do not have a gain in time, however, if we consider a more complex function of arguments in the TT-format, then the situation will change dramatically, since a general function cannot be calculated using simple operations in the  TT-format. For example:

  .. code-block:: python

    d = 5      # Dimension of the input tensors
    n = [10]*d # Mode sizes of the input tensors (it may be list)
    
    # Random TT-tensors (inputs):
    X1 = teneva.tensor_rand(n, r=3)
    X2 = teneva.tensor_rand(n, r=4)
    X3 = teneva.tensor_rand(n, r=5)

  .. code-block:: python

    def f(X):
        return np.exp(-0.1 * X[:, 0]**2) + X[:, 1] + 0.42 * np.sin(X[:, 2]**2)

  .. code-block:: python

    t = tpc()
    Y = teneva.tensor_rand(n, r=1)
    Y = teneva.cross_act(f, [X1, X2, X3], Y, log=True)
    Y = teneva.truncate(Y, e=1.E-10)
    t = tpc() - t
    
    print('\nResult:')
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # == cross-act #    1 | e:  1.3e+00 | r:  10.8
    # == cross-act #    2 | e:  5.5e-02 | r:  17.8
    # == cross-act #    3 | e:  6.2e-02 | r:  23.9
    # == cross-act #    4 | e:  6.8e-02 | r:  29.8
    # == cross-act #    5 | e:  6.7e-02 | r:  35.7
    # == cross-act #    6 | e:  7.3e-02 | r:  41.6
    # == cross-act #    7 | e:  6.4e-02 | r:  47.4
    # == cross-act #    8 | e:  5.8e-02 | r:  53.2
    # == cross-act #    9 | e:  4.9e-02 | r:  59.1
    # == cross-act #   10 | e:  2.8e-02 | r:  64.6
    # == cross-act #   11 | e:  9.5e-03 | r:  65.7
    # 
    # Result:
    #    10  10  10  10  10 
    #   / \ / \ / \ / \ / \ 
    #  1   10 100 100  10  1  
    # 
    # 

  We can check the accuracy from comparison with the full tensor:

  .. code-block:: python

    X1_full = teneva.full(X1).reshape(-1, order='F').reshape(-1, 1)
    X2_full = teneva.full(X2).reshape(-1, order='F').reshape(-1, 1)
    X3_full = teneva.full(X3).reshape(-1, order='F').reshape(-1, 1)
    
    Y_full = teneva.full(Y).reshape(-1, order='F')
    
    Y_real = f(np.hstack((X1_full, X2_full, X3_full)))
    
    e_nrm = np.linalg.norm(Y_full - Y_real) / np.linalg.norm(Y_real)
    e_max = np.max(np.abs(Y_full - Y_real))
    
    print(f'Error norm : {e_nrm:-7.1e}')
    print(f'Error max  : {e_max:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error norm : 2.2e-14
    # Error max  : 6.9e-12
    # 



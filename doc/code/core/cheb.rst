cheb: Chebyshev interpolation in the TT-format
----------------------------------------------


.. automodule:: teneva.core.cheb


-----


.. autofunction:: teneva.cheb_bld

  **Examples**:

  .. code-block:: python

    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CROSS
    e = 1.E-6                                 # Accuracy for TT-CROSS
    eps = 1.E-6                               # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                         # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    Y_full = teneva.cheb_bld_full(f, a, b, n) # Full tensor with function values

  .. code-block:: python

    Y_ref = teneva.full(Y)                    # Compute tensor in the full format
    e = np.linalg.norm(Y_full - Y_ref)        # Compare two methods
    e /= np.linalg.norm(Y_ref)
    print(f'Error     : {e:-8.2e}')           # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 8.22e-16
    # 


-----


.. autofunction:: teneva.cheb_get

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CROSS
    # (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CROSS
    e = 1.E-3                                 # Accuracy for TT-CROSS
    eps = 1.E-6                               # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                         # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation coefficients
    # (see cheb_int function for more details):
    
    A = teneva.cheb_int(Y)
    teneva.show(A)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points inside the bounds:
    # (the values for points outside the bounds will be set as "z")
    
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    Z = teneva.cheb_get(X, A, a, b, z=-1.)
    print(Z)       # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  2.27373675e-13 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    A_full = teneva.full(A)                          # Build full tensor
    Z = teneva.cheb_get_full(X, A_full, a, b, z=-1.) # Compute the values
    print(Z)       # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  8.18545232e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 


-----


.. autofunction:: teneva.cheb_gets

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CROSS
    # (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CROSS
    e = 1.E-3                                 # Accuracy for TT-CROSS
    eps = 1.E-6                               # Accuracy for truncation
    Y1 = teneva.cheb_bld(f, a, b, n,          # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                         # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y1)                           # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation coefficients
    # (see cheb_int function for more details):
    
    A1 = teneva.cheb_int(Y1)
    teneva.show(A1)                           # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    m = [7, 8, 9, 10]                         # New size of the grid
    Y2 = teneva.cheb_gets(A1, a, b, m)        # Compute tensor on finer grid
    teneva.show(Y2)

    # >>> ----------------------------------------
    # >>> Output:

    #   7  8  9 10 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # We can compute interpolation coefficients on the new grid:
    
    A2 = teneva.cheb_int(Y2)
    teneva.show(A2)                           # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   7  8  9 10 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points inside the bounds
    # for 2 different approximations:
    
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    Z1 = teneva.cheb_get(X, A1, a, b, z=-1.)
    Z2 = teneva.cheb_get(X, A2, a, b, z=-1.)
    print(Z1)      # Print the result
    print(Z2)      # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  2.27373675e-13 -1.00000000e+00]
    # [ 3.00000000e+00  5.40600000e+03 -3.35376171e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    A1_full = teneva.full(A1)                         # Full tensor of interpolation coefficients
    Y2_full = teneva.cheb_gets_full(A1_full, a, b, m) # Compute tensor on finer grid

  .. code-block:: python

    Y_ref = teneva.full(Y2)                           # Compute tensor in the full format
    e = np.linalg.norm(Y2_full - Y_ref)               # Compare two methods
    e /= np.linalg.norm(Y_ref)
    print(f'Error     : {e:-8.2e}')                   # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 3.73e-16
    # 


-----


.. autofunction:: teneva.cheb_int

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CROSS
    # (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CROSS
    e = 1.E-3                                 # Accuracy for TT-CROSS
    eps = 1.E-6                               # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                         # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Then we can compute the TT-tensor for Chebyshev interpolation coefficients:
    
    A = teneva.cheb_int(Y)
    teneva.show(A)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    Y_full = teneva.full(Y)                   # Full tensor with function values
    A_full = teneva.cheb_int_full(Y_full)     # Interpolation in the full format

  .. code-block:: python

    A_ref = teneva.full(A)                    # Compute tensor in the full format
    e = np.linalg.norm(A_full - A_ref)        # Compare two methods
    e /= np.linalg.norm(A_ref)
    print(f'Error     : {e:-8.2e}')          # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 4.51e-16
    # 


-----


.. autofunction:: teneva.cheb_pol

  **Examples**:

  .. code-block:: python

    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    X = np.array([                            # Two 4-dim points
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
    ])
    m = 3                                    # Maximum order of polynomial      
    T = teneva.cheb_pol(X, a, b, m)          # Compute polynomials

  .. code-block:: python

    print(T.shape)
    print(T[0, 0, 0])                        # 0-th order pol. is 1
    print(T[1, 0, 0])                        # 1-th order pol. is x (=0 for x=0)
    print(T[2, 0, 0])                        # 2-th order pol. is 2x^2-1 (=-1 for x=0)

    # >>> ----------------------------------------
    # >>> Output:

    # (3, 2, 4)
    # 1.0
    # 0.0
    # -1.0
    # 

  .. code-block:: python

    # Note that grid is scaled from [a, b] limits to [-1, 1] limit:
    
    print(T[0, 1, 3])                        # 0-th order pol. is 1
    print(T[1, 1, 3])                        # 1-th order pol. is x (=1 for x=2 with lim [-2, 2])
    print(T[2, 1, 3])                        # 2-th order pol. is 2x^2-1 (=1 for x=2 with lim [-2, 2])

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0
    # 1.0
    # 1.0
    # 


-----


.. autofunction:: teneva.cheb_sum

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CROSS
    # (see cheb_bld function for more details):
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -11.]              # Grid lower bounds
    b = [+12., +14., +13., +11.]              # Grid upper bounds
    n = [50, 50, 50, 50]                      # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CROSS
    e = 1.E-5                                 # Accuracy for TT-CROSS
    eps = 1.E-6                               # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                         # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #  50 50 50 50 
    #  / \/ \/ \/ \
    #  1  1  1  1  1 
    # 
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation coefficients
    # (see cheb_int function for more details):
    
    A = teneva.cheb_int(Y)
    teneva.show(A)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #  50 50 50 50 
    #  / \/ \/ \/ \
    #  1  1  1  1  1 
    # 
    # 

  .. code-block:: python

    # Finally we compute the integral:
    
    v = teneva.cheb_sum(A, a, b)
    print(v)       # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598721
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    A_full = teneva.full(A)
    v = teneva.cheb_sum_full(A_full, a, b)
    print(v)       # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598715
    # 



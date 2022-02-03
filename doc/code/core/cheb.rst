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
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CAM
    e = 1.E-3                                 # Accuracy for TT-CAM
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        Y0=Y0, e=e)                           # TT-CAM arguments (Y0 and e are required)
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 



-----

.. autofunction:: teneva.cheb_get

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CAM
    # (see teneva.core.cheb.cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CAM
    e = 1.E-3                                 # Accuracy for TT-CAM
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        Y0=Y0, e=e)                           # TT-CAM arguments
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
    # (see teneva.core.cheb.cheb_int function for more details):
    
    A = teneva.cheb_int(Y, e)
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

    # [ 3.00000000e+00  5.40600000e+03 -2.20268248e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 



-----

.. autofunction:: teneva.cheb_get_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CAM
    # (see teneva.core.cheb.cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    d = 4
    a = -2.                                   # Grid lower bounds (it works now only for constant bounds)
    b = +3.                                   # Grid upper bounds (it works now only for constant bounds)
    n = 5                                     # Grid size (it works now only for constant grid sizes)
    Y0 = teneva.rand([n]*d, r=2)              # Initial approximation for TT-CAM
    e = 1.E-3                                 # Accuracy for TT-CAM
    Y = teneva.cheb_bld(f, a, b, [n]*d,       # TT-tensor of values on the Cheb. grid
        Y0=Y0, e=e)                           # TT-CAM arguments
    teneva.show(Y)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation coefficients
    # (see teneva.core.cheb.cheb_int function for more details):
    
    A = teneva.cheb_int(Y, e)
    teneva.show(A)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 

  .. code-block:: python

    m = 10                                    # New size of the grid
    Z = teneva.cheb_get_full(A, a, b, m)      # Compute tensor on finer grid
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  10 10 10 10 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
    # 



-----

.. autofunction:: teneva.cheb_int

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid using TT-CAM
    # (see teneva.core.cheb.cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)                  # Target function
    
    a = [-2., -4., -3., -2.]                  # Grid lower bounds
    b = [+2., +3., +4., +2.]                  # Grid upper bounds
    n = [5, 6, 7, 8]                          # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CAM
    e = 1.E-3                                 # Accuracy for TT-CAM
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        Y0=Y0, e=e)                           # TT-CAM arguments
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
    
    A = teneva.cheb_int(Y, e)
    teneva.show(A)                            # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    #   5  6  7  8 
    #  / \/ \/ \/ \
    #  1  3  3  3  1 
    # 
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
    print(T[1, 1, 3])                        # 1-th order pol. is x (=0 for x=2 with lim [-2, 2])
    print(T[2, 1, 3])                        # 2-th order pol. is 2x^2-1 (=-1 for x=2 with lim [-2, 2])

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

    # In the beginning we compute the function values on the Chebyshev grid using TT-CAM
    # (see teneva.core.cheb.cheb_bld function for more details):
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -12.]              # Grid lower bounds
    b = [+12., +14., +13., +12.]              # Grid upper bounds
    n = [50, 50, 50, 50]                      # Grid size
    Y0 = teneva.rand(n, r=2)                  # Initial approximation for TT-CAM
    e = 1.E-5                                 # Accuracy for TT-CAM
    Y = teneva.cheb_bld(f, a, b, n,           # TT-tensor of values on the Cheb. grid
        Y0=Y0, e=e)                           # TT-CAM arguments
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
    # (see teneva.core.cheb.cheb_int function for more details):
    
    A = teneva.cheb_int(Y, e)
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

    # 1.0000000205312076
    # 


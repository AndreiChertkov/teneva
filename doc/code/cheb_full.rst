Module cheb_full: Chebyshev interpolation in the full format
------------------------------------------------------------


.. automodule:: teneva.cheb_full


-----




|
|

.. autofunction:: teneva.cheb_full.cheb_bld_full

  **Examples**:

  .. code-block:: python

    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    
    # Full tensor with function values:
    Y = teneva.cheb_bld_full(f, a, b, n)
    
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 




|
|

.. autofunction:: teneva.cheb_full.cheb_get_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid (see cheb_bld_full function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    
    # Array of values on the Cheb. grid:
    Y = teneva.cheb_bld_full(f, a, b, n)
    
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation coefficients
    # (see cheb_int_full function for more details):
    
    A = teneva.cheb_int_full(Y)
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
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
    Z = teneva.cheb_get_full(X, A, a, b, z=-1.)
    print(Z)       # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.86535248e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.cheb_full.cheb_gets_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # (see cheb_bld_full function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    
    # Array of values on the Cheb. grid:
    Y1 = teneva.cheb_bld_full(f, a, b, n)
    
    print(Y1.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation
    # coefficients (see cheb_int_full function for more details):
    A1 = teneva.cheb_int_full(Y1)
    
    print(A1.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    m = [7, 8, 9, 10] # New size of the grid
    
    # Compute tensor on finer grid:
    Y2 = teneva.cheb_gets_full(A1, a, b, m)
    
    print(Y2.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (7, 8, 9, 10)
    # 

  .. code-block:: python

    # We can compute interpolation coefficients on the new grid:
    A2 = teneva.cheb_int_full(Y2)
    
    print(A2.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (7, 8, 9, 10)
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points inside
    # the bounds for 2 different approximations:
    
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    Z1 = teneva.cheb_get_full(X, A1, a, b, z=-1.)
    Z2 = teneva.cheb_get_full(X, A2, a, b, z=-1.)
    print(Z1)      # Print the result
    print(Z2)      # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.86535248e-12 -1.00000000e+00]
    # [ 3.00000000e+00  5.40600000e+03  2.58637556e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.cheb_full.cheb_int_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid (see heb_bld_full function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    
    # Array of values on the Cheb. grid:
    Y = teneva.cheb_bld_full(f, a, b, n)
    
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we can compute the array for Chebyshev interpolation coefficients:
    A = teneva.cheb_int_full(Y)
    
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 




|
|

.. autofunction:: teneva.cheb_full.cheb_sum_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid
    # (see cheb_bld_full function for more details):
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -11.] # Grid lower bounds
    b = [+12., +14., +13., +11.] # Grid upper bounds
    n = [50, 50, 50, 50]         # Grid size
    
    # Array of values on the Cheb. grid:
    Y = teneva.cheb_bld_full(f, a, b, n)
    
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 50, 50, 50)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation
    # coefficients (see cheb_int_full function for more details):
    A = teneva.cheb_int_full(Y)
    
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 50, 50, 50)
    # 

  .. code-block:: python

    # Finally we compute the integral:
    v = teneva.cheb_sum_full(A, a, b)
    
    print(v)       # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598715
    # 




|
|


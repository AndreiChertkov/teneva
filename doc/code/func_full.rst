Module func_full: Functional full format including Chebyshev interpolation
--------------------------------------------------------------------------


.. automodule:: teneva.func_full


-----




|
|

.. autofunction:: teneva.func_full.func_get_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid:
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    I = teneva.grid_flat(n)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    Y = f(X).reshape(n, order='F')
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation coefficients:
    A = teneva.func_int_full(Y)
    
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
    
    Z = teneva.func_get_full(X, A, a, b, z=-1.)
    
    print(Z)       # Print the result
    print(f(X))    # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.86535248e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.func_full.func_gets_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid:
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    I = teneva.grid_flat(n)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    Y = f(X).reshape(n, order='F')
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation coefficients:
    A = teneva.func_int_full(Y)
    
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    m = [7, 8, 9, 10] # New size of the grid
    
    # Compute tensor on finer grid:
    Z = teneva.func_gets_full(A, a, b, m)
    
    print(Z.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (7, 8, 9, 10)
    # 

  .. code-block:: python

    # We can compute interpolation coefficients on the new grid:
    B = teneva.func_int_full(Z)
    
    print(B.shape)

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
    
    z1 = teneva.func_get_full(X, A, a, b, z=-1.)
    z2 = teneva.func_get_full(X, B, a, b, z=-1.)
    
    # We can check the result by comparing it to the true values:
    print(z1)
    print(z2)
    print(f(X)) 

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.86535248e-12 -1.00000000e+00]
    # [ 3.00000000e+00  5.40600000e+03  2.18847163e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.func_full.func_int_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid:
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T) # Target function
    
    a = [-2., -4., -3., -2.] # Grid lower bounds
    b = [+2., +3., +4., +2.] # Grid upper bounds
    n = [5, 6, 7, 8]         # Grid size
    I = teneva.grid_flat(n)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    Y = f(X).reshape(n, order='F')
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 

  .. code-block:: python

    # Then we can compute the array for Chebyshev interpolation coefficients:
    A = teneva.func_int_full(Y)
    
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (5, 6, 7, 8)
    # 




|
|

.. autofunction:: teneva.func_full.func_sum_full

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev grid:
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -11.] # Grid lower bounds
    b = [+12., +14., +13., +11.] # Grid upper bounds
    n = [50, 50, 50, 50]         # Grid size
    I = teneva.grid_flat(n)
    X = teneva.ind_to_poi(I, a, b, n, 'cheb')
    Y = f(X).reshape(n, order='F')
    print(Y.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 50, 50, 50)
    # 

  .. code-block:: python

    # Then we should compute the array for Chebyshev interpolation coefficients:
    A = teneva.func_int_full(Y)
    
    print(A.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (50, 50, 50, 50)
    # 

  .. code-block:: python

    # Finally we compute the integral:
    v = teneva.func_sum_full(A, a, b)
    
    print(v)       # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598715
    # 




|
|


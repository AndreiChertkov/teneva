Module cheb: Chebyshev interpolation in the TT-format
-----------------------------------------------------


.. automodule:: teneva.core.cheb


-----




|
|

.. autofunction:: teneva.cheb_bld

  **Examples**:

  .. code-block:: python

    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.tensor_rand(n, r=2) # Initial approximation for TT-CROSS
    e = 1.E-6                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n, # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)               # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                  # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    # Full tensor with function values:
    Y_full = teneva.cheb_bld_full(f, a, b, n)

  .. code-block:: python

    # Compute tensor in the full format:
    Y_ref = teneva.full(Y)
    
    # Compare two methods:
    e = np.linalg.norm(Y_full - Y_ref)        
    e /= np.linalg.norm(Y_ref)
    
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.04e-16
    # 




|
|

.. autofunction:: teneva.cheb_diff_matrix

  **Examples**:

  Let's build an analytic function for demonstration:

  .. code-block:: python

    a = -2.   # Grid lower bound
    b = +3.   # Grid upper bound
    n = 1000  # Grid size
    
    # Function and its first derivative:
    f     = lambda x: np.sin(x**3) + np.exp(-x**2)
    f_der = lambda x: 3. * x**2 * np.cos(x**3) - 2. * x * np.exp(-x**2)
    
    # Chebyshev grid and function values on the grid:
    i = np.arange(n)
    x = teneva.ind_to_poi(i, a, b, n, kind='cheb')
    y = f(x)

  We can compute the derivative for "y" by Chebyshev differential matrix:

  .. code-block:: python

    D = teneva.cheb_diff_matrix(a, b, n)
    z = D @ y

  Let check the result:

  .. code-block:: python

    z_real = f_der(x)
    
    e_nrm = np.linalg.norm(z - z_real) / np.linalg.norm(z_real)
    e_max = np.max(np.abs((z - z_real) / z_real))
    
    print(f'Error nrm : {e_nrm:-7.1e}')
    print(f'Error max : {e_max:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error nrm : 7.5e-13
    # Error max : 6.3e-10
    # 

  We can also calculate higher order derivatives:

  .. code-block:: python

    D1, D2, D3 = teneva.cheb_diff_matrix(a, b, n, m=3)
    z = [D1 @ y, D2 @ y, D3 @ y]

  Let check the result:

  .. code-block:: python

    z1_real = 3. * x**2 * np.cos(x**3) - 2. * x * np.exp(-x**2)
    
    z2_real = 6. * x * np.cos(x**3) - 9. * x**4 * np.sin(x**3)
    z2_real += - 2. * np.exp(-x**2) + 4. * x**2 * np.exp(-x**2)
    
    z3_real = 6. * np.cos(x**3) - 18. * x**3 * np.sin(x**3)
    z3_real += - 36. * x**3 * np.sin(x**3) - 27. * x**6 * np.cos(x**3)
    z3_real += 4. * x * np.exp(-x**2)
    z3_real += 8. * x * np.exp(-x**2) - 8. * x**3 * np.exp(-x**2)
    
    z_real = [z1_real, z2_real, z3_real]
    
    for k in range(3):
        e_nrm = np.linalg.norm(z[k] - z_real[k]) / np.linalg.norm(z_real[k])
        e_max = np.max(np.abs((z[k] - z_real[k]) / z_real[k]))
        print(f'Der # {k+1} | Error nrm : {e_nrm:-7.1e} | Error max : {e_max:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Der # 1 | Error nrm : 7.5e-13 | Error max : 6.3e-10
    # Der # 2 | Error nrm : 4.9e-09 | Error max : 4.3e-08
    # Der # 3 | Error nrm : 1.3e-05 | Error max : 1.4e-03
    # 




|
|

.. autofunction:: teneva.cheb_get

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-CROSS (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.tensor_rand(n, r=2) # Initial approximation for TT-CROSS
    e = 1.E-3                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n, # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)               # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                  # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see cheb_int function for more details):
    A = teneva.cheb_int(Y)
    
    teneva.show(A) # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points inside
    # the bounds (the values for points outside the bounds will be set as "z")
    
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    
    Z = teneva.cheb_get(X, A, a, b, z=-1.)
    
    print(Z)    # Print the result
    print(f(X)) # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.85114163e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    # Build full tensor:
    A_full = teneva.full(A)
    
    # Compute the values:
    Z = teneva.cheb_get_full(X, A_full, a, b, z=-1.) 
    
    # We can check the result by comparing it to the true values:
    print(Z)
    print(f(X))    

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  7.90123522e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.cheb_gets

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-CROSS (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)         # Target function
    
    a = [-2., -4., -3., -2.]         # Grid lower bounds
    b = [+2., +3., +4., +2.]         # Grid upper bounds
    n = [5, 6, 7, 8]                 # Grid size
    Y0 = teneva.tensor_rand(n, r=2)  # Initial approximation for TT-CROSS
    e = 1.E-3                        # Accuracy for TT-CROSS
    eps = 1.E-6                      # Accuracy for truncation
    Y1 = teneva.cheb_bld(f, a, b, n, # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)                # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y1)                  # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see cheb_int function for more details):
    A1 = teneva.cheb_int(Y1)
    
    teneva.show(A1) # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    m = [7, 8, 9, 10] # New size of the grid
    
    # Compute tensor on finer grid:
    Y2 = teneva.cheb_gets(A1, a, b, m)
    
    teneva.show(Y2)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |7| |8| |9| |10|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # We can compute interpolation coefficients on the new grid:
    
    A2 = teneva.cheb_int(Y2)
    
    teneva.show(A2)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |7| |8| |9| |10|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points
    # inside the bounds for 2 different approximations:
    
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    
    Z1 = teneva.cheb_get(X, A1, a, b, z=-1.)
    Z2 = teneva.cheb_get(X, A2, a, b, z=-1.)
    
    # We can check the result by comparing it to the true values:
    print(Z1)
    print(Z2)
    print(f(X))    

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  3.85114163e-12 -1.00000000e+00]
    # [ 3.00000000e+00  5.40600000e+03  1.29318778e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    # Full tensor of interpolation coefficients:
    A1_full = teneva.full(A1)
    
    # Compute tensor on finer grid:
    Y2_full = teneva.cheb_gets_full(A1_full, a, b, m)

  .. code-block:: python

    # Compute tensor in the full format:
    Y_ref = teneva.full(Y2)            
    
    # Compare two methods:
    e = np.linalg.norm(Y2_full - Y_ref)
    e /= np.linalg.norm(Y_ref)
    
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 3.31e-16
    # 




|
|

.. autofunction:: teneva.cheb_int

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-CROSS (see cheb_bld function for more details):
    
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.tensor_rand(n, r=2) # Initial approximation for TT-CROSS
    e = 1.E-3                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n, # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)               # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                  # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we can compute the TT-tensor for Chebyshev
    # interpolation coefficients:
    A = teneva.cheb_int(Y)
    
    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    # Full tensor with function values:
    Y_full = teneva.full(Y)
    
    # Interpolation in the full format:
    A_full = teneva.cheb_int_full(Y_full)

  .. code-block:: python

    # Compute tensor in the full format:
    A_ref = teneva.full(A)
    
    # Compare two methods:
    e = np.linalg.norm(A_full - A_ref)
    e /= np.linalg.norm(A_ref)
    
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.07e-16
    # 




|
|

.. autofunction:: teneva.cheb_pol

  **Examples**:

  .. code-block:: python

    X = np.array([           # Two 4-dim points
        [0., 0., 0., 0.],
        [1., 1., 1., 1.],
    ])
    
    m = 3                    # Maximum order of polynomial 
    
    # Compute polynomials:
    T = teneva.cheb_pol(X, m)

  .. code-block:: python

    print(T.shape)
    print(T)

    # >>> ----------------------------------------
    # >>> Output:

    # (3, 2, 4)
    # [[[ 1.  1.  1.  1.]
    #   [ 1.  1.  1.  1.]]
    # 
    #  [[ 0.  0.  0.  0.]
    #   [ 1.  1.  1.  1.]]
    # 
    #  [[-1. -1. -1. -1.]
    #   [ 1.  1.  1.  1.]]]
    # 




|
|

.. autofunction:: teneva.cheb_sum

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-CROSS (see cheb_bld function for more details):
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -11.]    # Grid lower bounds
    b = [+12., +14., +13., +11.]    # Grid upper bounds
    n = [50, 50, 50, 50]            # Grid size
    Y0 = teneva.tensor_rand(n, r=2) # Initial approximation for TT-CROSS
    e = 1.E-5                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    Y = teneva.cheb_bld(f, a, b, n, # TT-tensor of values on the Cheb. grid
        eps, Y0, e=e)               # TT-CROSS arguments (eps and Y0 are required)
    teneva.show(Y)                  # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |50| |50| |50| |50|
    # <rank>  =    1.0 :    \1/  \1/  \1/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see cheb_int function for more details):
    A = teneva.cheb_int(Y)
    
    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |50| |50| |50| |50|
    # <rank>  =    1.0 :    \1/  \1/  \1/
    # 

  .. code-block:: python

    # Finally we compute the integral:
    v = teneva.cheb_sum(A, a, b)
    
    print(v) # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598721
    # 

  There is also the realization of this function in the full (numpy) format:

  .. code-block:: python

    A_full = teneva.full(A)
    
    v = teneva.cheb_sum_full(A_full, a, b)
    
    print(v) # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0000000191598715
    # 




|
|


Module func: Functional TT-format including Chebyshev interpolation
-------------------------------------------------------------------


.. automodule:: teneva.func


-----




|
|

.. autofunction:: teneva.func.func_basis

  **Examples**:

  .. code-block:: python

    X = np.array([           # Two 4-dim points
        [0., 0., 0., 0.],
        [1., 1., 1., 1.],
    ])
    
    m = 3                    # Maximum order of polynomial 
    
    # Compute Chebyshev polynomials:
    T = teneva.func_basis(X, m)
    
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

.. autofunction:: teneva.func.func_diff_matrix

  **Examples**:

  Let build an analytic function for demonstration:

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

    D = teneva.func_diff_matrix(a, b, n)
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

    D1, D2, D3 = teneva.func_diff_matrix(a, b, n, m=3)
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

  We may also use the "sin" basis (DRAFT!!!):

  .. code-block:: python

    a = 0.       # Grid lower bound
    b = np.pi    # Grid upper bound
    n = 1000     # Grid size
    
    # Function and its first derivative:
    f     = lambda x: np.sin(x)
    f_der = lambda x: np.cos(x)
    
    # Uniform grid and function values on the grid:
    i = np.arange(n)
    x = teneva.ind_to_poi(i, a, b, n, kind='uni')
    y = f(x)

  .. code-block:: python

    D = teneva.func_diff_matrix(a, b, n, kind='sin')
    z = D @ y

  .. code-block:: python

    z_real = f_der(x)
    
    e_nrm = np.linalg.norm(z - z_real) / np.linalg.norm(z_real)
    e_max = np.max(np.abs((z - z_real) / z_real))
    
    print(f'Error nrm : {e_nrm:-7.1e}')
    print(f'Error max : {e_max:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error nrm : 5.3e+02
    # Error max : 3.2e+05
    # 




|
|

.. autofunction:: teneva.func.func_get

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-cross (see cross function for more details):
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.rand(n, r=2)        # Initial approximation for TT-cross
    e = 1.E-3                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    
    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.002 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.008 | evals: 3.12e+02 | rank:   4.0 | e: 5.6e+04 | 
    # #   2 | time:      0.013 | evals: 1.09e+03 | rank:   6.0 | e: 7.4e-09 | stop: e | 
    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see func_int function for more details):
    A = teneva.func_int(Y)
    
    teneva.show(A) # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Finally we compute the approximation in selected points inside
    # the bounds (the values for points outside the bounds will be set as "z"):
    X = np.array([
        [0., 0., 0., 0.],
        [0., 2., 3., 2.],
        [1., 1., 1., 1.],
        [1., 1., 1., 99999999],
    ])
    
    Z = teneva.func_get(X, A, a, b, z=-1.)
    
    print(Z)    # Print the result
    print(f(X)) # We can check the result by comparing it to the true values

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  8.24229573e-12 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 




|
|

.. autofunction:: teneva.func.func_gets

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-cross (see cross function for more details):
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.rand(n, r=2)        # Initial approximation for TT-cross
    e = 1.E-3                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    
    
    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.004 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.010 | evals: 3.12e+02 | rank:   4.0 | e: 7.0e+04 | 
    # #   2 | time:      0.016 | evals: 1.09e+03 | rank:   6.0 | e: 0.0e+00 | stop: e | 
    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see func_int function for more details):
    A = teneva.func_int(Y)
    
    teneva.show(A) # Show the result

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    m = [7, 8, 9, 10] # New size of the grid
    
    # Compute tensor on finer grid:
    Z = teneva.func_gets(A, m)
    
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |7| |8| |9| |10|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # We can compute interpolation coefficients on the new grid:
    B = teneva.func_int(Z)
    
    teneva.show(B)

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
    
    z1 = teneva.func_get(X, A, a, b, z=-1.)
    z2 = teneva.func_get(X, B, a, b, z=-1.)
    
    # We can check the result by comparing it to the true values:
    print(z1)
    print(z2)
    print(f(X))    

    # >>> ----------------------------------------
    # >>> Output:

    # [ 3.00000000e+00  5.40600000e+03  8.24229573e-12 -1.00000000e+00]
    # [ 3.00000000e+00  5.40600000e+03  1.21929133e-11 -1.00000000e+00]
    # [3.0000000e+00 5.4060000e+03 0.0000000e+00 9.9999996e+17]
    # 

  We may also use "sin" basis:

  .. code-block:: python

    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'uni')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    A = teneva.func_int(Y, kind='sin')
    Z = teneva.func_gets(A, m, kind='sin')
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.004 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.011 | evals: 3.12e+02 | rank:   4.0 | e: 6.0e+04 | 
    # #   2 | time:      0.016 | evals: 1.09e+03 | rank:   6.0 | e: 0.0e+00 | stop: e | 
    # TT-tensor     4D : |7| |8| |9| |10|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 




|
|

.. autofunction:: teneva.func.func_int

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-cross (see cross function for more details):
    from scipy.optimize import rosen
    f = lambda X: rosen(X.T)        # Target function
    
    a = [-2., -4., -3., -2.]        # Grid lower bounds
    b = [+2., +3., +4., +2.]        # Grid upper bounds
    n = [5, 6, 7, 8]                # Grid size
    Y0 = teneva.rand(n, r=2)        # Initial approximation for TT-cross
    e = 1.E-3                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    
    
    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.004 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.011 | evals: 3.12e+02 | rank:   4.0 | e: 9.2e+04 | 
    # #   2 | time:      0.018 | evals: 1.09e+03 | rank:   6.0 | e: 0.0e+00 | stop: e | 
    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we can compute the TT-tensor for Chebyshev
    # interpolation coefficients:
    A = teneva.func_int(Y)
    
    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  We may also use "sin" basis:

  .. code-block:: python

    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'uni')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.004 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.010 | evals: 3.12e+02 | rank:   4.0 | e: 7.8e+04 | 
    # #   2 | time:      0.016 | evals: 1.09e+03 | rank:   6.0 | e: 5.0e-09 | stop: e | 
    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 

  .. code-block:: python

    # Then we can compute the TT-tensor for Sin
    # interpolation coefficients:
    A = teneva.func_int(Y, kind='sin')
    
    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |5| |6| |7| |8|
    # <rank>  =    3.0 :   \3/ \3/ \3/
    # 




|
|

.. autofunction:: teneva.func.func_int_general

  **Examples**:

  .. code-block:: python

    a = -2. # Lower bound for continuous grid
    b = +3. # Upper bound for continuous grid
    d = 4   # Dimension of the grid
    n = 10  # Number of grid points
    
    # Build grid points:
    I = np.arange(n)
    X = teneva.ind_to_poi(I, a, b, n)
    
    # Random TT-tensor:
    Y = teneva.rand([n]*d, r=4)

  .. code-block:: python

    # basis_func = TODO
    # A = teneva.func_int_general(Y, X, basis_func, rcond=1e-6)




|
|

.. autofunction:: teneva.func.func_sum

  **Examples**:

  .. code-block:: python

    # In the beginning we compute the function values on the Chebyshev
    # grid using TT-cross (see cheb_bld function for more details):
                     
    d = 4
    def f(X): # Target function
        a = 2.
        r = np.exp(-np.sum(X*X, axis=1) / a) / (np.pi * a)**(d/2)
        return r.reshape(-1)
    
    a = [-12., -14., -13., -11.]    # Grid lower bounds
    b = [+12., +14., +13., +11.]    # Grid upper bounds
    n = [50, 50, 50, 50]            # Grid size
    Y0 = teneva.rand(n, r=2)        # Initial approximation for TT-cross
    e = 1.E-5                       # Accuracy for TT-CROSS
    eps = 1.E-6                     # Accuracy for truncation
    
    Y = teneva.cross(lambda I: f(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, e=e, log=True)
    Y = teneva.truncate(Y, eps)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.004 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.011 | evals: 2.40e+03 | rank:   4.0 | e: 1.0e+00 | 
    # #   2 | time:      0.026 | evals: 8.40e+03 | rank:   6.0 | e: 0.0e+00 | stop: e | 
    # TT-tensor     4D : |50| |50| |50| |50|
    # <rank>  =    1.0 :    \1/  \1/  \1/
    # 

  .. code-block:: python

    # Then we should compute the TT-tensor for Chebyshev interpolation
    # coefficients (see func_int function for more details):
    A = teneva.func_int(Y)
    
    teneva.show(A)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |50| |50| |50| |50|
    # <rank>  =    1.0 :    \1/  \1/  \1/
    # 

  .. code-block:: python

    # Finally we compute the integral:
    v = teneva.func_sum(A, a, b)
    
    print(v) # Print the result (the real value is 1.)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.000000019159871
    # 




|
|


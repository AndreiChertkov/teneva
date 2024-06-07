Module optima_func: estimate max for function
---------------------------------------------


.. automodule:: teneva.optima_func


-----




|
|

.. autofunction:: teneva.optima_func.optima_func_tt_beam

  **Examples**:

  First we create a coefficient tensor:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]           # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)   # Random TT-tensor with rank 4
    A = teneva.func_int(Y)             # TT-tensor of interpolation coefficients

  .. code-block:: python

    # Finding the maximum modulo point:
    x_opt = teneva.optima_func_tt_beam(A, k=3)     
    y_opt = teneva.func_get(x_opt, A, -1, 1)
    
    print(f'x opt appr :', x_opt)
    print(f'y opt appr : {y_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # x opt appr : [ 0.92074466 -0.50381115  0.88270924  0.48885584  0.21839684]
    # y opt appr : 19.522690205649386
    # 

  The function can also return all found candidates for the optimum:

  .. code-block:: python

    x_opt = teneva.optima_func_tt_beam(A, k=3, ret_all=True)     
    y_opt = teneva.func_get(x_opt, A, -1, 1)
    
    print(f'x opt appr :', x_opt)
    print(f'y opt appr : {y_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # x opt appr : [[ 0.92074466 -0.50381115  0.88270924  0.48885584  0.21839684]
    #  [ 0.92074466 -0.50381115  0.88270924  0.48885584 -0.15894822]
    #  [ 0.92074466 -0.80385377  0.76687945  0.41562491  0.19705068]]
    # y opt appr : [ 19.52269021 -16.92563497  14.99017353]
    # 

  We can solve the problem of optimizing a real function:

  .. code-block:: python

    # Target function:
    f = lambda x: 10. - np.sum(x**2)
    f_batch = lambda X: np.array([f(x) for x in X])
    
    d = 5                              # Dimension
    a = [-2.]*d                        # Grid lower bounds
    b = [+2.]*d                        # Grid upper bounds
    n = [201]*d                        # Grid size

  .. code-block:: python

    # We build very accurate approximation of the function:
    Y0 = teneva.rand(n, r=2, seed=42)  # Initial approximation for TT-cross
    Y = teneva.cross(lambda I: f_batch(teneva.ind_to_poi(I, a, b, n, 'cheb')),
        Y0, m=5.E+5, e=None, log=True)
    Y = teneva.truncate(Y, 1.E-9)
    
    # We compute the TT-tensor for Chebyshev interpolation coefficients:
    A = teneva.func_int(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # # pre | time:      0.007 | evals: 0.00e+00 | rank:   2.0 | 
    # #   1 | time:      0.100 | evals: 1.33e+04 | rank:   4.0 | e: 1.2e+01 | 
    # #   2 | time:      0.315 | evals: 4.74e+04 | rank:   6.0 | e: 0.0e+00 | 
    # #   3 | time:      0.865 | evals: 1.12e+05 | rank:   8.0 | e: 0.0e+00 | 
    # #   4 | time:      2.012 | evals: 2.17e+05 | rank:  10.0 | e: 0.0e+00 | 
    # #   5 | time:      3.869 | evals: 3.72e+05 | rank:  12.0 | e: 0.0e+00 | 
    # #   5 | time:      5.874 | evals: 4.74e+05 | rank:  13.2 | e: 1.8e-08 | stop: m | 
    # 

  .. code-block:: python

    # We find the maximum modulo point:
    x_opt = teneva.optima_func_tt_beam(A, k=10)     
    y_opt = teneva.func_get(x_opt, A, a, b)
    
    print(f'x opt appr :', x_opt)
    print(f'y opt appr :', y_opt)

    # >>> ----------------------------------------
    # >>> Output:

    # x opt appr : [ 0.54519297 -1.         -1.          1.         -1.        ]
    # y opt appr : 5.70276462626716
    # 




|
|


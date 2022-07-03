optima: estimate min and max value of tensor
--------------------------------------------


.. automodule:: teneva.core.optima


-----


.. autofunction:: teneva.optima_tt

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]             # Shape of the tensor
    Y = teneva.delta(n, k, v)            # Random TT-tensor with rank 4
    i_min, i_max = teneva.optima_tt(Y)   # Multi-indices of min and max
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {teneva.get(Y, i_min):-12.4e}')
    print(f'y max appr : {teneva.get(Y, i_max):-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [14  6 15 13  8]
    # i max appr : [14  6  1 13  8]
    # y min appr :  -2.7379e+02
    # y max appr :   3.1204e+02
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)              # Build tensor in full format
    i_min = np.argmin(Y_full)            # Multi-indices of min and max
    i_max = np.argmax(Y_full)
    
    i_min = np.unravel_index(i_min, n)
    i_max = np.unravel_index(i_max, n)
    
    print(f'i min real :', i_min)
    print(f'i max real :', i_max)
    print(f'y min real : {Y_full[i_min]:-12.4e}')
    print(f'y max real : {Y_full[i_max]:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min real : (14, 6, 15, 13, 8)
    # i max real : (14, 6, 1, 13, 8)
    # y min real :  -2.7379e+02
    # y max real :   3.1204e+02
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]           # Shape of the tensor
    
    for i in range(10):
        Y = teneva.rand(n, r=4)
        t = tpc()
        i_min_appr, i_max_appr = teneva.optima_tt(Y)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = np.abs(y_min_appr - y_min_real) / np.abs(y_min_real)
        e_max = np.abs(y_max_appr - y_max_real) / np.abs(y_max_real)
    
        print(f'-> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e} | Time {t:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    # -> Error for min 2.0e-16 | Error for max 1.8e-16 | Time  15.5049
    # -> Error for min 3.5e-16 | Error for max 1.4e-16 | Time  15.3083
    # -> Error for min 1.7e-16 | Error for max 1.5e-16 | Time  15.5759
    # -> Error for min 1.8e-16 | Error for max 1.6e-16 | Time  17.7801
    # -> Error for min 2.3e-16 | Error for max 0.0e+00 | Time  16.7488
    # -> Error for min 2.2e-16 | Error for max 1.2e-16 | Time  16.4675
    # -> Error for min 1.2e-16 | Error for max 0.0e+00 | Time  15.3547
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time  15.7807
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time  14.9045
    # -> Error for min 1.7e-16 | Error for max 1.9e-16 | Time  15.7040
    # 

  We can also check it for real data (we build TT-tensor using TT-SVD here). Note that we shift all functions up by $0.5$ to ensure that its min/max values are nonzero, since we compute the relative error for result.

  .. code-block:: python

    d = 6   # Dimension
    n = 16  # Grid size
    
    for func in teneva.func_demo_all(d, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')
        
        # Build full tensor on the grid:
        I_full = teneva.grid_flat(func.n)
        Y_full_real = func.get_f_ind(I_full).reshape(func.n, order='F')
    
        # Build TT-approximation by TT-SVD:
        Y = teneva.svd(Y_full_real, e=1.E-8)
        Y = teneva.truncate(Y, e=1.E-8)
        r = teneva.erank(Y)
    
        # Compute the exact min and max for TT-tensor:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)
    
        # Find the minimum and maximum of TT-tensor by opt_tt:
        t = tpc()
        i_min_appr, i_max_appr = teneva.optima_tt(Y)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
        
        # Check the accuracy of result:
        e_min = abs((y_min_real - y_min_appr) / y_min_real)
        e_max = abs((y_max_real - y_max_appr) / y_max_real)
        
        # Present the result:
        text = '-> ' + func.name + ' ' * max(0, 20 - len(func.name)) + ' | '
        text += f'TT-rank {r:-5.1f} | '
        text += f'Error for min {e_min:-7.1e} | '
        text += f'Error for max {e_max:-7.1e} | '
        text += f'Time {t:-8.4f} | '
        print(text)

    # >>> ----------------------------------------
    # >>> Output:

    # -> Ackley               | TT-rank   9.4 | Error for min 2.0e-16 | Error for max 0.0e+00 | Time 113.8575 | 
    # -> Alpine               | TT-rank   2.0 | Error for min 7.7e-16 | Error for max 0.0e+00 | Time   4.6771 | 
    # -> Brown                | TT-rank  10.3 | Error for min 1.9e-07 | Error for max 1.6e-15 | Time  14.9051 | 
    # -> Dixon                | TT-rank   6.4 | Error for min 1.8e-12 | Error for max 0.0e+00 | Time 181.5377 | 
    # -> Exponential          | TT-rank   2.0 | Error for min 1.1e-16 | Error for max 0.0e+00 | Time   4.7090 | 
    # -> Grienwank            | TT-rank   4.2 | Error for min 1.0e-15 | Error for max 0.0e+00 | Time  82.3997 | 
    # -> Michalewicz          | TT-rank   2.0 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   4.5191 | 
    # -> Qing                 | TT-rank   6.0 | Error for min 4.6e-08 | Error for max 1.6e-16 | Time   5.7403 | 
    # -> Rastrigin            | TT-rank   4.6 | Error for min 7.1e-16 | Error for max 0.0e+00 | Time   7.1644 | 
    # -> Rosenbrock           | TT-rank   6.4 | Error for min 4.2e-14 | Error for max 3.7e-16 | Time 113.6932 | 
    # -> Schaffer             | TT-rank   8.7 | Error for min 1.2e-16 | Error for max 0.0e+00 | Time 204.0390 | 
    # -> Schwefel             | TT-rank   4.8 | Error for min 4.9e-16 | Error for max 0.0e+00 | Time   6.9592 | 
    # 

  We can also check it for real data with TT-CROSS approach:

  .. code-block:: python

    d = 6   # Dimension
    n = 16  # Grid size
    
    for func in teneva.func_demo_all(d, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')
    
        # Build TT-approximation by TT-CROSS:
        Y = teneva.rand(func.n, r=1)
        Y = teneva.cross(func.get_f_ind, Y, m=1.E+5, dr_max=1, cache={})
        Y = teneva.truncate(Y, e=1.E-8)
        r = teneva.erank(Y)
    
        # Compute the exact min and max for TT-tensor:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)
        
        # Find the minimum and maximum of TT-tensor by opt_tt:
        t = tpc()
        i_min_appr, i_max_appr = teneva.optima_tt(Y)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
        
        # Check the accuracy of result:
        e_min = abs((y_min_real - y_min_appr) / y_min_real)
        e_max = abs((y_max_real - y_max_appr) / y_max_real)
        
        # Present the result:
        text = '-> ' + func.name + ' ' * max(0, 20 - len(func.name)) + ' | '
        text += f'TT-rank {r:-5.1f} | '
        text += f'Error for min {e_min:-7.1e} | '
        text += f'Error for max {e_max:-7.1e} | '
        text += f'Time {t:-8.4f} | '
        print(text)

    # >>> ----------------------------------------
    # >>> Output:

    # -> Ackley               | TT-rank  11.6 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time  51.2964 | 
    # -> Alpine               | TT-rank   5.9 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   5.4746 | 
    # -> Brown                | TT-rank  11.0 | Error for min 8.2e-10 | Error for max 1.4e-15 | Time  42.1256 | 
    # -> Dixon                | TT-rank   7.6 | Error for min 1.9e-12 | Error for max 2.4e-16 | Time 377.8651 | 
    # -> Exponential          | TT-rank   6.5 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time  12.8678 | 
    # -> Grienwank            | TT-rank   5.4 | Error for min 3.4e-14 | Error for max 2.1e-16 | Time  94.6897 | 
    # -> Michalewicz          | TT-rank   6.7 | Error for min 0.0e+00 | Error for max 5.1e-15 | Time  12.6210 | 
    # -> Qing                 | TT-rank   9.3 | Error for min 9.8e-09 | Error for max 3.3e-16 | Time  11.7420 | 
    # -> Rastrigin            | TT-rank   4.4 | Error for min 4.2e-14 | Error for max 0.0e+00 | Time   6.1641 | 
    # -> Rosenbrock           | TT-rank   3.7 | Error for min 1.5e-13 | Error for max 3.7e-16 | Time 110.6051 | 
    # -> Schaffer             | TT-rank  13.1 | Error for min 1.2e-16 | Error for max 0.0e+00 | Time 192.4195 | 
    # -> Schwefel             | TT-rank   6.7 | Error for min 4.9e-16 | Error for max 1.9e-16 | Time   8.0720 | 
    # 

  Note that the default value for the number of iterations "nswp" is 5 and the maximum TT-rank "r" is 50 (to accurately find a good optimum value for any tensors), because of this, the calculation time is significant, however, for most cases, we can choose smaller values for this parameters without loss of accuracy (in the calculation below we have poor accuracy only for the Dixon function):

  .. code-block:: python

    d = 6   # Dimension
    n = 16  # Grid size
    
    for func in teneva.func_demo_all(d, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')
    
        # Build TT-approximation by TT-CROSS:
        Y = teneva.rand(func.n, r=1)
        Y = teneva.cross(func.get_f_ind, Y, m=1.E+5, dr_max=1, cache={})
        Y = teneva.truncate(Y, e=1.E-8)
        r = teneva.erank(Y)
    
        # Compute the exact min and max for TT-tensor:
        Y_full = teneva.full(Y)
        y_min_real = np.min(Y_full)
        y_max_real = np.max(Y_full)
        
        # Find the minimum and maximum of TT-tensor by opt_tt
        # (note that we have reduced the number of sweeps):
        t = tpc()
        i_min_appr, i_max_appr = teneva.optima_tt(Y, nswp=1)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
        
        # Check the accuracy of result:
        e_min = abs((y_min_real - y_min_appr) / y_min_real)
        e_max = abs((y_max_real - y_max_appr) / y_max_real)
        
        # Present the result:
        text = '-> ' + func.name + ' ' * max(0, 20 - len(func.name)) + ' | '
        text += f'TT-rank {r:-5.1f} | '
        text += f'Error for min {e_min:-7.1e} | '
        text += f'Error for max {e_max:-7.1e} | '
        text += f'Time {t:-8.4f} | '
        print(text)

    # >>> ----------------------------------------
    # >>> Output:

    # -> Ackley               | TT-rank  11.6 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   5.0452 | 
    # -> Alpine               | TT-rank   5.9 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   1.8947 | 
    # -> Brown                | TT-rank  11.0 | Error for min 5.5e-08 | Error for max 1.4e-15 | Time   2.8061 | 
    # -> Dixon                | TT-rank   8.7 | Error for min 4.3e+01 | Error for max 0.0e+00 | Time   2.6867 | 
    # -> Exponential          | TT-rank   6.5 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   2.0724 | 
    # -> Grienwank            | TT-rank   5.4 | Error for min 3.4e-14 | Error for max 2.1e-16 | Time   2.0186 | 
    # -> Michalewicz          | TT-rank   6.4 | Error for min 1.3e-16 | Error for max 0.0e+00 | Time   2.1065 | 
    # -> Qing                 | TT-rank   9.3 | Error for min 9.8e-09 | Error for max 3.3e-16 | Time   2.7001 | 
    # -> Rastrigin            | TT-rank   4.5 | Error for min 1.5e-15 | Error for max 0.0e+00 | Time   2.0519 | 
    # -> Rosenbrock           | TT-rank   3.7 | Error for min 4.5e+00 | Error for max 3.7e-16 | Time   2.0493 | 
    # -> Schaffer             | TT-rank  11.9 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   3.3574 | 
    # -> Schwefel             | TT-rank   6.7 | Error for min 4.9e-16 | Error for max 1.9e-16 | Time   2.1940 | 
    # 

  We can also log the optimization process (note that for the Rosenbrock function it takes quite a lot of iterations to get a good minimum and maximum value):

  .. code-block:: python

    func = teneva.FuncDemoRosenbrock(d=6, dy=0.5)
    func.set_grid(n=16, kind='uni')
    
    Y = teneva.rand(func.n, r=1)
    Y = teneva.cross(func.get_f_ind, Y, m=1.E+5, dr_max=1, cache={})
    Y = teneva.truncate(Y, e=1.E-8)
    
    Y_full = teneva.full(Y)
    y_min = np.min(Y_full)
    y_max = np.max(Y_full)
    print(f'Real values for TT-tensor        | y_min = {y_min:-16.7e} | y_max = {y_max:-16.7e} |\n')
        
    i_min_appr, i_max_appr = teneva.optima_tt(Y, log=True)

    # >>> ----------------------------------------
    # >>> Output:

    # Real values for TT-tensor        | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 |
    # 
    # outer : pre | ... | rank =   3.7 | y_min =    1.2690465e+01 | y_max =    1.9530131e+04 | 
    # outer :   1 | ... | 
    # inner :   0 | MIN | rank =   4.7 | y_min =    1.2690465e+01 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MIN | rank =  11.1 | y_min =    7.7716161e+00 | y_max =    1.9530131e+04 | y_eps =    4.9188487e+00 | 
    # inner :   2 | MIN | rank =  16.2 | y_min =    7.7716161e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MIN | rank =  30.3 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    6.3668718e+00 | 
    # inner :   4 | MIN | rank =  54.4 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   0 | MAX | rank =   4.7 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MAX | rank =  11.4 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   2 | MAX | rank =  23.2 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MAX | rank =  40.6 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   4 | MAX | rank =  57.7 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # 



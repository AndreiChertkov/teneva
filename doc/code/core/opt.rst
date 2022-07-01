opt: estimate min and max value of tensor
-----------------------------------------


.. automodule:: teneva.core.opt


-----


.. autofunction:: teneva.opt_tt

  **Examples**:

  .. code-block:: python

    n = [ 20,  18,  16,  14,  12]     # Shape of the tensor
    Y = teneva.rand(n, r=4)           # Random TT-tensor with rank 4
    i_min, i_max = teneva.opt_tt(Y)   # Multi-indices of min and max
    
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

    Y_full = teneva.full(Y)           # Build tensor in full format
    i_min = np.argmin(Y_full)         # Multi-indices of min and max
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

    n = [ 20,  18,  16,  14,  12]    # Shape of the tensor
    
    for i in range(10):
        Y = teneva.rand(n, r=4)
        t = tpc()
        i_min_appr, i_max_appr = teneva.opt_tt(Y)
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

    # -> Error for min 2.0e-16 | Error for max 1.8e-16 | Time  14.5012
    # -> Error for min 3.5e-16 | Error for max 1.4e-16 | Time  15.2316
    # -> Error for min 1.7e-16 | Error for max 1.5e-16 | Time  14.6780
    # -> Error for min 1.8e-16 | Error for max 1.6e-16 | Time  14.5652
    # -> Error for min 2.3e-16 | Error for max 0.0e+00 | Time  14.6821
    # -> Error for min 2.2e-16 | Error for max 1.2e-16 | Time  15.5223
    # -> Error for min 1.2e-16 | Error for max 0.0e+00 | Time  14.2446
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time  13.6671
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time  13.7663
    # -> Error for min 1.7e-16 | Error for max 1.9e-16 | Time  14.2239
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
        i_min_appr, i_max_appr = teneva.opt_tt(Y)
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

    # -> Ackley               | TT-rank   9.4 | Error for min 2.0e-16 | Error for max 0.0e+00 | Time   8.5492 | 
    # -> Alpine               | TT-rank   2.0 | Error for min 7.7e-16 | Error for max 0.0e+00 | Time   8.2908 | 
    # -> Brown                | TT-rank  10.3 | Error for min 1.9e-07 | Error for max 1.6e-15 | Time  10.4127 | 
    # -> Dixon                | TT-rank   6.4 | Error for min 1.0e-12 | Error for max 0.0e+00 | Time  14.0481 | 
    # -> Exponential          | TT-rank   2.0 | Error for min 1.1e-16 | Error for max 0.0e+00 | Time   7.4840 | 
    # -> Grienwank            | TT-rank   4.2 | Error for min 1.0e-15 | Error for max 0.0e+00 | Time  11.5836 | 
    # -> Michalewicz          | TT-rank   2.0 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   7.8507 | 
    # -> Qing                 | TT-rank   6.0 | Error for min 4.6e-08 | Error for max 1.6e-16 | Time   9.2934 | 
    # -> Rastrigin            | TT-rank   4.6 | Error for min 4.8e-16 | Error for max 0.0e+00 | Time   9.0017 | 
    # -> Rosenbrock           | TT-rank   6.4 | Error for min 4.2e-14 | Error for max 3.7e-16 | Time  10.4305 | 
    # -> Schaffer             | TT-rank   8.7 | Error for min 1.2e-16 | Error for max 0.0e+00 | Time   6.7973 | 
    # -> Schwefel             | TT-rank   4.8 | Error for min 4.9e-16 | Error for max 0.0e+00 | Time   9.1098 | 
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
        i_min_appr, i_max_appr = teneva.opt_tt(Y)
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

    # -> Ackley               | TT-rank  11.6 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   7.8370 | 
    # -> Alpine               | TT-rank   5.9 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   9.2351 | 
    # -> Brown                | TT-rank  11.0 | Error for min 8.2e-10 | Error for max 1.4e-15 | Time   9.9420 | 
    # -> Dixon                | TT-rank   7.6 | Error for min 1.9e-12 | Error for max 2.4e-16 | Time  14.0661 | 
    # -> Exponential          | TT-rank   6.5 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   8.5580 | 
    # -> Grienwank            | TT-rank   5.4 | Error for min 7.7e-15 | Error for max 0.0e+00 | Time  11.5098 | 
    # -> Michalewicz          | TT-rank   6.7 | Error for min 0.0e+00 | Error for max 5.1e-15 | Time   8.8541 | 
    # -> Qing                 | TT-rank   9.3 | Error for min 9.8e-09 | Error for max 3.3e-16 | Time   9.7634 | 
    # -> Rastrigin            | TT-rank   4.4 | Error for min 4.0e-15 | Error for max 1.2e-16 | Time   9.2471 | 
    # -> Rosenbrock           | TT-rank   3.7 | Error for min 1.5e-13 | Error for max 3.7e-16 | Time  10.2144 | 
    # -> Schaffer             | TT-rank  13.1 | Error for min 1.2e-16 | Error for max 0.0e+00 | Time   9.5934 | 
    # -> Schwefel             | TT-rank   6.7 | Error for min 4.9e-16 | Error for max 1.9e-16 | Time   9.2415 | 
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
        
    i_min_appr, i_max_appr = teneva.opt_tt(Y, log=True)

    # >>> ----------------------------------------
    # >>> Output:

    # Real values for TT-tensor        | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 |
    # 
    # outer : pre | ... | rank =   3.7 | y_min =    1.2690465e+01 | y_max =    1.9530131e+04 | 
    # outer :   1 | ... | 
    # inner :   0 | MIN | rank =   4.7 | y_min =    1.2690465e+01 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MIN | rank =   6.0 | y_min =    9.4878651e+00 | y_max =    1.9530131e+04 | y_eps =    3.2025997e+00 | 
    # inner :   2 | MIN | rank =  13.1 | y_min =    9.4878651e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MIN | rank =  18.6 | y_min =    9.4878651e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   4 | MIN | rank =  22.6 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    8.0831208e+00 | 
    # inner :   0 | MAX | rank =   4.7 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MAX | rank =   5.9 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   2 | MAX | rank =  14.5 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MAX | rank =  25.0 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # outer :   2 | ... | 
    # inner :   0 | MIN | rank =   4.7 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MIN | rank =   5.9 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   2 | MIN | rank =  13.3 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MIN | rank =  18.6 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   4 | MIN | rank =  22.6 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   0 | MAX | rank =   4.7 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   1 | MAX | rank =   5.9 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   2 | MAX | rank =  14.5 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # inner :   3 | MAX | rank =  25.0 | y_min =    1.4047443e+00 | y_max =    1.9530131e+04 | y_eps =    0.0000000e+00 | 
    # 

  Additionally, we can carry out a more complicated check for essentially multidimensional tensors. We generate a random TT-tensor and manually set its minimum value:

  .. code-block:: python

    y_min_real  = 1.      # Value for min
    y_min_scale = 2.      # Scale
    
    def delta(n, cind=0):
        d = len(n)
        Y = []
        for i in range(d):
            G = np.zeros((1, n[i], 1))
            G[0, cind[i], 0] = 1
            Y.append(G)
        return Y
    
    for d in range(3, 11):    # Dimension of the tensor
        n = [20] * d          # Shape of the tensor
        r = 2                 # TT-rank of the tensor
        i_min_real = [2] * d  # Multi-index for min
        
    
        for i in range(3):
            Y = teneva.rand(n, r)
            Y = teneva.mul(Y, Y)
            Y = teneva.add(Y, y_min_real * y_min_scale)
            y = teneva.get(Y, i_min_real)
            D = delta(n, i_min_real)
            D = teneva.mul(D, y_min_real - y)
            Y = teneva.add(Y, D)
            Y = teneva.truncate(Y, 1.E-16)
    
            y_min_real = teneva.get(Y, i_min_real)
            r_real = teneva.erank(Y)
    
            t = tpc()
            i_min_appr, i_max_appr = teneva.opt_tt(Y, nswp_outer=2, nswp=20, r=50, e=1.E-8, log=False)
            y_min_appr = teneva.get(Y, i_min_appr)
            y_max_appr = teneva.get(Y, i_max_appr)
            t = tpc() - t
    
            e_min = np.abs((y_min_appr - y_min_real) / y_min_real)
    
            print(f'Dim {d:-2d} | Rank {r_real:-5.1f} | Error for min {e_min:-7.1e} | Time {t:-8.4f}')
        print()

    # >>> ----------------------------------------
    # >>> Output:

    # Dim  3 | Rank   5.5 | Error for min 0.0e+00 | Time  26.7064
    # Dim  3 | Rank   5.5 | Error for min 0.0e+00 | Time  28.1536
    # Dim  3 | Rank   6.0 | Error for min 0.0e+00 | Time  26.6550
    # 
    # Dim  4 | Rank   5.7 | Error for min 0.0e+00 | Time  71.4249
    # Dim  4 | Rank   5.5 | Error for min 0.0e+00 | Time  70.1516
    # Dim  4 | Rank   5.3 | Error for min 0.0e+00 | Time  60.2636
    # 
    # Dim  5 | Rank   5.4 | Error for min 0.0e+00 | Time  72.1397
    # Dim  5 | Rank   5.6 | Error for min 0.0e+00 | Time  91.2491
    # Dim  5 | Rank   5.2 | Error for min 0.0e+00 | Time 234.8456
    # 
    # Dim  6 | Rank   5.2 | Error for min 0.0e+00 | Time 266.9129
    # Dim  6 | Rank   5.5 | Error for min 0.0e+00 | Time 238.7489
    # Dim  6 | Rank   5.6 | Error for min 0.0e+00 | Time 122.5100
    # 
    # Dim  7 | Rank   5.7 | Error for min 0.0e+00 | Time  98.4463
    # Dim  7 | Rank   5.4 | Error for min 0.0e+00 | Time 318.2259
    # Dim  7 | Rank   5.7 | Error for min 0.0e+00 | Time 604.1358
    # 
    # Dim  8 | Rank   5.6 | Error for min 0.0e+00 | Time 411.4440
    # Dim  8 | Rank   5.4 | Error for min 0.0e+00 | Time 801.7985
    # Dim  8 | Rank   5.9 | Error for min 0.0e+00 | Time 701.7503
    # 
    # Dim  9 | Rank   5.7 | Error for min 0.0e+00 | Time 1047.0420
    # Dim  9 | Rank   5.1 | Error for min 0.0e+00 | Time 901.1661
    # Dim  9 | Rank   5.5 | Error for min 0.0e+00 | Time 292.5046
    # 
    # Dim 10 | Rank   5.8 | Error for min 0.0e+00 | Time 355.9778
    # Dim 10 | Rank   5.6 | Error for min 0.0e+00 | Time 861.2100
    # Dim 10 | Rank   5.3 | Error for min 0.0e+00 | Time 175.8720
    # 
    # 

  We can also investigate the dependency on parameters of the method:

  .. code-block:: python

    n      = [ 20,  18,  16,  14,  12]                   # Shape of the tensor
    Y_list = [teneva.rand(n, r=4) for _ in range(1000)]  # A list of random TT-tensors
    e_bad  = 1.E-10                                      # Error threshold for bad result

  .. code-block:: python

    # What if we do only preiteration (only maxvol for original tensor):
    ind_bad = []
    
    t = tpc()
    
    for ind in range(len(Y_list)):
        Y = Y_list[ind]
        i_min_appr, i_max_appr = teneva.opt_tt(Y, nswp_outer=0)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = np.abs(y_min_appr - y_min_real) / np.abs(y_min_real)
        e_max = np.abs(y_max_appr - y_max_real) / np.abs(y_max_real)
    
        if e_min > e_bad or e_max > e_bad:
            ind_bad.append(ind)
            # print(f'Bad case (# {ind+1:-5d}). -> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e}')
            
    k_all = len(Y_list)
    k_bad = len(ind_bad)
    t = (tpc() - t) / k_all if k_all else 0.
    
    print('-' * 70)
    print(f'Average time    : {t:-8.4f}')
    print(f'Total bad cases : {k_bad:-8d}')

    # >>> ----------------------------------------
    # >>> Output:

    # ----------------------------------------------------------------------
    # Average time    :   0.2860
    # Total bad cases :      515
    # 

  .. code-block:: python

    # Let consider bad results and do only "Y-y_ref" operation:
    ind_bad_2 = []
    
    t = tpc()
    
    for ind in ind_bad:
        Y = Y_list[ind]
        i_min_appr, i_max_appr = teneva.opt_tt(Y, nswp_outer=1, nswp=0)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = np.abs(y_min_appr - y_min_real) / np.abs(y_min_real)
        e_max = np.abs(y_max_appr - y_max_real) / np.abs(y_max_real)
    
        if e_min > e_bad or e_max > e_bad:
            ind_bad_2.append(ind)
            print(f'Bad case (# {ind+1:-5d}). -> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e}')
            
    k_all = len(ind_bad)
    k_bad = len(ind_bad_2)
    t = (tpc() - t) / k_all if k_all else 0.
    
    print('-' * 70)
    print(f'Average time    : {t:-8.4f}')
    print(f'Total bad cases : {k_bad:-8d}')

    # >>> ----------------------------------------
    # >>> Output:

    # Bad case (#     2). -> Error for min 0.0e+00 | Error for max 1.3e-01
    # Bad case (#     8). -> Error for min 0.0e+00 | Error for max 7.3e-02
    # Bad case (#     9). -> Error for min 0.0e+00 | Error for max 1.0e-01
    # Bad case (#    10). -> Error for min 1.6e-01 | Error for max 1.7e-01
    # Bad case (#    11). -> Error for min 7.5e-03 | Error for max 2.8e-02
    # Bad case (#    12). -> Error for min 8.2e-02 | Error for max 1.6e-16
    # Bad case (#    14). -> Error for min 0.0e+00 | Error for max 2.4e-02
    # Bad case (#    17). -> Error for min 4.6e-02 | Error for max 0.0e+00
    # Bad case (#    18). -> Error for min 2.1e-01 | Error for max 0.0e+00
    # Bad case (#    21). -> Error for min 2.2e-16 | Error for max 3.5e-02
    # Bad case (#    26). -> Error for min 1.1e-01 | Error for max 0.0e+00
    # Bad case (#    28). -> Error for min 7.3e-02 | Error for max 1.0e-01
    # Bad case (#    29). -> Error for min 4.5e-02 | Error for max 4.7e-02
    # Bad case (#    30). -> Error for min 1.8e-01 | Error for max 1.7e-02
    # Bad case (#    31). -> Error for min 2.5e-01 | Error for max 3.0e-02
    # Bad case (#    32). -> Error for min 1.4e-16 | Error for max 5.0e-02
    # Bad case (#    34). -> Error for min 0.0e+00 | Error for max 3.7e-02
    # Bad case (#    35). -> Error for min 8.3e-02 | Error for max 2.2e-16
    # Bad case (#    37). -> Error for min 1.2e-16 | Error for max 1.9e-04
    # Bad case (#    41). -> Error for min 2.2e-16 | Error for max 3.3e-01
    # Bad case (#    44). -> Error for min 8.4e-02 | Error for max 2.0e-16
    # Bad case (#    49). -> Error for min 2.1e-16 | Error for max 9.7e-02
    # Bad case (#    52). -> Error for min 1.0e-01 | Error for max 0.0e+00
    # Bad case (#    53). -> Error for min 0.0e+00 | Error for max 3.7e-02
    # Bad case (#    54). -> Error for min 4.6e-02 | Error for max 2.2e-16
    # Bad case (#    60). -> Error for min 2.8e-02 | Error for max 3.9e-16
    # Bad case (#    66). -> Error for min 0.0e+00 | Error for max 1.3e-01
    # Bad case (#    67). -> Error for min 0.0e+00 | Error for max 5.1e-02
    # Bad case (#    71). -> Error for min 0.0e+00 | Error for max 1.2e-01
    # Bad case (#    72). -> Error for min 8.8e-02 | Error for max 9.7e-02
    # Bad case (#    81). -> Error for min 1.4e-16 | Error for max 1.5e-01
    # Bad case (#    84). -> Error for min 1.1e-01 | Error for max 2.3e-01
    # Bad case (#    86). -> Error for min 0.0e+00 | Error for max 6.1e-02
    # Bad case (#    88). -> Error for min 5.4e-02 | Error for max 5.9e-02
    # Bad case (#    91). -> Error for min 3.6e-02 | Error for max 9.5e-02
    # Bad case (#    92). -> Error for min 0.0e+00 | Error for max 8.4e-02
    # Bad case (#    94). -> Error for min 0.0e+00 | Error for max 1.1e-02
    # Bad case (#    95). -> Error for min 3.9e-02 | Error for max 0.0e+00
    # Bad case (#    96). -> Error for min 1.2e-01 | Error for max 3.2e-01
    # Bad case (#    97). -> Error for min 2.8e-02 | Error for max 0.0e+00
    # Bad case (#   102). -> Error for min 4.2e-02 | Error for max 1.2e-16
    # Bad case (#   103). -> Error for min 0.0e+00 | Error for max 8.3e-02
    # Bad case (#   104). -> Error for min 2.2e-16 | Error for max 1.1e-01
    # Bad case (#   109). -> Error for min 0.0e+00 | Error for max 2.9e-02
    # Bad case (#   115). -> Error for min 3.4e-01 | Error for max 0.0e+00
    # Bad case (#   116). -> Error for min 3.8e-02 | Error for max 6.6e-02
    # Bad case (#   120). -> Error for min 0.0e+00 | Error for max 1.5e-01
    # Bad case (#   122). -> Error for min 1.5e-01 | Error for max 0.0e+00
    # Bad case (#   123). -> Error for min 1.1e-01 | Error for max 1.1e-01
    # Bad case (#   128). -> Error for min 4.2e-02 | Error for max 0.0e+00
    # Bad case (#   129). -> Error for min 6.6e-02 | Error for max 0.0e+00
    # Bad case (#   131). -> Error for min 1.3e-02 | Error for max 0.0e+00
    # Bad case (#   138). -> Error for min 5.5e-04 | Error for max 2.2e-16
    # Bad case (#   139). -> Error for min 8.5e-03 | Error for max 0.0e+00
    # Bad case (#   142). -> Error for min 8.1e-02 | Error for max 0.0e+00
    # Bad case (#   144). -> Error for min 3.2e-01 | Error for max 6.3e-02
    # Bad case (#   150). -> Error for min 3.9e-03 | Error for max 0.0e+00
    # Bad case (#   151). -> Error for min 1.4e-03 | Error for max 2.3e-16
    # Bad case (#   153). -> Error for min 2.6e-01 | Error for max 0.0e+00
    # Bad case (#   154). -> Error for min 1.8e-01 | Error for max 0.0e+00
    # Bad case (#   155). -> Error for min 2.9e-02 | Error for max 0.0e+00
    # Bad case (#   158). -> Error for min 1.7e-16 | Error for max 1.9e-02
    # Bad case (#   160). -> Error for min 4.4e-03 | Error for max 5.0e-03
    # Bad case (#   163). -> Error for min 1.7e-01 | Error for max 1.5e-16
    # Bad case (#   164). -> Error for min 1.1e-16 | Error for max 5.4e-02
    # Bad case (#   167). -> Error for min 1.9e-04 | Error for max 3.3e-16
    # Bad case (#   169). -> Error for min 1.7e-16 | Error for max 1.0e-01
    # Bad case (#   173). -> Error for min 1.9e-01 | Error for max 1.5e-01
    # Bad case (#   174). -> Error for min 1.7e-16 | Error for max 7.8e-02
    # Bad case (#   178). -> Error for min 1.8e-02 | Error for max 1.6e-02
    # Bad case (#   189). -> Error for min 1.1e-01 | Error for max 2.7e-16
    # Bad case (#   195). -> Error for min 6.6e-02 | Error for max 5.4e-02
    # Bad case (#   196). -> Error for min 2.6e-01 | Error for max 1.4e-01
    # Bad case (#   197). -> Error for min 0.0e+00 | Error for max 4.2e-03
    # Bad case (#   200). -> Error for min 0.0e+00 | Error for max 1.1e-01
    # Bad case (#   201). -> Error for min 0.0e+00 | Error for max 1.0e-01
    # Bad case (#   203). -> Error for min 1.2e-01 | Error for max 2.1e-16
    # Bad case (#   204). -> Error for min 1.2e-16 | Error for max 5.8e-02
    # Bad case (#   206). -> Error for min 1.3e-01 | Error for max 2.0e-01
    # Bad case (#   207). -> Error for min 1.8e-01 | Error for max 1.3e-02
    # Bad case (#   209). -> Error for min 6.6e-02 | Error for max 0.0e+00
    # Bad case (#   211). -> Error for min 4.7e-16 | Error for max 1.0e-01
    # Bad case (#   217). -> Error for min 2.6e-16 | Error for max 2.9e-02
    # Bad case (#   218). -> Error for min 3.9e-16 | Error for max 3.3e-02
    # Bad case (#   220). -> Error for min 1.2e-01 | Error for max 1.1e-16
    # Bad case (#   222). -> Error for min 0.0e+00 | Error for max 7.0e-02
    # Bad case (#   223). -> Error for min 0.0e+00 | Error for max 6.0e-02
    # Bad case (#   224). -> Error for min 1.7e-16 | Error for max 1.7e-01
    # Bad case (#   225). -> Error for min 1.4e-16 | Error for max 1.4e-01
    # Bad case (#   227). -> Error for min 3.9e-16 | Error for max 1.9e-01
    # Bad case (#   229). -> Error for min 3.8e-02 | Error for max 8.6e-02
    # Bad case (#   233). -> Error for min 0.0e+00 | Error for max 3.5e-02
    # Bad case (#   234). -> Error for min 1.4e-16 | Error for max 1.4e-02
    # Bad case (#   238). -> Error for min 2.2e-01 | Error for max 0.0e+00
    # Bad case (#   241). -> Error for min 0.0e+00 | Error for max 2.4e-02
    # Bad case (#   242). -> Error for min 1.3e-01 | Error for max 1.4e-16
    # Bad case (#   243). -> Error for min 7.1e-02 | Error for max 4.1e-02
    # Bad case (#   250). -> Error for min 2.3e-01 | Error for max 2.1e-01
    # Bad case (#   252). -> Error for min 1.3e-16 | Error for max 4.6e-02
    # Bad case (#   254). -> Error for min 0.0e+00 | Error for max 7.4e-03
    # Bad case (#   256). -> Error for min 6.9e-03 | Error for max 1.5e-16
    # Bad case (#   257). -> Error for min 4.8e-03 | Error for max 5.1e-02
    # Bad case (#   259). -> Error for min 2.0e-01 | Error for max 0.0e+00
    # Bad case (#   260). -> Error for min 1.1e-16 | Error for max 2.1e-01
    # Bad case (#   263). -> Error for min 3.5e-02 | Error for max 2.2e-01
    # Bad case (#   270). -> Error for min 4.6e-02 | Error for max 1.2e-16
    # Bad case (#   271). -> Error for min 2.4e-02 | Error for max 8.6e-02
    # Bad case (#   272). -> Error for min 1.8e-16 | Error for max 4.8e-02
    # Bad case (#   278). -> Error for min 1.8e-02 | Error for max 2.1e-16
    # Bad case (#   279). -> Error for min 7.2e-02 | Error for max 1.4e-16
    # Bad case (#   280). -> Error for min 3.1e-02 | Error for max 3.6e-16
    # Bad case (#   281). -> Error for min 1.5e-16 | Error for max 2.6e-01
    # Bad case (#   283). -> Error for min 2.8e-01 | Error for max 1.9e-01
    # Bad case (#   287). -> Error for min 0.0e+00 | Error for max 7.2e-02
    # Bad case (#   288). -> Error for min 1.4e-16 | Error for max 1.2e-01
    # Bad case (#   293). -> Error for min 2.4e-01 | Error for max 2.4e-01
    # Bad case (#   294). -> Error for min 0.0e+00 | Error for max 2.3e-02
    # Bad case (#   296). -> Error for min 7.3e-02 | Error for max 0.0e+00
    # Bad case (#   302). -> Error for min 0.0e+00 | Error for max 7.8e-02
    # Bad case (#   303). -> Error for min 6.2e-02 | Error for max 1.7e-16
    # Bad case (#   305). -> Error for min 8.4e-02 | Error for max 1.4e-01
    # Bad case (#   306). -> Error for min 2.6e-01 | Error for max 1.2e-16
    # Bad case (#   309). -> Error for min 1.6e-01 | Error for max 2.5e-16
    # Bad case (#   314). -> Error for min 1.3e-02 | Error for max 2.6e-16
    # Bad case (#   315). -> Error for min 5.6e-02 | Error for max 1.4e-16
    # Bad case (#   324). -> Error for min 2.3e-02 | Error for max 1.4e-16
    # Bad case (#   325). -> Error for min 6.7e-02 | Error for max 1.5e-16
    # Bad case (#   327). -> Error for min 3.6e-02 | Error for max 1.3e-16
    # Bad case (#   330). -> Error for min 1.1e-01 | Error for max 4.7e-02
    # Bad case (#   331). -> Error for min 9.0e-03 | Error for max 1.2e-01
    # Bad case (#   332). -> Error for min 1.9e-01 | Error for max 0.0e+00
    # Bad case (#   334). -> Error for min 3.6e-02 | Error for max 0.0e+00
    # Bad case (#   341). -> Error for min 1.2e-01 | Error for max 1.8e-01
    # Bad case (#   345). -> Error for min 7.8e-03 | Error for max 1.2e-16
    # Bad case (#   348). -> Error for min 1.8e-16 | Error for max 7.6e-02
    # Bad case (#   349). -> Error for min 4.2e-02 | Error for max 3.6e-16
    # Bad case (#   351). -> Error for min 1.9e-01 | Error for max 4.1e-02
    # Bad case (#   354). -> Error for min 1.4e-16 | Error for max 3.4e-02
    # Bad case (#   355). -> Error for min 2.2e-01 | Error for max 8.9e-02
    # Bad case (#   357). -> Error for min 2.2e-02 | Error for max 4.1e-16
    # Bad case (#   358). -> Error for min 1.4e-01 | Error for max 2.2e-16
    # Bad case (#   359). -> Error for min 0.0e+00 | Error for max 7.1e-02
    # Bad case (#   360). -> Error for min 4.0e-02 | Error for max 0.0e+00
    # Bad case (#   361). -> Error for min 4.1e-03 | Error for max 1.2e-16
    # Bad case (#   364). -> Error for min 3.3e-02 | Error for max 0.0e+00
    # Bad case (#   369). -> Error for min 2.4e-02 | Error for max 0.0e+00
    # Bad case (#   373). -> Error for min 3.0e-01 | Error for max 2.9e-16
    # Bad case (#   375). -> Error for min 0.0e+00 | Error for max 2.3e-02
    # Bad case (#   376). -> Error for min 5.4e-02 | Error for max 6.9e-02
    # Bad case (#   378). -> Error for min 0.0e+00 | Error for max 1.5e-01
    # Bad case (#   379). -> Error for min 6.6e-02 | Error for max 1.2e-01
    # Bad case (#   382). -> Error for min 0.0e+00 | Error for max 3.0e-02
    # Bad case (#   387). -> Error for min 4.6e-02 | Error for max 1.3e-01
    # Bad case (#   393). -> Error for min 4.3e-02 | Error for max 0.0e+00
    # Bad case (#   396). -> Error for min 7.2e-02 | Error for max 3.3e-02
    # Bad case (#   398). -> Error for min 1.9e-16 | Error for max 6.0e-02
    # Bad case (#   403). -> Error for min 3.3e-02 | Error for max 0.0e+00
    # Bad case (#   405). -> Error for min 1.2e-01 | Error for max 4.5e-16
    # Bad case (#   406). -> Error for min 1.7e-03 | Error for max 0.0e+00
    # Bad case (#   408). -> Error for min 3.0e-01 | Error for max 7.7e-03
    # Bad case (#   415). -> Error for min 1.6e-16 | Error for max 7.4e-02
    # Bad case (#   417). -> Error for min 6.8e-02 | Error for max 1.4e-01
    # Bad case (#   419). -> Error for min 4.0e-03 | Error for max 1.4e-16
    # Bad case (#   420). -> Error for min 9.5e-02 | Error for max 2.2e-16
    # Bad case (#   421). -> Error for min 1.1e-01 | Error for max 2.0e-16
    # Bad case (#   422). -> Error for min 0.0e+00 | Error for max 2.8e-02
    # Bad case (#   424). -> Error for min 2.6e-02 | Error for max 1.3e-03
    # Bad case (#   429). -> Error for min 3.1e-16 | Error for max 2.4e-01
    # Bad case (#   432). -> Error for min 1.2e-01 | Error for max 0.0e+00
    # Bad case (#   434). -> Error for min 0.0e+00 | Error for max 1.2e-01
    # Bad case (#   443). -> Error for min 2.3e-02 | Error for max 1.4e-16
    # Bad case (#   445). -> Error for min 8.3e-02 | Error for max 0.0e+00
    # Bad case (#   451). -> Error for min 0.0e+00 | Error for max 1.3e-02
    # Bad case (#   452). -> Error for min 0.0e+00 | Error for max 8.4e-02
    # Bad case (#   453). -> Error for min 5.7e-02 | Error for max 1.1e-01
    # Bad case (#   454). -> Error for min 1.1e-16 | Error for max 2.5e-02
    # Bad case (#   459). -> Error for min 4.0e-02 | Error for max 2.6e-16
    # Bad case (#   461). -> Error for min 8.7e-02 | Error for max 2.1e-01
    # Bad case (#   464). -> Error for min 1.6e-16 | Error for max 1.0e-02
    # Bad case (#   470). -> Error for min 0.0e+00 | Error for max 2.2e-01
    # Bad case (#   471). -> Error for min 1.2e-16 | Error for max 7.8e-02
    # Bad case (#   473). -> Error for min 1.6e-01 | Error for max 0.0e+00
    # Bad case (#   478). -> Error for min 0.0e+00 | Error for max 4.8e-02
    # Bad case (#   482). -> Error for min 8.9e-02 | Error for max 4.3e-02
    # Bad case (#   483). -> Error for min 4.6e-02 | Error for max 2.0e-16
    # Bad case (#   484). -> Error for min 1.1e-16 | Error for max 1.7e-01
    # Bad case (#   486). -> Error for min 9.4e-02 | Error for max 2.5e-16
    # Bad case (#   489). -> Error for min 1.6e-01 | Error for max 2.2e-02
    # Bad case (#   490). -> Error for min 0.0e+00 | Error for max 1.5e-02
    # Bad case (#   496). -> Error for min 1.3e-01 | Error for max 1.3e-01
    # Bad case (#   499). -> Error for min 1.6e-02 | Error for max 0.0e+00
    # Bad case (#   500). -> Error for min 0.0e+00 | Error for max 6.4e-02
    # Bad case (#   502). -> Error for min 2.3e-01 | Error for max 1.2e-16
    # Bad case (#   506). -> Error for min 0.0e+00 | Error for max 4.8e-02
    # Bad case (#   507). -> Error for min 2.1e-02 | Error for max 0.0e+00
    # Bad case (#   508). -> Error for min 0.0e+00 | Error for max 8.4e-04
    # Bad case (#   512). -> Error for min 4.6e-02 | Error for max 0.0e+00
    # Bad case (#   514). -> Error for min 2.9e-16 | Error for max 7.4e-02
    # Bad case (#   515). -> Error for min 1.5e-01 | Error for max 6.6e-02
    # Bad case (#   518). -> Error for min 1.5e-01 | Error for max 9.5e-02
    # Bad case (#   521). -> Error for min 1.5e-16 | Error for max 1.2e-01
    # Bad case (#   522). -> Error for min 2.5e-16 | Error for max 9.4e-02
    # Bad case (#   525). -> Error for min 0.0e+00 | Error for max 7.8e-02
    # Bad case (#   529). -> Error for min 2.5e-16 | Error for max 5.1e-02
    # Bad case (#   530). -> Error for min 1.9e-16 | Error for max 5.0e-02
    # Bad case (#   537). -> Error for min 0.0e+00 | Error for max 7.3e-02
    # Bad case (#   538). -> Error for min 2.0e-16 | Error for max 1.8e-02
    # Bad case (#   540). -> Error for min 0.0e+00 | Error for max 1.2e-02
    # Bad case (#   542). -> Error for min 6.7e-02 | Error for max 0.0e+00
    # Bad case (#   543). -> Error for min 3.9e-02 | Error for max 0.0e+00
    # Bad case (#   544). -> Error for min 1.3e-16 | Error for max 2.4e-01
    # Bad case (#   546). -> Error for min 0.0e+00 | Error for max 2.0e-01
    # Bad case (#   547). -> Error for min 1.5e-01 | Error for max 1.9e-01
    # Bad case (#   550). -> Error for min 0.0e+00 | Error for max 1.8e-02
    # Bad case (#   552). -> Error for min 1.6e-16 | Error for max 6.4e-02
    # Bad case (#   553). -> Error for min 1.2e-01 | Error for max 0.0e+00
    # Bad case (#   559). -> Error for min 2.5e-02 | Error for max 2.6e-16
    # Bad case (#   561). -> Error for min 1.8e-01 | Error for max 1.2e-01
    # Bad case (#   562). -> Error for min 1.2e-16 | Error for max 1.1e-01
    # Bad case (#   563). -> Error for min 1.2e-16 | Error for max 2.1e-01
    # Bad case (#   564). -> Error for min 1.4e-16 | Error for max 6.4e-02
    # Bad case (#   565). -> Error for min 9.6e-02 | Error for max 2.2e-16
    # Bad case (#   568). -> Error for min 7.4e-02 | Error for max 1.5e-16
    # Bad case (#   569). -> Error for min 4.8e-02 | Error for max 1.4e-16
    # Bad case (#   570). -> Error for min 0.0e+00 | Error for max 5.0e-02
    # Bad case (#   571). -> Error for min 2.6e-01 | Error for max 5.7e-02
    # Bad case (#   573). -> Error for min 1.5e-16 | Error for max 1.3e-03
    # Bad case (#   576). -> Error for min 1.1e-01 | Error for max 1.9e-16
    # Bad case (#   577). -> Error for min 6.9e-02 | Error for max 2.7e-16
    # Bad case (#   578). -> Error for min 1.8e-16 | Error for max 9.4e-02
    # Bad case (#   583). -> Error for min 4.2e-02 | Error for max 1.1e-01
    # Bad case (#   585). -> Error for min 1.4e-01 | Error for max 4.5e-02
    # Bad case (#   586). -> Error for min 4.0e-02 | Error for max 0.0e+00
    # Bad case (#   593). -> Error for min 1.4e-02 | Error for max 2.2e-16
    # Bad case (#   602). -> Error for min 1.6e-01 | Error for max 1.8e-01
    # Bad case (#   605). -> Error for min 0.0e+00 | Error for max 2.1e-02
    # Bad case (#   607). -> Error for min 3.2e-02 | Error for max 0.0e+00
    # Bad case (#   608). -> Error for min 8.7e-02 | Error for max 5.3e-02
    # Bad case (#   609). -> Error for min 6.4e-02 | Error for max 2.7e-16
    # Bad case (#   613). -> Error for min 5.2e-02 | Error for max 5.9e-02
    # Bad case (#   616). -> Error for min 2.1e-16 | Error for max 7.1e-03
    # Bad case (#   617). -> Error for min 1.9e-16 | Error for max 3.1e-02
    # Bad case (#   621). -> Error for min 1.4e-16 | Error for max 1.4e-02
    # Bad case (#   623). -> Error for min 0.0e+00 | Error for max 4.8e-03
    # Bad case (#   625). -> Error for min 0.0e+00 | Error for max 4.9e-02
    # Bad case (#   629). -> Error for min 6.2e-02 | Error for max 3.2e-16
    # Bad case (#   631). -> Error for min 1.1e-16 | Error for max 7.8e-02
    # Bad case (#   635). -> Error for min 1.3e-16 | Error for max 4.8e-02
    # Bad case (#   642). -> Error for min 0.0e+00 | Error for max 1.7e-01
    # Bad case (#   648). -> Error for min 2.1e-16 | Error for max 7.3e-02
    # Bad case (#   650). -> Error for min 5.4e-02 | Error for max 5.1e-02
    # Bad case (#   655). -> Error for min 9.2e-02 | Error for max 2.3e-16
    # Bad case (#   656). -> Error for min 0.0e+00 | Error for max 1.1e-02
    # Bad case (#   658). -> Error for min 1.6e-01 | Error for max 1.5e-02
    # Bad case (#   660). -> Error for min 0.0e+00 | Error for max 2.6e-02
    # Bad case (#   664). -> Error for min 1.6e-01 | Error for max 8.8e-02
    # Bad case (#   665). -> Error for min 9.8e-02 | Error for max 1.4e-01
    # Bad case (#   666). -> Error for min 2.2e-01 | Error for max 1.1e-16
    # Bad case (#   667). -> Error for min 0.0e+00 | Error for max 6.0e-02
    # Bad case (#   668). -> Error for min 2.3e-01 | Error for max 1.6e-16
    # Bad case (#   670). -> Error for min 3.4e-02 | Error for max 0.0e+00
    # Bad case (#   672). -> Error for min 1.3e-16 | Error for max 2.5e-02
    # Bad case (#   683). -> Error for min 2.2e-01 | Error for max 2.7e-01
    # Bad case (#   685). -> Error for min 1.5e-16 | Error for max 2.9e-02
    # Bad case (#   693). -> Error for min 6.7e-03 | Error for max 1.7e-16
    # Bad case (#   695). -> Error for min 4.2e-02 | Error for max 1.8e-01
    # Bad case (#   696). -> Error for min 1.4e-01 | Error for max 1.4e-01
    # Bad case (#   699). -> Error for min 2.8e-01 | Error for max 5.8e-02
    # Bad case (#   700). -> Error for min 5.5e-02 | Error for max 1.4e-16
    # Bad case (#   701). -> Error for min 2.2e-16 | Error for max 1.2e-01
    # Bad case (#   709). -> Error for min 4.0e-01 | Error for max 1.2e-16
    # Bad case (#   710). -> Error for min 2.1e-16 | Error for max 9.8e-02
    # Bad case (#   712). -> Error for min 3.1e-16 | Error for max 1.6e-01
    # Bad case (#   713). -> Error for min 2.2e-01 | Error for max 0.0e+00
    # Bad case (#   714). -> Error for min 5.5e-16 | Error for max 3.7e-02
    # Bad case (#   715). -> Error for min 1.4e-01 | Error for max 2.2e-02
    # Bad case (#   716). -> Error for min 3.8e-03 | Error for max 1.3e-16
    # Bad case (#   717). -> Error for min 0.0e+00 | Error for max 3.2e-02
    # Bad case (#   723). -> Error for min 0.0e+00 | Error for max 4.9e-03
    # Bad case (#   725). -> Error for min 1.5e-02 | Error for max 7.1e-02
    # Bad case (#   732). -> Error for min 1.2e-02 | Error for max 0.0e+00
    # Bad case (#   735). -> Error for min 5.9e-03 | Error for max 1.2e-16
    # Bad case (#   736). -> Error for min 4.7e-02 | Error for max 0.0e+00
    # Bad case (#   740). -> Error for min 2.4e-01 | Error for max 0.0e+00
    # Bad case (#   742). -> Error for min 0.0e+00 | Error for max 2.6e-01
    # Bad case (#   744). -> Error for min 1.4e-01 | Error for max 5.3e-02
    # Bad case (#   747). -> Error for min 2.5e-03 | Error for max 1.1e-16
    # Bad case (#   750). -> Error for min 1.3e-16 | Error for max 1.2e-02
    # Bad case (#   751). -> Error for min 1.6e-16 | Error for max 5.4e-02
    # Bad case (#   753). -> Error for min 8.4e-02 | Error for max 0.0e+00
    # Bad case (#   754). -> Error for min 4.1e-02 | Error for max 0.0e+00
    # Bad case (#   756). -> Error for min 6.1e-02 | Error for max 1.3e-16
    # Bad case (#   759). -> Error for min 0.0e+00 | Error for max 2.1e-02
    # Bad case (#   762). -> Error for min 1.7e-16 | Error for max 9.9e-02
    # Bad case (#   767). -> Error for min 1.5e-01 | Error for max 1.6e-01
    # Bad case (#   768). -> Error for min 2.6e-03 | Error for max 2.2e-02
    # Bad case (#   772). -> Error for min 7.4e-02 | Error for max 1.7e-01
    # Bad case (#   773). -> Error for min 2.1e-16 | Error for max 5.9e-03
    # Bad case (#   776). -> Error for min 1.6e-16 | Error for max 9.7e-02
    # Bad case (#   777). -> Error for min 1.1e-01 | Error for max 3.0e-02
    # Bad case (#   778). -> Error for min 4.3e-02 | Error for max 1.4e-02
    # Bad case (#   779). -> Error for min 3.4e-02 | Error for max 1.2e-01
    # Bad case (#   782). -> Error for min 1.3e-16 | Error for max 6.3e-02
    # Bad case (#   785). -> Error for min 1.6e-01 | Error for max 1.7e-16
    # Bad case (#   789). -> Error for min 7.1e-04 | Error for max 6.2e-02
    # Bad case (#   791). -> Error for min 2.4e-01 | Error for max 1.6e-01
    # Bad case (#   792). -> Error for min 1.2e-16 | Error for max 4.2e-04
    # Bad case (#   794). -> Error for min 3.6e-02 | Error for max 1.3e-02
    # Bad case (#   797). -> Error for min 2.3e-01 | Error for max 2.0e-01
    # Bad case (#   799). -> Error for min 7.9e-03 | Error for max 0.0e+00
    # Bad case (#   802). -> Error for min 1.5e-16 | Error for max 7.8e-03
    # Bad case (#   806). -> Error for min 0.0e+00 | Error for max 3.2e-02
    # Bad case (#   810). -> Error for min 2.2e-01 | Error for max 2.2e-16
    # Bad case (#   811). -> Error for min 8.7e-02 | Error for max 0.0e+00
    # Bad case (#   813). -> Error for min 1.2e-16 | Error for max 5.3e-02
    # Bad case (#   814). -> Error for min 0.0e+00 | Error for max 2.2e-02
    # Bad case (#   821). -> Error for min 2.2e-01 | Error for max 1.1e-16
    # Bad case (#   823). -> Error for min 1.8e-16 | Error for max 3.8e-02
    # Bad case (#   824). -> Error for min 0.0e+00 | Error for max 2.0e-02
    # Bad case (#   827). -> Error for min 3.0e-01 | Error for max 0.0e+00
    # Bad case (#   828). -> Error for min 2.4e-16 | Error for max 4.6e-02
    # Bad case (#   835). -> Error for min 1.4e-01 | Error for max 0.0e+00
    # Bad case (#   836). -> Error for min 9.8e-02 | Error for max 2.3e-16
    # Bad case (#   839). -> Error for min 0.0e+00 | Error for max 1.7e-01
    # Bad case (#   840). -> Error for min 1.6e-01 | Error for max 1.3e-16
    # Bad case (#   843). -> Error for min 0.0e+00 | Error for max 1.9e-01
    # Bad case (#   845). -> Error for min 7.6e-03 | Error for max 2.7e-02
    # Bad case (#   848). -> Error for min 9.1e-02 | Error for max 3.0e-01
    # Bad case (#   850). -> Error for min 1.8e-16 | Error for max 1.7e-03
    # Bad case (#   851). -> Error for min 1.4e-16 | Error for max 1.4e-02
    # Bad case (#   852). -> Error for min 2.0e-16 | Error for max 1.9e-02
    # Bad case (#   854). -> Error for min 1.9e-01 | Error for max 2.7e-03
    # Bad case (#   855). -> Error for min 3.1e-02 | Error for max 1.3e-16
    # Bad case (#   862). -> Error for min 1.7e-02 | Error for max 1.4e-16
    # Bad case (#   864). -> Error for min 1.1e-01 | Error for max 1.9e-01
    # Bad case (#   866). -> Error for min 1.8e-16 | Error for max 1.5e-01
    # Bad case (#   868). -> Error for min 2.4e-16 | Error for max 4.5e-02
    # Bad case (#   870). -> Error for min 1.8e-16 | Error for max 4.9e-02
    # Bad case (#   872). -> Error for min 1.4e-16 | Error for max 3.9e-02
    # Bad case (#   875). -> Error for min 1.2e-02 | Error for max 0.0e+00
    # Bad case (#   876). -> Error for min 6.0e-02 | Error for max 1.6e-01
    # Bad case (#   877). -> Error for min 0.0e+00 | Error for max 3.2e-03
    # Bad case (#   879). -> Error for min 1.4e-01 | Error for max 1.4e-16
    # Bad case (#   884). -> Error for min 1.6e-01 | Error for max 1.2e-01
    # Bad case (#   885). -> Error for min 1.2e-03 | Error for max 3.7e-02
    # Bad case (#   887). -> Error for min 9.1e-03 | Error for max 0.0e+00
    # Bad case (#   889). -> Error for min 1.2e-16 | Error for max 1.6e-01
    # Bad case (#   890). -> Error for min 2.4e-01 | Error for max 1.9e-16
    # Bad case (#   897). -> Error for min 4.0e-02 | Error for max 3.1e-16
    # Bad case (#   898). -> Error for min 1.6e-16 | Error for max 2.4e-01
    # Bad case (#   903). -> Error for min 2.4e-01 | Error for max 1.2e-01
    # Bad case (#   906). -> Error for min 8.3e-02 | Error for max 1.1e-16
    # Bad case (#   910). -> Error for min 1.9e-02 | Error for max 6.7e-02
    # Bad case (#   911). -> Error for min 6.1e-02 | Error for max 0.0e+00
    # Bad case (#   916). -> Error for min 8.8e-02 | Error for max 1.2e-01
    # Bad case (#   917). -> Error for min 1.3e-16 | Error for max 1.6e-02
    # Bad case (#   919). -> Error for min 2.6e-01 | Error for max 3.1e-16
    # Bad case (#   923). -> Error for min 4.2e-02 | Error for max 2.8e-01
    # Bad case (#   924). -> Error for min 1.4e-16 | Error for max 1.2e-02
    # Bad case (#   926). -> Error for min 6.8e-02 | Error for max 1.2e-01
    # Bad case (#   927). -> Error for min 2.4e-16 | Error for max 1.3e-01
    # Bad case (#   931). -> Error for min 1.5e-01 | Error for max 3.3e-01
    # Bad case (#   932). -> Error for min 0.0e+00 | Error for max 1.3e-01
    # Bad case (#   933). -> Error for min 1.5e-02 | Error for max 2.3e-01
    # Bad case (#   936). -> Error for min 4.8e-02 | Error for max 7.9e-02
    # Bad case (#   947). -> Error for min 6.9e-02 | Error for max 2.6e-16
    # Bad case (#   954). -> Error for min 7.2e-03 | Error for max 1.7e-16
    # Bad case (#   955). -> Error for min 8.2e-02 | Error for max 0.0e+00
    # Bad case (#   962). -> Error for min 2.4e-16 | Error for max 1.8e-02
    # Bad case (#   966). -> Error for min 1.1e-02 | Error for max 1.2e-16
    # Bad case (#   967). -> Error for min 7.4e-02 | Error for max 0.0e+00
    # Bad case (#   968). -> Error for min 1.0e-01 | Error for max 2.2e-16
    # Bad case (#   971). -> Error for min 2.2e-01 | Error for max 1.3e-01
    # Bad case (#   972). -> Error for min 1.5e-01 | Error for max 2.4e-01
    # Bad case (#   973). -> Error for min 0.0e+00 | Error for max 6.6e-02
    # Bad case (#   974). -> Error for min 0.0e+00 | Error for max 1.1e-01
    # Bad case (#   976). -> Error for min 1.4e-01 | Error for max 2.1e-16
    # Bad case (#   980). -> Error for min 7.2e-02 | Error for max 2.3e-16
    # Bad case (#   981). -> Error for min 1.4e-01 | Error for max 1.1e-16
    # Bad case (#   986). -> Error for min 0.0e+00 | Error for max 1.8e-01
    # Bad case (#   988). -> Error for min 1.3e-16 | Error for max 3.0e-02
    # Bad case (#   989). -> Error for min 3.2e-03 | Error for max 6.0e-02
    # Bad case (#   994). -> Error for min 5.9e-02 | Error for max 0.0e+00
    # Bad case (#   996). -> Error for min 2.9e-02 | Error for max 2.3e-16
    # Bad case (#  1000). -> Error for min 1.9e-01 | Error for max 4.0e-16
    # ----------------------------------------------------------------------
    # Average time    :   0.9220
    # Total bad cases :      385
    # 

  .. code-block:: python

    # Let consider bad results and do only one inner iteration
    # (i.e., maxvol for "(Y-y_ref)^2"):
    ind_bad_3 = []
    
    t = tpc()
    
    for ind in ind_bad_2:
        Y = Y_list[ind]
        i_min_appr, i_max_appr = teneva.opt_tt(Y, nswp_outer=1, nswp=1, r=100)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = np.abs(y_min_appr - y_min_real) / np.abs(y_min_real)
        e_max = np.abs(y_max_appr - y_max_real) / np.abs(y_max_real)
    
        if e_min > e_bad or e_max > e_bad:
            ind_bad_3.append(ind)
            print(f'Bad case (# {ind+1:-5d}). -> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e}')
                  
    k_all = len(ind_bad_2)
    k_bad = len(ind_bad_3)
    t = (tpc() - t) / k_all if k_all else 0.
    
    print('-' * 70)
    print(f'Average time    : {t:-8.4f}')
    print(f'Total bad cases : {k_bad:-8d}')

    # >>> ----------------------------------------
    # >>> Output:

    # Bad case (#    37). -> Error for min 1.2e-16 | Error for max 1.9e-04
    # Bad case (#   116). -> Error for min 3.8e-02 | Error for max 2.1e-16
    # Bad case (#   128). -> Error for min 4.2e-02 | Error for max 0.0e+00
    # Bad case (#   164). -> Error for min 1.1e-16 | Error for max 7.7e-03
    # Bad case (#   254). -> Error for min 0.0e+00 | Error for max 7.4e-03
    # Bad case (#   361). -> Error for min 4.1e-03 | Error for max 1.2e-16
    # Bad case (#   382). -> Error for min 0.0e+00 | Error for max 3.0e-02
    # Bad case (#   387). -> Error for min 1.3e-16 | Error for max 3.0e-02
    # Bad case (#   451). -> Error for min 0.0e+00 | Error for max 1.3e-02
    # Bad case (#   512). -> Error for min 6.6e-03 | Error for max 0.0e+00
    # Bad case (#   562). -> Error for min 1.2e-16 | Error for max 1.1e-01
    # Bad case (#   621). -> Error for min 1.4e-16 | Error for max 1.4e-02
    # Bad case (#   650). -> Error for min 1.2e-16 | Error for max 3.8e-02
    # Bad case (#   664). -> Error for min 0.0e+00 | Error for max 5.3e-02
    # Bad case (#   903). -> Error for min 0.0e+00 | Error for max 3.6e-02
    # Bad case (#   936). -> Error for min 4.8e-02 | Error for max 0.0e+00
    # Bad case (#   974). -> Error for min 0.0e+00 | Error for max 1.1e-01
    # ----------------------------------------------------------------------
    # Average time    :   1.8239
    # Total bad cases :       17
    # 

  .. code-block:: python

    # Let consider bad results and do two inner iterations
    # (i.e., maxvol for "(Y-y_ref)^2" and then for "(Y-y_ref)^4"):
    ind_bad_4 = []
    
    t = tpc()
    
    for ind in ind_bad_3:
        Y = Y_list[ind]
        i_min_appr, i_max_appr = teneva.opt_tt(Y, nswp_outer=1, nswp=2, r=100)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = np.abs(y_min_appr - y_min_real) / np.abs(y_min_real)
        e_max = np.abs(y_max_appr - y_max_real) / np.abs(y_max_real)
    
        if e_min > e_bad or e_max > e_bad:
            ind_bad_4.append(ind)
            print(f'Bad case (# {ind+1:-5d}). -> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e}')
                  
    k_all = len(ind_bad_3)
    k_bad = len(ind_bad_4)
    t = (tpc() - t) / k_all if k_all else 0.
    
    print('-' * 70)
    print(f'Average time    : {t:-8.4f}')
    print(f'Total bad cases : {k_bad:-8d}')

    # >>> ----------------------------------------
    # >>> Output:

    # ----------------------------------------------------------------------
    # Average time    :   7.5682
    # Total bad cases :        0
    # 



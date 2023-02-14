Module optima: estimate min and max value of the tensor
-------------------------------------------------------


.. automodule:: teneva.core.optima


-----




|
|

.. autofunction:: teneva.optima_qtt

  **Examples**:

  .. code-block:: python

    d = 5                    # Dimension
    q = 4                    # Mode size factor
    n = [2**q]*d             # Shape of the tensor
    Y = teneva.rand(n, r=4)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_qtt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [ 5 12 12  6  9]
    # i max appr : [10 12  3  0 12]
    # y min appr :  -1.1638e+01
    # y max appr :   1.2187e+01
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)   # Transform the TT-tensor to full format
    i_min = np.argmin(Y_full) # Multi-index of the minimum
    i_max = np.argmax(Y_full) # Multi-index of the maximum
    
    i_min = np.unravel_index(i_min, n)
    i_max = np.unravel_index(i_max, n)
    
    print(f'i min real :', i_min)
    print(f'i max real :', i_max)
    print(f'y min real : {Y_full[i_min]:-12.4e}')
    print(f'y max real : {Y_full[i_max]:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min real : (5, 12, 12, 6, 9)
    # i max real : (10, 12, 3, 0, 12)
    # y min real :  -1.1638e+01
    # y max real :   1.2187e+01
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    d = 5        # Dimension
    q = 4        # Mode size factor
    n = [2**q]*d # Shape of the tensor
    
    for i in range(10):
        Y = teneva.rand(n, r=4)
        t = tpc()
        i_min_appr, y_min_appr, i_max_appr, y_max_appr = teneva.optima_qtt(Y)
        t = tpc() - t
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = abs(y_min_appr - y_min_real)
        e_max = abs(y_max_appr - y_max_real)
    
        print(f'-> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e} | Time {t:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    # -> Error for min 0.0e+00 | Error for max 3.6e-15 | Time   0.0597
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0543
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0558
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0499
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0570
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0512
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0529
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0530
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0560
    # -> Error for min 3.6e-15 | Error for max 1.8e-15 | Time   0.0522
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):

  .. code-block:: python

    # NOTE : "func" module will be removed soon!!!
    
    d = 6        # Dimension
    q = 4        # Mode size factor
    n = [2**q]*d # Shape of the tensor
    
    for func in teneva.func_demo_all(d):#, dy=0.5):
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
        i_min_appr, y_min_appr, i_max_appr, y_max_appr = teneva.optima_qtt(Y)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
        
        # Check the accuracy of result:
        e_min = abs(y_min_real - y_min_appr)
        e_max = abs(y_max_real - y_max_appr)
        
        # Present the result:
        text = '-> ' + func.name + ' ' * max(0, 20 - len(func.name)) + ' | '
        text += f'TT-rank {r:-5.1f} | '
        text += f'Error for min {e_min:-7.1e} | '
        text += f'Error for max {e_max:-7.1e} | '
        text += f'Time {t:-8.4f} | '
        print(text)

    # >>> ----------------------------------------
    # >>> Output:

    # -> Ackley               | TT-rank   9.7 | Error for min 1.8e-15 | Error for max 1.8e-14 | Time   1.2194 | 
    # -> Alpine               | TT-rank   2.6 | Error for min 4.4e-02 | Error for max 0.0e+00 | Time   0.0483 | 
    # -> Dixon                | TT-rank   5.2 | Error for min 1.7e-11 | Error for max 2.3e-10 | Time   0.1450 | 
    # -> Exponential          | TT-rank   3.9 | Error for min 1.1e-16 | Error for max 6.9e-17 | Time   0.0842 | 
    # -> Grienwank            | TT-rank   3.4 | Error for min 4.7e-14 | Error for max 3.4e-13 | Time   0.0527 | 
    # -> Michalewicz          | TT-rank   6.7 | Error for min 0.0e+00 | Error for max 4.9e-13 | Time   0.3839 | 
    # -> Qing                 | TT-rank   4.5 | Error for min 1.1e-05 | Error for max 6.1e-05 | Time   0.1265 | 
    # -> Rastrigin            | TT-rank   4.5 | Error for min 6.8e-13 | Error for max 1.4e-13 | Time   0.1201 | 
    # -> Rosenbrock           | TT-rank   6.1 | Error for min 2.0e-13 | Error for max 0.0e+00 | Time   0.3018 | 
    # -> Schaffer             | TT-rank  13.6 | Error for min 2.3e-14 | Error for max 7.7e-13 | Time   4.0549 | 
    # -> Schwefel             | TT-rank   6.4 | Error for min 7.1e-14 | Error for max 0.0e+00 | Time   0.3240 | 
    # 




|
|

.. autofunction:: teneva.optima_tt

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12] # Shape of the tensor
    Y = teneva.rand(n, r=4)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [14  9 11  3  2]
    # i max appr : [17 12  0  6  2]
    # y min appr :  -1.1549e+01
    # y max appr :   1.2922e+01
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)   # Transform the TT-tensor to full format
    i_min = np.argmin(Y_full) # Multi-index of the minimum
    i_max = np.argmax(Y_full) # Multi-index of the maximum
    
    i_min = np.unravel_index(i_min, n)
    i_max = np.unravel_index(i_max, n)
    
    print(f'i min real :', i_min)
    print(f'i max real :', i_max)
    print(f'y min real : {Y_full[i_min]:-12.4e}')
    print(f'y max real : {Y_full[i_max]:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min real : (14, 9, 11, 3, 2)
    # i max real : (17, 12, 0, 6, 2)
    # y min real :  -1.1549e+01
    # y max real :   1.2922e+01
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]
    
    for i in range(10):
        Y = teneva.rand(n, r=4)
        t = tpc()
        i_min_appr, y_min_appr, i_max_appr, y_max_appr = teneva.optima_tt(Y)
        t = tpc() - t
    
        Y_full = teneva.full(Y)
        i_min_real = np.unravel_index(np.argmin(Y_full), n)
        i_max_real = np.unravel_index(np.argmax(Y_full), n)
        y_min_real = Y_full[i_min_real]
        y_max_real = Y_full[i_max_real]
        
        e_min = abs(y_min_appr - y_min_real)
        e_max = abs(y_max_appr - y_max_real)
    
        print(f'-> Error for min {e_min:-7.1e} | Error for max {e_max:-7.1e} | Time {t:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0219
    # -> Error for min 1.8e-15 | Error for max 3.6e-15 | Time   0.0131
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0136
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0136
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0134
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0137
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0148
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0136
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0136
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0133
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):

  .. code-block:: python

    # NOTE : "func" module will be removed soon!!!
    
    d = 6   # Dimension
    n = 16  # Grid size
    
    for func in teneva.func_demo_all(d):#, dy=0.5):
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
        i_min_appr, y_min_appr, i_max_appr, y_max_appr = teneva.optima_tt(Y)
        y_min_appr = teneva.get(Y, i_min_appr)
        y_max_appr = teneva.get(Y, i_max_appr)
        t = tpc() - t
        
        # Check the accuracy of result:
        e_min = abs(y_min_real - y_min_appr)
        e_max = abs(y_max_real - y_max_appr)
        
        # Present the result:
        text = '-> ' + func.name + ' ' * max(0, 20 - len(func.name)) + ' | '
        text += f'TT-rank {r:-5.1f} | '
        text += f'Error for min {e_min:-7.1e} | '
        text += f'Error for max {e_max:-7.1e} | '
        text += f'Time {t:-8.4f} | '
        print(text)

    # >>> ----------------------------------------
    # >>> Output:

    # -> Ackley               | TT-rank  10.7 | Error for min 0.0e+00 | Error for max 7.1e-15 | Time   0.2186 | 
    # -> Alpine               | TT-rank   2.6 | Error for min 2.2e-16 | Error for max 0.0e+00 | Time   0.0181 | 
    # -> Dixon                | TT-rank   3.8 | Error for min 1.6e-11 | Error for max 0.0e+00 | Time   0.0252 | 
    # -> Exponential          | TT-rank   3.9 | Error for min 0.0e+00 | Error for max 1.1e-16 | Time   0.0439 | 
    # -> Grienwank            | TT-rank   3.4 | Error for min 1.6e-14 | Error for max 0.0e+00 | Time   0.0265 | 
    # -> Michalewicz          | TT-rank   6.7 | Error for min 0.0e+00 | Error for max 2.9e-16 | Time   0.0717 | 
    # -> Qing                 | TT-rank   4.5 | Error for min 1.1e-05 | Error for max 6.1e-05 | Time   0.0319 | 
    # -> Rastrigin            | TT-rank   4.5 | Error for min 8.9e-16 | Error for max 5.7e-14 | Time   0.0316 | 
    # -> Rosenbrock           | TT-rank   6.1 | Error for min 2.0e-13 | Error for max 0.0e+00 | Time   0.0512 | 
    # -> Schaffer             | TT-rank  10.2 | Error for min 2.2e-16 | Error for max 6.4e-03 | Time   0.1925 | 
    # -> Schwefel             | TT-rank   6.4 | Error for min 7.1e-14 | Error for max 0.0e+00 | Time   0.0709 | 
    # 




|
|

.. autofunction:: teneva.optima_tt_beam

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]  # Shape of the tensor
    Y = teneva.rand(n, r=4)   # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [11  6 12  1 10]
    # y opt appr :   1.1935e+01
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)            # Transform the TT-tensor to full format
    
    i_opt = np.argmax(np.abs(Y_full))  # Multi-index of the maximum modulo item
    i_opt = np.unravel_index(i_opt, n)
    y_opt = Y_full[i_opt]              # The related tensor value
    
    print(f'i opt real :', i_opt)
    print(f'y opt real : {Y_full[i_opt]:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt real : (11, 6, 12, 1, 10)
    # y opt real :   1.1935e+01
    # 

  This function may also return the "top-k" candidates for the optimum:

  .. code-block:: python

    n = [20, 18, 16, 14, 12] # Shape of the tensor
    Y = teneva.rand(n, r=4)  # Random TT-tensor with rank 4
    
    I_opt = teneva.optima_tt_beam(Y, k=10, ret_all=True)
    
    for i_opt in I_opt:
        y_opt = abs(teneva.get(Y, i_opt))
        print(f'y : {y_opt:-12.4e} | i : {i_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # y :   1.4574e+01 | i : [7 3 2 6 7]
    # y :   1.4516e+01 | i : [7 3 2 6 1]
    # y :   1.4436e+01 | i : [15  7  8  4  1]
    # y :   1.2795e+01 | i : [7 4 2 6 7]
    # y :   1.2686e+01 | i : [18  5  8  4  1]
    # y :   1.2210e+01 | i : [7 4 2 6 1]
    # y :   1.2145e+01 | i : [14 12  1  9  1]
    # y :   1.2008e+01 | i : [14  3  1  9  1]
    # y :   1.0953e+01 | i : [15  7  8  4  7]
    # y :   9.5068e+00 | i : [ 7 13 14 10  3]
    # 




|
|

.. autofunction:: teneva.optima_tt_max

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12] # Shape of the tensor
    Y = teneva.rand(n, r=4)  # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [17  0  4  1  9]
    # y opt appr :  -8.0435e+00
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)            # Transform the TT-tensor to full format
    
    i_opt = np.argmax(np.abs(Y_full))  # Multi-index of the maximum modulo item
    i_opt = np.unravel_index(i_opt, n)
    y_opt = Y_full[i_opt]              # The related tensor value
    
    print(f'i opt real :', i_opt)
    print(f'y opt real : {Y_full[i_opt]:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt real : (17, 0, 4, 1, 9)
    # y opt real :  -8.0435e+00
    # 




|
|


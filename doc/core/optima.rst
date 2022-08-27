Module optima: estimate min and max value of the tensor
-------------------------------------------------------


.. automodule:: teneva.core.optima


-----


.. autofunction:: teneva.optima_qtt

  **Examples**:

  .. code-block:: python

    d = 5                           # Dimension
    q = 4                           # Mode size factor
    n = [2**q]*d                    # Shape of the tensor
    Y = teneva.tensor_rand(n, r=4)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_qtt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [ 3  7 11  7 12]
    # i max appr : [ 3  7  7 11 12]
    # y min appr :  -1.6496e+02
    # y max appr :   1.9745e+02
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

    # i min real : (3, 7, 11, 7, 12)
    # i max real : (3, 7, 7, 11, 12)
    # y min real :  -1.6496e+02
    # y max real :   1.9745e+02
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    d = 5        # Dimension
    q = 4        # Mode size factor
    n = [2**q]*d # Shape of the tensor
    
    for i in range(10):
        Y = teneva.tensor_rand(n, r=4)
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

    # -> Error for min 5.7e-14 | Error for max 2.8e-14 | Time   0.0495
    # -> Error for min 0.0e+00 | Error for max 5.7e-14 | Time   0.0455
    # -> Error for min 5.7e-14 | Error for max 0.0e+00 | Time   0.0432
    # -> Error for min 0.0e+00 | Error for max 2.8e-14 | Time   0.0456
    # -> Error for min 5.7e-14 | Error for max 2.8e-14 | Time   0.0441
    # -> Error for min 0.0e+00 | Error for max 5.7e-14 | Time   0.0450
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0430
    # -> Error for min 5.7e-14 | Error for max 0.0e+00 | Time   0.0433
    # -> Error for min 2.8e-14 | Error for max 2.8e-14 | Time   0.0450
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0437
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):

  .. code-block:: python

    d = 6        # Dimension
    q = 4        # Mode size factor
    n = [2**q]*d # Shape of the tensor
    
    for func in teneva.func_demo_all(d):#, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')
    
        # Build TT-approximation by TT-CROSS:
        Y = teneva.tensor_rand(func.n, r=1)
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

    # -> Ackley               | TT-rank  10.6 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   1.1472 | 
    # -> Alpine               | TT-rank   2.7 | Error for min 4.4e-02 | Error for max 0.0e+00 | Time   0.0458 | 
    # -> Dixon                | TT-rank   6.1 | Error for min 6.5e-12 | Error for max 1.2e-10 | Time   0.2434 | 
    # -> Exponential          | TT-rank   3.7 | Error for min 1.1e-16 | Error for max 0.0e+00 | Time   0.0621 | 
    # -> Grienwank            | TT-rank   5.9 | Error for min 2.7e-13 | Error for max 1.1e-13 | Time   0.2002 | 
    # -> Michalewicz          | TT-rank   4.5 | Error for min 1.8e-15 | Error for max 4.9e-13 | Time   0.0946 | 
    # -> Qing                 | TT-rank   5.0 | Error for min 3.8e-07 | Error for max 1.2e-04 | Time   0.1602 | 
    # -> Rastrigin            | TT-rank   5.0 | Error for min 6.0e-13 | Error for max 2.0e-13 | Time   0.1325 | 
    # -> Rosenbrock           | TT-rank   5.9 | Error for min 2.4e-13 | Error for max 0.0e+00 | Time   0.1973 | 
    # -> Schaffer             | TT-rank  12.6 | Error for min 4.1e-13 | Error for max 7.5e-12 | Time   2.2929 | 
    # -> Schwefel             | TT-rank   6.9 | Error for min 2.0e-13 | Error for max 0.0e+00 | Time   0.3280 | 
    # 


.. autofunction:: teneva.optima_tt

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]        # Shape of the tensor
    Y = teneva.tensor_rand(n, r=4)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [6 4 3 0 7]
    # i max appr : [6 4 3 0 9]
    # y min appr :  -2.6654e+02
    # y max appr :   2.9885e+02
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

    # i min real : (6, 4, 3, 0, 7)
    # i max real : (6, 4, 3, 0, 9)
    # y min real :  -2.6654e+02
    # y max real :   2.9885e+02
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]
    
    for i in range(10):
        Y = teneva.tensor_rand(n, r=4)
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

    # -> Error for min 2.8e-14 | Error for max 2.8e-14 | Time   0.0187
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0116
    # -> Error for min 5.7e-14 | Error for max 2.8e-14 | Time   0.0108
    # -> Error for min 0.0e+00 | Error for max 2.8e-14 | Time   0.0117
    # -> Error for min 5.7e-14 | Error for max 0.0e+00 | Time   0.0114
    # -> Error for min 2.8e-14 | Error for max 2.8e-14 | Time   0.0112
    # -> Error for min 5.7e-14 | Error for max 0.0e+00 | Time   0.0107
    # -> Error for min 0.0e+00 | Error for max 5.7e-14 | Time   0.0109
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0115
    # -> Error for min 0.0e+00 | Error for max 2.8e-14 | Time   0.0138
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):

  .. code-block:: python

    d = 6   # Dimension
    n = 16  # Grid size
    
    for func in teneva.func_demo_all(d):#, dy=0.5):
        # Set the uniform grid:
        func.set_grid(n, kind='uni')
    
        # Build TT-approximation by TT-CROSS:
        Y = teneva.tensor_rand(func.n, r=1)
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

    # -> Ackley               | TT-rank  10.6 | Error for min 7.1e-15 | Error for max 3.6e-15 | Time   0.1495 | 
    # -> Alpine               | TT-rank   2.7 | Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0188 | 
    # -> Dixon                | TT-rank   5.7 | Error for min 6.0e-13 | Error for max 0.0e+00 | Time   0.0352 | 
    # -> Exponential          | TT-rank   3.7 | Error for min 2.2e-16 | Error for max 0.0e+00 | Time   0.0205 | 
    # -> Grienwank            | TT-rank   5.9 | Error for min 2.7e-15 | Error for max 1.1e-13 | Time   0.0459 | 
    # -> Michalewicz          | TT-rank   4.5 | Error for min 1.8e-15 | Error for max 9.9e-17 | Time   0.0290 | 
    # -> Qing                 | TT-rank   5.0 | Error for min 3.8e-07 | Error for max 1.2e-04 | Time   0.0329 | 
    # -> Rastrigin            | TT-rank   5.0 | Error for min 7.1e-15 | Error for max 1.4e-13 | Time   0.0306 | 
    # -> Rosenbrock           | TT-rank   5.9 | Error for min 2.4e-13 | Error for max 0.0e+00 | Time   0.0397 | 
    # -> Schaffer             | TT-rank  13.4 | Error for min 3.3e-16 | Error for max 6.4e-03 | Time   0.4579 | 
    # -> Schwefel             | TT-rank   6.9 | Error for min 2.0e-13 | Error for max 0.0e+00 | Time   0.0557 | 
    # 


.. autofunction:: teneva.optima_tt_beam

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]       # Shape of the tensor
    Y = teneva.tensor_rand(n, r=4) # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [ 1 12  9 11  0]
    # y opt appr :  -2.8787e+02
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

    # i opt real : (1, 12, 9, 11, 0)
    # y opt real :  -2.8787e+02
    # 

  This function may also return the "top-k" candidates for the optimum:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]       # Shape of the tensor
    Y = teneva.tensor_rand(n, r=4) # Random TT-tensor with rank 4
    
    I_opt = teneva.optima_tt_beam(Y, k=10, ret_all=True)
    
    for i_opt in I_opt:
        y_opt = abs(teneva.get(Y, i_opt))
        print(f'y : {y_opt:-12.4e} | i : {i_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # y :   2.5220e+02 | i : [3 1 1 2 3]
    # y :   2.0979e+02 | i : [3 8 8 2 3]
    # y :   2.0328e+02 | i : [ 3  8 12  3  9]
    # y :   1.9756e+02 | i : [15  7  8  2  3]
    # y :   1.9386e+02 | i : [ 3 16  9  4  3]
    # y :   1.8796e+02 | i : [ 3  8 12  2  3]
    # y :   1.8516e+02 | i : [3 8 2 4 9]
    # y :   1.8339e+02 | i : [3 8 2 4 3]
    # y :   1.8338e+02 | i : [3 8 2 2 3]
    # y :   1.7781e+02 | i : [ 3  8 14  2  3]
    # 


.. autofunction:: teneva.optima_tt_max

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]       # Shape of the tensor
    Y = teneva.tensor_rand(n, r=4) # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [ 8 10 11 10  7]
    # y opt appr :  -2.2694e+02
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

    # i opt real : (8, 10, 11, 10, 7)
    # y opt real :  -2.2694e+02
    # 



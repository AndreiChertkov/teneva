Module optima: estimate min and max value of the tensor
-------------------------------------------------------


.. automodule:: teneva.optima


-----




|
|

.. autofunction:: teneva.optima.optima_qtt

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

    # -> Error for min 0.0e+00 | Error for max 3.6e-15 | Time   0.0669
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0520
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0512
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0531
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0596
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0479
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0473
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0524
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0549
    # -> Error for min 3.6e-15 | Error for max 1.8e-15 | Time   0.0478
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):




|
|

.. autofunction:: teneva.optima.optima_tt

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

    # i min appr : [19 13  8  6  2]
    # i max appr : [11  9  3 13  2]
    # y min appr :  -1.2604e+01
    # y max appr :   1.4029e+01
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

    # i min real : (19, 13, 8, 6, 2)
    # i max real : (11, 9, 3, 13, 2)
    # y min real :  -1.2604e+01
    # y max real :   1.4029e+01
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

    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0169
    # -> Error for min 3.6e-15 | Error for max 0.0e+00 | Time   0.0181
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0177
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0141
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0119
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0120
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0119
    # -> Error for min 0.0e+00 | Error for max 3.6e-15 | Time   0.0118
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0116
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0124
    # 




|
|

.. autofunction:: teneva.optima.optima_tt_beam

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

    # i opt appr : [1 1 6 4 3]
    # y opt appr :   8.6876e+00
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

    # i opt real : (1, 1, 6, 4, 3)
    # y opt real :   8.6876e+00
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

    # y :   1.3060e+01 | i : [18 15  1  7  5]
    # y :   1.2174e+01 | i : [18 15  1  7  9]
    # y :   1.1727e+01 | i : [18 10 13  1  7]
    # y :   1.1572e+01 | i : [ 8 15  1  7  5]
    # y :   1.0987e+01 | i : [18 10 13 12  2]
    # y :   1.0732e+01 | i : [ 8 15  1  7  9]
    # y :   1.0497e+01 | i : [18 10 13  3  7]
    # y :   1.0260e+01 | i : [18 10  1  7  5]
    # y :   1.0230e+01 | i : [18 10  1  7  9]
    # y :   1.0213e+01 | i : [18 15 10 12  2]
    # 




|
|

.. autofunction:: teneva.optima.optima_tt_max

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

    # i opt appr : [16 12 10  2  6]
    # y opt appr :  -1.1537e+01
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

    # i opt real : (16, 12, 10, 2, 6)
    # y opt real :  -1.1537e+01
    # 




|
|


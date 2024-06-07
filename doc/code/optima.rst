Module optima: estimate min and max value of the tensor
-------------------------------------------------------


.. automodule:: teneva.optima


-----




|
|

.. autofunction:: teneva.optima.optima_qtt

  **Examples**:

  .. code-block:: python

    d = 5                             # Dimension
    q = 4                             # Mode size factor
    n = [2**q]*d                      # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_qtt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [ 4  0 15  9 15]
    # i max appr : [12  8 15  9 15]
    # y min appr :  -1.2605e+01
    # y max appr :   1.2871e+01
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

    # i min real : (4, 0, 15, 9, 15)
    # i max real : (12, 8, 15, 9, 15)
    # y min real :  -1.2605e+01
    # y max real :   1.2871e+01
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    d = 5        # Dimension
    q = 4        # Mode size factor
    n = [2**q]*d # Shape of the tensor
    
    for i in range(10):
        Y = teneva.rand(n, r=4, seed=i)
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

    # -> Error for min 1.8e-15 | Error for max 3.6e-15 | Time   0.0569
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0601
    # -> Error for min 4.5e-01 | Error for max 1.8e-15 | Time   0.0581
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0534
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0571
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0543
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0572
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0599
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.1248
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0839
    # 

  We can also check it for real data (we build TT-tensor using TT-cross method here):




|
|

.. autofunction:: teneva.optima.optima_tt

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]          # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)  # Random TT-tensor with rank 4
    
    i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
    
    print(f'i min appr :', i_min)
    print(f'i max appr :', i_max)
    print(f'y min appr : {y_min:-12.4e}')
    print(f'y max appr : {y_max:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i min appr : [11 16  3  5  6]
    # i max appr : [11 16  3  5  5]
    # y min appr :  -1.1443e+01
    # y max appr :   1.0128e+01
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

    # i min real : (11, 16, 3, 5, 6)
    # i max real : (11, 16, 3, 5, 5)
    # y min real :  -1.1443e+01
    # y max real :   1.0128e+01
    # 

  We can check results for many random TT-tensors:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]
    
    for i in range(10):
        Y = teneva.rand(n, r=4, seed=i)
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

    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0181
    # -> Error for min 3.6e-15 | Error for max 3.6e-15 | Time   0.0163
    # -> Error for min 1.8e-15 | Error for max 1.8e-15 | Time   0.0201
    # -> Error for min 1.8e-15 | Error for max 3.6e-15 | Time   0.0165
    # -> Error for min 3.6e-15 | Error for max 0.0e+00 | Time   0.0164
    # -> Error for min 0.0e+00 | Error for max 1.8e-15 | Time   0.0143
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0127
    # -> Error for min 3.6e-15 | Error for max 0.0e+00 | Time   0.0132
    # -> Error for min 1.8e-15 | Error for max 0.0e+00 | Time   0.0133
    # -> Error for min 0.0e+00 | Error for max 0.0e+00 | Time   0.0128
    # 




|
|

.. autofunction:: teneva.optima.optima_tt_beam

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]          # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)  # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [11 16  3  5  6]
    # y opt appr :  -1.1443e+01
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

    # i opt real : (11, 16, 3, 5, 6)
    # y opt real :  -1.1443e+01
    # 

  This function may also return the "top-k" candidates for the optimum:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]          # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)  # Random TT-tensor with rank 4
    
    I_opt = teneva.optima_tt_beam(Y, k=10, ret_all=True)
    
    for i_opt in I_opt:
        y_opt = abs(teneva.get(Y, i_opt))
        print(f'y : {y_opt:-12.4e} | i : {i_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # y :   1.1443e+01 | i : [11 16  3  5  6]
    # y :   1.0383e+01 | i : [11 16  3  1  0]
    # y :   1.0128e+01 | i : [11 16  3  5  5]
    # y :   1.0047e+01 | i : [ 8  5 11  4  5]
    # y :   9.9418e+00 | i : [ 8 16  3  5  6]
    # y :   9.5700e+00 | i : [11 17 11  4  5]
    # y :   9.4352e+00 | i : [11 16  3  5  9]
    # y :   9.4341e+00 | i : [11 16  3  1 11]
    # y :   8.9518e+00 | i : [ 8 16  3  1 11]
    # y :   8.6305e+00 | i : [5 2 7 1 5]
    # 




|
|

.. autofunction:: teneva.optima.optima_tt_max

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]          # Shape of the tensor
    Y = teneva.rand(n, r=4, seed=42)  # Random TT-tensor with rank 4
    
    i_opt = teneva.optima_tt_beam(Y)
    y_opt = teneva.get(Y, i_opt)
    
    print(f'i opt appr :', i_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # i opt appr : [11 16  3  5  6]
    # y opt appr :  -1.1443e+01
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

    # i opt real : (11, 16, 3, 5, 6)
    # y opt real :  -1.1443e+01
    # 




|
|


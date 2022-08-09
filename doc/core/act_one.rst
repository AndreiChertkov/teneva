Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva.core.act_one


-----


.. autofunction:: teneva.copy

  **Examples**:

  .. code-block:: python

    Y = teneva.tensor_rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    
    Z = teneva.copy(Y)         # The copy of Y
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -1.2208436499710222
    # -1.2208436499710222
    # 

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -1.2208436499710222
    # 42.0
    # 

  It also supports numbers for convenience:

  .. code-block:: python

    teneva.copy(42.)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.get

  **Examples**:

  .. code-block:: python

    n = [10] * 5              # Shape of the tensor      
    Y0 = np.random.randn(*n)  # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)       # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)           # Print the TT-tensor
    k = [1, 2, 3, 4, 5]       # Select some tensor element
    y1 = teneva.get(Y1, k)    # Compute the element of the TT-tensor
    y0 = Y0[tuple(k)]         # Compute the same element of the original tensor
    abs(np.max(y1-y0))        # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    #    10  10  10  10  10 
    #   / \ / \ / \ / \ / \ 
    #  1   10 100 100  10  1  
    # 
    # 

  This function is also support batch mode:

  .. code-block:: python

    K = [
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    y1 = teneva.get(Y1, K)
    y0 = [Y0[tuple(k)] for k in K]
    abs(np.max(y1-y0))

    # >>> ----------------------------------------
    # >>> Output:

    # 9.992007221626409e-16
    # 


.. autofunction:: teneva.get_many

  **Examples**:

  .. code-block:: python

    n = [10] * 5                    # Shape of the tensor      
    Y0 = np.random.randn(*n)        # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)             # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)                 # Print the TT-tensor
    K = [                           # Select some tensor elements
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]     
    y1 = teneva.get_many(Y1, K)     # Compute the element of the TT-tensor
    y0 = [Y0[tuple(k)] for k in K]  # Compute the same element of the original tensor
    abs(np.max(y1-y0))              # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    #    10  10  10  10  10 
    #   / \ / \ / \ / \ / \ 
    #  1   10 100 100  10  1  
    # 
    # 


.. autofunction:: teneva.getter

  **Examples**:

  .. code-block:: python

    n = [10] * 5              # Shape of the tensor      
    Y0 = np.random.randn(*n)  # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)       # Compute TT-tensor from Y0 by TT-SVD
    get = teneva.getter(Y1)   # Build (compile) function to compute the element of the TT-tensor
    k = (1, 2, 3, 4, 5)       # Select some tensor element
    y1 = get(k)               # Compute the element of the TT-tensor
    y0 = Y0[k]                # Compute the same element of the original tensor
    abs(np.max(y1-y0))        # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    # 8.881784197001252e-16
    # 

  We can compare the calculation time using the base function and the function accelerated with numba:

  .. code-block:: python

    n = [100] * 40
    Y = teneva.tensor_rand(n, r=4)
    
    get1 = lambda i: teneva.get(Y, i)
    get2 = teneva.getter(Y)
    
    I = teneva.sample_lhs(n, m=1000)
    
    t1 = tpc()
    for i in I:
        y1 = get1(i)
    t1 = tpc() - t1
    
    t2 = tpc()
    for i in I:
        y2 = get2(i)
    t2 = tpc() - t2
    
    print(f'Time for "simple" : {t1:-8.4f} sec')
    print(f'Time for "numba"  : {t2:-8.4f} sec')

    # >>> ----------------------------------------
    # >>> Output:

    # Time for "simple" :   0.1312 sec
    # Time for "numba"  :   0.0161 sec
    # 


.. autofunction:: teneva.mean

  **Examples**:

  .. code-block:: python

    Y = teneva.tensor_rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    m = teneva.mean(Y)                # The mean value

  .. code-block:: python

    Y_full = teneva.full(Y)           # Compute tensor in the full format to check the result
    m_full = np.mean(Y_full)          # The mean value for the numpy array
    e = abs(m - m_full)               # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 4.88e-19
    # 

  The probability of tensor inputs my be also set:

  .. code-block:: python

    n = [5]*10                        # Shape of the tensor
    Y = teneva.tensor_rand(n, 2)      # 10-dim random TT-tensor with TT-rank 2
    P = [np.zeros(k) for k in n]      # The "probability"
    teneva.mean(Y, P)                 # The mean value

    # >>> ----------------------------------------
    # >>> Output:

    # 0.0
    # 


.. autofunction:: teneva.norm

  **Examples**:

  .. code-block:: python

    Y = teneva.tensor_rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2

  .. code-block:: python

    v = teneva.norm(Y)                # Compute the Frobenius norm
    print(v)                          # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # 27798.44414412251
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)           # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                     # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)      # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')   # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 27798.444144122514
    # Error     : 1.31e-16
    # 


.. autofunction:: teneva.qtt_to_tt

  **Examples**:

  .. code-block:: python

    d = 4                         # Dimension of the tensor
    q = 5                         # Quantization value (n=2^q)
    r = [                         # TT-ranks of the QTT-tensor
        1,
        3, 4, 5, 6, 7,
        5, 4, 3, 6, 7,
        5, 4, 3, 6, 7,
        5, 4, 3, 6, 1,
    ]      
    
    # Random QTT-tensor:
    Y = teneva.tensor_rand([2]*(d*q), r)
    
    # Related TT-tensor:
    Z = teneva.qtt_to_tt(Y, q)
    
    teneva.show(Y)                # Show QTT-tensor
    teneva.show(Z)                # Show TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #   2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  4  5  6  7  5  4  3  6  7  5  4  3  6  7  5  4  3  6  1 
    # 
    #  32 32 32 32 
    #  / \/ \/ \/ \
    #  1  7  7  7  1 
    # 
    # 

  We can check that values of the QTT-tensor and TT-tensor are the same:

  .. code-block:: python

    # Multi-index for QTT-tensor:
    i = [
        0, 1, 1, 0, 0,
        0, 0, 1, 1, 0,
        0, 1, 1, 1, 1,
        0, 1, 1, 1, 0,
    ]
    
    # Related multi-index for TT-tensor:
    j = teneva.ind_qtt_to_tt(i, q)
    
    print(f'QTT value : {teneva.get(Y, i):-14.6f}')
    print(f' TT value : {teneva.get(Z, j):-14.6f}')

    # >>> ----------------------------------------
    # >>> Output:

    # QTT value :  182994.666782
    #  TT value :  182994.666782
    # 

  We can also transform the TT-tensor back into QTT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.tt_to_qtt(Z)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.1361884846794984e-08
    # 


.. autofunction:: teneva.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.tensor_rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                                 # Sum of the TT-tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # -497.3325879631785
    # 

  .. code-block:: python

    Z = teneva.full(Y) # Compute tensors in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # -497.33258796317836
    # 


.. autofunction:: teneva.tt_to_qtt

  **Examples**:

  .. code-block:: python

    d = 4                         # Dimension of the tensor
    n = [32] * d                  # Shape of the tensor
    r = [1, 4, 3, 6, 1]           # TT-ranks of the tensor
    Y = teneva.tensor_rand(n, r)  # Random TT-tensor
    Z = teneva.tt_to_qtt(Y)       # Related QTT-tensor
    
    teneva.show(Y)                # Show TT-tensor
    teneva.show(Z)                # Show QTT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  32 32 32 32 
    #  / \/ \/ \/ \
    #  1  4  3  6  1 
    # 
    #   2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  4  8  8  4  8 16 12  6  3  6 12 24 12  6 12  8  4  2  1 
    # 
    # 

  We can check that values of the TT-tensor and QTT-tensor are the same:

  .. code-block:: python

    # Multi-index for TT-tensor:
    i = [5, 10, 20, 30]
    
    # Related multi-index for QTT-tensor:
    j = teneva.ind_tt_to_qtt(i, n[0])
    
    print(f' TT value : {teneva.get(Y, i):-14.6f}')
    print(f'QTT value : {teneva.get(Z, j):-14.6f}')

    # >>> ----------------------------------------
    # >>> Output:

    #  TT value :      -1.272940
    # QTT value :      -1.272940
    # 

  We can also transform the QTT-tensor back into TT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.qtt_to_tt(Z, q)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 0.0
    # 

  We can also perform the transformation with limited precision: 

  .. code-block:: python

    Z = teneva.tt_to_qtt(Y, r=20)
    teneva.show(Z)
    
    U = teneva.qtt_to_tt(Z, q)
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    #   2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  4  8  8  4  8 16 12  6  3  6 12 20 12  6 12  8  4  2  1 
    # 
    # 



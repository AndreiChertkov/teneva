Module tensor: basic operations with TT-tensors
-----------------------------------------------


.. automodule:: teneva.core.tensor


-----


.. autofunction:: teneva.accuracy

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2)                 # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.add(Y1, teneva.mul(1.E-4, Y1))  # The TT-tensor Y1 + eps * Y1 (eps = 1.E-4)
    eps = teneva.accuracy(Y1, Y2)               # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 1.00e-04
    # 

  Note that this function works correctly even for very large dimension values due to the use of balancing in the scalar product:

  .. code-block:: python

    for d in [10, 50, 100, 250, 1000, 10000]:
        Y1 = teneva.rand([10]*d, r=2)
        Y2 = teneva.add(Y1, Y1)
    
        eps = teneva.accuracy(Y1, Y2)
    
        print(f'd = {d:-5d} | eps = {eps:-8.1e} | expected value 0.5')

    # >>> ----------------------------------------
    # >>> Output:

    # d =    10 | eps =  5.0e-01 | expected value 0.5
    # d =    50 | eps =  5.0e-01 | expected value 0.5
    # d =   100 | eps =  5.0e-01 | expected value 0.5
    # d =   250 | eps =  5.0e-01 | expected value 0.5
    # d =  1000 | eps =  5.0e-01 | expected value 0.5
    # d = 10000 | eps =  5.0e-01 | expected value 0.5
    # 


.. autofunction:: teneva.accuracy_on_data

  **Examples**:

  .. code-block:: python

    m = 100                                     # Size of the dataset
    n = [5] * 10                                # Shape of the tensor
    Y = teneva.rand(n, 2)                       # Random TT-tensor with TT-rank 2
    I_data = teneva.sample_lhs(n, m)            # Let build toy dataset
    Y_data = [teneva.get(Y, i) for i in I_data]
    Y_data = np.array(Y_data)
    Y_data = Y_data + 1.E-3*np.random.randn(m)  # Add add some noise
    
    # Compute the accuracy:
    eps = teneva.accuracy_on_data(Y, I_data, Y_data)
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 4.88e-05
    # 


.. autofunction:: teneva.add

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 3 = 5)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  5  5  5  5  5  5  5  5  5  1 
    # 
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1)                       # Compute tensors in the full format
    Y2_full = teneva.full(Y2)                       # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    e = np.linalg.norm(Y_full - Z_full)             # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)
    
    print(f'Error     : {e:-8.2e}')                 # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 8.54e-17
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # 

  .. code-block:: python

    Y1_full = 42.                       # Compute tensors in the full format
    Y2_full = teneva.full(Y2)           # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    e = np.linalg.norm(Y_full - Z_full) # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)
    
    print(f'Error     : {e:-8.2e}')     # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 4.92e-16
    # 

  If both arguments are numbers, then function returns the sum of numbers:

  .. code-block:: python

    Y1 = 40.                    # Just a number
    Y2 = 2                      # Just a number
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    print(Y)                    # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.add_many

  **Examples**:

  .. code-block:: python

    Y_all = [teneva.rand([5]*10, 2) for _ in range(10)]     # 10 random TT-tensors with TT-rank 2
    Y = teneva.add_many(Y_all, e=1.E-4, r=50, trunc_freq=2)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  5 20 20 20 20 20 20 20  5  1 
    # 
    # 

  This function also supports float arguments:

  .. code-block:: python

    Y_all = [
        42.,
        teneva.rand([5]*10, 2),
        33.,
        teneva.rand([5]*10, 4)
    ]
    Y = teneva.add_many(Y_all, e=1.E-4, r=50, trunc_freq=2)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  5  7  7  7  7  7  7  7  5  1 
    # 
    # 

  If all arguments are numbers, then function returns the sum of numbers:

  .. code-block:: python

    Y_all = [10., 20., 2., 10.]
    Y = teneva.add_many(Y_all, e=1.E-4, r=50, trunc_freq=2)
    print(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.copy

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Z = teneva.copy(Y)         # The copy of Y
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.14469232266028922
    # -0.14469232266028922
    # 

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.14469232266028922
    # 42.0
    # 

  It also supports numbers for convenience:

  .. code-block:: python

    teneva.copy(42.)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.erank

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    teneva.erank(Y)            # The effective TT-rank

    # >>> ----------------------------------------
    # >>> Output:

    # 2.0
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
    
    y1 = teneva.get(Y1, k)
    y0 = [Y0[tuple(k)] for k in K]
    abs(np.max(y1-y0))

    # >>> ----------------------------------------
    # >>> Output:

    # 0.7217290025096638
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

    # 5.551115123125783e-17
    # 


.. autofunction:: teneva.mean

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2)   # 10-dim random TT-tensor with TT-rank 2
    m = teneva.mean(Y)           # The mean value

  .. code-block:: python

    Y_full = teneva.full(Y)      # Compute tensor in the full format to check the result
    m_full = np.mean(Y_full)     # The mean value for the numpy array
    e = abs(m - m_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 2.17e-19
    # 

  The probability of tensor inputs my be also set:

  .. code-block:: python

    n = [5]*10                   # Shape of the tensor
    Y = teneva.rand(n, 2)        # 10-dim random TT-tensor with TT-rank 2
    P = [np.zeros(k) for k in n] # The "probability"
    teneva.mean(Y, P)            # The mean value

    # >>> ----------------------------------------
    # >>> Output:

    # 0.0
    # 


.. autofunction:: teneva.mul

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.mul(Y1, Y2)      # Compute the product of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 x 3 = 6)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  6  6  6  6  6  6  6  6  6  1 
    # 
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1)                       # Compute tensors in the full format
    Y2_full = teneva.full(Y2)                       # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full * Y2_full
    
    e = np.linalg.norm(Y_full - Z_full)             # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)                     #
    
    print(f'Error     : {e:-8.2e}')                 # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 4.82e-16
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    Y = teneva.mul(Y1, Y2)      # Compute the product of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 x 1 = 2)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  1 
    # 
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y = teneva.mul(Y1, Y2)      # Compute the product of Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 x 1 = 2)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  1 
    # 
    # 

  .. code-block:: python

    Y1 = 21.                    # Just a number
    Y2 = 2                      # Just a number
    Y = teneva.mul(Y1, Y2)      # Compute the product of Y1 and Y2
    print(Y)                    # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.mul_scalar

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2)           # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3)           # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    v = teneva.mul_scalar(Y1, Y2)         # Compute the product of Y1 and Y2
    print(v)                              # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # -494064.2967292642
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1)             # Compute tensors in the full format
    Y2_full = teneva.full(Y2)             # to check the result
    
    v_full = np.sum(Y1_full * Y2_full)
    print(v_full)                         # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')       # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # -494064.29672926356
    # Error     : 1.30e-15
    # 

  We can also set a flag "use_stab", in which case a value that is 2^p times smaller than the real value will be returned:

  .. code-block:: python

    v, p = teneva.mul_scalar(Y1, Y2, use_stab=True)
    print(v)
    print(p)
    print(v*2**p)

    # >>> ----------------------------------------
    # >>> Output:

    # -1.8847057217760628
    # 18
    # -494064.2967292642
    # 


.. autofunction:: teneva.norm

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2)            # 10-dim random TT-tensor with TT-rank 2

  .. code-block:: python

    v = teneva.norm(Y)                    # Compute the Frobenius norm
    print(v)                              # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # 54764.181919363546
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)               # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                         # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')       # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 54764.18191936357
    # Error     : 3.99e-16
    # 


.. autofunction:: teneva.rand

  **Examples**:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]    # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]      # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)       # Build random TT-tensor
    teneva.show(Y)              # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  If all inner TT-ranks are equal, we may pass it as a number:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]    # Shape of the tensor
    r = 5                       # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)       # Build random TT-tensor
    teneva.show(Y)              # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  5  5  5  5  1 
    # 
    # 


.. autofunction:: teneva.ranks

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.ranks(Y)                        # TT-ranks of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # array([1, 2, 2, 2, 2, 1])
    # 


.. autofunction:: teneva.shape

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.shape(Y)                        # Shape of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # array([10, 12,  8,  8, 30])
    # 


.. autofunction:: teneva.show

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.show(Y)                         # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  10 12  8  8 30 
    #  / \/ \/ \/ \/ \
    #  1  2  2  2  2  1 
    # 
    # 


.. autofunction:: teneva.size

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.size(Y)                         # Size of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # 192
    # 


.. autofunction:: teneva.sub

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.sub(Y1, Y2)      # Compute the difference between Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 3 = 5)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  5  5  5  5  5  5  5  5  5  1 
    # 
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1)                       # Compute tensors in the full format
    Y2_full = teneva.full(Y2)                       # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full - Y2_full
    
    e = np.linalg.norm(Y_full - Z_full)             # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)                     
    
    print(f'Error     : {e:-8.2e}')                 # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 8.81e-17
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    Y = teneva.sub(Y1, Y2)      # Compute the difference between Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y = teneva.sub(Y1, Y2)      # Compute the difference between Y1 and Y2
    teneva.show(Y)              # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # 

  .. code-block:: python

    Y1 = 44.                    # Just a number
    Y2 = 2                      # Just a number
    Y = teneva.sub(Y1, Y2)      # Compute the difference between Y1 and Y2
    print(Y)                    # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 


.. autofunction:: teneva.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                          # Sum of the TT-tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # -814.3947987359592
    # 

  .. code-block:: python

    Z = teneva.full(Y)                     # Compute tensors in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # -814.3947987359597
    # 



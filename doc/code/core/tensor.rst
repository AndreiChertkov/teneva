tensor: basic operations with TT-tensors
----------------------------------------


.. automodule:: teneva.core.tensor

---


.. autofunction:: teneva.core.tensor.accuracy

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

---


.. autofunction:: teneva.core.tensor.add

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

    # Error     : 1.08e-16
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

    Y1_full = 42.                                   # Compute tensors in the full format
    Y2_full = teneva.full(Y2)                       # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    e = np.linalg.norm(Y_full - Z_full)             # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)
    
    print(f'Error     : {e:-8.2e}')                 # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 9.12e-17
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

---


.. autofunction:: teneva.core.tensor.add_many

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

---


.. autofunction:: teneva.core.tensor.const

  **Examples**:

  .. code-block:: python

    n = [10] * 5                     # Shape of the tensor  
    Y = teneva.const(n, v=42.)       # Tensor of all 42
    teneva.show(Y)                   # Print the resulting TT-tensor
    Y_full = teneva.full(Y)
    print(f'Min value : {np.min(Y_full)}')
    print(f'Max value : {np.max(Y_full)}')

    # >>> ----------------------------------------
    # >>> Output:

    #  10 10 10 10 10 
    #  / \/ \/ \/ \/ \
    #  1  1  1  1  1  1 
    # 
    # Min value : 42.0
    # Max value : 42.0
    # 

---


.. autofunction:: teneva.core.tensor.copy

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Z = teneva.copy(Y)         # The copy of Y
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # 0.5136144642310962
    # 0.5136144642310962
    # 

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # 0.5136144642310962
    # 42.0
    # 

  It also supports numbers for convenience:

  .. code-block:: python

    teneva.copy(42.)

---


.. autofunction:: teneva.core.tensor.erank

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    teneva.erank(Y)            # The effective TT-rank

---


.. autofunction:: teneva.core.tensor.full

  **Examples**:

  .. code-block:: python

    n = [10] * 5              # Shape of the tensor      
    Y0 = np.random.randn(*n)  # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)       # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)           # Print the TT-tensor
    Y2 = teneva.full(Y1)      # Compute full tensor from the TT-tensor
    abs(np.max(Y2-Y0))        # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    #    10  10  10  10  10 
    #   / \ / \ / \ / \ / \ 
    #  1   10 100 100  10  1  
    # 
    # 

---


.. autofunction:: teneva.core.tensor.get

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

---


.. autofunction:: teneva.core.tensor.getter

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

---


.. autofunction:: teneva.core.tensor.mean

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2)   # 10-dim random TT-tensor with TT-rank 2
    teneva.mean(Y)               # The mean value

  The probability of tensor inputs my be also set:

  .. code-block:: python

    n = [5]*10                   # Shape of the tensor
    Y = teneva.rand(n, 2)        # 10-dim random TT-tensor with TT-rank 2
    P = [np.zeros(k) for k in n] # The "probability"
    teneva.mean(Y, P)            # The mean value

---


.. autofunction:: teneva.core.tensor.mul

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

    # Error     : 4.41e-16
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

---


.. autofunction:: teneva.core.tensor.mul_scalar

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2)           # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3)           # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    v = teneva.mul_scalar(Y1, Y2)         # Compute the product of Y1 and Y2
    print(v)                              # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # -376850.76200628746
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

    # -376850.76200628735
    # Error     : 3.09e-16
    # 

---


.. autofunction:: teneva.core.tensor.norm

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2)            # 10-dim random TT-tensor with TT-rank 2

  .. code-block:: python

    v = teneva.norm(Y)                    # Compute the Frobenius norm
    print(v)                              # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # 46625.68010134915
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)               # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                         # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')       # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 46625.680101349135
    # Error     : 3.12e-16
    # 

---


.. autofunction:: teneva.core.tensor.orthogonalize

  **Examples**:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.orthogonalize(Y, d-1)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  .. code-block:: python

    for G in Z:
        print(sum([G[:, i, :].T @ G[:, i, :] for i in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[1.00000000e+00 3.20923843e-17]
    #  [3.20923843e-17 1.00000000e+00]]
    # [[ 1.00000000e+00 -1.73472348e-17 -5.55111512e-17]
    #  [-1.73472348e-17  1.00000000e+00 -3.46944695e-18]
    #  [-5.55111512e-17 -3.46944695e-18  1.00000000e+00]]
    # [[ 1.00000000e+00 -8.67361738e-17  5.55111512e-17  2.77555756e-17]
    #  [-8.67361738e-17  1.00000000e+00  6.93889390e-18  5.55111512e-17]
    #  [ 5.55111512e-17  6.93889390e-18  1.00000000e+00 -6.93889390e-18]
    #  [ 2.77555756e-17  5.55111512e-17 -6.93889390e-18  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.30104261e-18 -3.12250226e-17 -6.93889390e-18
    #    4.85722573e-17]
    #  [-1.30104261e-18  1.00000000e+00 -1.04083409e-17  1.21430643e-17
    #    1.73472348e-17]
    #  [-3.12250226e-17 -1.04083409e-17  1.00000000e+00 -6.20163643e-17
    #    0.00000000e+00]
    #  [-6.93889390e-18  1.21430643e-17 -6.20163643e-17  1.00000000e+00
    #    1.73472348e-17]
    #  [ 4.85722573e-17  1.73472348e-17  0.00000000e+00  1.73472348e-17
    #    1.00000000e+00]]
    # [[52656524.31621235]]
    # 

---


.. autofunction:: teneva.core.tensor.rand

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

---


.. autofunction:: teneva.core.tensor.ranks

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.ranks(Y)                        # TT-ranks of the TT-tensor

---


.. autofunction:: teneva.core.tensor.shape

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.shape(Y)                        # Shape of the TT-tensor

---


.. autofunction:: teneva.core.tensor.show

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

---


.. autofunction:: teneva.core.tensor.size

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.size(Y)                         # Size of the TT-tensor

---


.. autofunction:: teneva.core.tensor.sub

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

    # Error     : 1.16e-16
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

---


.. autofunction:: teneva.core.tensor.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                          # Sum of the TT-tensor elements

  .. code-block:: python

    Z = teneva.full(Y)                     # Compute tensors in the full format to check the result
    np.sum(Z)

---


.. autofunction:: teneva.core.tensor.truncate

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 3)           # 10-dim random TT-tensor with TT-rank 3
    Y = teneva.add(Y, teneva.add(Y, Y))  # Compute Y + Y + Y (the real TT-rank is still 3)
    teneva.show(Y)                       # Print the resulting TT-tensor (note that it has TT-rank 3 + 3 + 3 = 9)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  9  9  9  9  9  9  9  9  9  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.truncate(Y, e=1.E-2)      # Truncate (round) the TT-tensor
    teneva.show(Z)                       # Print the resulting TT-tensor (note that it has TT-rank 3)
    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # Accuracy     : 0.00e+00
    # 

  .. code-block:: python

    Z = teneva.truncate(Y, e=1.E-6, r=2) # Truncate (round) the TT-tensor
    teneva.show(Z)                       # Print the resulting TT-tensor (note that it has TT-rank 2)
    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  1 
    # 
    # Accuracy     : 1.27e+00
    # 

---

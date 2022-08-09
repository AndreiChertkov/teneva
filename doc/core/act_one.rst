Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva.core.act_one


-----


.. autofunction:: teneva.copy

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
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
    
    y1 = teneva.get(Y1, k)
    y0 = [Y0[tuple(k)] for k in K]
    abs(np.max(y1-y0))

    # >>> ----------------------------------------
    # >>> Output:

    # 0.027701036030878035
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

    # Error     : 1.95e-18
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


.. autofunction:: teneva.norm

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2)            # 10-dim random TT-tensor with TT-rank 2

  .. code-block:: python

    v = teneva.norm(Y)                    # Compute the Frobenius norm
    print(v)                              # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # 19008.059560699174
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)               # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                         # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')       # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 19008.05956069918
    # Error     : 3.83e-16
    # 


.. autofunction:: teneva.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                          # Sum of the TT-tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # 1332.4536946055384
    # 

  .. code-block:: python

    Z = teneva.full(Y)                     # Compute tensors in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # 1332.4536946055393
    # 



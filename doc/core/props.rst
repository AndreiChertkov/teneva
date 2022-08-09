Module props: various properties (mean, norm, etc.) of TT-tensors
-----------------------------------------------------------------


.. automodule:: teneva.core.props


-----


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

    # Accuracy     : 1.35e-04
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

    # Error     : 1.48e-18
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

    # 73636.62749118447
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)               # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                         # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)          # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')       # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 73636.62749118448
    # Error     : 1.98e-16
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


.. autofunction:: teneva.size

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.size(Y)                         # Size of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # 192
    # 



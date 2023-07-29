Module act_one: single TT-tensor operations
-------------------------------------------


.. automodule:: teneva.act_one


-----




|
|

.. autofunction:: teneva.act_one.copy

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([5]*10, 2)
    
    Z = teneva.copy(Y) # The copy of Y         
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.6210572818314286
    # -0.6210572818314286
    # 

  Note that changes to the copy will not affect the original tensor:

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.6210572818314286
    # 42.0
    # 

  Note that this function also supports numbers and numpy arrays for convenience:

  .. code-block:: python

    a = teneva.copy(42.)
    b = teneva.copy(np.array([1, 2, 3]))




|
|

.. autofunction:: teneva.act_one.get

  **Examples**:

  .. code-block:: python

    n = [10] * 5              # Shape of the tensor      
    Y0 = np.random.randn(*n)  # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)       # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)           # Print the TT-tensor
    
    k = [1, 2, 3, 4, 5]       # Select some tensor element
    y1 = teneva.get(Y1, k)    # Compute the element of the TT-tensor
    y0 = Y0[tuple(k)]         # Compute the same element of the original tensor
    abs(y1-y0)                # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10|  |10|   |10|   |10|  |10|
    # <rank>  =   63.0 :    \10/  \100/  \100/  \10/
    # 

  This function is also support batch mode (in the case of batch, it calls the function "get_many"):

  .. code-block:: python

    # Select some tensor elements:
    K = [
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    # Compute the element of the TT-tensor:
    y1 = teneva.get(Y1, K)
    
    # Compute the same element of the original tensor:
    y0 = [Y0[tuple(k)] for k in K]
    
    # Compare original tensor and reconstructed tensor:
    e = np.max(np.abs(y1-y0))
    print(f'Error   : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error   : 1.3e-14
    # 




|
|

.. autofunction:: teneva.act_one.get_and_grad

  **Examples**:

  .. code-block:: python

    l = 1.E-4                # Learning rate
    n = [4, 5, 6, 7]         # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index for gradient
    y, dY = teneva.get_and_grad(Y, i)
    
    Z = teneva.copy(Y)
    for k in range(len(n)):
        Z[k] -= l * dY[k]
    
    z = teneva.get(Z, i)
    e = teneva.accuracy(Y, Z)
    
    print(f'Old value at multi-index : {y:-12.5e}')
    print(f'New value at multi-index : {z:-12.5e}')
    print(f'Difference for tensors   : {e:-12.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Old value at multi-index : -5.41306e-01
    # New value at multi-index : -5.41558e-01
    # Difference for tensors   :      7.8e-05
    # 




|
|

.. autofunction:: teneva.act_one.get_many

  **Examples**:

  .. code-block:: python

    n = [10] * 5             # Shape of the tensor      
    Y0 = np.random.randn(*n) # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)      # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)          # Print the TT-tensor
    
    # Select some tensor elements:
    K = [
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    # Compute the element of the TT-tensor:
    y1 = teneva.get_many(Y1, K)
    
    # Compute the same element of the original tensor:
    y0 = [Y0[tuple(k)] for k in K]
    
    # Compare original tensor and reconstructed tensor:
    e = np.max(np.abs(y1-y0))
    print(f'Error   : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10|  |10|   |10|   |10|  |10|
    # <rank>  =   63.0 :    \10/  \100/  \100/  \10/
    # Error   : 1.5e-14
    # 




|
|

.. autofunction:: teneva.act_one.getter

  **Examples**:

  .. code-block:: python

    # Note that numba package is required for this function
    
    n = [10] * 5              # Shape of the tensor      
    Y0 = np.random.randn(*n)  # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)       # Compute TT-tensor from Y0 by TT-SVD
    get = teneva.getter(Y1)   # Build (compile) function to compute the element of the TT-tensor
    k = (1, 2, 3, 4, 5)       # Select some tensor element
    y1 = get(k)               # Compute the element of the TT-tensor
    y0 = Y0[k]                # Compute the same element of the original tensor
    np.max(np.max(y1-y0))     # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    # -5.218048215738236e-15
    # 




|
|

.. autofunction:: teneva.act_one.interface

  **Examples**:

  .. code-block:: python

    n = [4, 5, 6, 7]         # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index
    phi_r = teneva.interface(Y, idx=i, ltr=False)
    phi_l = teneva.interface(Y, idx=i, ltr=True)
    
    print('Right:')
    for phi in phi_r:
        print(phi)
        
    print('Left:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # Right:
    # [-1.]
    # [-0.88773099 -0.02639246 -0.45960541]
    # [-0.27406482  0.83976061  0.46871163]
    # [-0.30430509 -0.63579329 -0.70934146]
    # [1.]
    # Left:
    # [1.]
    # [ 0.77612132  0.56519037 -0.27963466]
    # [ 0.12306665 -0.63983521 -0.75859442]
    # [0.61434194 0.53139406 0.58327038]
    # [-1.]
    # 




|
|

.. autofunction:: teneva.act_one.mean

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    m = teneva.mean(Y)         # The mean value

  .. code-block:: python

    Y_full = teneva.full(Y)    # Compute tensor in the full format to check the result
    m_full = np.mean(Y_full)   # The mean value for the numpy array
    e = abs(m - m_full)        # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 3.31e-21
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




|
|

.. autofunction:: teneva.act_one.norm

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2

  .. code-block:: python

    v = teneva.norm(Y)                # Compute the Frobenius norm
    print(v)                          # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # 246.44115016110976
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)           # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                     # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)      # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')   # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 246.44115016110973
    # Error     : 1.15e-16
    # 




|
|

.. autofunction:: teneva.act_one.qtt_to_tt

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
    Y = teneva.rand([2]*(d*q), r)
    
    # Related TT-tensor:
    Z = teneva.qtt_to_tt(Y, q)
    
    teneva.show(Y)                # Show QTT-tensor
    print()
    teneva.show(Z)                # Show TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    20D : |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2|
    # <rank>  =    5.0 :   \3/ \4/ \5/ \6/ \7/ \5/ \4/ \3/ \6/ \7/ \5/ \4/ \3/ \6/ \7/ \5/ \4/ \3/ \6/
    # 
    # TT-tensor     4D : |32| |32| |32| |32|
    # <rank>  =    7.0 :    \7/  \7/  \7/
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

    # QTT value :      13.583833
    #  TT value :      13.583833
    # 

  We can also transform the TT-tensor back into QTT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.tt_to_qtt(Z)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 3.0420906858911156e-08
    # 




|
|

.. autofunction:: teneva.act_one.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                          # Sum of the TT-tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # -5.98349790825715
    # 

  .. code-block:: python

    Z = teneva.full(Y) # Compute tensors in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # -5.983497908257153
    # 




|
|

.. autofunction:: teneva.act_one.tt_to_qtt

  **Examples**:

  .. code-block:: python

    d = 4                         # Dimension of the tensor
    n = [32] * d                  # Shape of the tensor
    r = [1, 4, 3, 6, 1]           # TT-ranks of the tensor
    Y = teneva.rand(n, r)         # Random TT-tensor
    Z = teneva.tt_to_qtt(Y)       # Related QTT-tensor
    
    teneva.show(Y)                # Show TT-tensor
    print()
    teneva.show(Z)                # Show QTT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     4D : |32| |32| |32| |32|
    # <rank>  =    4.0 :    \4/  \3/  \6/
    # 
    # TT-tensor    20D : |2| |2| |2| |2| |2| |2| |2|  |2|  |2| |2| |2| |2|  |2|  |2|  |2| |2|  |2| |2| |2| |2|
    # <rank>  =    9.2 :   \2/ \4/ \8/ \8/ \4/ \8/ \16/ \12/ \6/ \3/ \6/ \12/ \24/ \12/ \6/ \12/ \8/ \4/ \2/
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

    #  TT value :      -1.643231
    # QTT value :      -1.643231
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

    # TT-tensor    20D : |2| |2| |2| |2| |2| |2| |2|  |2|  |2| |2| |2| |2|  |2|  |2|  |2| |2|  |2| |2| |2| |2|
    # <rank>  =    8.9 :   \2/ \4/ \8/ \8/ \4/ \8/ \16/ \12/ \6/ \3/ \6/ \12/ \20/ \12/ \6/ \12/ \8/ \4/ \2/
    # 




|
|


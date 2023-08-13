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

    # -0.40243492107357137
    # -0.40243492107357137
    # 

  Note that changes to the copy will not affect the original tensor:

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # -0.40243492107357137
    # 42.0
    # 

  Note that this function also supports numbers and numpy arrays for convenience:

  .. code-block:: python

    a = teneva.copy(42.)
    b = teneva.copy(np.array([1, 2, 3]))
    c = teneva.copy(None)




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

    # Old value at multi-index :  1.23997e-01
    # New value at multi-index :  1.23897e-01
    # Difference for tensors   :      4.7e-05
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
    phi_r = teneva.interface(Y)
    phi_l = teneva.interface(Y, ltr=True)
    
    print('\nRight:')
    for phi in phi_r:
        print(phi)
        
    print('\nLeft:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # Right:
    # [1.]
    # [ 0.91827356  0.01969008 -0.39545666]
    # [0.27596253 0.88243108 0.38099878]
    # [0.25073765 0.86488016 0.43487117]
    # [1.]
    # 
    # Left:
    # [1.]
    # [0.70880216 0.03760129 0.70440445]
    # [-0.35614742  0.91707463 -0.17925719]
    # [0.42238547 0.51390831 0.7466517 ]
    # [1.]
    # 

  .. code-block:: python

    n = [4, 5, 6, 7]         # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index
    phi_r = teneva.interface(Y, i=i)
    phi_l = teneva.interface(Y, i=i, ltr=True)
    
    print('\nRight:')
    for phi in phi_r:
        print(phi)
        
    print('\nLeft:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # Right:
    # [-1.]
    # [ 0.45668125 -0.53773627  0.70871852]
    # [-0.18818141  0.15340933 -0.97007903]
    # [-0.72371761 -0.58742218 -0.36217124]
    # [1.]
    # 
    # Left:
    # [1.]
    # [ 0.17094215  0.35940745 -0.91739036]
    # [ 0.75337601 -0.454257    0.47547362]
    # [ 0.78641158 -0.30084399  0.53949024]
    # [-1.]
    # 

  .. code-block:: python

    n = [4, 5, 6, 7]         # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index
    P = [                    # Weight for all modes
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
    phi_r = teneva.interface(Y, P, i)
    phi_l = teneva.interface(Y, P, i, ltr=True)
    
    print('\nRight:')
    for phi in phi_r:
        print(phi)
        
    print('\nLeft:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # Right:
    # [1.]
    # [ 0.25451724 -0.93532114  0.24575464]
    # [-0.43380958 -0.81302931  0.38832021]
    # [ 0.90700559 -0.175054   -0.38301039]
    # [1.]
    # 
    # Left:
    # [1.]
    # [-0.83116354 -0.53502569  0.15137598]
    # [-0.7057236  -0.41653531 -0.57310779]
    # [0.59937809 0.7644333  0.23746081]
    # [1.]
    # 

  .. code-block:: python

    n = [7] * 4              # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index
    p = [                    # Weight for all modes (equal)
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    phi_r = teneva.interface(Y, p, i)
    phi_l = teneva.interface(Y, p, i, ltr=True)
    
    print('\nRight:')
    for phi in phi_r:
        print(phi)
        
    print('\nLeft:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # Right:
    # [1.]
    # [-0.91749354 -0.38737649  0.09024998]
    # [-0.98518646  0.10887469 -0.13249132]
    # [-0.75041279  0.42979407 -0.50215306]
    # [1.]
    # 
    # Left:
    # [1.]
    # [-0.39132182  0.44443091 -0.80582157]
    # [-0.37711509  0.47245875  0.7965971 ]
    # [-0.36842955 -0.55684806 -0.7444326 ]
    # [1.]
    # 

  .. code-block:: python

    n = [7] * 4              # Shape of the tensor
    Y = teneva.rand(n, r=3)  # Create 4-dim random TT-tensor
    i = [2, 3, 4, 5]         # Targer multi-index
    p = [                    # Weight for all modes (equal)
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    phi_r = teneva.interface(Y, p, i, norm=None)
    phi_l = teneva.interface(Y, p, i, norm=None, ltr=True)
    
    print('\nRight:')
    for phi in phi_r:
        print(phi)
        
    print('\nLeft:')
    for phi in phi_l:
        print(phi)

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # Right:
    # [0.03670165]
    # [ 0.07263102 -0.06344386 -0.01487898]
    # [ 0.26831568 -0.13243834 -0.06017433]
    # [-0.17640024 -0.12505031 -0.28719092]
    # [1.]
    # 
    # Left:
    # [1.]
    # [ 0.21526752 -0.27396234 -0.24768622]
    # [ 0.03480896 -0.16310117 -0.09573867]
    # [-0.02303626  0.06480931 -0.14186545]
    # [0.03670165]
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

    # Error     : 3.71e-21
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

    # 223.7667820576122
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)           # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                     # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)      # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')   # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 223.7667820576121
    # Error     : 5.08e-16
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

    # QTT value :       0.732299
    #  TT value :       0.732299
    # 

  We can also transform the TT-tensor back into QTT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.tt_to_qtt(Z)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.1622984106102536e-08
    # 




|
|

.. autofunction:: teneva.act_one.sum

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 9, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.sum(Y)                          # Sum of the TT-tensor elements

    # >>> ----------------------------------------
    # >>> Output:

    # 63.547547128159685
    # 

  .. code-block:: python

    Z = teneva.full(Y) # Compute tensor in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # 63.547547128159685
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

    #  TT value :       0.016735
    # QTT value :       0.016735
    # 

  We can also transform the QTT-tensor back into TT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.qtt_to_tt(Z, q)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 7.26168549613891e-09
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


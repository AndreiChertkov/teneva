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

    # 0.1643445611288208
    # 0.1643445611288208
    # 

  Note that changes to the copy will not affect the original tensor:

  .. code-block:: python

    Z[2][1, 2, 0] = 42.
    
    print(Y[2][1, 2, 0])
    print(Z[2][1, 2, 0])

    # >>> ----------------------------------------
    # >>> Output:

    # 0.1643445611288208
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
    
    i = [1, 2, 3, 4, 5]       # Select some tensor element
    y1 = teneva.get(Y1, i)    # Compute the element of the TT-tensor
    y0 = Y0[tuple(i)]         # Compute the same element of the original tensor
    abs(y1-y0)                # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10|  |10|   |10|   |10|  |10|
    # <rank>  =   63.0 :    \10/  \100/  \100/  \10/
    # 

  This function is also support batch mode (in the case of batch, it calls the function "get_many"):

  .. code-block:: python

    # Select some tensor elements:
    I = [
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    # Compute the element of the TT-tensor:
    y1 = teneva.get(Y1, I)
    
    # Compute the same element of the original tensor:
    y0 = [Y0[tuple(i)] for i in I]
    
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

    lr = 1.E-4                        # Learning rate
    n = [4, 5, 6, 7]                  # Shape of the tensor
    Y = teneva.rand(n, r=3, seed=44)  # Random TT-tensor
    i = [2, 3, 4, 5]                  # Targer multi-index for gradient
    y, dY = teneva.get_and_grad(Y, i)
    
    Z = teneva.copy(Y)                # Simulating gradient descent
    for k in range(len(n)):
        Z[k] -= lr * dY[k]
    
    z = teneva.get(Z, i)
    e = teneva.accuracy(Y, Z)
    
    print(f'Old value at multi-index : {y:-12.5e}')
    print(f'New value at multi-index : {z:-12.5e}')
    print(f'Difference for tensors   : {e:-12.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Old value at multi-index :  2.91493e-01
    # New value at multi-index :  2.90991e-01
    # Difference for tensors   :      8.1e-05
    # 

  We can also perform several GD steps:

  .. code-block:: python

    Z = teneva.copy(Y)
    for step in range(100):
        for k in range(len(n)):
            Z[k] -= lr * dY[k]
    
    z = teneva.get(Z, i)
    e = teneva.accuracy(Y, Z)
    
    print(f'Old value at multi-index : {y:-12.5e}')
    print(f'New value at multi-index : {z:-12.5e}')
    print(f'Difference for tensors   : {e:-12.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Old value at multi-index :  2.91493e-01
    # New value at multi-index :  2.41494e-01
    # Difference for tensors   :      8.1e-03
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
    I = [
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1],
    ]
    
    # Compute the element of the TT-tensor:
    y1 = teneva.get_many(Y1, I)
    
    # Compute the same element of the original tensor:
    y0 = [Y0[tuple(i)] for i in I]
    
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
    # [-1.]
    # [0.68813332 0.53462172 0.49056309]
    # [ 0.02724276  0.17567491 -0.98407122]
    # [ 0.28219429 -0.45639302 -0.84384346]
    # [1.]
    # 
    # Left:
    # [1.]
    # [-0.82889095 -0.55022389 -0.10096271]
    # [ 0.55175562 -0.81512821  0.1764419 ]
    # [ 0.65082799 -0.37736438  0.65880123]
    # [-1.]
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
    # [0.3736717  0.13703245 0.917383  ]
    # [ 0.15999998  0.97874472 -0.1282918 ]
    # [-0.73153711  0.26369271 -0.62874447]
    # [1.]
    # 
    # Left:
    # [1.]
    # [ 0.72667917 -0.39223735 -0.56399224]
    # [ 0.54988977 -0.5076079   0.66329139]
    # [ 0.54448258 -0.61483454 -0.57054116]
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
    # [-1.]
    # [ 0.02712957  0.79077339 -0.61150751]
    # [0.30447033 0.7558563  0.57963702]
    # [0.87461345 0.48475263 0.0081361 ]
    # [1.]
    # 
    # Left:
    # [1.]
    # [ 0.55886258 -0.31423024  0.76741903]
    # [-0.96060732 -0.24796613 -0.12548457]
    # [-0.81379032  0.03462715 -0.58012609]
    # [-1.]
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
    # [-0.32868849  0.94421494  0.02054309]
    # [-0.99678302  0.00501843 -0.07999011]
    # [ 0.55197584 -0.68846358 -0.47046846]
    # [1.]
    # 
    # Left:
    # [1.]
    # [-0.45780124  0.79627937 -0.39542028]
    # [-0.76947865 -0.16291856 -0.61754364]
    # [ 0.06665077 -0.90158264  0.4274417 ]
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
    # [-0.04710111]
    # [-0.05334143 -0.11429042 -0.11918024]
    # [ 0.05111053 -0.15246171  0.26067213]
    # [ 0.58188778  0.419016   -0.11394976]
    # [1.]
    # 
    # Left:
    # [1.]
    # [0.19713772 0.03388943 0.27447725]
    # [ 0.05784957  0.03471187 -0.17173144]
    # [-0.05027864 -0.05818472 -0.05735637]
    # [-0.04710111]
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

    # Error     : 1.91e-21
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

    # 283.64341295400476
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)           # Compute tensor in the full format to check the result
    
    v_full = np.linalg.norm(Y_full)
    print(v_full)                     # Print the resulting value from full tensor
    
    e = abs((v - v_full)/v_full)      # Compute error for TT-tensor vs full tensor 
    print(f'Error     : {e:-8.2e}')   # Rel. error

    # >>> ----------------------------------------
    # >>> Output:

    # 283.6434129540049
    # Error     : 4.01e-16
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

    # QTT value :       4.067825
    #  TT value :       4.067825
    # 

  We can also transform the TT-tensor back into QTT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.tt_to_qtt(Z)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.3084361360868113e-08
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

    # -10.421669993532463
    # 

  .. code-block:: python

    Z = teneva.full(Y) # Compute tensor in the full format to check the result
    np.sum(Z)

    # >>> ----------------------------------------
    # >>> Output:

    # -10.421669993532458
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

    #  TT value :      -0.144598
    # QTT value :      -0.144598
    # 

  We can also transform the QTT-tensor back into TT-tensor:

  .. code-block:: python

    q = int(np.log2(n[0]))
    U = teneva.qtt_to_tt(Z, q)
    
    teneva.accuracy(Y, U)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.9914054150840573e-08
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


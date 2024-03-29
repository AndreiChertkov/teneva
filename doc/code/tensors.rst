Module tensors: collection of explicit useful TT-tensors
--------------------------------------------------------


.. automodule:: teneva.tensors


-----




|
|

.. autofunction:: teneva.tensors.const

  **Examples**:

  .. code-block:: python

    n = [10] * 5               # Shape of the tensor  
    Y = teneva.const(n, v=42.) # A tensor of all 42
    teneva.show(Y)             # Print the resulting TT-tensor
    Y_full = teneva.full(Y)
    print()
    print(f'Min value : {np.min(Y_full)}')
    print(f'Max value : {np.max(Y_full)}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10| |10| |10| |10| |10|
    # <rank>  =    1.0 :    \1/  \1/  \1/  \1/
    # 
    # Min value : 42.00000000000003
    # Max value : 42.00000000000003
    # 

  We can, among other things, build the TT-tensor equal to zero everywhere:

  .. code-block:: python

    n = [10] * 5              # Shape of the tensor  
    Y = teneva.const(n, v=0.) # A tensor of all zeros
    teneva.show(Y)            # Print the resulting TT-tensor
    Y_full = teneva.full(Y)
    print()
    print(f'Min value : {np.min(Y_full)}')
    print(f'Max value : {np.max(Y_full)}')

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10| |10| |10| |10| |10|
    # <rank>  =    1.0 :    \1/  \1/  \1/  \1/
    # 
    # Min value : 0.0
    # Max value : 0.0
    # 

  Note that the given value is distributed evenly over the TT-cores:

  .. code-block:: python

    Y = teneva.const([10] * 100, v=42**100)
    print(Y[5].reshape(-1))

    # >>> ----------------------------------------
    # >>> Output:

    # [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
    # 

  We can also set multi-indices in which the tensor is forced to zero (note that this will also necessarily lead to the appearance of other zero elements):

  .. code-block:: python

    n = [10] * 5           # Shape of the tensor
    I = [                  # Multi-indices for zeros
        [0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5],
        [9, 9, 9, 9, 9],
    ]
    Y = teneva.const(n, v=42., I_zero=I)
    
    print(f'Y at I[0]           :', teneva.get(Y, I[0]))
    print(f'Y at I[1]           :', teneva.get(Y, I[1]))
    print(f'Y at I[2]           :', teneva.get(Y, I[2]))
    
    Y_full = teneva.full(Y)
    
    print(f'Num of zero items   :', np.sum(Y_full < 1.E-20))
    print(f'Mean non zero value :', np.sum(Y_full) / np.sum(Y_full > 1.E-20))

    # >>> ----------------------------------------
    # >>> Output:

    # Y at I[0]           : 0.0
    # Y at I[1]           : 0.0
    # Y at I[2]           : 0.0
    # Num of zero items   : 27100
    # Mean non zero value : 42.00000000000001
    # 

  Then we specify multi-indices in which the tensor is forced to zero, we can also set one multi-index, which will not affected by zero multi-indices:

  .. code-block:: python

    n = [10] * 5            # Shape of the tensor
    i = [5, 5, 5, 5, 5]     # Multi-index for non-zero item
    I = [                   # Multi-indices for zeros
        [0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5],
        [9, 9, 9, 9, 9],
    ]
    Y = teneva.const(n, v=42., I_zero=I, i_non_zero=i)
    
    print(f'Y at i              :', teneva.get(Y, i))
    print(f'Y at I[0]           :', teneva.get(Y, I[0]))
    print(f'Y at I[1]           :', teneva.get(Y, I[1]))
    print(f'Y at I[2]           :', teneva.get(Y, I[2]))
    
    Y_full = teneva.full(Y)
    
    print(f'Num of zero items   :', np.sum(Y_full < 1.E-20))
    print(f'Mean non zero value :', np.sum(Y_full) / np.sum(Y_full > 1.E-20))

    # >>> ----------------------------------------
    # >>> Output:

    # Y at i              : 42.00000000000003
    # Y at I[0]           : 0.0
    # Y at I[1]           : 0.0
    # Y at I[2]           : 0.0
    # Num of zero items   : 27100
    # Mean non zero value : 42.00000000000001
    # 

  Note, if we set too many multi-indices in which the tensor is forced to zero (under which it will be impossible to keep a non-zero item), then it will lead to error:

  .. code-block:: python

    n = [2] * 5                    # Shape of the tensor
    i = [1, 1, 1, 1, 1]            # Multi-index for non-zero item
    I = teneva.sample_lhs(n, 100)  # Multi-indices for zeros
    
    try:
        Y = teneva.const(n, v=42., I_zero=I, i_non_zero=i)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Can not set zero items
    # 




|
|

.. autofunction:: teneva.tensors.delta

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]  # Shape of the tensor
    i = [ 1,  2,  3,  4,  5]  # The multi-index for the nonzero element
    v = 42.                   # A value of the tensor at the multi-index "i"
    Y = teneva.delta(n, i, v) # Build the TT-tensor
    
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |20| |18| |16| |14| |12|
    # <rank>  =    1.0 :    \1/  \1/  \1/  \1/
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)    # Transform the TT-tensor to the full format
    i_max = np.argmax(Y_full)  # Find the multi-index and the value for max
    i_max = np.unravel_index(i_max, n)
    y_max = Y_full[i_max]
    
    # Find a number of nonzero tensor items:
    s = len([y for y in Y_full.flatten() if abs(y) > 1.E-10])
        
    print(f'The max value multi-index:', i_max)
    print(f'The max value            :', y_max)
    print(f'Number of nonzero items  :', s)

    # >>> ----------------------------------------
    # >>> Output:

    # The max value multi-index: (1, 2, 3, 4, 5)
    # The max value            : 42.00000000000003
    # Number of nonzero items  : 1
    # 

  We can also build some multidimensional TT-tensor by the "delta" function and check the norm of the result:

  .. code-block:: python

    d = 100                   # Dimension of the tensor
    n = [20] * d              # Shape of the tensor
    i = [3] * d               # The multi-index for the nonzero element
    v = 42.                   # The value of the tensor at the multi-index "k"
    Y = teneva.delta(n, i, v) # Build the TT-tensor
    
    teneva.norm(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.00000000000021
    # 




|
|

.. autofunction:: teneva.tensors.poly

  **Examples**:

  .. code-block:: python

    n = [10] * 5                      # Shape of the tensor
    shift = np.array([2, 3, 2, 3, 2]) # Shift value
    scale = 5.                        # Scale
    power = 3                         # Power
    Y = teneva.poly(n, shift, power, scale)
    teneva.show(Y)                    # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10| |10| |10| |10| |10|
    # <rank>  =    2.0 :    \2/  \2/  \2/  \2/
    # 

  We can check the result:

  .. code-block:: python

    i = [2, 3, 3, 4, 5]
    
    y_appr = teneva.get(Y, i)
    y_real = scale * np.sum((i + shift)**power)
    
    print(y_appr)
    print(y_real)

    # >>> ----------------------------------------
    # >>> Output:

    # 5455.0
    # 5455.0
    # 

  .. code-block:: python

    i = np.zeros(5)
    
    y_appr = teneva.get(Y, i)
    y_real = scale * np.sum((i + shift)**power)
    
    print(y_appr)
    print(y_real)

    # >>> ----------------------------------------
    # >>> Output:

    # 390.0
    # 390.0
    # 

  The value of the "shift" may be also a scalar:

  .. code-block:: python

    Y = teneva.poly(n, 42., power, scale)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |10| |10| |10| |10| |10|
    # <rank>  =    2.0 :    \2/  \2/  \2/  \2/
    # 




|
|

.. autofunction:: teneva.tensors.rand

  **Examples**:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]           # TT-ranks for the TT-tensor
    Y = teneva.rand(n, r)            # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    3.6 :    \2/  \3/  \4/  \5/
    # 

  If all inner TT-ranks are equal, we may pass it as a number:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    Y = teneva.rand(n, r)            # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/
    # 

  We may use custom limits:

  .. code-block:: python

    n = [4] * 5                      # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    a = 0.99                         # Minimum value
    b = 1.                           # Maximum value
    Y = teneva.rand(n, r, a, b)      # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor
    print(Y[0])                      # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4| |4| |4| |4| |4|
    # <rank>  =    5.0 :   \5/ \5/ \5/ \5/
    # [[[0.99927218 0.99093213 0.99799328 0.9995912  0.99683778]
    #   [0.99596874 0.99060295 0.99602039 0.99490159 0.99912654]
    #   [0.99325016 0.99544655 0.99448139 0.99497898 0.9961929 ]
    #   [0.99645428 0.99537899 0.99154167 0.9918711  0.99215704]]]
    # 

  Note that we can also set a random seed value:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    Y = teneva.rand(n, r, seed=42)   # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/
    # 




|
|

.. autofunction:: teneva.tensors.rand_custom

  **Examples**:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]           # TT-ranks for the TT-tensor
    f = lambda sz: [42]*sz           # Sampling function
    Y = teneva.rand_custom(n, r, f)  # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor
    print(Y[0])                      # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    3.6 :    \2/  \3/  \4/  \5/
    # [[[42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]
    #   [42. 42.]]]
    # 

  If all inner TT-ranks are equal, we may pass it as a number:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    f = lambda sz: [42]*sz           # Sampling function
    Y = teneva.rand_custom(n, r, f)  # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/
    # 




|
|

.. autofunction:: teneva.tensors.rand_norm

  **Examples**:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]           # TT-ranks for the TT-tensor
    Y = teneva.rand_norm(n, r)       # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    3.6 :    \2/  \3/  \4/  \5/
    # 

  If all inner TT-ranks are equal, we may pass it as a number:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]         # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    Y = teneva.rand_norm(n, r)       # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    5.0 :    \5/  \5/  \5/  \5/
    # 

  We may use custom limits:

  .. code-block:: python

    n = [4] * 5                      # Shape of the tensor
    r = 5                            # TT-ranks for the TT-tensor
    m = 42.                          # Mean ("centre")
    s = 0.0001                       # Standard deviation
    Y = teneva.rand_norm(n, r, m, s) # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor
    print(Y[0])                      # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4| |4| |4| |4| |4|
    # <rank>  =    5.0 :   \5/ \5/ \5/ \5/
    # [[[42.00014161 42.00005536 42.00008364 42.00002766 41.99995429]
    #   [42.00004742 41.99996626 42.00007688 41.99997376 41.99998834]
    #   [42.00001833 42.00008305 41.99980699 42.00005218 42.00014115]
    #   [41.99992415 41.99994802 41.99995973 42.00007551 42.00011347]]]
    # 

  Note that we can also set a random seed value:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]            # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]              # TT-ranks for the TT-tensor
    Y = teneva.rand_norm(n, r, seed=42) # Build the random TT-tensor
    teneva.show(Y)                      # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |12| |13| |14| |15| |16|
    # <rank>  =    3.6 :    \2/  \3/  \4/  \5/
    # 




|
|

.. autofunction:: teneva.tensors.rand_stab

  **Examples**:

  .. code-block:: python

    d = 10000                  # Dimension of the tensor
    n = [10] * d               # Shape of the tensor
    r = 5                      # TT-rank for the TT-tensor
    Y = teneva.rand_stab(n, r) # Build the random TT-tensor
    teneva.get(Y, [0]*d)       # Compute element of the tensor         

    # >>> ----------------------------------------
    # >>> Output:

    # 0.9999999999999921
    # 

  Please note that the usual method of generating a random TT-tensor will result in infinite or zero tensor values at any indices at very high dimensions:

  .. code-block:: python

    Y = teneva.rand(n, r)
    teneva.get(Y, [0]*d) 

    # >>> ----------------------------------------
    # >>> Output:

    # /Users/andrei/opt/anaconda3/envs/teneva/lib/python3.8/site-packages/teneva-0.14.6-py3.8.egg/teneva/act_one.py:61: RuntimeWarning: overflow encountered in matmul
    # /Users/andrei/opt/anaconda3/envs/teneva/lib/python3.8/site-packages/teneva-0.14.6-py3.8.egg/teneva/act_one.py:61: RuntimeWarning: invalid value encountered in matmul
    # 




|
|


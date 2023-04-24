Module act_two: operations with a pair of TT-tensors
----------------------------------------------------


.. automodule:: teneva.act_two


-----




|
|

.. autofunction:: teneva.act_two.accuracy

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2)   # 10-dim random TT-tensor with TT-rank 2
    Z1 = teneva.mul(1.E-4, Y1)    # The TT-tensor Y1 + eps * Y1 (eps = 1.E-4)
    
    Y2 = teneva.add(Y1, Z1) 
    
    eps = teneva.accuracy(Y1, Y2) # The relative difference ("accuracy")
    
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




|
|

.. autofunction:: teneva.act_two.add

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3
    
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 + 3 = 5):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    5.0 :   \5/ \5/ \5/ \5/ \5/ \5/ \5/ \5/ \5/
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 8.70e-17
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    3.0 :   \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    
    Y = teneva.add(Y1, Y2)      # Compute the sum of Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    3.0 :   \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/
    # 

  .. code-block:: python

    Y1_full = 42.             # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full + Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 5.14e-16
    # 

  If both arguments are numbers, then function returns the sum of numbers:

  .. code-block:: python

    Y1 = 40.               # Just a number
    Y2 = 2                 # Just a number
    Y = teneva.add(Y1, Y2) # Compute the sum of Y1 and Y2
    print(Y)               # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 




|
|

.. autofunction:: teneva.act_two.mul

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.mul(Y1, Y2) # Compute the product of Y1 and Y2
    teneva.show(Y)         # Print the resulting TT-tensor (note that it has TT-rank 2 x 3 = 6)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    6.0 :   \6/ \6/ \6/ \6/ \6/ \6/ \6/ \6/ \6/
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full * Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 4.00e-16
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    
    Y = teneva.mul(Y1, Y2) # Compute the product of Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 x 1 = 2):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    2.0 :   \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    
    Y = teneva.mul(Y1, Y2) # Compute the product of Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 x 1 = 2):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    2.0 :   \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/
    # 

  .. code-block:: python

    Y1 = 21.               # Just a number
    Y2 = 2                 # Just a number
    
    Y = teneva.mul(Y1, Y2) # Compute the product of Y1 and Y2
    print(Y)               # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 




|
|

.. autofunction:: teneva.act_two.mul_scalar

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3
    
    v = teneva.mul_scalar(Y1, Y2) # Compute the product of Y1 and Y2
    
    print(v)                      # Print the resulting value

    # >>> ----------------------------------------
    # >>> Output:

    # -3.460467948446013
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    
    v_full = np.sum(Y1_full * Y2_full)
    
    print(v_full) # Print the resulting value from full tensor
    
    # Compute error for TT-tensor vs full tensor :
    e = abs((v - v_full)/v_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # -3.4604679484459884
    # Error     : 7.06e-15
    # 

  We can also set a flag "use_stab", in which case a value that is 2^p times smaller than the real value will be returned:

  .. code-block:: python

    v, p = teneva.mul_scalar(Y1, Y2, use_stab=True)
    print(v)
    print(p)
    print(v * 2**p)

    # >>> ----------------------------------------
    # >>> Output:

    # -1.7302339742230064
    # 1
    # -3.460467948446013
    # 




|
|

.. autofunction:: teneva.act_two.outer

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([4]*5, 2) # 5-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([3]*5, 3) # 5-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.outer(Y1, Y2) # Compute the outer product of Y1 and Y2
    teneva.show(Y)           # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |4| |4| |4| |4| |4| |3| |3| |3| |3| |3|
    # <rank>  =    2.3 :   \2/ \2/ \2/ \2/ \1/ \3/ \3/ \3/ \3/
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = np.tensordot(Y1_full, Y2_full, 0)
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 2.09e-16
    # 




|
|

.. autofunction:: teneva.act_two.sub

  **Examples**:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = teneva.rand([5]*10, 3) # 10-dim random TT-tensor with TT-rank 3

  .. code-block:: python

    Y = teneva.sub(Y1, Y2) # Compute the difference between Y1 and Y2
    teneva.show(Y)         # Print the resulting TT-tensor (note that it has TT-rank 2 + 3 = 5)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    5.0 :   \5/ \5/ \5/ \5/ \5/ \5/ \5/ \5/ \5/
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y_full = teneva.full(Y)
    
    Z_full = Y1_full - Y2_full
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)                     
    print(f'Error     : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 8.65e-17
    # 

  This function also supports float argument:

  .. code-block:: python

    Y1 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    Y2 = 42.                    # Just a number
    
    Y = teneva.sub(Y1, Y2)     # Compute the difference between Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    3.0 :   \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/
    # 

  .. code-block:: python

    Y1 = 42.                    # Just a number
    Y2 = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    
    Y = teneva.sub(Y1, Y2)      # Compute the difference between Y1 and Y2
    
    # Print the resulting TT-tensor (note that it has TT-rank 2 + 1 = 3):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    3.0 :   \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/ \3/
    # 

  .. code-block:: python

    Y1 = 44.               # Just a number
    Y2 = 2                 # Just a number
    
    Y = teneva.sub(Y1, Y2) # Compute the difference between Y1 and Y2
    
    print(Y)               # The result is a number

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 




|
|


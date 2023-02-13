Module act_many: operations with a set of TT-tensors
----------------------------------------------------


.. automodule:: teneva.core.act_many


-----




|
|

.. autofunction:: teneva.add_many

  **Examples**:

  .. code-block:: python

    # 10 random TT-tensors with TT-rank 2:
    Y_all = [teneva.tensor_rand([5]*10, 2) for _ in range(10)]
    
    # Compute the sum:
    Y = teneva.add_many(Y_all, e=1.E-4, r=50, trunc_freq=2)
    
    # Show the result:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5|  |5|  |5|  |5|  |5|  |5|  |5|  |5| |5|
    # <rank>  =   17.9 :   \5/ \20/ \20/ \20/ \20/ \20/ \20/ \20/ \5/
    # 

  This function also supports float arguments:

  .. code-block:: python

    Y_all = [
        42.,
        teneva.tensor_rand([5]*10, 2),
        33.,
        teneva.tensor_rand([5]*10, 4)
    ]
    Y = teneva.add_many(Y_all, e=1.E-4, r=50, trunc_freq=2)
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    10D : |5| |5| |5| |5| |5| |5| |5| |5| |5| |5|
    # <rank>  =    6.7 :   \5/ \7/ \7/ \7/ \7/ \7/ \7/ \7/ \5/
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




|
|

.. autofunction:: teneva.outer_many

  **Examples**:

  .. code-block:: python

    Y1 = teneva.tensor_rand([4]*5, 2) # 5-dim random TT-tensor with TT-rank 2
    Y2 = teneva.tensor_rand([3]*5, 3) # 5-dim random TT-tensor with TT-rank 3
    Y3 = teneva.tensor_rand([2]*5, 4) # 5-dim random TT-tensor with TT-rank 4

  .. code-block:: python

    Y = teneva.outer_many([Y1, Y2, Y3]) # Compute the outer product
    teneva.show(Y)                      # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    15D : |4| |4| |4| |4| |4| |3| |3| |3| |3| |3| |2| |2| |2| |2| |2|
    # <rank>  =    2.6 :   \2/ \2/ \2/ \2/ \1/ \3/ \3/ \3/ \3/ \1/ \4/ \4/ \4/ \4/
    # 

  .. code-block:: python

    Y1_full = teneva.full(Y1) # Compute tensors in the full format
    Y2_full = teneva.full(Y2) # to check the result
    Y3_full = teneva.full(Y3)
    Y_full = teneva.full(Y)
    
    Z_full = np.tensordot(Y1_full, Y2_full, 0)
    Z_full = np.tensordot(Z_full, Y3_full, 0)
    
    e = np.linalg.norm(Y_full - Z_full) # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)         #
    
    print(f'Error     : {e:-8.2e}')     # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Error     : 3.11e-16
    # 




|
|


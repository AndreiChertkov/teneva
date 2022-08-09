Module act_many: operations with a set of TT-tensors
----------------------------------------------------


.. automodule:: teneva.core.act_many


-----


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

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  5 20 20 20 20 20 20 20  5  1 
    # 
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



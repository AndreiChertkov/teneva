Module vis: visualization methods for tensors
---------------------------------------------


.. automodule:: teneva.core.vis


-----


.. autofunction:: teneva.show

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.tensor_rand([10, 12, 8, 8, 30], 2)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #  10 12  8  8 30 
    #  / \/ \/ \/ \/ \
    #  1  2  2  2  2  1 
    # 
    # 

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.tensor_rand([2**14]*10, 12)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #    16384 16384 16384 16384 16384 16384 16384 16384 16384 16384 
    #     / \   / \   / \   / \   / \   / \   / \   / \   / \   / \  
    #   1     12    12    12    12    12    12    12    12    12    1   
    # 
    # 



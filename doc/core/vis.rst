Module vis: visualization methods for tensors
---------------------------------------------


.. automodule:: teneva.core.vis


-----


.. autofunction:: teneva.show

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.show(Y)                         # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  10 12  8  8 30 
    #  / \/ \/ \/ \/ \
    #  1  2  2  2  2  1 
    # 
    # 



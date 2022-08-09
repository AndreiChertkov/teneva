Module tensor: basic operations with TT-tensors
-----------------------------------------------


.. automodule:: teneva.core.tensor


-----


.. autofunction:: teneva.rand

  **Examples**:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]    # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]      # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)       # Build random TT-tensor
    teneva.show(Y)              # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  If all inner TT-ranks are equal, we may pass it as a number:

  .. code-block:: python

    n = [12, 13, 14, 15, 16]    # Shape of the tensor
    r = 5                       # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)       # Build random TT-tensor
    teneva.show(Y)              # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  5  5  5  5  1 
    # 
    # 



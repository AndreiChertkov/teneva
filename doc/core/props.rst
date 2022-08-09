Module props: various properties (mean, norm, etc.) of TT-tensors
-----------------------------------------------------------------


.. automodule:: teneva.core.props


-----


.. autofunction:: teneva.erank

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 2) # 10-dim random TT-tensor with TT-rank 2
    teneva.erank(Y)            # The effective TT-rank

    # >>> ----------------------------------------
    # >>> Output:

    # 2.0
    # 


.. autofunction:: teneva.ranks

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.ranks(Y)                        # TT-ranks of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # array([1, 2, 2, 2, 2, 1])
    # 


.. autofunction:: teneva.shape

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.shape(Y)                        # Shape of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # array([10, 12,  8,  8, 30])
    # 


.. autofunction:: teneva.size

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([10, 12, 8, 8, 30], 2) # 5-dim random TT-tensor with TT-rank 2
    teneva.size(Y)                         # Size of the TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # 192
    # 



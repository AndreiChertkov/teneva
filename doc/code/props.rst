Module props: various simple properties of TT-tensors
-----------------------------------------------------


.. automodule:: teneva.props


-----




|
|

.. autofunction:: teneva.props.erank

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([5]*10, 2)
    
    # The effective TT-rank:
    teneva.erank(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 2.0
    # 

  Note that it also works for 2-dimensional arrays (i.e., matrices):

  .. code-block:: python

    # 2-dim random TT-tensor (matrix) with TT-rank 20:
    Y = teneva.rand([5]*2, 20)
    
    # The effective TT-rank:
    teneva.erank(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 20
    # 




|
|

.. autofunction:: teneva.props.ranks

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([10, 12, 8, 8, 30], 2)
    
    # TT-ranks of the TT-tensor:
    teneva.ranks(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # array([1, 2, 2, 2, 2, 1])
    # 




|
|

.. autofunction:: teneva.props.shape

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([10, 12, 8, 8, 30], 2)
    
    # Shape of the TT-tensor:
    teneva.shape(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # array([10, 12,  8,  8, 30])
    # 




|
|

.. autofunction:: teneva.props.size

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([10, 12, 8, 8, 30], 2)
    
    # Size of the TT-tensor:
    teneva.size(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 192
    # 




|
|


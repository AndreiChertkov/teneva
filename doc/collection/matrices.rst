Module matrices: collection of explicit useful QTT-matrices (draft)
-------------------------------------------------------------------


.. automodule:: teneva.collection.matrices


-----


.. autofunction:: teneva.matrix_delta

  **Examples**:

  .. code-block:: python

    q = 5                               # Quantization level (the size is 2^q)
    i = 2                               # The col index for nonzero element
    j = 4                               # The row index for nonzero element
    v = 42.                             # The value of the matrix at indices "i, j"
    Y = teneva.matrix_delta(q, i, j, v) # Build QTT-matrix

  We can also build some big QTT-matrix by "delta" function and check the norm of the result:

  .. code-block:: python

    q = 100                             # Quantization level (the size is 2^q)
    i = 2                               # The col index for nonzero element
    j = 4                               # The row index for nonzero element
    v = 42.                             # The value of the matrix at indices "i, j"
    Y = teneva.matrix_delta(q, i, j, v) # Build QTT-matrix
    
    teneva.norm(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 



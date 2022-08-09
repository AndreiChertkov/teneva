Module vectors: collection of explicit useful QTT-vectors (draft)
-----------------------------------------------------------------


.. automodule:: teneva.collection.vectors


-----


.. autofunction:: teneva.vector_delta

  **Examples**:

  .. code-block:: python

    q = 5                            # Quantization level (the size is 2^q)
    i = 2                            # The index for nonzero element
    v = 42.                          # The value of the vector at index "i"
    Y = teneva.vector_delta(q, i, v) # Build QTT-vector
    
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #   2  2  2  2  2 
    #  / \/ \/ \/ \/ \
    #  1  1  1  1  1  1 
    # 
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)          # Transform QTT-vector to full format
    Y_full = Y_full.flatten('F')
    i_max = np.argmax(Y_full)        # Find index and value for max
    y_max = Y_full[i_max]
    
    # Find number of nonzero vector items:
    s = len([y for y in Y_full if abs(y) > 1.E-10])                          
        
    print(f'The max value index      :', i_max)
    print(f'The max value            :', y_max)
    print(f'Number of nonzero items  :', s)

    # >>> ----------------------------------------
    # >>> Output:

    # The max value index      : 2
    # The max value            : 42.0
    # Number of nonzero items  : 1
    # 

  We can also build some big QTT-vector by "delta" function and check the norm of the result:

  .. code-block:: python

    q = 100                          # Quantization level (the size is 2^q)
    i = 99                           # The index for nonzero element
    v = 42.                          # The value of the vector at index "i"
    Y = teneva.vector_delta(q, i, v) # Build QTT-vector
    
    teneva.norm(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 



tensors: collection of explicit useful TT-tensors
-------------------------------------------------


.. automodule:: teneva.collection.tensors


-----


.. autofunction:: teneva.tensor_delta

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]         # Shape of the tensor
    k = [ 1,  2,  3,  4,  5]         # The multi-index for nonzero element
    v = 42.                          # The value of the tensor at multi-index "k"
    Y = teneva.tensor_delta(n, k, v) # Build TT-tensor
    
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #  20 18 16 14 12 
    #  / \/ \/ \/ \/ \
    #  1  1  1  1  1  1 
    # 
    # 

  Let check the result:

  .. code-block:: python

    Y_full = teneva.full(Y)          # Transform TT-tensor to full format
    i_max = np.argmax(Y_full)        # Find multi-index and value for max
    i_max = np.unravel_index(i_max, n)
    y_max = Y_full[i_max]
    
    s = 0                            # Find number of nonzero tensor items
    for y in Y_full.flatten():
        if abs(y) > 1.E-10:
            s += 1
        
    print(f'The max value multi-index:', i_max)
    print(f'The max value            :', y_max)
    print(f'Number of nonzero items  :', s)

    # >>> ----------------------------------------
    # >>> Output:

    # The max value multi-index: (1, 2, 3, 4, 5)
    # The max value            : 42.0
    # Number of nonzero items  : 1
    # 

  We can also build some multidimensional TT-tensor by "delta" function and check the norm of the result:

  .. code-block:: python

    d = 100                          # Dimension of the tensor
    n = [20] * d                     # Shape of the tensor
    k = [3] * d                      # The multi-index for nonzero element
    v = 42.                          # The value of the tensor at multi-index "k"
    Y = teneva.tensor_delta(n, k, v) # Build TT-tensor
    
    teneva.norm(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # 42.0
    # 



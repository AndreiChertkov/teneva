transformation: orthogonalization and truncation of TT-tensors
--------------------------------------------------------------


.. automodule:: teneva.core.transformation


-----


.. autofunction:: teneva.orthogonalize

  **Examples**:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.orthogonalize(Y, d-1)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  .. code-block:: python

    for G in Z:
        print(sum([G[:, i, :].T @ G[:, i, :] for i in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[1. 0.]
    #  [0. 1.]]
    # [[ 1.00000000e+00 -4.85722573e-17  1.45716772e-16]
    #  [-4.85722573e-17  1.00000000e+00  2.94902991e-17]
    #  [ 1.45716772e-16  2.94902991e-17  1.00000000e+00]]
    # [[ 1.00000000e+00  2.77555756e-17  3.46944695e-18 -3.46944695e-17]
    #  [ 2.77555756e-17  1.00000000e+00  6.93889390e-17  1.38777878e-17]
    #  [ 3.46944695e-18  6.93889390e-17  1.00000000e+00  3.46944695e-18]
    #  [-3.46944695e-17  1.38777878e-17  3.46944695e-18  1.00000000e+00]]
    # [[ 1.00000000e+00  6.93889390e-18  2.08166817e-17  6.07153217e-18
    #   -4.29344060e-17]
    #  [ 6.93889390e-18  1.00000000e+00 -4.33680869e-17 -4.33680869e-17
    #   -1.14491749e-16]
    #  [ 2.08166817e-17 -4.33680869e-17  1.00000000e+00  4.85722573e-17
    #   -4.07660017e-17]
    #  [ 6.07153217e-18 -4.33680869e-17  4.85722573e-17  1.00000000e+00
    #    2.16840434e-17]
    #  [-4.29344060e-17 -1.14491749e-16 -4.07660017e-17  2.16840434e-17
    #    1.00000000e+00]]
    # [[74734922.71019751]]
    # 

  We can also orthogonalize for the first mode:

  .. code-block:: python

    Z = teneva.orthogonalize(Y, 0)

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  .. code-block:: python

    for G in Z:
        print(sum([G[:, i, :] @ G[:, i, :].T for i in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[74734922.71019748]]
    # [[1.0000000e+00 6.9388939e-18]
    #  [6.9388939e-18 1.0000000e+00]]
    # [[1.00000000e+00 0.00000000e+00 0.00000000e+00]
    #  [0.00000000e+00 1.00000000e+00 2.08166817e-17]
    #  [0.00000000e+00 2.08166817e-17 1.00000000e+00]]
    # [[ 1.00000000e+00 -3.98986399e-17  7.11236625e-17  5.63785130e-18]
    #  [-3.98986399e-17  1.00000000e+00  2.08166817e-17 -7.97972799e-17]
    #  [ 7.11236625e-17  2.08166817e-17  1.00000000e+00  2.42861287e-17]
    #  [ 5.63785130e-18 -7.97972799e-17  2.42861287e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -5.03069808e-17  4.91143584e-17  0.00000000e+00
    #   -1.38777878e-17]
    #  [-5.03069808e-17  1.00000000e+00  2.75387352e-17 -1.38777878e-17
    #   -4.51028104e-17]
    #  [ 4.91143584e-17  2.75387352e-17  1.00000000e+00 -5.59448321e-17
    #   -5.50774704e-17]
    #  [ 0.00000000e+00 -1.38777878e-17 -5.59448321e-17  1.00000000e+00
    #    2.77555756e-17]
    #  [-1.38777878e-17 -4.51028104e-17 -5.50774704e-17  2.77555756e-17
    #    1.00000000e+00]]
    # 


-----


.. autofunction:: teneva.orthogonalize_left

  **Examples**:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.orthogonalize_left(Y, d-2)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 


-----


.. autofunction:: teneva.orthogonalize_right

  **Examples**:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.orthogonalize_right(Y, d-1)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 1.06e-08
    # 


-----


.. autofunction:: teneva.truncate

  **Examples**:

  .. code-block:: python

    Y = teneva.rand([5]*10, 3)            # 10-dim random TT-tensor with TT-rank 3
    Y = teneva.add(Y, teneva.add(Y, Y))   # Compute Y + Y + Y (the real TT-rank is still 3)
    teneva.show(Y)                        # Print the resulting TT-tensor (note that it has TT-rank 3 + 3 + 3 = 9)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  9  9  9  9  9  9  9  9  9  1 
    # 
    # 

  .. code-block:: python

    Z = teneva.truncate(Y, e=1.E-2)       # Truncate (round) the TT-tensor
    teneva.show(Z)                        # Print the resulting TT-tensor (note that it has TT-rank 3)
    eps = teneva.accuracy(Y, Z)           # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # Accuracy     : 1.80e-08
    # 

  We can also specify the desired TT-rank of truncated TT-tensor:

  .. code-block:: python

    Z = teneva.truncate(Y, e=1.E-6, r=3)  # Truncate (round) the TT-tensor
    teneva.show(Z)                        # Print the resulting TT-tensor (note that it has TT-rank 3)
    eps = teneva.accuracy(Y, Z)           # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # Accuracy     : 1.80e-08
    # 

  If we choose a lower TT-rank value, then precision will be (predictably) lost:

  .. code-block:: python

    Z = teneva.truncate(Y, e=1.E-6, r=2)  # Truncate (round) the TT-tensor
    teneva.show(Z)                        # Print the resulting TT-tensor (note that it has TT-rank 2)
    eps = teneva.accuracy(Y, Z)           # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  1 
    # 
    # Accuracy     : 1.02e+00
    # 



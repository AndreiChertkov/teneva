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

    # [[1.00000000e+00 4.16333634e-17]
    #  [4.16333634e-17 1.00000000e+00]]
    # [[ 1.00000000e+00  5.55111512e-17 -5.55111512e-17]
    #  [ 5.55111512e-17  1.00000000e+00 -4.16333634e-17]
    #  [-5.55111512e-17 -4.16333634e-17  1.00000000e+00]]
    # [[ 1.00000000e+00  3.13876529e-17 -5.20417043e-17  2.08166817e-17]
    #  [ 3.13876529e-17  1.00000000e+00 -3.81639165e-17  1.73472348e-18]
    #  [-5.20417043e-17 -3.81639165e-17  1.00000000e+00  4.85722573e-17]
    #  [ 2.08166817e-17  1.73472348e-18  4.85722573e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -2.08166817e-17 -6.93889390e-18 -1.38777878e-17
    #    5.55111512e-17]
    #  [-2.08166817e-17  1.00000000e+00  1.38777878e-17  6.50521303e-18
    #    2.77555756e-17]
    #  [-6.93889390e-18  1.38777878e-17  1.00000000e+00 -5.55111512e-17
    #    6.93889390e-18]
    #  [-1.38777878e-17  6.50521303e-18 -5.55111512e-17  1.00000000e+00
    #    6.24500451e-17]
    #  [ 5.55111512e-17  2.77555756e-17  6.93889390e-18  6.24500451e-17
    #    1.00000000e+00]]
    # [[53653047.87891468]]
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

    # Accuracy     : 7.48e-09
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

    # Accuracy     : 1.35e-08
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
    # Accuracy     : 2.17e-08
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
    # Accuracy     : 2.17e-08
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
    # Accuracy     : 7.20e-01
    # 



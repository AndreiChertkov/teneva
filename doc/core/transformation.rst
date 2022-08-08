Module transformation: orthogonalization and truncation of TT-tensors
---------------------------------------------------------------------


.. automodule:: teneva.core.transformation


-----


.. autofunction:: teneva.orthogonalize

  **Examples**:

  We set the values of parameters and build a random TT-tensor:

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

  We perform "left" orthogonalization for all TT-cores except the last one:

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

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  And we can make sure that all TT-cores, except the last one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    for G in Z:
        print(sum([G[:, j, :].T @ G[:, j, :] for j in range(G.shape[1])]))

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

  We can also perform "right" orthogonalization for all TT-cores except the first one:

  .. code-block:: python

    Z = teneva.orthogonalize(Y, 0)

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 3.12e-08
    # 

  And we can make sure that all TT-cores, except the first one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    for G in Z:
        print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[53653047.87891469]]
    # [[1.00000000e+00 2.77555756e-17]
    #  [2.77555756e-17 1.00000000e+00]]
    # [[ 1.00000000e+00 -1.38777878e-17 -3.20923843e-17]
    #  [-1.38777878e-17  1.00000000e+00 -6.07153217e-18]
    #  [-3.20923843e-17 -6.07153217e-18  1.00000000e+00]]
    # [[ 1.00000000e+00  6.24500451e-17 -4.16333634e-17  1.04083409e-17]
    #  [ 6.24500451e-17  1.00000000e+00  0.00000000e+00 -8.32667268e-17]
    #  [-4.16333634e-17  0.00000000e+00  1.00000000e+00  2.08166817e-17]
    #  [ 1.04083409e-17 -8.32667268e-17  2.08166817e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -4.16333634e-17  0.00000000e+00  9.19403442e-17
    #    5.20417043e-17]
    #  [-4.16333634e-17  1.00000000e+00  6.93889390e-17 -3.12250226e-17
    #    0.00000000e+00]
    #  [ 0.00000000e+00  6.93889390e-17  1.00000000e+00 -1.43114687e-17
    #   -6.93889390e-17]
    #  [ 9.19403442e-17 -3.12250226e-17 -1.43114687e-17  1.00000000e+00
    #    4.33680869e-18]
    #  [ 5.20417043e-17  0.00000000e+00 -6.93889390e-17  4.33680869e-18
    #    1.00000000e+00]]
    # 

  We can perform "left" orthogonalization for all TT-cores until i-th and "right" orthogonalization for all TT-cores after i-th:

  .. code-block:: python

    i = 2
    Z = teneva.orthogonalize(Y, i)
    
    for G in Z[:i]:
        print(sum([G[:, j, :].T @ G[:, j, :] for j in range(G.shape[1])]))
    
    G = Z[i]
    print('-' * 10 + ' i-th core :')
    print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))
    print('-' * 10)
    
    for G in Z[i+1:]:
        print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[1.00000000e+00 4.16333634e-17]
    #  [4.16333634e-17 1.00000000e+00]]
    # [[ 1.00000000e+00  5.55111512e-17 -5.55111512e-17]
    #  [ 5.55111512e-17  1.00000000e+00 -4.16333634e-17]
    #  [-5.55111512e-17 -4.16333634e-17  1.00000000e+00]]
    # ---------- i-th core :
    # [[20810484.53516915  1635934.46063803  7184477.8245978 ]
    #  [ 1635934.46063803 17777741.01428014   343326.90719904]
    #  [ 7184477.8245978    343326.90719904 15064822.32946539]]
    # ----------
    # [[ 1.00000000e+00  6.24500451e-17 -4.16333634e-17  1.04083409e-17]
    #  [ 6.24500451e-17  1.00000000e+00  0.00000000e+00 -8.32667268e-17]
    #  [-4.16333634e-17  0.00000000e+00  1.00000000e+00  2.08166817e-17]
    #  [ 1.04083409e-17 -8.32667268e-17  2.08166817e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -4.16333634e-17  0.00000000e+00  9.19403442e-17
    #    5.20417043e-17]
    #  [-4.16333634e-17  1.00000000e+00  6.93889390e-17 -3.12250226e-17
    #    0.00000000e+00]
    #  [ 0.00000000e+00  6.93889390e-17  1.00000000e+00 -1.43114687e-17
    #   -6.93889390e-17]
    #  [ 9.19403442e-17 -3.12250226e-17 -1.43114687e-17  1.00000000e+00
    #    4.33680869e-18]
    #  [ 5.20417043e-17  0.00000000e+00 -6.93889390e-17  4.33680869e-18
    #    1.00000000e+00]]
    # 

  We can also set a flag "use_stab", in which case a tensor that is 2^p times smaller than the original tensor will be returned (this allows us to preserve the stability of the operation for essentially multidimensional tensors):

  .. code-block:: python

    Z, p = teneva.orthogonalize(Y, 2, use_stab=True)
    Z = teneva.mul(Z, 2**p)
    eps = teneva.accuracy(Y, Z)
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 


.. autofunction:: teneva.orthogonalize_left

  **Examples**:

  We set the values of parameters and build a random TT-tensor:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    i = d - 2                            # The TT-core for orthogonalization
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  We perform "left" orthogonalization for the i-th TT-core:

  .. code-block:: python

    Z = teneva.orthogonalize_left(Y, i)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 7.48e-09
    # 

  And we can make sure that the updated TT-core have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    G = Z[i]
    print(sum([G[:, j, :].T @ G[:, j, :] for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.00000000e+00 -6.93889390e-17 -7.63278329e-17  5.55111512e-17
    #    1.21430643e-16]
    #  [-6.93889390e-17  1.00000000e+00 -4.16333634e-17  2.77555756e-17
    #    6.93889390e-17]
    #  [-7.63278329e-17 -4.16333634e-17  1.00000000e+00 -3.46944695e-17
    #    0.00000000e+00]
    #  [ 5.55111512e-17  2.77555756e-17 -3.46944695e-17  1.00000000e+00
    #   -3.80554963e-17]
    #  [ 1.21430643e-16  6.93889390e-17  0.00000000e+00 -3.80554963e-17
    #    1.00000000e+00]]
    # 


.. autofunction:: teneva.orthogonalize_right

  **Examples**:

  We set the values of parameters and build a random TT-tensor:

  .. code-block:: python

    d = 5                                # Dimension of the tensor
    n = [12, 13, 14, 15, 16]             # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]               # TT-ranks for TT-tensor
    i = d - 2                            # The TT-core for orthogonalization
    Y = teneva.rand(n, r)                # Build random TT-tensor
    teneva.show(Y)                       # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  We perform "right" orthogonalization for the i-th TT-core:

  .. code-block:: python

    Z = teneva.orthogonalize_right(Y, i)
    teneva.show(Z)

    # >>> ----------------------------------------
    # >>> Output:

    #  12 13 14 15 16 
    #  / \/ \/ \/ \/ \
    #  1  2  3  4  5  1 
    # 
    # 

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    eps = teneva.accuracy(Y, Z)          # The relative difference ("accuracy")
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  And we can make sure that the updated TT-core have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    G = Z[i]
    print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.00000000e+00  1.38777878e-17 -2.42861287e-17  0.00000000e+00]
    #  [ 1.38777878e-17  1.00000000e+00  1.17961196e-16 -5.55111512e-17]
    #  [-2.42861287e-17  1.17961196e-16  1.00000000e+00 -3.46944695e-18]
    #  [ 0.00000000e+00 -5.55111512e-17 -3.46944695e-18  1.00000000e+00]]
    # 


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



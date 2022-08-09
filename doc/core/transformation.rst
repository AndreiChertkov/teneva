Module transformation: orthogonalization, truncation and other transformations of the TT-tensors
------------------------------------------------------------------------------------------------


.. automodule:: teneva.core.transformation


-----


.. autofunction:: teneva.full

  **Examples**:

  .. code-block:: python

    n = [10] * 5             # Shape of the tensor      
    Y0 = np.random.randn(*n) # Create 5-dim random numpy tensor
    Y1 = teneva.svd(Y0)      # Compute TT-tensor from Y0 by TT-SVD
    teneva.show(Y1)          # Print the TT-tensor
    Y2 = teneva.full(Y1)     # Compute full tensor from the TT-tensor
    abs(np.max(Y2-Y0))       # Compare original tensor and reconstructed tensor

    # >>> ----------------------------------------
    # >>> Output:

    #    10  10  10  10  10 
    #   / \ / \ / \ / \ / \ 
    #  1   10 100 100  10  1  
    # 
    # 


.. autofunction:: teneva.orthogonalize

  **Examples**:

  We set the values of parameters and build a random TT-tensor:

  .. code-block:: python

    d = 5                        # Dimension of the tensor
    n = [12, 13, 14, 15, 16]     # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]       # TT-ranks for TT-tensor
    Y = teneva.tensor_rand(n, r) # Build random TT-tensor
    teneva.show(Y)               # Print the resulting TT-tensor

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

    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 1.61e-08
    # 

  And we can make sure that all TT-cores, except the last one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    for G in Z:
        print(sum([G[:, j, :].T @ G[:, j, :] for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.00000000e+00 -2.77555756e-17]
    #  [-2.77555756e-17  1.00000000e+00]]
    # [[1.00000000e+00 6.93889390e-18 8.32667268e-17]
    #  [6.93889390e-18 1.00000000e+00 7.63278329e-17]
    #  [8.32667268e-17 7.63278329e-17 1.00000000e+00]]
    # [[ 1.00000000e+00  3.20923843e-17 -1.04083409e-17  6.93889390e-18]
    #  [ 3.20923843e-17  1.00000000e+00  2.42861287e-17 -5.89805982e-17]
    #  [-1.04083409e-17  2.42861287e-17  1.00000000e+00 -2.08166817e-17]
    #  [ 6.93889390e-18 -5.89805982e-17 -2.08166817e-17  1.00000000e+00]]
    # [[ 1.00000000e+00  2.77555756e-17 -1.25767452e-17  2.77555756e-17
    #    2.77555756e-17]
    #  [ 2.77555756e-17  1.00000000e+00 -6.93889390e-18  0.00000000e+00
    #   -2.77555756e-17]
    #  [-1.25767452e-17 -6.93889390e-18  1.00000000e+00  1.38777878e-17
    #    3.81639165e-17]
    #  [ 2.77555756e-17  0.00000000e+00  1.38777878e-17  1.00000000e+00
    #   -2.08166817e-17]
    #  [ 2.77555756e-17 -2.77555756e-17  3.81639165e-17 -2.08166817e-17
    #    1.00000000e+00]]
    # [[86536252.64677295]]
    # 

  We can also perform "right" orthogonalization for all TT-cores except the first one:

  .. code-block:: python

    Z = teneva.orthogonalize(Y, 0)

  We can verify that the values of the orthogonalized tensor have not changed:

  .. code-block:: python

    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  And we can make sure that all TT-cores, except the first one, have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    for G in Z:
        print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[86536252.64677292]]
    # [[ 1.00000000e+00 -1.38777878e-17]
    #  [-1.38777878e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.03216047e-16 -1.38777878e-17]
    #  [-1.03216047e-16  1.00000000e+00 -1.04083409e-17]
    #  [-1.38777878e-17 -1.04083409e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -6.59194921e-17  1.56125113e-17  4.16333634e-17]
    #  [-6.59194921e-17  1.00000000e+00 -1.38777878e-17  1.38777878e-17]
    #  [ 1.56125113e-17 -1.38777878e-17  1.00000000e+00  4.33680869e-17]
    #  [ 4.16333634e-17  1.38777878e-17  4.33680869e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.56125113e-17 -1.11672824e-16  7.63278329e-17
    #   -4.16333634e-17]
    #  [-1.56125113e-17  1.00000000e+00  6.39679282e-17  4.16333634e-17
    #    1.04083409e-16]
    #  [-1.11672824e-16  6.39679282e-17  1.00000000e+00  4.16333634e-17
    #   -3.46944695e-18]
    #  [ 7.63278329e-17  4.16333634e-17  4.16333634e-17  1.00000000e+00
    #   -5.55111512e-17]
    #  [-4.16333634e-17  1.04083409e-16 -3.46944695e-18 -5.55111512e-17
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

    # [[ 1.00000000e+00 -2.77555756e-17]
    #  [-2.77555756e-17  1.00000000e+00]]
    # [[1.00000000e+00 6.93889390e-18 8.32667268e-17]
    #  [6.93889390e-18 1.00000000e+00 7.63278329e-17]
    #  [8.32667268e-17 7.63278329e-17 1.00000000e+00]]
    # ---------- i-th core :
    # [[23081344.01711273 -5306017.07921848  9384122.28817653]
    #  [-5306017.07921848 32691745.16366326 -9550300.34658735]
    #  [ 9384122.28817653 -9550300.34658735 30763163.46599693]]
    # ----------
    # [[ 1.00000000e+00 -6.59194921e-17  1.56125113e-17  4.16333634e-17]
    #  [-6.59194921e-17  1.00000000e+00 -1.38777878e-17  1.38777878e-17]
    #  [ 1.56125113e-17 -1.38777878e-17  1.00000000e+00  4.33680869e-17]
    #  [ 4.16333634e-17  1.38777878e-17  4.33680869e-17  1.00000000e+00]]
    # [[ 1.00000000e+00 -1.56125113e-17 -1.11672824e-16  7.63278329e-17
    #   -4.16333634e-17]
    #  [-1.56125113e-17  1.00000000e+00  6.39679282e-17  4.16333634e-17
    #    1.04083409e-16]
    #  [-1.11672824e-16  6.39679282e-17  1.00000000e+00  4.16333634e-17
    #   -3.46944695e-18]
    #  [ 7.63278329e-17  4.16333634e-17  4.16333634e-17  1.00000000e+00
    #   -5.55111512e-17]
    #  [-4.16333634e-17  1.04083409e-16 -3.46944695e-18 -5.55111512e-17
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

    d = 5                        # Dimension of the tensor
    n = [12, 13, 14, 15, 16]     # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]       # TT-ranks for TT-tensor
    i = d - 2                    # The TT-core for orthogonalization
    Y = teneva.tensor_rand(n, r) # Build random TT-tensor
    teneva.show(Y)               # Print the resulting TT-tensor

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

    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 0.00e+00
    # 

  And we can make sure that the updated TT-core have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    G = Z[i]
    print(sum([G[:, j, :].T @ G[:, j, :] for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.00000000e+00  2.08166817e-17 -3.46944695e-17 -5.89805982e-17
    #    1.73472348e-17]
    #  [ 2.08166817e-17  1.00000000e+00 -1.73472348e-17 -2.77555756e-17
    #   -3.81639165e-17]
    #  [-3.46944695e-17 -1.73472348e-17  1.00000000e+00  6.93889390e-18
    #   -2.08166817e-17]
    #  [-5.89805982e-17 -2.77555756e-17  6.93889390e-18  1.00000000e+00
    #    5.98479599e-17]
    #  [ 1.73472348e-17 -3.81639165e-17 -2.08166817e-17  5.98479599e-17
    #    1.00000000e+00]]
    # 


.. autofunction:: teneva.orthogonalize_right

  **Examples**:

  We set the values of parameters and build a random TT-tensor:

  .. code-block:: python

    d = 5                        # Dimension of the tensor
    n = [12, 13, 14, 15, 16]     # Shape of the tensor
    r = [1, 2, 3, 4, 5, 1]       # TT-ranks for TT-tensor
    i = d - 2                    # The TT-core for orthogonalization
    Y = teneva.tensor_rand(n, r) # Build random TT-tensor
    teneva.show(Y)               # Print the resulting TT-tensor

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

    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 1.17e-08
    # 

  And we can make sure that the updated TT-core have become orthogonalized (in terms of the TT-format):

  .. code-block:: python

    G = Z[i]
    print(sum([G[:, j, :] @ G[:, j, :].T for j in range(G.shape[1])]))

    # >>> ----------------------------------------
    # >>> Output:

    # [[ 1.00000000e+00  0.00000000e+00 -4.33680869e-17  9.71445147e-17]
    #  [ 0.00000000e+00  1.00000000e+00 -1.12757026e-17 -6.93889390e-18]
    #  [-4.33680869e-17 -1.12757026e-17  1.00000000e+00 -4.51028104e-17]
    #  [ 9.71445147e-17 -6.93889390e-18 -4.51028104e-17  1.00000000e+00]]
    # 


.. autofunction:: teneva.truncate

  **Examples**:

  .. code-block:: python

    # 10-dim random TT-tensor with TT-rank 3:
    Y = teneva.tensor_rand([5]*10, 3)
    
    # Compute Y + Y + Y (the real TT-rank is still 3):
    Y = teneva.add(Y, teneva.add(Y, Y))
    
    # Print the resulting TT-tensor
    # (note that it has TT-rank 3 + 3 + 3 = 9):
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  9  9  9  9  9  9  9  9  9  1 
    # 
    # 

  .. code-block:: python

    # Truncate (round) the TT-tensor:
    Z = teneva.truncate(Y, e=1.E-2)
    
    # Print the resulting TT-tensor (note that it has TT-rank 3):
    teneva.show(Z)
    
    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # Accuracy     : 0.00e+00
    # 

  We can also specify the desired TT-rank of truncated TT-tensor:

  .. code-block:: python

    # Truncate (round) the TT-tensor:
    Z = teneva.truncate(Y, e=1.E-6, r=3)
    
    # Print the resulting TT-tensor (note that it has TT-rank 3):
    teneva.show(Z)
    
    # The relative difference ("accuracy"):
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  3  3  3  3  3  3  3  3  3  1 
    # 
    # Accuracy     : 0.00e+00
    # 

  If we choose a lower TT-rank value, then precision will be (predictably) lost:

  .. code-block:: python

    # Truncate (round) the TT-tensor:
    Z = teneva.truncate(Y, e=1.E-6, r=2)
    
    # Print the resulting TT-tensor (note that it has TT-rank 2):
    teneva.show(Z)
    
    # The relative difference ("accuracy")
    eps = teneva.accuracy(Y, Z)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    #   5  5  5  5  5  5  5  5  5  5 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  1 
    # 
    # Accuracy     : 9.82e-01
    # 



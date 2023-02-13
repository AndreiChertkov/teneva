Module svd: SVD-based algorithms for matrices and tensors
---------------------------------------------------------


.. automodule:: teneva.core.svd


-----




|
|

.. autofunction:: teneva.matrix_skeleton

  **Examples**:

  .. code-block:: python

    # Shape of the matrix:
    m, n = 100, 30
    
    # Build random matrix, which has rank 3,
    # as a sum of rank-1 matrices:
    A = np.outer(np.random.randn(m), np.random.randn(n))
    A += np.outer(np.random.randn(m), np.random.randn(n)) 
    A += np.outer(np.random.randn(m), np.random.randn(n))

  .. code-block:: python

    # Compute skeleton decomp.:
    U, V = teneva.matrix_skeleton(A, e=1.E-10)
    
    # Approximation error
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 9.00e-16
    # 

  .. code-block:: python

    # Compute skeleton decomp with small rank:
    U, V = teneva.matrix_skeleton(A, r=2)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 30)
    # Error      : 4.60e-01
    # 




|
|

.. autofunction:: teneva.matrix_svd

  **Examples**:

  .. code-block:: python

    # Shape of the matrix:
    m, n = 100, 30
    
    # Build random matrix, which has rank 3,
    # as a sum of rank-1 matrices:
    A = np.outer(np.random.randn(m), np.random.randn(n))
    A += np.outer(np.random.randn(m), np.random.randn(n)) 
    A += np.outer(np.random.randn(m), np.random.randn(n))

  .. code-block:: python

    # Compute SVD-decomp.:
    U, V = teneva.matrix_svd(A, e=1.E-10)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 17)
    # Shape of V : (17, 30)
    # Error      : 6.73e-16
    # 

  .. code-block:: python

    # Compute SVD-decomp.:
    U, V = teneva.matrix_svd(A, r=3)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 6.64e-16
    # 

  .. code-block:: python

    # Compute SVD-decomp.:
    U, V = teneva.matrix_svd(A, e=1.E-2)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 6.64e-16
    # 

  .. code-block:: python

    # Compute SVD-decomp.:
    U, V = teneva.matrix_svd(A, r=2)
    
    # Approximation error:
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)
    
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 30)
    # Error      : 4.37e-01
    # 




|
|

.. autofunction:: teneva.svd

  **Examples**:

  .. code-block:: python

    d = 20              # Dimension number
    t = np.arange(2**d) # Tensor will be 2^d
    
    # Construct d-dim full array:
    Z_full = np.cos(t).reshape([2] * d, order='F')

  .. code-block:: python

    # Construct TT-tensor by TT-SVD:
    Y = teneva.svd(Z_full)
    
    # Convert it back to numpy to check result:
    Y_full = teneva.full(Y)
    
    # Compute error for TT-tensor vs full tensor:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)

  .. code-block:: python

    print(f'Size (np) : {Z_full.size:-8d}')       # Size of original tensor
    print(f'Size (tt) : {teneva.size(Y):-8d}')    # Size of the TT-tensor
    print(f'Erank     : {teneva.erank(Y):-8.2f}') # Eff. rank of the TT-tensor
    print(f'Error     : {e:-8.2e}')               # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :  1048576
    # Size (tt) :      152
    # Erank     :     2.00
    # Error     : 1.84e-14
    # 




|
|

.. autofunction:: teneva.svd_matrix

  **Examples**:

  .. code-block:: python

    q = 10   # Matrix size factor
    n = 2**q # Matrix mode size
    
    # Construct some matrix:
    Z_full = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z_full[i, j] = np.cos(i) * j**2

  .. code-block:: python

    # Construct QTT-matrix / TT-tensor by TT-SVD:
    Y = teneva.svd_matrix(Z_full, e=1.E-6)
    
    # Convert it back to numpy to check result:
    Y_full = teneva.full_matrix(Y)
    
    # Compute error for QTT-matrix / TT-tensor vs full matrix:
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)

  .. code-block:: python

    print(f'Size (np) : {Z_full.size:-8d}')       # Size of original tensor
    print(f'Size (tt) : {teneva.size(Y):-8d}')    # Size of the QTT-matrix
    print(f'Erank     : {teneva.erank(Y):-8.2f}') # Eff. rank of the QTT-matrix
    print(f'Error     : {e:-8.2e}')               # Rel. error for QTT-matrix vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :  1048576
    # Size (tt) :     1088
    # Erank     :     5.71
    # Error     : 3.64e-12
    # 




|
|

.. autofunction:: teneva.svd_incomplete

  **Examples**:

  .. code-block:: python

    d = 20              # Dimension number
    n = [2] * d         # Shape of the tensor/grid
    t = np.arange(2**d) # Tensor will be 2^d
    
    # Construct d-dim full array:
    Z_full = np.cos(t).reshape([2] * d, order='F')

  .. code-block:: python

    m = 4 # The expected TT-rank
    
    # Generate special samples (indices) for the tensor:
    I_trn, idx, idx_many = teneva.sample_tt(n, m)

  .. code-block:: python

    # Compute tensor values in I multiindices:
    Y_trn = np.array([Z_full[tuple(i)] for i in I_trn])

  .. code-block:: python

    Y = teneva.svd_incomplete(I_trn, Y_trn,
        idx, idx_many, e=1.E-10, r=3) # Construct TT-tensor
    teneva.show(Y)                    # Show the tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor    20D : |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2| |2|
    # <rank>  =    2.0 :   \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/ \2/
    # 

  .. code-block:: python

    # Convert it back to numpy to check result:
    Y_full = teneva.full(Y)                          
    
    # Compute error for TT-tensor vs full tensor :
    e = np.linalg.norm(Y_full - Z_full)
    e /= np.linalg.norm(Z_full)

  .. code-block:: python

    print(f'Size (np) : {Z_full.size:-8d}')       # Size of original tensor
    print(f'Size (tt) : {teneva.size(Y):-8d}')    # Size of the TT-tensor
    print(f'Erank     : {teneva.erank(Y):-8.2f}') # Eff. rank of the TT-tensor
    print(f'Error     : {e:-8.2e}')               # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :  1048576
    # Size (tt) :      152
    # Erank     :     2.00
    # Error     : 2.50e-15
    # 




|
|


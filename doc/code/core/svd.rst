svd: SVD-based algorithms for matrices and tensors
--------------------------------------------------


.. automodule:: teneva.core.svd

---


.. autofunction:: teneva.core.svd.matrix_skeleton

  **Examples**:

  .. code-block:: python

    m, n = 100, 30                                        # Shape of the matrix
    A = np.outer(np.random.randn(m), np.random.randn(n))  # Build random matrix,
    A += np.outer(np.random.randn(m), np.random.randn(n)) # which has rank 3,
    A += np.outer(np.random.randn(m), np.random.randn(n)) # as a sum of rank-1 matrices

  .. code-block:: python

    U, V = teneva.matrix_skeleton(A, e=1.E-10)            # Compute skeleton decomp.
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 8.92e-16
    # 

  .. code-block:: python

    U, V = teneva.matrix_skeleton(A, r=2)                 # Compute skeleton decomp with small rank
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 30)
    # Error      : 4.37e-01
    # 

---


.. autofunction:: teneva.core.svd.matrix_svd

  **Examples**:

  .. code-block:: python

    m, n = 100, 30                                        # Shape of the matrix
    A = np.outer(np.random.randn(m), np.random.randn(n))  # Build random matrix,
    A += np.outer(np.random.randn(m), np.random.randn(n)) # which has rank 3,
    A += np.outer(np.random.randn(m), np.random.randn(n)) # as a sum of rank-1 matrices

  .. code-block:: python

    U, V = teneva.matrix_svd(A, e=1.E-10)                 # Compute SVD-decomp.
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 17)
    # Shape of V : (17, 30)
    # Error      : 4.98e-16
    # 

  .. code-block:: python

    U, V = teneva.matrix_svd(A, r=3)                      # Compute SVD-decomp.
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 4.78e-16
    # 

  .. code-block:: python

    U, V = teneva.matrix_svd(A, e=1.E-2)                  # Compute SVD-decomp.
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 3)
    # Shape of V : (3, 30)
    # Error      : 4.78e-16
    # 

  .. code-block:: python

    U, V = teneva.matrix_svd(A, r=2)                      # Compute SVD-decomp.
    e = np.linalg.norm(A - U @ V) / np.linalg.norm(A)     # Approximation error
    print(f'Shape of U :', U.shape)
    print(f'Shape of V :',V.shape)
    print(f'Error      : {e:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape of U : (100, 2)
    # Shape of V : (2, 30)
    # Error      : 3.79e-01
    # 

---


.. autofunction:: teneva.core.svd.svd

  **Examples**:

  .. code-block:: python

    d = 20                                          # Dimension number
    t = np.arange(2**d)                             # Tensor will be 2^d
    Z_full = np.cos(t).reshape([2] * d, order='F')  # Construct d-dim full array

  .. code-block:: python

    Y = teneva.svd(Z_full)                          # Construct TT-tensor by TT-SVD
    Y_full = teneva.full(Y)                         # Convert it back to numpy to chech result
    e = np.linalg.norm(Y_full - Z_full)             # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)                     #

  .. code-block:: python

    print(f'Size (np) : {Z_full.size:-8d}')         # Size of original tensor
    print(f'Size (tt) : {teneva.size(Y):-8d}')      # Size of the TT-tensor
    print(f'Erank     : {teneva.erank(Y):-8.2f}')   # Eff. rank of the TT-tensor
    print(f'Error     : {e:-8.2e}')                 # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :  1048576
    # Size (tt) :      152
    # Erank     :     2.00
    # Error     : 1.58e-14
    # 

---


.. autofunction:: teneva.core.svd.svd_incomplete

  **Examples**:

  .. code-block:: python

    d = 20                                              # Dimension number
    n = [2] * d                                         # Shape of the tensor/grid
    t = np.arange(2**d)                                 # Tensor will be 2^d
    Z_full = np.cos(t).reshape([2] * d, order='F')      # Construct d-dim full array

  .. code-block:: python

    m = 4                                               # The expected TT-rank
    I_trn, idx, idx_many = teneva.sample_tt(n, m)       # Generate special samples (indices) for the tensor

  .. code-block:: python

    Y_trn = np.array([Z_full[tuple(i)] for i in I_trn]) # Compute tensor values in I multiindices

  .. code-block:: python

    Y = teneva.svd_incomplete(I_trn, Y_trn,
        idx, idx_many, e=1.E-10, r=3)                   # Construct TT-tensor
    teneva.show(Y)                                      # Show the tensor

    # >>> ----------------------------------------
    # >>> Output:

    #   2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 
    #  / \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \
    #  1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  1 
    # 
    # 

  .. code-block:: python

    Y_full = teneva.full(Y)                          # Convert it back to numpy to chech result
    e = np.linalg.norm(Y_full - Z_full)              # Compute error for TT-tensor vs full tensor 
    e /= np.linalg.norm(Z_full)                      #

  .. code-block:: python

    print(f'Size (np) : {Z_full.size:-8d}')          # Size of original tensor
    print(f'Size (tt) : {teneva.size(Y):-8d}')       # Size of the TT-tensor
    print(f'Erank     : {teneva.erank(Y):-8.2f}')    # Eff. rank of the TT-tensor
    print(f'Error     : {e:-8.2e}')                  # Rel. error for TT-tensor vs full tensor

    # >>> ----------------------------------------
    # >>> Output:

    # Size (np) :  1048576
    # Size (tt) :      152
    # Erank     :     2.00
    # Error     : 2.25e-15
    # 

---

Module core: operations with individual TT-cores
------------------------------------------------


.. automodule:: teneva.core.core


-----


.. autofunction:: teneva.core_qtt_to_tt

  **Examples**:

  .. code-block:: python

    # TT-ranks for cores:
    r_list = [4, 3, 5, 8, 18, 2, 4, 3]
    
    # Create random QTT-cores:
    Q_list = []
    for i in range(1, len(r_list)):
        Q = np.random.randn(r_list[i-1], 2, r_list[i]) 
        Q_list.append(Q)
    
    # Transform the QTT-cores into one TT-core:
    G = teneva.core_qtt_to_tt(Q_list)
    
    print(f'Shape : {G.shape}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape : (4, 128, 3)
    # 


.. autofunction:: teneva.core_stab

  **Examples**:

  .. code-block:: python

    r = 4   # Left TT-rank
    n = 10  # Mode size
    q = 5   # Right TT-rank
    
    # Create random TT-core:
    G = np.random.randn(r, n, q)
    
    # Perform scaling:
    Q, p = teneva.core_stab(G)
    
    print(p)
    print(np.max(np.abs(Q)))
    print(np.max(np.abs(G - 2**p * Q)))

    # >>> ----------------------------------------
    # >>> Output:

    # 1
    # 1.3484433214707858
    # 0.0
    # 

  For convenience, we can set an initial value for the power-factor:

  .. code-block:: python

    p0 = 2
    Q, p = teneva.core_stab(G, p0)
    
    print(p)
    print(np.max(np.abs(Q)))
    print(np.max(np.abs(G - 2**(p-p0) * Q)))

    # >>> ----------------------------------------
    # >>> Output:

    # 3
    # 1.3484433214707858
    # 0.0
    # 


.. autofunction:: teneva.core_tt_to_qtt

  **Examples**:

  .. code-block:: python

    r = 3      # Left TT-rank
    n = 2**10  # Mode size
    q = 5      # Right TT-rank
    
    # Create random TT-core:
    G = np.random.randn(r, n, q)
    
    # Transform the core to QTT:
    Q_list = teneva.core_tt_to_qtt(G)
    
    print('Len  : ', len(Q_list))
    print('Q  1 : ', Q_list[0].shape)
    print('Q  2 : ', Q_list[1].shape)
    print('Q 10 : ', Q_list[-1].shape)

    # >>> ----------------------------------------
    # >>> Output:

    # Len  :  10
    # Q  1 :  (3, 2, 6)
    # Q  2 :  (6, 2, 12)
    # Q 10 :  (10, 2, 5)
    # 

  We can check the result if transform the list of the QTT-cores back:

  .. code-block:: python

    G_new = teneva.core_qtt_to_tt(Q_list)
    
    eps = np.max(np.abs(G_new - G))
    
    print(f'Shape : {G_new.shape}')
    print(f'Eps   : {eps:-7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Shape : (3, 1024, 5)
    # Eps   : 1.8e-14
    # 



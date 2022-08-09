Module core: operations with individual TT-cores
------------------------------------------------


.. automodule:: teneva.core.core


-----


.. autofunction:: teneva.core_stab

  **Examples**:

  .. code-block:: python

    r = 4                            # Left TT-rank
    n = 10                           # Mode size
    q = 5                            # Right TT-rank
    G = np.random.randn(r, n, q)     # Create random TT-core
    
    Q, p = teneva.core_stab(G)       # Perform scaling
    
    print(p)
    print(np.max(np.abs(Q)))
    print(np.max(np.abs(G - 2**p * Q)))

    # >>> ----------------------------------------
    # >>> Output:

    # 1
    # 1.3600845832948094
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
    # 1.3600845832948094
    # 0.0
    # 


.. autofunction:: teneva.core_tt_to_qtt

  **Examples**:

  .. code-block:: python

    r = 4                            # Left TT-rank
    n = 2**10                        # Mode size
    q = 5                            # Right TT-rank
    G = np.random.randn(r, n, q)     # Create random TT-core
    
    Y = teneva.core_tt_to_qtt(G)     # Transform the core to QTT
    
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    #    2   2   2   2   2   2   2   2   2   2  
    #   / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ 
    #  1   8   16  32  64 128  80  40  20  10  5  
    # 
    # 

  .. code-block:: python

    for G in Y:
        print(G.shape)

    # >>> ----------------------------------------
    # >>> Output:

    # (4, 2, 8)
    # (8, 2, 16)
    # (16, 2, 32)
    # (32, 2, 64)
    # (64, 2, 128)
    # (128, 2, 80)
    # (80, 2, 40)
    # (40, 2, 20)
    # (20, 2, 10)
    # (10, 2, 5)
    # 



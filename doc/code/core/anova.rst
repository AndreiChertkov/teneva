anova: construct TT-tensor by TT-ANOVA
--------------------------------------


.. automodule:: teneva.core.anova

---


.. autofunction:: teneva.core.anova.anova

  **Examples**:

  .. code-block:: python

    d         = 10          # Dimension of the function
    A         = [-5.] * d   # Lower bound for spatial grid
    B         = [+5.] * d   # Upper bound for spatial grid
    N         = [10] * d    # Shape of the tensor (it may be non-uniform)
    M_tst     = 10000       # Number of test points

  .. code-block:: python

    evals     = 10000       # Number of calls to target function
    order     = 1           # Order of ANOVA decomposition (1 or 2)
    r         = 3           # TT-rank of the resulting tensor

  .. code-block:: python

    # Target function:
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind2poi(I, A, B, N)
        return rosen(X.T)

  .. code-block:: python

    # Train data:
    
    I_trn = teneva.sample_lhs(N, evals) 
    Y_trn = func(I_trn)

  .. code-block:: python

    # Test data:
    
    I_tst = np.vstack([np.random.choice(N[i], M_tst) for i in range(d)]).T
    Y_tst = func(I_tst)

  .. code-block:: python

    # Build tensor:
    
    t = tpc()
    Y = teneva.anova(I_trn, Y_trn, r, order)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :       0.01
    # 

  .. code-block:: python

    # Check result:
    
    get = teneva.getter(Y)
    
    Z = np.array([get(i) for i in I_trn])
    e_trn = np.linalg.norm(Z - Y_trn) / np.linalg.norm(Y_trn)
    
    Z = np.array([get(i) for i in I_tst])
    e_tst = np.linalg.norm(Z - Y_tst) / np.linalg.norm(Y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   9.59e-02
    # Error on test  :   9.68e-02
    # 

---

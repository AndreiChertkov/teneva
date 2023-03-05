Module als: construct TT-tensor by TT-ALS
-----------------------------------------


.. automodule:: teneva.core_jax.als


-----




|
|

.. autofunction:: teneva.core_jax.als.als

  **Examples**:

  .. code-block:: python

    d = 50             # Dimension of the function
    n = 20             # Shape of the tensor
    r = 3              # TT-rank of the initial random tensor
    nswp = 50          # Sweep number for ALS iterations
    m = int(1.E+4)     # Number of calls to target function
    m_tst = int(1.E+4) # Number of test points

  We set the target function (the function takes as input a multi-index i of the shape [dimension], which is transformed into point x of a uniform spatial grid):

  .. code-block:: python

    a = -2.048 # Lower bound for the spatial grid
    b = +2.048 # Upper bound for the spatial grid
    
    def func_base(i):
        """Michalewicz function."""
        x = i / n * (b - a) + a
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return np.sum(y1 + y2)
    
        y1 = np.sin(((np.arange(d) + 1) * x**2 / np.pi))
        return -np.sum(np.sin(x) * y1**(2 * 10))
    
    func = jax.vmap(func_base)

  We prepare train data from the LHS random distribution:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    I_trn = teneva.sample_lhs(d, n, m, key)
    y_trn = func(I_trn)

  We prepare test data from a random tensor multi-indices:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    I_tst = teneva.sample_rand(d, n, m_tst, key)
    y_tst = func(I_tst)

  We build the initial approximation by the TT-ANOVA method:

  .. code-block:: python

    # TODO: replace with jax-version!
    Y_anova_base = teneva_base.anova(I_trn, y_trn, r)
    Y_anova = teneva.convert(Y_anova_base)

  And now we will build the TT-tensor, which approximates the target function by the TT-ALS method:

  .. code-block:: python

    t = tpc()
    Y = teneva.als(I_trn, y_trn, Y_anova, nswp)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :      11.86
    # 

  We can check the accuracy of the result:

  .. code-block:: python

    # Compute approximation in train points:
    y_our = teneva.get_many(Y, I_trn)
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    # Compute approximation in test points:
    y_our = teneva.get_many(Y, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on train :   4.24e-02
    # Error on test  :   4.24e+05
    # 

  We can compare the result with the base (numpy) ALS method (we run it on the same train data with the same initial approximation and parameters):

  .. code-block:: python

    t = tpc()
    Y = teneva_base.als(I_trn, y_trn, Y_anova_base, nswp, e=-1.)
    t = tpc() - t
    
    print(f'Build time     : {t:-10.2f}')
    
    # Compute approximation in train points:
    y_our = teneva_base.get_many(Y, I_trn)
    
    # Accuracy of the result for train points:
    e_trn = np.linalg.norm(y_our - y_trn)          
    e_trn /= np.linalg.norm(y_trn)
    
    # Compute approximation in test points:
    y_our = teneva_base.get_many(Y, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst)          
    e_tst /= np.linalg.norm(y_tst)
    
    print(f'Error on train : {e_trn:-10.2e}')
    print(f'Error on test  : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time     :      19.64
    # Error on train :   2.05e-02
    # Error on test  :   3.26e-01
    # 




|
|


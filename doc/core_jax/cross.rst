Module cross: construct TT-tensor by TT-cross
---------------------------------------------


.. automodule:: teneva.core_jax.cross


-----




|
|

.. autofunction:: teneva.core_jax.cross.cross

  **Examples**:

  .. code-block:: python

    d = 10             # Dimension of the function
    n = 5              # Shape of the tensor
    r = 3              # TT-rank of the initial random tensor
    nswp = 5           # Sweep number for TT-cross iterations
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
    
    func = jax.jit(jax.vmap(func_base))

  We prepare test data from random tensor multi-indices:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    I_tst = teneva.sample_rand(d, n, m_tst, key)
    y_tst = func(I_tst)

  We build a random initial approximation:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y0 = teneva.rand(d, n, r, key)

  And now we run the TT-cross method:

  .. code-block:: python

    from functools import partial
    
    
    def cross(f, Y0, nswp=10):
        Y = teneva.copy(Y0)
        d = len(Y[1]) + 2
        n = Y[0].shape[1]
    
        Ir = [np.zeros((1, 0)) for i in range(d+1)]
        Ic = [np.zeros((1, 0)) for i in range(d+1)]
    
        Y = teneva.convert(Y)
    
        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            G = np.tensordot(Y[i], R, 1)
            Y[i], R, Ic[i] = _iter_rtl(G, Ic[i+1])
        Y[0] = np.tensordot(R, Y[0], 1)
    
        for fff in Ic:
            print(fff)
        Icl = Ic[1]
        Icm = np.vstack(Ic[2:-1])
        
        @partial(jax.jit, static_argnums=[2])
        def _func(Ir, Ic, ig):
            n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
            I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
            I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
            I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
            return np.reshape(f(I), (r1, n, r2), order='F')
    
        @jax.jit
        def _iter_ltr_body(args, Ic):
            R, Ir, ig = args
            #Z = _func(Ir, Ic, ig)
            
            n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
            I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
            I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
            I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
            Z = np.reshape(f(I), (r1, n, r2), order='F')
        
            G, R, Ir = _iter_ltr(Z, Ir)
            return (R, Ir, ig), (G, Ir)
        
        @jax.jit
        def _iter_rtl_body(args, Ir):
            R, Ic, ig = args
            #Z = _func(Ir, Ic, ig)
            
            n, r1, r2 = ig.shape[0], Ir.shape[0], Ic.shape[0]
            I = np.kron(np.kron(np.ones(r2), ig), np.ones(r1)).reshape((-1,1))
            I = np.hstack((np.kron(np.ones((n*r2, 1)), Ir), I))
            I = np.hstack((I, np.kron(Ic, np.ones((r1*n, 1)))))
            Z = np.reshape(f(I), (r1, n, r2), order='F')
            
            G, R, Ic = _iter_rtl(Z, Ic)
            return (R, Ic, ig), (G, Ic)
    
        ig = np.arange(n)
    
        for _ in range(nswp):
            (R, _, _), (Yl, Irl) = _iter_ltr_body(
                (None, np.zeros((1, 0)), ig), Icl)
            (R, _, _), (Ym, Irm) = jax.lax.scan(_iter_ltr_body,
                (R, Irl, ig), Icm)
            Irm, Irr = _shift_ltr(Irl, Irm)
            (R, _, _), (Yr, _) = _iter_ltr_body(
                (R, Irr, ig), np.zeros((1, 0)))
            Yr = np.tensordot(Yr, R, 1)
            
            (R, _, _), (Yr, Icr) = _iter_rtl_body(
                (None, Irr, ig), np.zeros((1, 0)))
            (R, _, _), (Ym, Icm) = jax.lax.scan(_iter_rtl_body,
                (R, Icr, ig), Irm, reverse=True)
            Icl, Icm = _shift_rtl(Icm, Icr)
            (R, _, _), (Yl, _) = _iter_rtl_body(
                (R, Icl, ig), np.zeros((1, 0)))
            Yl = np.tensordot(R, Yl, 1)
            
        import numpy as onp
        Y = [onp.array(G) for G in Y]
        return teneva.convert(Y)
    
    
    def _iter_ltr(Z, Ir):
        r1, n, r2 = Z.shape
    
        I = np.kron(np.arange(n), np.ones(r1)).reshape((-1,1))
        I = np.hstack((np.kron(np.ones((n, 1)), Ir), I))
    
        Q, R = np.linalg.qr(np.reshape(Z, (r1 * n, r2), order='F'))
        ind, B = teneva.maxvol(Q)
        G = np.reshape(B, (r1, n, -1), order='F')
        R = Q[ind, :] @ R
    
        return G, R, I[ind, :]
    
    
    def _iter_rtl(Z, Il):
        r1, n, r2 = Z.shape
    
        I = np.kron(np.ones(r2), np.arange(n)).reshape((-1,1))
        I = np.hstack((I, np.kron(Il, np.ones((n, 1)))))
    
        Q, R = np.linalg.qr(np.reshape(Z, (r1, n * r2), order='F').T)
        ind, B = teneva.maxvol(Q)
        G = np.reshape(B.T, (-1, n, r2), order='F')
        R = (Q[ind, :] @ R).T
    
        return G, R, I[ind, :]
    
    
    @jax.jit
    def _shift_ltr(Zl_ltr, Zm_ltr):
        return np.vstack((Zl_ltr[None], Zm_ltr[:-1])), Zm_ltr[-1]
    
    
    @jax.jit
    def _shift_rtl(Zm_rtl, Zr_rtl):
        return Zm_rtl[0], np.vstack((Zm_rtl[1:], Zr_rtl[None]))

  .. code-block:: python

    t = tpc()
    Y = cross(func, Y0, nswp)
    t = tpc() - t
    
    print(f'Build time           : {t:-10.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # [[0. 3. 4. 0. 3. 2. 2. 2. 0. 4.]]
    # [[3. 4. 0. 3. 2. 2. 2. 0. 4.]
    #  [0. 4. 0. 3. 2. 2. 2. 0. 4.]
    #  [3. 1. 0. 3. 2. 2. 2. 0. 4.]]
    # [[4. 0. 3. 2. 2. 2. 0. 4.]
    #  [0. 4. 3. 2. 2. 2. 0. 4.]
    #  [1. 0. 3. 2. 2. 2. 0. 4.]]
    # [[4. 3. 2. 2. 2. 0. 4.]
    #  [3. 3. 2. 2. 2. 0. 4.]
    #  [0. 3. 2. 2. 2. 0. 4.]]
    # [[0. 0. 1. 2. 0. 4.]
    #  [3. 2. 2. 2. 0. 4.]
    #  [4. 2. 2. 2. 0. 4.]]
    # [[0. 1. 2. 0. 4.]
    #  [2. 2. 2. 0. 4.]
    #  [4. 2. 2. 0. 4.]]
    # [[2. 2. 0. 4.]
    #  [0. 4. 4. 1.]
    #  [1. 2. 0. 4.]]
    # [[2. 0. 4.]
    #  [4. 4. 1.]
    #  [4. 1. 2.]]
    # [[0. 4.]
    #  [4. 1.]
    #  [1. 2.]]
    # [[2.]
    #  [4.]
    #  [1.]]
    # []
    # 

  Then we can check the result:

  .. code-block:: python

    # Compute approximation in test points:
    y_our = teneva.get_many(Y, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    print(f'Error on test        : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error on test        :   2.72e-15
    # 

  We can compare the result with the base (numpy) TT-cross method (we run it with the same initial approximation and parameters):

  .. code-block:: python

    t = tpc()
    info, cache = {}, {}
    Y0_base = teneva.convert(Y0)
    Y_base = teneva_base.cross(func, Y0_base, nswp=nswp, dr_max=0, info=info)
    t = tpc() - t
    
    y_our = teneva_base.get_many(Y_base, I_tst)
    
    # Accuracy of the result for test points:
    e_tst = np.linalg.norm(y_our - y_tst) / np.linalg.norm(y_tst)
    
    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Error on test        : {e_tst:-10.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Build time           :       0.18
    # Evals func           :       3900
    # Iter accuracy        :   2.52e-08
    # Error on test        :   1.46e-15
    # 




|
|


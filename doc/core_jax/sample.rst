Module sample: random sampling for/from the TT-tensor
-----------------------------------------------------


.. automodule:: teneva.core_jax.sample


-----




|
|

.. autofunction:: teneva.core_jax.sample.sample

  **Examples**:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=8, n=5, r=4, key=key)
    zl, zm = teneva.interface_rtl(Y)
    
    rng, key = jax.random.split(rng)
    i = teneva.sample(Y, zm, key)
    print(i)

    # >>> ----------------------------------------
    # >>> Output:

    # [2 1 4 0 4 1 3 0]
    # 

  And now let check this function for big random TT-tensor:

  .. code-block:: python

    interface_rtl = jax.jit(teneva.interface_rtl)
    sample = jax.jit(jax.vmap(teneva.sample, (None, None, 0)))

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = teneva.rand(d=1000, n=100, r=10, key=key)

  .. code-block:: python

    zl, zm = interface_rtl(Y)
    
    m = 10  # Number of samples
    rng, key = jax.random.split(rng)
    I = sample(Y, zm, jax.random.split(key, m))
    
    for i in I: # i is a sample of the length d = 1000
        print(len(i), np.mean(i))

    # >>> ----------------------------------------
    # >>> Output:

    # 1000 49.323
    # 1000 50.120003
    # 1000 48.634003
    # 1000 50.989002
    # 1000 49.136
    # 1000 48.072002
    # 1000 49.901
    # 1000 49.281002
    # 1000 50.443
    # 1000 50.769
    # 

  Let compare this function with numpy realization:

  .. code-block:: python

    d = 25       # Dimension of the tensor
    n = 10       # Mode size of the tensor
    r = 5        # Rank of the tensor
    m = 100000   # Number of samples

  .. code-block:: python

    Y_base = teneva_base.rand([n]*d, r)

  .. code-block:: python

    t = tpc()
    I_base = teneva_base.sample(Y_base, m)
    t = tpc() - t
    
    print(f'Time : {t:-8.2f}')
    print(f'Mean : {np.mean(I_base):-8.2f}')
    print(f'Var  : {np.var(I_base):-8.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Time :    53.21
    # Mean :     4.38
    # Var  :     8.40
    # 

  .. code-block:: python

    Y = teneva.convert(Y_base) # Convert it to the jax version

  .. code-block:: python

    t = tpc()
    interface_rtl = jax.jit(teneva.interface_rtl)
    sample = jax.jit(jax.vmap(teneva.sample, (None, None, 0)))
    
    zl, zm = interface_rtl(Y)
    rng, key = jax.random.split(rng)
    I = sample(Y, zm, jax.random.split(key, m))
    t = tpc() - t
    
    print(f'Time : {t:-8.2f}')
    print(f'Mean : {np.mean(I):-8.2f}')
    print(f'Var  : {np.var(I):-8.2f}')

    # >>> ----------------------------------------
    # >>> Output:

    # Time :     1.53
    # Mean :     4.42
    # Var  :     8.30
    # 




|
|


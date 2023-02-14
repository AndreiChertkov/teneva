Module tensors: collection of explicit useful TT-tensors
--------------------------------------------------------


.. automodule:: teneva.core_jax.tensors


-----




|
|

.. autofunction:: teneva.core_jax.tensors.rand

  **Examples**:

  .. code-block:: python

    d = 6                            # Dimension of the tensor
    n = 5                            # Shape of the tensor
    r = 4                            # TT-rank for the TT-tensor
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key)    # Build the random TT-tensor
    teneva.show(Y)                   # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # 

  We may use custom limits:

  .. code-block:: python

    d = 6                            # Dimension of the tensor
    n = 5                            # Shape of the tensor
    r = 4                            # TT-rank for the TT-tensor
    a = 0.99                         # Minimum value
    b = 1.                           # Maximum value
    rng, key = jax.random.split(rng)
    Y = teneva.rand(d, n, r, key, a, b)
    print(Y[0])                      # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # [[[0.99508137 0.99743134 0.9958516  0.99606663]
    #   [0.9949319  0.9946317  0.99657923 0.9994617 ]
    #   [0.9993697  0.9918885  0.995916   0.9996112 ]
    #   [0.9966089  0.99671346 0.999126   0.9900023 ]
    #   [0.9951171  0.99752164 0.9920615  0.99856436]]]
    # 




|
|

.. autofunction:: teneva.core_jax.tensors.rand_norm

  **Examples**:

  .. code-block:: python

    d = 6                               # Dimension of the tensor
    n = 5                               # Shape of the tensor
    r = 4                               # TT-rank for the TT-tensor
    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d, n, r, key)  # Build the random TT-tensor
    teneva.show(Y)                      # Print the resulting TT-tensor

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor-jax | d =     6 | n =     5 | r =     4 |
    # 

  We may use custom limits:

  .. code-block:: python

    d = 6                               # Dimension of the tensor
    n = 5                               # Shape of the tensor
    r = 4                               # TT-rank for the TT-tensor
    m = 42.                             # Mean ("centre")
    s = 0.0001                          # Standard deviation
    rng, key = jax.random.split(rng)
    Y = teneva.rand_norm(d, n, r, key, m, s)
    print(Y[0])                         # Print the first TT-core

    # >>> ----------------------------------------
    # >>> Output:

    # [[[41.999935 42.       41.999977 41.999935]
    #   [42.000004 42.000088 41.99993  41.99992 ]
    #   [42.00024  42.000114 41.99994  41.999992]
    #   [42.000084 41.99994  42.000153 42.00012 ]
    #   [42.000057 42.000114 42.0001   41.999905]]]
    # 




|
|


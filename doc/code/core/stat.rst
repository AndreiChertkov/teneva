stat: helper functions for processing statistics
------------------------------------------------


.. automodule:: teneva.core.stat


-----


.. autofunction:: teneva.cdf_confidence

  **Examples**:

  .. code-block:: python

    points = np.random.randn(15)                     # Statistical points
    cdf_min, cdf_max = teneva.cdf_confidence(points) # Compute the confidence
    for p, c_min, c_max in zip(points, cdf_min, cdf_max):
        print(f'{p:-8.4f} | {c_min:-8.4f} | {c_max:-8.4f}')

    # >>> ----------------------------------------
    # >>> Output:

    #   0.4967 |   0.1461 |   0.8474
    #  -0.1383 |   0.0000 |   0.2124
    #   0.6477 |   0.2970 |   0.9983
    #   1.5230 |   1.0000 |   1.0000
    #  -0.2342 |   0.0000 |   0.1165
    #  -0.2341 |   0.0000 |   0.1165
    #   1.5792 |   1.0000 |   1.0000
    #   0.7674 |   0.4168 |   1.0000
    #  -0.4695 |   0.0000 |   0.0000
    #   0.5426 |   0.1919 |   0.8932
    #  -0.4634 |   0.0000 |   0.0000
    #  -0.4657 |   0.0000 |   0.0000
    #   0.2420 |   0.0000 |   0.5926
    #  -1.9133 |   0.0000 |   0.0000
    #  -1.7249 |   0.0000 |   0.0000
    # 


-----


.. autofunction:: teneva.cdf_getter

  **Examples**:

  .. code-block:: python

    x = np.random.randn(1000)      # Statistical points
    cdf = teneva.cdf_getter(x)     # Build the CDF getter

  .. code-block:: python

    z = -9999  # Point for CDF computations
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # 0.0
    # 

  .. code-block:: python

    z = +9999  # Point for CDF computations
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # 1.0
    # 

  .. code-block:: python

    z = [-10000, -10, -1, 0, 100]  # Several points for CDF computations
    cdf(z)

    # >>> ----------------------------------------
    # >>> Output:

    # array([0.   , 0.   , 0.145, 0.485, 1.   ])
    # 



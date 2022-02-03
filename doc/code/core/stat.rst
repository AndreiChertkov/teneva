stat: helper functions for processing statistics
------------------------------------------------


.. automodule:: teneva.core.stat

---


.. autofunction:: teneva.core.stat.cdf_confidence

  **Examples**:

  .. code-block:: python

    x = np.random.randn(10)                     # Statistical points
    cdf_min, cdf_max = teneva.cdf_confidence(x) # Compute the confidence
    print(f'{cdf_min}')
    print(f'{cdf_max}')

    # >>> ----------------------------------------
    # >>> Output:

    # [0.06724474 0.         0.21821913 1.         0.         0.
    #  1.         0.33796532 0.         0.11309064]
    # [0.92618356 0.29120511 1.         1.         0.19531603 0.19533245
    #  1.         1.         0.         0.97202945]
    # 

---


.. autofunction:: teneva.core.stat.cdf_getter

  **Examples**:

  .. code-block:: python

    x = np.random.randn(1000)      # Statistical points
    cdf = teneva.cdf_getter(x)     # Build the CDF getter
    z = [-10000, -10, -1, 0, 100]  # Points for CDF computations
    cdf(z)

---

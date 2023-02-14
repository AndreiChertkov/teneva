Module optima_contin: estimate max for function
-----------------------------------------------


.. automodule:: teneva.core.optima_contin


-----




|
|

.. autofunction:: teneva.optima_contin_tt_beam

  **Examples**:

  .. code-block:: python

    # DRAFT
    
    n = [20, 18, 16, 14, 12]  # Shape of the tensor
    Y = teneva.rand(n, r=4)   # Random TT-tensor with rank 4
    A = teneva.cheb_int(Y)
    x_opt = teneva.optima_contin_tt_beam(A, k=1)
    y_opt = teneva.cheb_get(x_opt, A)
    
    print(f'x opt appr :', x_opt)
    print(f'y opt appr : {y_opt:-12.4e}')

    # >>> ----------------------------------------
    # >>> Output:

    # 
    # 




|
|


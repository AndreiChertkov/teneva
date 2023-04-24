Module optima_func: estimate max for function
---------------------------------------------


.. automodule:: teneva.optima_func


-----




|
|

.. autofunction:: teneva.optima_func.optima_func_tt_beam

  **Examples**:

  .. code-block:: python

    n = [20, 18, 16, 14, 12]  # Shape of the tensor
    Y = teneva.rand(n, r=4)   # Random TT-tensor with rank 4
    A = teneva.func_int(Y)    # TT-tensor of interpolation coefficients
    
    x_opt = teneva.optima_func_tt_beam(A, k=3)
    y_opt = teneva.func_get(x_opt, A, -1, 1)
    
    print(f'x opt appr :', x_opt)
    print(f'y opt appr : {y_opt}')

    # >>> ----------------------------------------
    # >>> Output:

    # x opt appr : [[-0.32574766  0.60650301  0.8935902   0.57796563 -0.94833635]
    #  [-0.32574766  0.60650301  0.8935902   0.57796563  0.15852408]
    #  [-0.32574766  0.60650301  0.74642396 -0.9809905  -0.64170661]]
    # y opt appr : [-16.60387436  15.62801787  14.50242635]
    # 




|
|


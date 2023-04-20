Module vis: visualization methods for tensors
---------------------------------------------


.. automodule:: teneva.vis


-----




|
|

.. autofunction:: teneva.vis.show

  **Examples**:

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 12:
    Y = teneva.rand([4]*5, 12)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |4|  |4|  |4|  |4|  |4|
    # <rank>  =   12.0 :   \12/ \12/ \12/ \12/
    # 

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 2:
    Y = teneva.rand([2000, 2, 20000, 20, 200], 2)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |2000| |2| |20000| |20| |200|
    # <rank>  =    2.0 :      \2/ \2/     \2/  \2/
    # 

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 122:
    Y = teneva.rand([2000, 2, 20000, 20, 200], 122)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |2000|   |2|   |20000|   |20|   |200|
    # <rank>  =  122.0 :      \122/ \122/     \122/  \122/
    # 

  .. code-block:: python

    # 5-dim random TT-tensor with TT-rank 12:
    Y = teneva.rand([2**14]*5, 12)
    
    # Print the resulting TT-tensor:
    teneva.show(Y)

    # >>> ----------------------------------------
    # >>> Output:

    # TT-tensor     5D : |16384|  |16384|  |16384|  |16384|  |16384|
    # <rank>  =   12.0 :       \12/     \12/     \12/     \12/
    # 

  If an incorrect TT-tensor is passed to the function (the correctness of the shape of all cores is explicitly checked), then an error will be generated:

  .. code-block:: python

    Y = []
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid TT-tensor
    # 

  .. code-block:: python

    Y = [42.]
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid core for TT-tensor
    # 

  .. code-block:: python

    Y = [np.zeros((1, 5, 7)), np.zeros((42, 7, 1))]
    
    try:
        teneva.show(Y)
    except ValueError as e:
        print('Error :', e)

    # >>> ----------------------------------------
    # >>> Output:

    # Error : Invalid shape of core for TT-tensor
    # 




|
|


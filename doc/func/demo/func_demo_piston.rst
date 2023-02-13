Module func_demo_piston: Piston function for demo and tests
-----------------------------------------------------------


.. automodule:: teneva.func.demo.func_demo_piston


-----




|
|

.. autoclass:: teneva.FuncDemoPiston
  :members: 

**Examples**:

.. code-block:: python

  func = teneva.FuncDemoPiston()
  
  X_spec = (func.a + func.b) / 2
  X_spec = X_spec.reshape(1, -1)
  
  print(func.get_f_poi(X_spec))
  print(func.get_f_poi(X_spec[0]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.46439702]
  # 0.4643970224718025
  # 

It is also possible to calculate the function in the PyTorch format:

.. code-block:: python

  func = teneva.FuncDemoPiston(d=7)
  
  x1 = func.a + np.random.uniform(size=func.d) * (func.b - func.a)
  y1 = func._calc(x1)
  
  import torch
  # torch.set_default_dtype(torch.float64)
  
  x2 = torch.tensor(x1)
  y2 = func._calc_pt(x2)
  
  print(y1)
  print(y2.numpy())

  # >>> ----------------------------------------
  # >>> Output:

  # 0.38368320860039756
  # 0.38368321927735555
  # 




|
|


Module func_demo_dixon: Dixon function for demo and tests
---------------------------------------------------------


.. automodule:: teneva.func.demo.func_demo_dixon


-----




|
|

.. autoclass:: teneva.FuncDemoDixon
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  
  func = teneva.FuncDemoDixon(d=X.shape[1])
  
  print(func.get_f_poi(X))
  print(func.get_f_poi(X[0]), func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [1.000e+00 5.000e+00 1.810e+02 1.129e+03]
  # 1.0 1129.0
  # 

Minimum value:

.. code-block:: python

  print(func.x_min)                   # Argument for exact minimum
  print(func.get_f_poi(func.x_min))   # Computed minimum
  print(func.y_min)                   # Value of exact minimum

  # >>> ----------------------------------------
  # >>> Output:

  # [1.         0.70710678 0.59460356]
  # 1.355854680848614e-31
  # 0.0
  # 

Let try to check the min-max values for 2D case by simple brute-force:

.. code-block:: python

  func = teneva.FuncDemoDixon(d=2)
  func.set_grid(n=5000, kind='uni')
  
  I = teneva.grid_flat(func.n)
  Y = func.get_f_ind(I).reshape(func.n, order='F')
  
  i_min = np.unravel_index(np.argmin(Y), Y.shape)
  i_max = np.unravel_index(np.argmax(Y), Y.shape)
  
  x_min = teneva.ind_to_poi(i_min, func.a, func.b, func.n)
  x_max = teneva.ind_to_poi(i_max, func.a, func.b, func.n)
  
  y_min = func.get_f_poi(x_min)
  y_max = func.get_f_poi(x_max)
  
  print(f'Function   : {func.name}')
  print(f'y_min real = {func.y_min:-13.7e}; x_min real = {func.x_min}')
  print(f'y_min appr = {y_min:-13.7e}; x_min appr = {x_min}')
  print(f'y_max appr = {y_max:-13.7e}; x_max appr = {x_max}')

  # >>> ----------------------------------------
  # >>> Output:

  # Function   : Dixon
  # y_min real = 0.0000000e+00; x_min real = [1.         0.70710678]
  # y_min appr = 4.9665293e-06; x_min appr = [ 0.99819964 -0.70614123]
  # y_max appr = 8.8321000e+04; x_max appr = [-10. -10.]
  # 

We can plot the function for 2D case:

.. code-block:: python

  teneva.FuncDemoDixon(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.

It is also possible to calculate the function in the PyTorch format:

.. code-block:: python

  func = teneva.FuncDemoDixon(d=10)
  
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

  # 467550.67037377093
  # 467550.67037377093
  # 




|
|


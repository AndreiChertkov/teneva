Module func_demo_qing: Qing function for demo and tests
-------------------------------------------------------


.. automodule:: teneva.func.demo.func_demo_qing


-----




|
|

.. autoclass:: teneva.FuncDemoQing
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  
  func = teneva.FuncDemoQing(d=X.shape[1])
  
  print(func.get_f_poi(X))
  print(func.get_f_poi(X[0]), func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [ 14.   5.  14. 149.]
  # 14.0 149.0
  # 

Minimum value:

.. code-block:: python

  print(func.x_min)                   # Argument for exact minimum
  print(func.get_f_poi(func.x_min))   # Computed minimum
  print(func.y_min)                   # Value of exact minimum

  # >>> ----------------------------------------
  # >>> Output:

  # [1.         1.41421356 1.73205081]
  # 3.944304526105059e-31
  # 0.0
  # 

Let try to check the min-max values for 2D case by simple brute-force:

.. code-block:: python

  func = teneva.FuncDemoQing(d=2)
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

  # Function   : Qing
  # y_min real = 0.0000000e+00; x_min real = [1.         1.41421356]
  # y_min appr = 1.5380363e-03; x_min appr = [1.00020004 1.40028006]
  # y_max appr = 1.2499850e+11; x_max appr = [500. 500.]
  # 

We can plot the function for 2D case:

.. code-block:: python

  teneva.FuncDemoQing(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.

Note that for this function, we can construct the TT-cores of its TT-decomposition explicitly:

.. code-block:: python

  func = teneva.FuncDemoQing(d=100)
  func.set_grid(n=2**14, kind='uni')
  func.cores()

We can check the accuracy of approximation:

.. code-block:: python

  func.build_tst_ind(m=1.E+4)
  func.check()
  func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Qing            [CORES        ] > error: 2.8e-16 | rank:  2.0 | time:   0.095
  # 

And we can also check the accuracy of its minimum (i.e. the real minimum value compared to the value of the nearest element of the TT-tensor):

.. code-block:: python

  y_min_real = func.y_min
  x_min_real = func.x_min
  i_min_real = teneva.poi_to_ind(x_min_real, func.a, func.b, func.n, func.kind)
  y_min_appr = func.get_ind(i_min_real)
  
  print(f'Function   : {func.name}')
  print(f'y_min real = {y_min_real:-13.7e}')
  print(f'y_min appr = {y_min_appr:-13.7e}')

  # >>> ----------------------------------------
  # >>> Output:

  # Function   : Qing
  # y_min real = 0.0000000e+00
  # y_min appr = 1.3731984e+00
  # 

It is also possible to calculate the function in the PyTorch format:

.. code-block:: python

  func = teneva.FuncDemoQing(d=10)
  
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

  # 96665123284.63771
  # 96665123284.63773
  # 




|
|


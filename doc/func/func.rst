Module func: wrapper for multivariable function with approximation methods
--------------------------------------------------------------------------


.. automodule:: teneva.func.func


-----


.. autoclass:: teneva.Func
  :members: 

**Examples**:

First, let's denote in a compact form a convenient approach for approximating user-defined functions, and then consider the class interface in more detail:

.. code-block:: python

  # Prepare simple function (y = x_1^2 + x_2^2 + ... + x_5^2):
  func = teneva.Func(d=5, f_comp=lambda X: np.sum(X**2, axis=1), name='Sphere')
  func.set_lim(-2., +3.)
  func.set_grid(32, kind='cheb')
  func.build_tst_poi(m=1.E+4)

.. code-block:: python

  # Approximate the function by TT-CROSS method:
  func.rand(r=2)
  func.cross(m=1.E+5, e=1.E-10, log=True)

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      1.209 | evals: 0.00e+00 (+ 0.00e+00) | rank:   2.0 | 
  # #   1 | time:      1.230 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 4.0e+00 | 
  # #   2 | time:      1.261 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.1e-08 | 
  # #   3 | time:      1.390 | evals: 1.46e+04 (+ 4.24e+03) | rank:   8.8 | eps: 0.0e+00 | stop: e | 
  # 
  # 

.. code-block:: python

  # Check accuracy of the approximation on the train and test datasets:
  func.check()
  func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Sphere          [CRO          ] > error: 1.2e-15 / 7.4e-16 | rank:  2.0 | time:   1.397
  # 

.. code-block:: python

  # Usage of approximation to compute the discretized function:
  i = np.array([1] * func.d, dtype=int)
  print(f'Tensor multi-index :', i)
  print(f'Approximated value : {func[i]:.16f}')
  print(f'Real (exact) value : {func.get_f_ind(i):.16f}')

  # >>> ----------------------------------------
  # >>> Output:

  # Tensor multi-index : [1 1 1 1 1]
  # Approximated value : 44.6160218744688848
  # Real (exact) value : 44.6160218744689203
  # 

.. code-block:: python

  # Usage of approximation to compute the original function:
  x = np.array([0.5] * func.d, dtype=float)
  print(f'Spatial point      :', x)
  print(f'Approximated value : {func(x):.16f}')
  print(f'Real (exact) value : {func.get_f_poi(x):.16f}')

  # >>> ----------------------------------------
  # >>> Output:

  # Spatial point      : [0.5 0.5 0.5 0.5 0.5]
  # Approximated value : 1.2500000000000084
  # Real (exact) value : 1.2500000000000000
  # 

And now let consider the "Func" class interface in more detail. We set simple 5-dimensional analytical function (note that we can set only one function "f_calc" / "f_comp" or both):

.. code-block:: python

  d = 5                           # Number of dimensions                   
  name = 'Sphere'                 # Optional display name of the function
  
  def f_calc(x):                  # Calculate function in one point (optional)  
      return np.sum(x**2)         # (x is the 1D array of the shape [d])
  
  def f_comp(X):                  # Compute function in many points
      return np.sum(X**2, axis=1) # (X is the 2D array of the shape [samples, d])

.. code-block:: python

  # Create class instance of the function:
  func = teneva.Func(d, f_calc, f_comp, name)
  
  # Set lower (-2) and upper (+3) spatial bounds (number or list):
  func.set_lim(-2., [+3., +3., +3., +3., +3.])
  
  # Set number of grid points (number or list)
  # and kind of the grid ('uni' or 'cheb'):
  func.set_grid(32, kind='cheb')
  
  # Prepare test data (random 1.E+5 tensor indices):
  func.build_tst_ind(m=1.E+5)
  
  # Prepare test data (random 1.E+5 spatial points):
  func.build_tst_poi(m=1.E+5)

Then we can approximate the function by TT-CROSS (note that the train data will be collected from the cache of the TT-CROSS requests to the target function):

.. code-block:: python

  func.rand(r=2)               # Build initial approximation (random TT-tensor of the rank r)
  func.cross(m=1.E+5,          # Build TT-approximation by TT-CROSS
      e=1.E-10, log=True)          
  func.check_trn_ind()         # Check accuracy on the train data (tensor indices from TT-CROSS)
  func.check_tst_ind()         # Check accuracy on the test data (random tensor indices)
  func.check_tst_poi()         # Check accuracy on the test data (random spatial points)
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      0.002 | evals: 0.00e+00 (+ 0.00e+00) | rank:   2.0 | 
  # #   1 | time:      0.016 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 8.0e+00 | 
  # #   2 | time:      0.042 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.1e-08 | 
  # #   3 | time:      0.121 | evals: 1.45e+04 (+ 4.27e+03) | rank:   8.8 | eps: 0.0e+00 | stop: e | 
  # 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : CRO
  # 
  # Evals function      : 1.5e+04
  # Evals cache         : 4.3e+03
  # TT-rank             :     2.0
  # Number of params    : 5.1e+02
  # 
  # Samples trn ind     : 1.5e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 1.2e-15
  # Error tst ind       : 7.8e-16
  # Error tst poi       : 7.4e-16
  # 
  # Time approximation  :   0.123
  # Time trn check ind  :   0.028
  # Time tst build ind  :   0.031
  # Time tst check ind  :   0.412
  # Time tst build poi  :   0.925
  # Time tst check poi  :   3.475
  # 
  # Sweeps              :       3
  # ==================================================
  # 

We can also approximate the function by TT-ANOVA:

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+5)  # Prepare train data (random tensor indices)
  func.anova()                 # Build TT-approximation by TT-ANOVA
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # ==================================================
  # ------------------- | Sphere function
  # Method              : ANO
  # 
  # Evals function      : 1.0e+05
  # TT-rank             :     2.0
  # Number of params    : 5.1e+02
  # 
  # Samples trn ind     : 1.0e+05
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 1.3e-02
  # Error tst ind       : 1.3e-02
  # Error tst poi       : 1.8e-02
  # 
  # Time approximation  :   0.043
  # Time trn build ind  :   0.030
  # Time trn check ind  :   0.194
  # Time tst build ind  :   0.031
  # Time tst check ind  :   0.445
  # Time tst build poi  :   0.925
  # Time tst check poi  :   3.584
  # 
  # ==================================================
  # 

And we can approximate the function by TT-ALS:

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+5)  # Prepare train data (random tensor indices)
  func.rand(r=2)               # Build initial approximation
  func.als(log=True)           # Build TT-approximation by TT-ALS
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      0.035 | rank:   2.0 | 
  # #   1 | time:      0.234 | rank:   2.0 | eps: 1.0e+00 | 
  # #   2 | time:      0.407 | rank:   2.0 | eps: 1.0e+00 | 
  # #   3 | time:      0.572 | rank:   2.0 | eps: 1.0e+00 | 
  # #   4 | time:      0.748 | rank:   2.0 | eps: 5.0e-01 | 
  # #   5 | time:      0.911 | rank:   2.0 | eps: 2.8e+00 | 
  # #   6 | time:      1.129 | rank:   2.0 | eps: 1.4e+00 | 
  # #   7 | time:      1.302 | rank:   2.0 | eps: 6.2e-02 | 
  # #   8 | time:      1.472 | rank:   2.0 | eps: 1.6e-02 | 
  # #   9 | time:      1.636 | rank:   2.0 | eps: 2.4e-04 | 
  # #  10 | time:      1.809 | rank:   2.0 | eps: 2.7e-06 | 
  # #  11 | time:      1.976 | rank:   2.0 | eps: 6.0e-08 | 
  # #  12 | time:      2.145 | rank:   2.0 | eps: 2.1e-08 | 
  # #  13 | time:      2.315 | rank:   2.0 | eps: 2.1e-08 | 
  # #  14 | time:      2.486 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : ALS
  # 
  # Evals function      : 1.0e+05
  # TT-rank             :     2.0
  # Number of params    : 5.1e+02
  # 
  # Samples trn ind     : 1.0e+05
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 1.3e-14
  # Error tst ind       : 1.4e-14
  # Error tst poi       : 9.8e-15
  # 
  # Time approximation  :   2.487
  # Time trn build ind  :   0.028
  # Time trn check ind  :   0.192
  # Time tst build ind  :   0.031
  # Time tst check ind  :   0.405
  # Time tst build poi  :   0.925
  # Time tst check poi  :   3.820
  # 
  # Sweeps              :      14
  # ==================================================
  # 

We can approximate the function by TT-ANOVA + TT-ALS (we may combine the approximation methods; in this example we use the result of the TT-ANOVA as an initial approximation for the TT-ALS algorithm):

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+5)  # Prepare train data (random tensor indices)
  func.anova(r=2, order=1)     # Apply TT-ANOVA
  func.als(log=True)           # Apply TT-ALS, using TT-ANOVA as initial approximation
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      0.032 | rank:   2.0 | 
  # #   1 | time:      0.216 | rank:   2.0 | eps: 1.1e-02 | 
  # #   2 | time:      0.408 | rank:   2.0 | eps: 2.4e-04 | 
  # #   3 | time:      0.594 | rank:   2.0 | eps: 2.7e-06 | 
  # #   4 | time:      0.779 | rank:   2.0 | eps: 2.1e-08 | 
  # #   5 | time:      0.957 | rank:   2.0 | eps: 1.5e-08 | 
  # #   6 | time:      1.146 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : ANO-ALS
  # 
  # Evals function      : 1.0e+05
  # TT-rank             :     2.0
  # Number of params    : 5.1e+02
  # 
  # Samples trn ind     : 1.0e+05
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 3.5e-13
  # Error tst ind       : 3.5e-13
  # Error tst poi       : 4.1e-13
  # 
  # Time approximation  :   1.198
  # Time trn build ind  :   0.040
  # Time trn check ind  :   0.206
  # Time tst build ind  :   0.031
  # Time tst check ind  :   0.447
  # Time tst build poi  :   0.925
  # Time tst check poi  :   3.755
  # 
  # Sweeps              :       6
  # ==================================================
  # 

We can approximate the function by TT-ANOVA + TT-CROSS (note that the total number of function queries, i.e. 1.E+4 + 9.E+4) will be the same as above):

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+4)  # Prepare train data (random tensor indices)
  func.anova(r=2, order=1)     # Apply TT-ANOVA
  func.cross(m=9.E+4, log=True)# Apply TT-CROSS, using TT-ANOVA as initial approximation   
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      0.002 | evals: 0.00e+00 (+ 0.00e+00) | rank:   2.0 | 
  # #   1 | time:      0.016 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 3.1e-02 | 
  # #   2 | time:      0.043 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.1e-08 | 
  # #   3 | time:      0.137 | evals: 1.46e+04 (+ 4.24e+03) | rank:   8.8 | eps: 0.0e+00 | 
  # #   4 | time:      0.366 | evals: 2.90e+04 (+ 1.34e+04) | rank:  12.7 | eps: 2.1e-08 | 
  # #   5 | time:      0.894 | evals: 5.54e+04 (+ 3.02e+04) | rank:  16.7 | eps: 1.5e-08 | 
  # #   5 | time:      1.163 | evals: 8.35e+04 (+ 5.77e+04) | rank:  20.3 | eps: 1.5e-08 | stop: m | 
  # 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : ANO-CRO
  # 
  # Evals function      : 9.4e+04
  # Evals cache         : 5.8e+04
  # TT-rank             :     3.0
  # Number of params    : 1.1e+03
  # 
  # Samples trn ind     : 8.4e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 3.3e-15
  # Error tst ind       : 1.9e-15
  # Error tst poi       : 2.0e-15
  # 
  # Time approximation  :   1.175
  # Time trn check ind  :   0.169
  # Time tst build ind  :   0.031
  # Time tst check ind  :   0.515
  # Time tst build poi  :   0.925
  # Time tst check poi  :   3.884
  # 
  # Sweeps              :       5
  # ==================================================
  # 

We can compute the value of approximation for any point inside the grid bounds:

.. code-block:: python

  x = np.array([0.5] * 5)     # Spatial point [0.5, ..., 0.5]
  y_appr = func.get_poi(x)    # Approximated value
  y_real = func.get_f_poi(x)  # Real (exact) value
  
  print(f'Approximated value : {y_appr:-18.10f}')
  print(f'Real (exact) value : {y_real:-18.10f}')

  # >>> ----------------------------------------
  # >>> Output:

  # Approximated value :       1.2500000000
  # Real (exact) value :       1.2500000000
  # 

We can compute the value of approximation for any batch of points inside the grid bounds:

.. code-block:: python

  X = np.array([                # Spatial points
      [0.] * 5,
      [0.5] * 5,
      [1.] * 5,
      [1.5] * 5,
  ])   
  Y_appr = func.get_poi(X)      # Approximated values
  Y_real = func.get_f_poi(X)    # Real (exact) values
  
  for y_appr, y_real in zip(Y_appr, Y_real):
      print(f'Appr : {y_appr:-18.10f} | Real: {y_real:-18.10f}')

  # >>> ----------------------------------------
  # >>> Output:

  # Appr :      -0.0000000000 | Real:       0.0000000000
  # Appr :       1.2500000000 | Real:       1.2500000000
  # Appr :       5.0000000000 | Real:       5.0000000000
  # Appr :      11.2500000000 | Real:      11.2500000000
  # 

We may also use the "call" notation:

.. code-block:: python

  Y_appr = func(X)
  print(Y_appr)

  # >>> ----------------------------------------
  # >>> Output:

  # [-3.64153152e-14  1.25000000e+00  5.00000000e+00  1.12500000e+01]
  # 

We can compute the value of approximation for any tensor multi-indices:

.. code-block:: python

  # Tensor multiindices:
  I = np.array([            
      [0] * 5,
      [1] * 5,
      [20] * 5,
      [25] * 5,
  ])
  
  # Related spatial points:
  X = teneva.ind_to_poi(I, func.a, func.b, func.n, func.kind)
  
  Y_appr = func.get_ind(I)    # Values of the tensor items
  Y_real = func.get_f_poi(X)  # Real (exact) values in the related points
  
  for y_appr, y_real in zip(Y_appr, Y_real):
      print(f'Appr : {y_appr:-18.10f} | Real: {y_real:-18.10f}')

  # >>> ----------------------------------------
  # >>> Output:

  # Appr :      45.0000000000 | Real:      45.0000000000
  # Appr :      44.6160218745 | Real:      44.6160218745
  # Appr :       1.8059171282 | Real:       1.8059171282
  # Appr :      12.0421015606 | Real:      12.0421015606
  # 

We may also use the "getitem" notation:

.. code-block:: python

  Y_appr = func[I]
  print(Y_appr)

  # >>> ----------------------------------------
  # >>> Output:

  # [45.         44.61602187  1.80591713 12.04210156]
  # 

And for one multi-index:

.. code-block:: python

  i = I[0, :]
  Y_appr = func[i]
  print(Y_appr)

  # >>> ----------------------------------------
  # >>> Output:

  # 45.00000000000045
  # 

We can also approximate all demo functions (benchmarks) by any method, e.x. TT-CROSS:

.. code-block:: python

  for func in teneva.func_demo_all(d=10):
      func.clear()
      func.set_grid(32, kind='cheb')
      func.build_tst_ind(m=1.E+5)
      func.build_tst_poi(m=1.E+5)
      func.rand(r=2)
      func.cross(m=1.E+5)
      func.check()
      func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Ackley          [CRO          ] > error: 3.7e-05 / 3.6e-07 / 1.6e-02 | rank:  9.2 | time:   0.699
  # Alpine          [CRO          ] > error: 2.1e-14 / 7.5e-15 / 3.5e-02 | rank:  3.6 | time:   1.139
  # Dixon           [CRO          ] > error: 3.6e-12 / 2.8e-12 / 9.6e-12 | rank:  4.8 | time:   0.790
  # Exponential     [CRO          ] > error: 1.5e-15 / 1.4e-15 / 1.2e-15 | rank:  3.6 | time:   0.874
  # Grienwank       [CRO          ] > error: 5.9e-12 / 9.9e-13 / 1.6e-04 | rank:  3.8 | time:   0.730
  # Michalewicz     [CRO          ] > error: 6.4e-15 / 2.4e-15 / 4.0e-01 | rank:  4.8 | time:   1.104
  # Qing            [CRO          ] > error: 4.7e-15 / 3.3e-15 / 3.6e-15 | rank:  4.3 | time:   1.100
  # Rastrigin       [CRO          ] > error: 5.3e-14 / 1.7e-14 / 6.9e-02 | rank:  3.0 | time:   0.848
  # Rosenbrock      [CRO          ] > error: 8.9e-15 / 8.6e-15 / 1.6e-14 | rank:  5.0 | time:   0.845
  # Schaffer        [CRO          ] > error: 2.7e-04 / 1.0e-03 / 5.5e-02 | rank: 12.3 | time:   0.794
  # Schwefel        [CRO          ] > error: 1.0e-13 / 2.6e-14 / 9.1e-03 | rank:  3.8 | time:   0.888
  # 

And for the finer grid and more requests to the target function, we will have the more accurate result. Note that the errors are displayed in the following order (if there is no corresponding data, then the value is skipped): training set (index, then point), validation set (index, then point), test set (index, then point).

.. code-block:: python

  for func in teneva.func_demo_all(d=10):
      func.clear()
      func.set_grid(256, kind='cheb')
      func.build_tst_ind(m=1.E+5)
      func.build_tst_poi(m=1.E+5)
      func.rand(r=3)
      func.cross(m=1.E+6)
      func.check()
      func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Ackley          [CRO          ] > error: 1.6e-03 / 1.3e-06 / 3.5e-04 | rank:  9.9 | time:  10.284
  # Alpine          [CRO          ] > error: 6.5e-14 / 2.1e-14 / 1.3e-03 | rank:  2.4 | time:   9.162
  # Dixon           [CRO          ] > error: 2.4e-12 / 8.8e-13 / 2.4e-12 | rank:  5.9 | time:   9.484
  # Exponential     [CRO          ] > error: 4.4e-15 / 1.9e-15 / 1.9e-15 | rank:  6.9 | time:   9.905
  # Grienwank       [CRO          ] > error: 2.8e-11 / 2.3e-12 / 1.3e-04 | rank:  6.2 | time:  10.900
  # Michalewicz     [CRO          ] > error: 9.7e-15 / 6.2e-15 / 7.2e-04 | rank:  3.5 | time:  12.643
  # Qing            [CRO          ] > error: 5.4e-15 / 3.5e-15 / 4.5e-15 | rank:  2.9 | time:   9.600
  # Rastrigin       [CRO          ] > error: 5.4e-14 / 2.0e-14 / 3.2e-14 | rank:  4.8 | time:   9.401
  # Rosenbrock      [CRO          ] > error: 4.3e-14 / 1.5e-14 / 3.3e-14 | rank:  4.6 | time:   9.308
  # Schaffer        [CRO          ] > error: 1.3e-02 / 1.8e-02 / 2.5e-02 | rank: 14.2 | time:  10.621
  # Schwefel        [CRO          ] > error: 1.0e-13 / 2.2e-14 / 1.3e-05 | rank:  4.4 | time:   9.304
  # 

We can also form a list of functions for which the explicit form of their TT-cores is known and build them on an essentially multidimensional grid (100-dimensional with 1024 mode size). Note that the errors are displayed below for the random test multi-indices and then for the random test points.

.. code-block:: python

  for func in teneva.func_demo_all(d=100, only_with_cores=True):
      func.clear()
      func.set_grid(2**10, kind='cheb')
      func.build_tst_ind(m=1.E+4)
      func.build_tst_poi(m=1.E+4)
      func.cores()
      func.check()
      func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Alpine          [CORES        ] > error: 9.9e-17 / 5.8e-05 | rank:  2.0 | time:   0.008
  # Exponential     [CORES        ] > error: 6.0e-15 / 8.6e-15 | rank:  1.0 | time:   0.006
  # Grienwank       [CORES        ] > error: 4.1e-16 / 4.8e-15 | rank:  3.0 | time:   0.017
  # Michalewicz     [CORES        ] > error: 1.0e+00 / 1.0e+00 | rank:  1.0 | time:   0.008
  # Qing            [CORES        ] > error: 0.0e+00 / 4.8e-15 | rank:  2.0 | time:   0.006
  # Rastrigin       [CORES        ] > error: 6.5e-17 / 4.5e-15 | rank:  2.0 | time:   0.007
  # Rosenbrock      [CORES        ] > error: 4.1e-16 / 4.9e-15 | rank:  3.0 | time:   0.010
  # Schwefel        [CORES        ] > error: 8.8e-17 / 9.8e-07 | rank:  2.0 | time:   0.006
  # 



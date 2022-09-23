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

  # # pre | time:      1.171 | evals: 0.00e+00 (+ 0.00e+00) | rank:   2.0 | 
  # #   1 | time:      1.185 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 5.3e+00 | 
  # #   2 | time:      1.212 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.9e-08 | 
  # #   3 | time:      1.276 | evals: 1.46e+04 (+ 4.24e+03) | rank:   8.8 | eps: 6.4e-09 | 
  # #   4 | time:      1.432 | evals: 2.90e+04 (+ 1.34e+04) | rank:  12.7 | eps: 1.6e-08 | 
  # #   5 | time:      1.898 | evals: 5.54e+04 (+ 3.01e+04) | rank:  16.7 | eps: 8.5e-09 | 
  # #   6 | time:      3.045 | evals: 9.25e+04 (+ 6.23e+04) | rank:  20.7 | eps: 1.0e-08 | 
  # #   6 | time:      3.125 | evals: 9.65e+04 (+ 7.44e+04) | rank:  21.7 | eps: 1.0e-08 | stop: m | 
  # 
  # 

.. code-block:: python

  # Check accuracy of the approximation on the train and test datasets:
  func.check()
  func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Sphere          [CRO          ] > error: 2.6e-15 / 4.8e-15 | rank:  6.0 | time:   3.128
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
  # Approximated value : 44.6160218744688208
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
  # Approximated value : 1.2500000000001172
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

  # # pre | time:      0.003 | evals: 0.00e+00 (+ 0.00e+00) | rank:   2.0 | 
  # #   1 | time:      0.016 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 6.1e+00 | 
  # #   2 | time:      0.041 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.9e-08 | 
  # #   3 | time:      0.102 | evals: 1.45e+04 (+ 4.27e+03) | rank:   8.8 | eps: 6.4e-09 | 
  # #   4 | time:      0.252 | evals: 2.89e+04 (+ 1.35e+04) | rank:  12.7 | eps: 1.6e-08 | 
  # #   5 | time:      0.731 | evals: 5.53e+04 (+ 3.02e+04) | rank:  16.7 | eps: 8.5e-09 | 
  # #   6 | time:      1.929 | evals: 9.25e+04 (+ 6.23e+04) | rank:  20.7 | eps: 1.0e-08 | 
  # #   6 | time:      2.000 | evals: 9.64e+04 (+ 7.44e+04) | rank:  21.7 | eps: 1.0e-08 | stop: m | 
  # 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : CRO
  # 
  # Evals function      : 9.6e+04
  # Evals cache         : 7.4e+04
  # TT-rank             :     6.0
  # Number of params    : 3.9e+03
  # 
  # Samples trn ind     : 9.6e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 2.6e-15
  # Error tst ind       : 2.7e-15
  # Error tst poi       : 4.9e-15
  # 
  # Time approximation  :   2.004
  # Time trn check ind  :   0.218
  # Time tst build ind  :   0.032
  # Time tst check ind  :   0.492
  # Time tst build poi  :   1.004
  # Time tst check poi  :   4.237
  # 
  # Sweeps              :       6
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
  # Time approximation  :   0.046
  # Time trn build ind  :   0.035
  # Time trn check ind  :   0.199
  # Time tst build ind  :   0.032
  # Time tst check ind  :   0.523
  # Time tst build poi  :   1.004
  # Time tst check poi  :   3.883
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

  # # pre | time:      0.034 | rank:   2.0 | 
  # #   1 | time:      0.202 | rank:   2.0 | eps: 1.1e+00 | 
  # #   2 | time:      0.406 | rank:   2.0 | eps: 8.0e-01 | 
  # #   3 | time:      0.649 | rank:   2.0 | eps: 5.1e-01 | 
  # #   4 | time:      0.950 | rank:   2.0 | eps: 4.1e-01 | 
  # #   5 | time:      1.209 | rank:   2.0 | eps: 3.3e+00 | 
  # #   6 | time:      1.484 | rank:   2.0 | eps: 1.6e+00 | 
  # #   7 | time:      1.754 | rank:   2.0 | eps: 7.0e-02 | 
  # #   8 | time:      1.999 | rank:   2.0 | eps: 1.8e-02 | 
  # #   9 | time:      2.293 | rank:   2.0 | eps: 2.1e-04 | 
  # #  10 | time:      2.516 | rank:   2.0 | eps: 3.1e-06 | 
  # #  11 | time:      2.721 | rank:   2.0 | eps: 6.0e-08 | 
  # #  12 | time:      2.947 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
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
  # Error trn ind       : 2.6e-11
  # Error tst ind       : 2.7e-11
  # Error tst poi       : 2.0e-11
  # 
  # Time approximation  :   2.948
  # Time trn build ind  :   0.037
  # Time trn check ind  :   0.206
  # Time tst build ind  :   0.032
  # Time tst check ind  :   0.422
  # Time tst build poi  :   1.004
  # Time tst check poi  :   3.824
  # 
  # Sweeps              :      12
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

  # # pre | time:      0.031 | rank:   2.0 | 
  # #   1 | time:      0.269 | rank:   2.0 | eps: 1.2e-02 | 
  # #   2 | time:      0.499 | rank:   2.0 | eps: 2.7e-04 | 
  # #   3 | time:      0.734 | rank:   2.0 | eps: 2.4e-06 | 
  # #   4 | time:      0.955 | rank:   2.0 | eps: 3.6e-08 | 
  # #   5 | time:      1.202 | rank:   2.0 | eps: 2.2e-08 | 
  # #   6 | time:      1.430 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
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
  # Time approximation  :   1.471
  # Time trn build ind  :   0.029
  # Time trn check ind  :   0.198
  # Time tst build ind  :   0.032
  # Time tst check ind  :   0.433
  # Time tst build poi  :   1.004
  # Time tst check poi  :   3.749
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
  # #   1 | time:      0.017 | evals: 1.94e+03 (+ 1.68e+02) | rank:   4.0 | eps: 3.7e-02 | 
  # #   2 | time:      0.041 | evals: 5.78e+03 (+ 1.77e+03) | rank:   6.0 | eps: 1.9e-08 | 
  # #   3 | time:      0.108 | evals: 1.46e+04 (+ 4.24e+03) | rank:   8.8 | eps: 6.4e-09 | 
  # #   4 | time:      0.260 | evals: 2.90e+04 (+ 1.34e+04) | rank:  12.7 | eps: 1.6e-08 | 
  # #   5 | time:      0.722 | evals: 5.54e+04 (+ 3.02e+04) | rank:  16.7 | eps: 8.5e-09 | 
  # #   5 | time:      0.910 | evals: 8.35e+04 (+ 5.77e+04) | rank:  20.3 | eps: 8.5e-09 | stop: m | 
  # 
  # ==================================================
  # ------------------- | Sphere function
  # Method              : ANO-CRO
  # 
  # Evals function      : 9.4e+04
  # Evals cache         : 5.8e+04
  # TT-rank             :     3.2
  # Number of params    : 1.2e+03
  # 
  # Samples trn ind     : 8.4e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 1.9e-15
  # Error tst ind       : 1.6e-15
  # Error tst poi       : 2.6e-15
  # 
  # Time approximation  :   0.919
  # Time trn check ind  :   0.167
  # Time tst build ind  :   0.032
  # Time tst check ind  :   0.454
  # Time tst build poi  :   1.004
  # Time tst check poi  :   4.154
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

  # [-7.50510765e-14  1.25000000e+00  5.00000000e+00  1.12500000e+01]
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

  # 45.00000000000001
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

  # Ackley          [CRO          ] > error: 3.4e-05 / 3.6e-07 / 1.6e-02 | rank:  9.2 | time:   0.649
  # Alpine          [CRO          ] > error: 1.4e-14 / 7.4e-15 / 3.5e-02 | rank:  3.3 | time:   0.966
  # Dixon           [CRO          ] > error: 2.8e-12 / 2.0e-12 / 6.6e-12 | rank:  4.2 | time:   0.665
  # Exponential     [CRO          ] > error: 1.4e-15 / 1.3e-15 / 1.3e-15 | rank:  4.0 | time:   0.606
  # Grienwank       [CRO          ] > error: 7.3e-12 / 1.6e-12 / 1.6e-04 | rank:  4.0 | time:   0.663
  # Michalewicz     [CRO          ] > error: 2.5e-15 / 2.6e-15 / 4.0e-01 | rank:  4.6 | time:   0.847
  # Qing            [CRO          ] > error: 5.5e-15 / 1.9e-15 / 2.7e-15 | rank:  4.3 | time:   0.853
  # Rastrigin       [CRO          ] > error: 2.9e-14 / 1.2e-14 / 6.9e-02 | rank:  3.0 | time:   0.623
  # Rosenbrock      [CRO          ] > error: 1.4e-14 / 1.1e-14 / 2.7e-14 | rank:  5.0 | time:   0.633
  # Schaffer        [CRO          ] > error: 2.7e-04 / 1.0e-03 / 5.5e-02 | rank: 12.3 | time:   0.656
  # Schwefel        [CRO          ] > error: 1.2e-13 / 2.3e-14 / 9.1e-03 | rank:  3.9 | time:   0.635
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

  # Ackley          [CRO          ] > error: 1.5e-03 / 1.3e-06 / 3.5e-04 | rank:  9.9 | time:   8.704
  # Alpine          [CRO          ] > error: 5.9e-14 / 1.5e-14 / 1.3e-03 | rank:  2.2 | time:   7.882
  # Dixon           [CRO          ] > error: 2.8e-12 / 1.1e-12 / 3.0e-12 | rank:  6.1 | time:   8.369
  # Exponential     [CRO          ] > error: 2.4e-15 / 1.6e-15 / 2.1e-15 | rank:  7.2 | time:   8.412
  # Grienwank       [CRO          ] > error: 2.9e-11 / 2.3e-12 / 1.3e-04 | rank:  5.8 | time:   9.748
  # Michalewicz     [CRO          ] > error: 3.5e-14 / 6.5e-15 / 7.2e-04 | rank:  3.5 | time:  11.192
  # Qing            [CRO          ] > error: 1.3e-13 / 3.7e-15 / 4.5e-15 | rank:  3.0 | time:   8.297
  # Rastrigin       [CRO          ] > error: 4.8e-14 / 1.8e-14 / 3.0e-14 | rank:  4.6 | time:   8.361
  # Rosenbrock      [CRO          ] > error: 2.5e-14 / 1.6e-14 / 3.5e-14 | rank:  4.6 | time:   7.863
  # Schaffer        [CRO          ] > error: 1.3e-02 / 1.8e-02 / 2.5e-02 | rank: 14.2 | time:   9.370
  # Schwefel        [CRO          ] > error: 6.4e-14 / 1.3e-14 / 1.3e-05 | rank:  4.7 | time:   7.719
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

  # Alpine          [CORES        ] > error: 9.9e-17 / 5.8e-05 | rank:  2.0 | time:   0.007
  # Exponential     [CORES        ] > error: 6.0e-15 / 8.6e-15 | rank:  1.0 | time:   0.011
  # Grienwank       [CORES        ] > error: 4.1e-16 / 4.8e-15 | rank:  3.0 | time:   0.013
  # Michalewicz     [CORES        ] > error: 0.0e+00 / 2.9e-02 | rank:  2.0 | time:   0.011
  # Qing            [CORES        ] > error: 0.0e+00 / 4.8e-15 | rank:  2.0 | time:   0.005
  # Rastrigin       [CORES        ] > error: 6.5e-17 / 4.5e-15 | rank:  2.0 | time:   0.008
  # Rosenbrock      [CORES        ] > error: 4.1e-16 / 4.9e-15 | rank:  3.0 | time:   0.012
  # Schwefel        [CORES        ] > error: 8.8e-17 / 9.8e-07 | rank:  2.0 | time:   0.010
  # 

We can also approximate all demo functions (benchmarks) by TT-ALS method with TT-ANOVA:

.. code-block:: python

  for func in teneva.func_demo_all(d=5):
      func.clear()
      func.set_grid(32, kind='cheb')
      func.build_trn_ind(m=1.E+4)
      func.build_tst_ind(m=1.E+5)
      func.build_tst_poi(m=1.E+5)
      func.anova(r=5)
      func.als(nswp=20)
      func.check()
      func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Ackley          [ANO-ALS      ] > error: 1.3e-03 / 3.9e-03 / 2.5e-02 | rank:  5.0 | time:   1.002
  # Alpine          [ANO-ALS      ] > error: 1.2e-07 / 1.9e-07 / 4.8e-02 | rank:  5.0 | time:   1.030
  # Dixon           [ANO-ALS      ] > error: 2.2e-05 / 3.2e-05 / 5.4e-05 | rank:  5.0 | time:   1.029
  # Exponential     [ANO-ALS      ] > error: 2.9e-09 / 2.6e-08 / 3.2e-08 | rank:  5.0 | time:   0.992
  # Grienwank       [ANO-ALS      ] > error: 3.2e-04 / 1.3e-03 / 1.9e-03 | rank:  5.0 | time:   1.034
  # Michalewicz     [ANO-ALS      ] > error: 1.1e-04 / 1.8e-03 / 2.7e-01 | rank:  5.0 | time:   1.001
  # Qing            [ANO-ALS      ] > error: 3.6e-03 / 4.7e-02 / 2.1e-02 | rank:  5.0 | time:   1.022
  # Rastrigin       [ANO-ALS      ] > error: 1.7e-08 / 3.0e-07 / 9.6e-02 | rank:  5.0 | time:   1.028
  # Rosenbrock      [ANO-ALS      ] > error: 1.5e-02 / 2.3e-02 / 3.6e-02 | rank:  5.0 | time:   0.983
  # Schaffer        [ANO-ALS      ] > error: 7.5e-03 / 8.6e-02 / 2.1e-01 | rank:  5.0 | time:   0.992
  # Schwefel        [ANO-ALS      ] > error: 1.2e-07 / 9.8e-07 / 1.3e-02 | rank:  5.0 | time:   0.999
  # 

And we can try rank-adaptive TT-ALS method with TT-ANOVA:

.. code-block:: python

  for func in teneva.func_demo_all(d=5):
      func.clear()
      func.set_grid(32, kind='cheb')
      func.build_trn_ind(m=1.E+4)
      func.build_tst_ind(m=1.E+5)
      func.build_tst_poi(m=1.E+5)
      func.anova(r=2)
      func.als(nswp=20, log=True)
      func.check()
      func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # # pre | time:      0.004 | rank:   2.0 | 
  # #   1 | time:      0.031 | rank:   2.0 | eps: 8.0e-03 | 
  # #   2 | time:      0.057 | rank:   2.0 | eps: 4.8e-03 | 
  # #   3 | time:      0.084 | rank:   2.0 | eps: 1.1e-03 | 
  # #   4 | time:      0.110 | rank:   2.0 | eps: 4.2e-04 | 
  # #   5 | time:      0.137 | rank:   2.0 | eps: 2.0e-04 | 
  # #   6 | time:      0.162 | rank:   2.0 | eps: 1.1e-04 | 
  # #   7 | time:      0.188 | rank:   2.0 | eps: 6.4e-05 | 
  # #   8 | time:      0.216 | rank:   2.0 | eps: 4.0e-05 | 
  # #   9 | time:      0.245 | rank:   2.0 | eps: 2.7e-05 | 
  # #  10 | time:      0.273 | rank:   2.0 | eps: 2.0e-05 | 
  # #  11 | time:      0.299 | rank:   2.0 | eps: 1.5e-05 | 
  # #  12 | time:      0.325 | rank:   2.0 | eps: 1.2e-05 | 
  # #  13 | time:      0.353 | rank:   2.0 | eps: 9.6e-06 | 
  # #  14 | time:      0.380 | rank:   2.0 | eps: 7.8e-06 | 
  # #  15 | time:      0.405 | rank:   2.0 | eps: 6.4e-06 | 
  # #  16 | time:      0.434 | rank:   2.0 | eps: 5.2e-06 | 
  # #  17 | time:      0.461 | rank:   2.0 | eps: 4.3e-06 | 
  # #  18 | time:      0.489 | rank:   2.0 | eps: 3.5e-06 | 
  # #  19 | time:      0.515 | rank:   2.0 | eps: 2.9e-06 | 
  # #  20 | time:      0.543 | rank:   2.0 | eps: 2.4e-06 | stop: nswp | 
  # Ackley          [ANO-ALS      ] > error: 3.2e-03 / 3.9e-03 / 2.3e-02 | rank:  2.0 | time:   0.553
  # # pre | time:      0.003 | rank:   2.0 | 
  # #   1 | time:      0.030 | rank:   2.0 | eps: 3.3e-02 | 
  # #   2 | time:      0.057 | rank:   2.0 | eps: 2.4e-03 | 
  # #   3 | time:      0.087 | rank:   2.0 | eps: 2.1e-04 | 
  # #   4 | time:      0.114 | rank:   2.0 | eps: 2.4e-05 | 
  # #   5 | time:      0.141 | rank:   2.0 | eps: 3.5e-06 | 
  # #   6 | time:      0.169 | rank:   2.0 | eps: 5.9e-07 | 
  # #   7 | time:      0.197 | rank:   2.0 | eps: 1.1e-07 | 
  # #   8 | time:      0.225 | rank:   2.0 | eps: 2.1e-08 | 
  # #   9 | time:      0.254 | rank:   2.0 | eps: 3.1e-08 | 
  # #  10 | time:      0.281 | rank:   2.0 | eps: 1.9e-08 | 
  # #  11 | time:      0.308 | rank:   2.0 | eps: 3.8e-09 | 
  # #  12 | time:      0.334 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Alpine          [ANO-ALS      ] > error: 9.9e-12 / 1.2e-11 / 4.8e-02 | rank:  2.0 | time:   0.344
  # # pre | time:      0.003 | rank:   2.0 | 
  # #   1 | time:      0.032 | rank:   2.0 | eps: 5.4e-02 | 
  # #   2 | time:      0.060 | rank:   2.0 | eps: 6.6e-03 | 
  # #   3 | time:      0.099 | rank:   2.0 | eps: 4.6e-04 | 
  # #   4 | time:      0.198 | rank:   2.0 | eps: 4.8e-05 | 
  # #   5 | time:      0.240 | rank:   2.0 | eps: 5.9e-06 | 
  # #   6 | time:      0.277 | rank:   2.0 | eps: 7.8e-07 | 
  # #   7 | time:      0.309 | rank:   2.0 | eps: 1.1e-07 | 
  # #   8 | time:      0.339 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Dixon           [ANO-ALS      ] > error: 1.8e-02 / 1.9e-02 / 2.4e-02 | rank:  2.0 | time:   0.349
  # # pre | time:      0.004 | rank:   2.0 | 
  # #   1 | time:      0.037 | rank:   2.0 | eps: 1.0e-01 | 
  # #   2 | time:      0.064 | rank:   2.0 | eps: 5.6e-03 | 
  # #   3 | time:      0.093 | rank:   2.0 | eps: 4.0e-04 | 
  # #   4 | time:      0.120 | rank:   2.0 | eps: 3.6e-05 | 
  # #   5 | time:      0.147 | rank:   2.0 | eps: 3.5e-06 | 
  # #   6 | time:      0.176 | rank:   2.0 | eps: 4.2e-07 | 
  # #   7 | time:      0.205 | rank:   2.0 | eps: 6.9e-08 | 
  # #   8 | time:      0.233 | rank:   2.0 | eps: 1.7e-08 | 
  # #   9 | time:      0.260 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Exponential     [ANO-ALS      ] > error: 1.7e-10 / 2.0e-10 / 2.3e-10 | rank:  2.0 | time:   0.269
  # # pre | time:      0.003 | rank:   2.0 | 
  # #   1 | time:      0.031 | rank:   2.0 | eps: 3.2e-02 | 
  # #   2 | time:      0.058 | rank:   2.0 | eps: 2.4e-03 | 
  # #   3 | time:      0.087 | rank:   2.0 | eps: 1.7e-04 | 
  # #   4 | time:      0.115 | rank:   2.0 | eps: 1.8e-05 | 
  # #   5 | time:      0.142 | rank:   2.0 | eps: 2.4e-06 | 
  # #   6 | time:      0.170 | rank:   2.0 | eps: 3.3e-07 | 
  # #   7 | time:      0.197 | rank:   2.0 | eps: 4.1e-08 | 
  # #   8 | time:      0.228 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Grienwank       [ANO-ALS      ] > error: 5.9e-04 / 6.3e-04 / 1.1e-03 | rank:  2.0 | time:   0.238
  # # pre | time:      0.005 | rank:   2.0 | 
  # #   1 | time:      0.047 | rank:   2.0 | eps: 8.9e-02 | 
  # #   2 | time:      0.085 | rank:   2.0 | eps: 7.3e-03 | 
  # #   3 | time:      0.122 | rank:   2.0 | eps: 9.9e-04 | 
  # #   4 | time:      0.158 | rank:   2.0 | eps: 3.2e-04 | 
  # #   5 | time:      0.188 | rank:   2.0 | eps: 1.4e-04 | 
  # #   6 | time:      0.223 | rank:   2.0 | eps: 6.8e-05 | 
  # #   7 | time:      0.265 | rank:   2.0 | eps: 3.6e-05 | 
  # #   8 | time:      0.304 | rank:   2.0 | eps: 2.0e-05 | 
  # #   9 | time:      0.347 | rank:   2.0 | eps: 1.1e-05 | 
  # #  10 | time:      0.387 | rank:   2.0 | eps: 6.4e-06 | 
  # #  11 | time:      0.428 | rank:   2.0 | eps: 3.7e-06 | 
  # #  12 | time:      0.466 | rank:   2.0 | eps: 2.2e-06 | 
  # #  13 | time:      0.498 | rank:   2.0 | eps: 1.3e-06 | 
  # #  14 | time:      0.526 | rank:   2.0 | eps: 7.5e-07 | 
  # #  15 | time:      0.554 | rank:   2.0 | eps: 4.4e-07 | 
  # #  16 | time:      0.582 | rank:   2.0 | eps: 2.6e-07 | 
  # #  17 | time:      0.609 | rank:   2.0 | eps: 1.6e-07 | 
  # #  18 | time:      0.639 | rank:   2.0 | eps: 9.7e-08 | 
  # #  19 | time:      0.669 | rank:   2.0 | eps: 6.0e-08 | 
  # #  20 | time:      0.697 | rank:   2.0 | eps: 2.3e-08 | stop: nswp | 
  # Michalewicz     [ANO-ALS      ] > error: 2.8e-08 / 4.8e-08 / 2.7e-01 | rank:  2.0 | time:   0.704
  # # pre | time:      0.004 | rank:   2.0 | 
  # #   1 | time:      0.035 | rank:   2.0 | eps: 9.8e-01 | 
  # #   2 | time:      0.063 | rank:   2.0 | eps: 2.2e-01 | 
  # #   3 | time:      0.092 | rank:   2.0 | eps: 9.8e-02 | 
  # #   4 | time:      0.120 | rank:   2.0 | eps: 6.7e-02 | 
  # #   5 | time:      0.150 | rank:   2.0 | eps: 2.8e-02 | 
  # #   6 | time:      0.188 | rank:   2.0 | eps: 1.1e-02 | 
  # #   7 | time:      0.231 | rank:   2.0 | eps: 3.7e-03 | 
  # #   8 | time:      0.274 | rank:   2.0 | eps: 2.1e-03 | 
  # #   9 | time:      0.305 | rank:   2.0 | eps: 1.7e-03 | 
  # #  10 | time:      0.333 | rank:   2.0 | eps: 1.4e-03 | 
  # #  11 | time:      0.361 | rank:   2.0 | eps: 1.2e-03 | 
  # #  12 | time:      0.390 | rank:   2.0 | eps: 9.9e-04 | 
  # #  13 | time:      0.419 | rank:   2.0 | eps: 8.6e-04 | 
  # #  14 | time:      0.449 | rank:   2.0 | eps: 7.5e-04 | 
  # #  15 | time:      0.480 | rank:   2.0 | eps: 6.6e-04 | 
  # #  16 | time:      0.509 | rank:   2.0 | eps: 5.8e-04 | 
  # #  17 | time:      0.537 | rank:   2.0 | eps: 5.2e-04 | 
  # #  18 | time:      0.564 | rank:   2.0 | eps: 4.7e-04 | 
  # #  19 | time:      0.593 | rank:   2.0 | eps: 4.2e-04 | 
  # #  20 | time:      0.623 | rank:   2.0 | eps: 3.8e-04 | stop: nswp | 
  # Qing            [ANO-ALS      ] > error: 7.9e-03 / 8.1e-03 / 6.6e-03 | rank:  2.0 | time:   0.632
  # # pre | time:      0.003 | rank:   2.0 | 
  # #   1 | time:      0.030 | rank:   2.0 | eps: 2.2e-02 | 
  # #   2 | time:      0.059 | rank:   2.0 | eps: 1.5e-03 | 
  # #   3 | time:      0.086 | rank:   2.0 | eps: 1.5e-04 | 
  # #   4 | time:      0.114 | rank:   2.0 | eps: 2.1e-05 | 
  # #   5 | time:      0.140 | rank:   2.0 | eps: 3.7e-06 | 
  # #   6 | time:      0.167 | rank:   2.0 | eps: 7.2e-07 | 
  # #   7 | time:      0.193 | rank:   2.0 | eps: 1.5e-07 | 
  # #   8 | time:      0.222 | rank:   2.0 | eps: 1.2e-08 | 
  # #   9 | time:      0.255 | rank:   2.0 | eps: 3.2e-08 | 
  # #  10 | time:      0.283 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Rastrigin       [ANO-ALS      ] > error: 3.4e-10 / 4.1e-10 / 9.6e-02 | rank:  2.0 | time:   0.292
  # # pre | time:      0.003 | rank:   2.0 | 
  # #   1 | time:      0.032 | rank:   2.0 | eps: 1.7e-01 | 
  # #   2 | time:      0.062 | rank:   2.0 | eps: 4.7e-02 | 
  # #   3 | time:      0.093 | rank:   2.0 | eps: 1.4e-02 | 
  # #   4 | time:      0.122 | rank:   2.0 | eps: 7.4e-03 | 
  # #   5 | time:      0.153 | rank:   2.0 | eps: 4.1e-03 | 
  # #   6 | time:      0.189 | rank:   2.0 | eps: 2.3e-03 | 
  # #   7 | time:      0.231 | rank:   2.0 | eps: 1.4e-03 | 
  # #   8 | time:      0.276 | rank:   2.0 | eps: 8.4e-04 | 
  # #   9 | time:      0.313 | rank:   2.0 | eps: 5.4e-04 | 
  # #  10 | time:      0.347 | rank:   2.0 | eps: 3.7e-04 | 
  # #  11 | time:      0.376 | rank:   2.0 | eps: 2.6e-04 | 
  # #  12 | time:      0.404 | rank:   2.0 | eps: 1.8e-04 | 
  # #  13 | time:      0.432 | rank:   2.0 | eps: 1.3e-04 | 
  # #  14 | time:      0.474 | rank:   2.0 | eps: 9.7e-05 | 
  # #  15 | time:      0.520 | rank:   2.0 | eps: 7.1e-05 | 
  # #  16 | time:      0.566 | rank:   2.0 | eps: 5.2e-05 | 
  # #  17 | time:      0.611 | rank:   2.0 | eps: 3.8e-05 | 
  # #  18 | time:      0.709 | rank:   2.0 | eps: 2.8e-05 | 
  # #  19 | time:      0.752 | rank:   2.0 | eps: 2.1e-05 | 
  # #  20 | time:      0.792 | rank:   2.0 | eps: 1.5e-05 | stop: nswp | 
  # Rosenbrock      [ANO-ALS      ] > error: 1.2e-01 / 1.2e-01 / 1.6e-01 | rank:  2.0 | time:   0.802
  # # pre | time:      0.005 | rank:   2.0 | 
  # #   1 | time:      0.040 | rank:   2.0 | eps: 2.0e-02 | 
  # #   2 | time:      0.068 | rank:   2.0 | eps: 1.1e-02 | 
  # #   3 | time:      0.097 | rank:   2.0 | eps: 1.1e-02 | 
  # #   4 | time:      0.129 | rank:   2.0 | eps: 1.2e-02 | 
  # #   5 | time:      0.159 | rank:   2.0 | eps: 1.4e-02 | 
  # #   6 | time:      0.194 | rank:   2.0 | eps: 1.5e-02 | 
  # #   7 | time:      0.233 | rank:   2.0 | eps: 1.8e-02 | 
  # #   8 | time:      0.270 | rank:   2.0 | eps: 2.1e-02 | 
  # #   9 | time:      0.302 | rank:   2.0 | eps: 2.2e-02 | 
  # #  10 | time:      0.330 | rank:   2.0 | eps: 2.1e-02 | 
  # #  11 | time:      0.359 | rank:   2.0 | eps: 2.0e-02 | 
  # #  12 | time:      0.388 | rank:   2.0 | eps: 2.0e-02 | 
  # #  13 | time:      0.416 | rank:   2.0 | eps: 2.0e-02 | 
  # #  14 | time:      0.445 | rank:   2.0 | eps: 2.1e-02 | 
  # #  15 | time:      0.476 | rank:   2.0 | eps: 2.2e-02 | 
  # #  16 | time:      0.502 | rank:   2.0 | eps: 2.2e-02 | 
  # #  17 | time:      0.532 | rank:   2.0 | eps: 2.3e-02 | 
  # #  18 | time:      0.559 | rank:   2.0 | eps: 2.3e-02 | 
  # #  19 | time:      0.587 | rank:   2.0 | eps: 2.4e-02 | 
  # #  20 | time:      0.615 | rank:   2.0 | eps: 2.4e-02 | stop: nswp | 
  # Schaffer        [ANO-ALS      ] > error: 1.8e-02 / 4.5e-01 / 1.2e+00 | rank:  2.0 | time:   0.626
  # # pre | time:      0.004 | rank:   2.0 | 
  # #   1 | time:      0.031 | rank:   2.0 | eps: 2.4e-02 | 
  # #   2 | time:      0.057 | rank:   2.0 | eps: 1.6e-03 | 
  # #   3 | time:      0.084 | rank:   2.0 | eps: 1.4e-04 | 
  # #   4 | time:      0.112 | rank:   2.0 | eps: 1.7e-05 | 
  # #   5 | time:      0.139 | rank:   2.0 | eps: 2.3e-06 | 
  # #   6 | time:      0.166 | rank:   2.0 | eps: 3.7e-07 | 
  # #   7 | time:      0.194 | rank:   2.0 | eps: 6.7e-08 | 
  # #   8 | time:      0.221 | rank:   2.0 | eps: 0.0e+00 | stop: e | 
  # Schwefel        [ANO-ALS      ] > error: 2.2e-09 / 2.6e-09 / 1.3e-02 | rank:  2.0 | time:   0.230
  # 



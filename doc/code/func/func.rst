func: wrapper for multivariable function with approximation methods
-------------------------------------------------------------------


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
  func.cross(m=1.E+5)

.. code-block:: python

  # Check accuracy of the approximation on the train and test datasets:
  func.check()
  func.info()

  # >>> ----------------------------------------
  # >>> Output:

  # Sphere          [CRO          ] > error: 2.3e-15 / 2.3e-15 | rank:  5.4 | time:   2.276
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
  # Approximated value : 44.6160218744687640
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
  # Approximated value : 1.2500000000000737
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
  func.cross(m=1.E+5)          # Build TT-approximation by TT-CROSS
  func.check_trn_ind()         # Check accuracy on the train data (tensor indices from TT-CROSS)
  func.check_tst_ind()         # Check accuracy on the test data (random tensor indices)
  func.check_tst_poi()         # Check accuracy on the test data (random spatial points)
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # ==================================================
  # ------------------- | Sphere function
  # Method              : CRO
  # 
  # Evals function      : 9.6e+04
  # Evals cache         : 8.0e+04
  # TT-rank             :     5.4
  # Number of params    : 3.2e+03
  # 
  # Samples trn ind     : 9.6e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 2.3e-15
  # Error tst ind       : 1.5e-15
  # Error tst poi       : 2.3e-15
  # 
  # Time approximation  :   2.226
  # Time trn check ind  :   0.279
  # Time tst build ind  :   0.037
  # Time tst check ind  :   0.553
  # Time tst build poi  :   1.089
  # Time tst check poi  :   3.577
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
  # Time approximation  :   0.060
  # Time trn build ind  :   0.037
  # Time trn check ind  :   0.286
  # Time tst build ind  :   0.037
  # Time tst check ind  :   0.572
  # Time tst build poi  :   1.089
  # Time tst check poi  :   3.344
  # 
  # ==================================================
  # 

And we can approximate the function by TT-ALS:

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+5)  # Prepare train data (random tensor indices)
  func.rand(r=2)               # Build initial approximation
  func.als(nswp=30)            # Build TT-approximation by TT-ALS
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

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
  # Error trn ind       : 1.7e-15
  # Error tst ind       : 1.7e-15
  # Error tst poi       : 2.4e-15
  # 
  # Time approximation  :   5.274
  # Time trn build ind  :   0.035
  # Time trn check ind  :   0.258
  # Time tst build ind  :   0.037
  # Time tst check ind  :   0.496
  # Time tst build poi  :   1.089
  # Time tst check poi  :   3.230
  # 
  # Sweeps              :      30
  # ==================================================
  # 

We can approximate the function by TT-ANOVA + TT-ALS (we may combine the approximation methods; in this example we use the result of the TT-ANOVA as an initial approximation for the TT-ALS algorithm):

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+5)  # Prepare train data (random tensor indices)
  func.anova(r=2, order=1)     # Apply TT-ANOVA
  func.als(nswp=30)            # Apply TT-ALS, using TT-ANOVA as initial approximation
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

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
  # Error trn ind       : 1.9e-15
  # Error tst ind       : 2.0e-15
  # Error tst poi       : 2.3e-15
  # 
  # Time approximation  :   5.214
  # Time trn build ind  :   0.036
  # Time trn check ind  :   0.257
  # Time tst build ind  :   0.037
  # Time tst check ind  :   0.559
  # Time tst build poi  :   1.089
  # Time tst check poi  :   3.319
  # 
  # Sweeps              :      30
  # ==================================================
  # 

We can approximate the function by TT-ANOVA + TT-CROSS (note that the total number of function queries, i.e. 1.E+4 + 9.E+4) will be the same as above):

.. code-block:: python

  func.clear()                 # Remove results of the previous approximation
  func.build_trn_ind(m=1.E+4)  # Prepare train data (random tensor indices)
  func.anova(r=2, order=1)     # Apply TT-ANOVA
  func.cross(m=9.E+4)          # Apply TT-CROSS, using TT-ANOVA as initial approximation   
  func.check()                 # Check accuracy on all available data
  func.info_full()             # Print the result

  # >>> ----------------------------------------
  # >>> Output:

  # ==================================================
  # ------------------- | Sphere function
  # Method              : ANO-CRO
  # 
  # Evals function      : 9.4e+04
  # Evals cache         : 6.3e+04
  # TT-rank             :     3.0
  # Number of params    : 1.1e+03
  # 
  # Samples trn ind     : 8.4e+04
  # Samples tst ind     : 1.0e+05
  # Samples tst poi     : 1.0e+05
  # 
  # Error trn ind       : 2.2e-15
  # Error tst ind       : 1.1e-15
  # Error tst poi       : 1.3e-15
  # 
  # Time approximation  :   1.067
  # Time trn check ind  :   0.245
  # Time tst build ind  :   0.037
  # Time tst check ind  :   0.554
  # Time tst build poi  :   1.089
  # Time tst check poi  :   3.459
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

  # [-1.15463195e-14  1.25000000e+00  5.00000000e+00  1.12500000e+01]
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

  # 44.9999999999996
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

  # Ackley          [CRO          ] > error: 3.8e-05 / 3.2e-07 / 1.6e-02 | rank:  9.3 | time:   0.665
  # Brown           [CRO          ] > error: 9.2e-10 / 7.3e-09 / 2.1e-04 | rank:  7.0 | time:   0.760
  # Grienwank       [CRO          ] > error: 3.9e-11 / 3.3e-12 / 1.5e-04 | rank:  4.0 | time:   0.719
  # Michalewicz     [CRO          ] > error: 5.7e-15 / 3.3e-15 / 4.1e-01 | rank:  5.0 | time:   0.910
  # Rastrigin       [CRO          ] > error: 4.5e-14 / 1.9e-14 / 6.9e-02 | rank:  3.5 | time:   0.906
  # Rosenbrock      [CRO          ] > error: 7.5e-15 / 9.1e-15 / 2.2e-14 | rank:  4.4 | time:   0.749
  # Schaffer        [CRO          ] > error: 1.5e-04 / 8.3e-04 / 5.5e-02 | rank: 12.8 | time:   0.727
  # Schwefel        [CRO          ] > error: 3.1e-14 / 1.3e-14 / 9.2e-03 | rank:  2.8 | time:   0.722
  # 

And for the finer grid and more requests to the target function, we will have the more accurate result:

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

  # Ackley          [CRO          ] > error: 1.9e-03 / 1.9e-06 / 3.5e-04 | rank:  9.8 | time:  10.332
  # Brown           [CRO          ] > error: 4.3e-09 / 7.9e-09 / 4.4e-08 | rank:  9.1 | time:  10.938
  # Grienwank       [CRO          ] > error: 7.1e-12 / 2.2e-12 / 1.4e-04 | rank:  6.0 | time:  10.171
  # Michalewicz     [CRO          ] > error: 5.4e-15 / 2.3e-15 / 7.2e-04 | rank:  2.6 | time:   8.215
  # Rastrigin       [CRO          ] > error: 1.0e-13 / 1.5e-14 / 2.6e-14 | rank:  5.6 | time:   9.141
  # Rosenbrock      [CRO          ] > error: 2.2e-14 / 1.0e-14 / 2.2e-14 | rank:  5.8 | time:   8.772
  # Schaffer        [CRO          ] > error: 1.3e-02 / 2.2e-02 / 2.9e-02 | rank: 14.0 | time:   9.661
  # Schwefel        [CRO          ] > error: 4.7e-14 / 1.3e-14 / 1.3e-05 | rank:  4.3 | time:   8.757
  # 



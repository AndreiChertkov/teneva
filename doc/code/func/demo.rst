demo: analytical functions for demo and tests
---------------------------------------------


.. autofunction:: teneva.func_demo_all

  **Examples**:

  .. code-block:: python

    funcs = teneva.func_demo_all(d=5, with_piston=True)
    for func in funcs:
        print(func.name)

    # >>> ----------------------------------------
    # >>> Output:

    # Ackley
    # Alpine
    # Brown
    # Dixon
    # Exponential
    # Grienwank
    # Michalewicz
    # Piston
    # Qing
    # Rastrigin
    # Rosenbrock
    # Schaffer
    # Schwefel
    # 

  We can manually specify the list of names of the desired functions (a complete list of available benchmarks is given in the documentation for the "func_demo_all" function):

  .. code-block:: python

    funcs = teneva.func_demo_all(d=4, names=['Ackley', 'rosenbrock', 'PISTON'])
    for func in funcs:
        print(func.name)

    # >>> ----------------------------------------
    # >>> Output:

    # Ackley
    # Piston
    # Rosenbrock
    # 


-----


.. autoclass:: teneva.FuncDemoAckley
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoAckley(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.         3.62538494 6.59359908 9.02376728]
  # 0.0 9.023767278119472
  # 

.. code-block:: python

  teneva.FuncDemoAckley(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoAlpine
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoAlpine(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.         2.82441295 6.05578456 2.17008007]
  # 0.0 2.1700800725388047
  # 

.. code-block:: python

  teneva.FuncDemoAlpine(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoBrown
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoBrown(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.00000000e+00 4.00000000e+00 4.09600000e+03 1.39471376e+10]
  # 0.0 13947137604.0
  # 

.. code-block:: python

  teneva.FuncDemoBrown(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


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
  Func = teneva.FuncDemoDixon(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [  1.   0.  21. 184.]
  # 1.0 184.0
  # 

.. code-block:: python

  teneva.FuncDemoDixon(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoExponential
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoExponential(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [-1.00000000e+00 -2.23130160e-01 -2.47875218e-03 -1.37095909e-06]
  # -1.0 -1.3709590863840845e-06
  # 

.. code-block:: python

  teneva.FuncDemoExponential(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoGrienwank
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoGrienwank(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.         0.65656774 1.02923026 1.08990201]
  # 0.0 1.0899020113755438
  # 

.. code-block:: python

  teneva.FuncDemoGrienwank(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoMichalewicz
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoMichalewicz(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [-0.00000000e+00 -1.45382977e-02 -3.70232531e-01 -3.26333212e-04]
  # -0.0 -0.000326333211876712
  # 

.. code-block:: python

  teneva.FuncDemoMichalewicz(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoPiston
  :members: 

**Examples**:

.. code-block:: python

  Func = teneva.FuncDemoPiston()
  X_spec = (Func.a + Func.b) / 2
  X_spec = X_spec.reshape(1, -1)
  print(Func.get_f_poi(X_spec))
  print(Func.get_f_poi(X_spec[0]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.46439702]
  # 0.4643970224718025
  # 


-----


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
  Func = teneva.FuncDemoQing(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [ 14.   5.  14. 149.]
  # 14.0 149.0
  # 

.. code-block:: python

  teneva.FuncDemoQing(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoRastrigin
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoRastrigin(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [ 0.  3. 12. 27.]
  # 0.0 27.0
  # 

.. code-block:: python

  teneva.FuncDemoRastrigin(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoRosenbrock
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoRosenbrock(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [2.000e+00 0.000e+00 8.020e+02 7.208e+03]
  # 2.0 7208.0
  # 

.. code-block:: python

  teneva.FuncDemoRosenbrock(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoSchaffer
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoSchaffer(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [0.         1.94756906 0.20262542 1.56950769]
  # 0.0 1.5695076886237076
  # 

.. code-block:: python

  teneva.FuncDemoSchaffer(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.


-----


.. autoclass:: teneva.FuncDemoSchwefel
  :members: 

**Examples**:

.. code-block:: python

  X = np.array([
      [0., 0., 0.],
      [1., 1., 1.],
      [2., 2., 2.],
      [3., 3., 3.],
  ])
  Func = teneva.FuncDemoSchwefel(d=X.shape[1])
  print(Func.get_f_poi(X))
  print(Func.get_f_poi(X[0]), Func.get_f_poi(X[-1]))

  # >>> ----------------------------------------
  # >>> Output:

  # [1256.9487     1254.42428705 1251.02210432 1248.0654602 ]
  # 1256.9487 1248.0654601950866
  # 

.. code-block:: python

  teneva.FuncDemoSchwefel(d=2).plot()

  # >>> ----------------------------------------
  # >>> Output:

  # <Figure size 720x720 with 2 Axes>
  # 

  # >>> ----------------------------------------
  # >>> Output:

  # Display of images is not supported in the docs. See related ipynb file.



Module func_builder: helpers to build benchmarks
------------------------------------------------


.. automodule:: teneva.func.func_builder


-----


.. autofunction:: teneva.func_demo

  **Examples**:

  .. code-block:: python

    func = teneva.func_demo(d=5, name='Ackley')
    print(func.name)    # Name of the function
    print(func.a)       # Grid lower bound
    print(func.b)       # Grid upper bound
    print(func.x_min)   # Argument for exact minimum
    print(func.y_min)   # Value of exact minimum

    # >>> ----------------------------------------
    # >>> Output:

    # Ackley
    # [-32.768 -32.768 -32.768 -32.768 -32.768]
    # [32.768 32.768 32.768 32.768 32.768]
    # [0. 0. 0. 0. 0.]
    # 0.0
    # 


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

  We can also collect a list of functions for which the explicit form of their TT-cores is known:

  .. code-block:: python

    funcs = teneva.func_demo_all(d=100, only_with_cores=True)
    for func in funcs:
        print(func.name)

    # >>> ----------------------------------------
    # >>> Output:

    # Alpine
    # Exponential
    # Grienwank
    # Michalewicz
    # Qing
    # Rastrigin
    # Rosenbrock
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



Module data: functions for working with datasets
------------------------------------------------


.. automodule:: teneva.data


-----




|
|

.. autofunction:: teneva.data.accuracy_on_data

  **Examples**:

  .. code-block:: python

    m = 100       # Size of the dataset
    n = [5] * 10  # Shape of the tensor
    
    # Random TT-tensor with TT-rank 2:
    Y = teneva.rand(n, 2)
    
    # Let build toy dataset:
    I_data = teneva.sample_lhs(n, m)
    y_data = [teneva.get(Y, i) for i in I_data]
    y_data = np.array(y_data)
    
    # Add add some noise:
    y_data = y_data + 1.E-3*np.random.randn(m)
    
    # Compute the accuracy:
    eps = teneva.accuracy_on_data(Y, I_data, y_data)
    
    print(f'Accuracy     : {eps:-8.2e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Accuracy     : 3.09e-03
    # 




|
|

.. autofunction:: teneva.data.cache_to_data

  **Examples**:

  Let apply TT-cross for benchmark function:

  .. code-block:: python

    a = [-5., -4., -3., -2., -1.] # Lower bounds for spatial grid
    b = [+6., +3., +3., +1., +2.] # Upper bounds for spatial grid
    n = [ 20,  18,  16,  14,  12] # Shape of the tensor
    m = 8.E+3                     # Number of calls to function
    r = 3                         # TT-rank of the initial tensor
    
    from scipy.optimize import rosen
    def func(I): 
        X = teneva.ind_to_poi(I, a, b, n)
        return rosen(X.T)
    
    cache = {}
    Y = teneva.rand(n, r)
    Y = teneva.cross(func, Y, m, cache=cache)

  Now cache contains the requested function values and related tensor multi-indices:

  .. code-block:: python

    I_trn, y_trn = teneva.cache_to_data(cache)
    
    print(I_trn.shape)
    print(y_trn.shape)
    
    i = I_trn[0, :] # The 1th multi-index
    y = y_trn[0]    # Saved value in cache
    
    print(i)
    print(y)
    print(func(i))

    # >>> ----------------------------------------
    # >>> Output:

    # (7956, 5)
    # (7956,)
    # [0 0 0 4 3]
    # 130615.73557017733
    # 130615.73557017733
    # 




|
|


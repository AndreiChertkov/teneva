Module data: functions for working with datasets
------------------------------------------------


.. automodule:: teneva.core.data


-----




|
|

.. autofunction:: teneva.accuracy_on_data

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

